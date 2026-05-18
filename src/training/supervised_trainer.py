from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from src.utils.logger import get_logger, success
from .trainer_utils import (
    load_checkpoint, save_checkpoint,
    train_one_epoch, validate, make_criterion,
)


def train_supervised(model, train_loader, val_loader, config, device, save_path, logger=None):
    logger = logger or get_logger()

    train_cfg = config.get('training', {})
    epochs        = train_cfg['num_epochs']
    lr            = train_cfg['learning_rate']
    use_amp       = bool(train_cfg.get('use_amp', False)) and device.type == 'cuda'
    grad_clip     = float(train_cfg.get('grad_clip', 1.0))
    label_smooth  = float(train_cfg.get('label_smoothing', 0.1))
    warmup_epochs = int(train_cfg.get('warmup_epochs', 3))
    use_compile   = bool(train_cfg.get('torch_compile', False)) and hasattr(torch, 'compile')
    weight_decay  = float(train_cfg.get('weight_decay', 1e-4))

    output_dir = Path(config.get('output_dir', '.'))
    checkpoint_dir = output_dir / 'results' / 'checkpoints' / 'supervised'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_interval = train_cfg.get('save_interval') or config.get('logging', {}).get('save_interval', 5)

    # ── Loss, optimiser, scheduler ────────────────────────────────────────────
    criterion = make_criterion(
        n_classes=config['model']['n_classes'],
        label_smoothing=label_smooth,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Linear warmup → cosine anneal
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=lr * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                              milestones=[warmup_epochs])

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Optional torch.compile ────────────────────────────────────────────────
    if use_compile:
        logger.info("Compiling model with torch.compile (mode='reduce-overhead') …")
        model = torch.compile(model, mode='reduce-overhead')

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=config['logging']['log_dir'])

    # ── Resume ────────────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    start_epoch   = 0

    resume_from = train_cfg.get('resume_from')
    resume_path = None
    if resume_from and str(resume_from).lower() != 'auto':
        resume_path = Path(resume_from)
    else:
        last_ckpt = checkpoint_dir / 'last.pth'
        if last_ckpt.exists():
            resume_path = last_ckpt

    if resume_path is not None:
        if resume_path.exists():
            ckpt = load_checkpoint(resume_path, model, optimizer, scheduler, device=device)
            start_epoch   = ckpt.get('epoch', -1) + 1
            best_val_loss = ckpt.get('best_val_loss', best_val_loss)
            logger.info("Resuming supervised training from %s (epoch %d/%d)",
                        resume_path, start_epoch + 1, epochs)
        else:
            logger.warning("Resume checkpoint not found at %s — starting fresh.", resume_path)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        current_lr = optimizer.param_groups[0]['lr']
        logger.info("Epoch %d/%d  |  lr=%.2e", epoch + 1, epochs, current_lr)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            use_amp=use_amp, scaler=scaler, grad_clip=grad_clip,
        )
        val_loss = validate(model, val_loader, criterion, device, use_amp=use_amp)
        scheduler.step()

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val',   val_loss,   epoch)
        writer.add_scalar('LR',         current_lr, epoch)
        logger.info("Train Loss=%.4f | Val Loss=%.4f", train_loss, val_loss)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            raw_model = getattr(model, '_orig_mod', model)
            torch.save(raw_model.state_dict(), save_path)
            success(logger, "New best supervised model → %s", save_path)

        # Periodic checkpoint
        if save_interval and (epoch + 1) % save_interval == 0:
            save_checkpoint(checkpoint_dir / f"epoch_{epoch + 1:04d}.pth",
                            model, optimizer, scheduler, epoch, best_val_loss)

        # Always save last
        save_checkpoint(checkpoint_dir / 'last.pth',
                        model, optimizer, scheduler, epoch, best_val_loss)

    writer.close()