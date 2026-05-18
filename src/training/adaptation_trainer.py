from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm

from src.utils.logger import get_logger, success
from .trainer_utils import (
    entropy_loss, load_checkpoint, save_checkpoint,
    validate, make_criterion, _autocast_ctx,
)


def train_adaptation(model, src_loader, tgt_loader, val_loader,
                     config, device, save_path, logger=None):
    logger = logger or get_logger()

    train_cfg     = config.get('training', {})
    uda_cfg       = config.get('uda', {})
    epochs        = train_cfg['num_epochs']
    alpha         = float(uda_cfg.get('alpha', 0.1))
    lr            = float(uda_cfg.get('learning_rate', 5e-5))
    use_amp       = bool(train_cfg.get('use_amp', False)) and device.type == 'cuda'
    grad_clip     = float(train_cfg.get('grad_clip', 1.0))
    label_smooth  = float(train_cfg.get('label_smoothing', 0.1))
    warmup_epochs = int(train_cfg.get('warmup_epochs', 2))
    weight_decay  = float(train_cfg.get('weight_decay', 1e-4))

    output_dir = Path(config.get('output_dir', '.'))
    checkpoint_dir = output_dir / 'results' / 'checkpoints' / 'adaptation'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_interval = train_cfg.get('save_interval') or config.get('logging', {}).get('save_interval', 5)

    # ── Loss, optimiser, scheduler ────────────────────────────────────────────
    criterion = make_criterion(
        n_classes=config['model']['n_classes'],
        label_smoothing=label_smooth,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=lr * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                              milestones=[warmup_epochs])

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

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
            logger.info("Resuming adaptation from %s (epoch %d/%d)",
                        resume_path, start_epoch + 1, epochs)
        else:
            logger.warning("Resume checkpoint not found at %s — starting fresh.", resume_path)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        model.train()
        total_sup_loss = 0.0
        total_ent_loss = 0.0

        num_batches = min(len(src_loader), len(tgt_loader))
        src_iter    = iter(src_loader)
        tgt_iter    = iter(tgt_loader)

        current_lr = optimizer.param_groups[0]['lr']
        pbar = tqdm(
            range(num_batches),
            desc=f"Adaptation {epoch + 1}/{epochs}",
            leave=False,
            colour="green",
        )

        for _ in pbar:
            try:
                src_images, src_masks = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                src_images, src_masks = next(src_iter)
            try:
                tgt_images, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_images, _ = next(tgt_iter)

            src_images = src_images.to(device, non_blocking=True)
            src_masks  = src_masks.to(device, non_blocking=True)
            if src_masks.dim() == 4 and src_masks.shape[1] == 1:
                src_masks = src_masks.squeeze(1)
            tgt_images = tgt_images.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with _autocast_ctx(device, use_amp):
                src_logits = model(src_images)
                sup_loss   = criterion(src_logits, src_masks)

                tgt_logits = model(tgt_images)
                ent        = entropy_loss(tgt_logits)

                loss = sup_loss + alpha * ent

            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_sup_loss += sup_loss.item()
            total_ent_loss += ent.item()
            pbar.set_postfix(sup=f"{sup_loss.item():.4f}", ent=f"{ent.item():.4f}")

        avg_sup = total_sup_loss / num_batches
        avg_ent = total_ent_loss / num_batches
        logger.info(
            "Epoch %d/%d | lr=%.2e | Sup=%.4f | Ent=%.4f",
            epoch + 1, epochs, current_lr, avg_sup, avg_ent,
        )

        val_loss = validate(model, val_loader, criterion, device, use_amp=use_amp)
        logger.info("Target Val Loss=%.4f", val_loss)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            raw_model = getattr(model, '_orig_mod', model)
            torch.save(raw_model.state_dict(), save_path)
            success(logger, "New best adapted model → %s", save_path)

        if save_interval and (epoch + 1) % save_interval == 0:
            save_checkpoint(checkpoint_dir / f"epoch_{epoch + 1:04d}.pth",
                            model, optimizer, scheduler, epoch, best_val_loss)

        save_checkpoint(checkpoint_dir / 'last.pth',
                        model, optimizer, scheduler, epoch, best_val_loss)