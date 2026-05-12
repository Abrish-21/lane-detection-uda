from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.logger import get_logger, success
from torch.utils.tensorboard import SummaryWriter
from .trainer_utils import load_checkpoint, save_checkpoint, train_one_epoch, validate

def train_supervised(model, train_loader, val_loader, config, device, save_path, logger=None):
    logger = logger or get_logger()
    epochs = config['training']['num_epochs']
    lr = config['training']['learning_rate']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_dir = Path(config.get('output_dir', '.'))
    checkpoint_dir = output_dir / 'results' / 'checkpoints' / 'supervised'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_interval = config.get('training', {}).get('save_interval')
    if save_interval is None:
        save_interval = config.get('logging', {}).get('save_interval', 1)

    writer = SummaryWriter(log_dir=config['logging']['log_dir'])

    best_val_loss = float('inf')
    start_epoch = 0
    resume_from = config.get('training', {}).get('resume_from')
    resume_path = None
    if resume_from and str(resume_from).lower() != 'auto':
        resume_path = Path(resume_from)
    else:
        last_checkpoint = checkpoint_dir / 'last.pth'
        if last_checkpoint.exists():
            resume_path = last_checkpoint

    if resume_path is not None:
        if resume_path.exists():
            checkpoint = load_checkpoint(resume_path, model, optimizer, scheduler, device=device)
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
            logger.info("Resuming supervised training from %s (epoch %d)", resume_path, start_epoch + 1)
        else:
            logger.warning("Resume checkpoint not found at %s. Starting fresh.", resume_path)

    for epoch in range(start_epoch, epochs):
        logger.info("Epoch %d/%d", epoch + 1, epochs)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)

        logger.info("Train Loss=%.4f | Val Loss=%.4f", train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            success(logger, "Saved best supervised model to %s", save_path)

        if save_interval and (epoch + 1) % save_interval == 0:
            interval_path = checkpoint_dir / f"epoch_{epoch + 1:04d}.pth"
            save_checkpoint(interval_path, model, optimizer, scheduler, epoch, best_val_loss)

        last_path = checkpoint_dir / 'last.pth'
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_val_loss)

    writer.close()