import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, best_val_loss):
    # Unwrap torch.compile if needed
    raw_model = getattr(model, '_orig_mod', model)
    checkpoint = {
        "epoch": epoch,
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "best_val_loss": best_val_loss,
        "rng_state": {
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Unwrap torch.compile if needed
    raw_model = getattr(model, '_orig_mod', model)
    raw_model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and checkpoint.get("optimizer_state"):
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scheduler is not None and checkpoint.get("scheduler_state"):
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    rng = checkpoint.get("rng_state", {})
    try:
        if rng.get("torch") is not None:
            torch.random.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("python") is not None:
            random.setstate(rng["python"])
    except Exception:
        pass  # Non-fatal: RNG mismatch on device change

    return checkpoint


# ── Loss helpers ──────────────────────────────────────────────────────────────

def entropy_loss(logits):
    """Mean pixel entropy over a batch (for UDA)."""
    probs = torch.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return entropy.mean()


def make_criterion(n_classes=2, label_smoothing=0.1):
    """Cross-entropy with optional label smoothing (smoothing=0 → standard CE)."""
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


# ── Training loop helpers ─────────────────────────────────────────────────────

def _autocast_ctx(device, use_amp):
    if use_amp:
        return torch.amp.autocast(device_type=device.type, dtype=torch.float16)
    return nullcontext()


def train_one_epoch(model, dataloader, optimizer, criterion, device,
                    use_amp=False, scaler=None, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Training", leave=False, colour="green")

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)
        if masks.dim() == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

        with _autocast_ctx(device, use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)

        if use_amp and scaler is not None:
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

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation", leave=False, colour="cyan"):
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)
            if masks.dim() == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)

            with _autocast_ctx(device, use_amp):
                outputs = model(images)
                loss = criterion(outputs, masks)

            total_loss += loss.item()

    return total_loss / len(dataloader)