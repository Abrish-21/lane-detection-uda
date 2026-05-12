import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, best_val_loss):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    optimizer_state = checkpoint.get("optimizer_state")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = checkpoint.get("scheduler_state")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    rng_state = checkpoint.get("rng_state", {})
    torch_state = rng_state.get("torch")
    if torch_state is not None:
        torch.random.set_rng_state(torch_state)

    cuda_state = rng_state.get("cuda")
    if torch.cuda.is_available() and cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)

    numpy_state = rng_state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)

    python_state = rng_state.get("python")
    if python_state is not None:
        random.setstate(python_state)

    return checkpoint

def train_one_epoch(model, dataloader, optimizer, criterion, device, use_amp=False, scaler=None):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Training", colour="green")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        if masks.dim() == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation", colour="green"):
            images = images.to(device)
            masks = masks.to(device)
            if masks.dim() == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)
            if use_amp:
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def entropy_loss(logits):
    probs = torch.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return entropy.mean()