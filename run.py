#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single entry point for the Lane Detection UDA pipeline.
Usage:
    python run.py --config configs/default.yaml --mode all
Modes:
    all          - run supervised, adaptation, evaluation, conversion, report
    supervised   - only train supervised model
    adaptation   - only train adaptation (requires supervised model)
    evaluate     - evaluate models and select final model
    convert      - convert final model to ONNX & TorchScript
    report       - generate HTML performance report
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import LaneSegmentationDataset, load_split_txt
from src.data.transforms import get_train_transform, get_val_transform
from src.models.unet_resnet import UNetWithResNetEncoder
from src.training.supervised_trainer import train_supervised
from src.training.adaptation_trainer import train_adaptation
from src.evaluation.evaluate import evaluate_model
from src.utils.logger import setup_logger, banner, section, success
from scripts.convert_models import convert_to_onnx, convert_to_torchscript
from scripts.generate_report import generate_html_report

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def convert_numerics(config):
    """Convert numeric strings in config to appropriate types."""
    if isinstance(config['training'].get('learning_rate'), str):
        config['training']['learning_rate'] = float(config['training']['learning_rate'])
    if isinstance(config['uda'].get('learning_rate'), str):
        config['uda']['learning_rate'] = float(config['uda']['learning_rate'])
    int_fields = ['batch_size', 'num_epochs', 'seed', 'img_height', 'img_width', 'n_classes', 'save_interval']
    for field in int_fields:
        for section in ['data', 'training', 'model', 'logging']:
            if section in config and field in config[section]:
                if isinstance(config[section][field], str):
                    config[section][field] = int(config[section][field])
    return config

def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        config = convert_numerics(config)
    set_seed(config['training']['seed'])
    device = torch.device(config['training']['device']
                          if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        if config.get('training', {}).get('enable_cudnn_benchmark', True):
            torch.backends.cudnn.benchmark = True
        if config.get('training', {}).get('enable_tf32', True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    logger = setup_logger(log_dir=config['logging']['log_dir'])
    banner(logger, "Lane Detection Pipeline", f"Mode={args.mode} • Device={device}")

    # -------------------- Data preparation ------------------------------------------
    data_root = config['data']['root']
    splits_root = config['data']['splits_root']
    img_h, img_w = config['data']['img_height'], config['data']['img_width']

    # Load split files (format: image_path [mask_path])
    source_train_samples = load_split_txt(os.path.join(splits_root, 'source_train.txt'), has_labels=True)
    source_val_samples   = load_split_txt(os.path.join(splits_root, 'source_val.txt'), has_labels=True)
    target_train_samples = load_split_txt(os.path.join(splits_root, 'target_train.txt'), has_labels=False)
    target_val_samples   = load_split_txt(os.path.join(splits_root, 'target_val.txt'), has_labels=True)
    target_test_samples  = load_split_txt(os.path.join(splits_root, 'target_test.txt'), has_labels=True)
    logger.info(
        "Dataset loaded: source(train=%d, val=%d) | target(train=%d, val=%d, test=%d)",
        len(source_train_samples), len(source_val_samples),
        len(target_train_samples), len(target_val_samples), len(target_test_samples)
    )

    # Transforms
    train_transform = get_train_transform(img_h, img_w)
    val_transform   = get_val_transform(img_h, img_w)

    # Datasets
    source_train_dataset = LaneSegmentationDataset(data_root, source_train_samples,
                                                   transform=train_transform, is_labeled=True)
    source_val_dataset   = LaneSegmentationDataset(data_root, source_val_samples,
                                                   transform=val_transform, is_labeled=True)
    target_train_dataset = LaneSegmentationDataset(data_root, target_train_samples,
                                                   transform=train_transform, is_labeled=False)
    target_val_dataset   = LaneSegmentationDataset(data_root, target_val_samples,
                                                   transform=val_transform, is_labeled=True)
    target_test_dataset  = LaneSegmentationDataset(data_root, target_test_samples,
                                                   transform=val_transform, is_labeled=True)
    # Dataloaders
    batch_size = config['training']['batch_size']
    data_cfg = config.get('data', {})
    num_workers = int(data_cfg.get('num_workers', 2))
    pin_memory = bool(data_cfg.get('pin_memory', device.type == 'cuda'))
    persistent_workers = bool(data_cfg.get('persistent_workers', num_workers > 0))
    prefetch_factor = int(data_cfg.get('prefetch_factor', 2))

    def build_loader(dataset, shuffle):
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
        }
        if pin_memory:
            loader_kwargs['pin_memory'] = True
        if num_workers > 0:
            loader_kwargs['persistent_workers'] = persistent_workers
            loader_kwargs['prefetch_factor'] = prefetch_factor
        return DataLoader(dataset, **loader_kwargs)

    source_train_loader = build_loader(source_train_dataset, shuffle=True)
    source_val_loader = build_loader(source_val_dataset, shuffle=False)
    target_train_loader = build_loader(target_train_dataset, shuffle=True)
    target_val_loader = build_loader(target_val_dataset, shuffle=False)
    target_test_loader = build_loader(target_test_dataset, shuffle=False)

    # ----- Model definition -------------------------------------------------
    n_classes = config['model']['n_classes']
    model = UNetWithResNetEncoder(n_classes=n_classes).to(device)

    # ----- Output directories -----------------------------------------------
    output_dir = Path(config.get('output_dir', '.'))
    model_dir = output_dir / 'models' / 'pretrained'
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / 'results' / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / 'results' / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)

    best_source_path = model_dir / 'best_source_model.pth'
    best_adapted_path = model_dir / 'best_adapted_model.pth'
    final_model_path = model_dir / 'final_lane_model.pth'

    # ----- Step 1: Supervised training --------------------------------------
    if args.mode in ('all', 'supervised'):
        section(logger, "Step 1/5 • Supervised training")
        train_supervised(model, source_train_loader, source_val_loader,
                         config, device, best_source_path, logger=logger)
    else:
        if best_source_path.exists():
            model.load_state_dict(torch.load(best_source_path, map_location=device))
            success(logger, "Loaded supervised model from %s", best_source_path)
        else:
            raise FileNotFoundError("Supervised model not found. Run supervised training first.")

    # ----- Step 2: Adaptation -----------------------------------------------
    if args.mode in ('all', 'adaptation'):
        section(logger, "Step 2/5 • Domain adaptation")
        adapt_model = UNetWithResNetEncoder(n_classes=n_classes).to(device)
        adapt_model.load_state_dict(torch.load(best_source_path, map_location=device))
        train_adaptation(adapt_model, source_train_loader, target_train_loader,
                         target_val_loader, config, device, best_adapted_path, logger=logger)
    else:
        if best_adapted_path.exists():
            adapt_model = UNetWithResNetEncoder(n_classes=n_classes).to(device)
            adapt_model.load_state_dict(torch.load(best_adapted_path, map_location=device))
            success(logger, "Loaded adapted model from %s", best_adapted_path)
        else:
            adapt_model = None

    # ----- Step 3: Evaluation & final model selection -----------------------
    if args.mode in ('all', 'evaluate'):
        section(logger, "Step 3/5 • Evaluation and model selection")
        # Evaluate source model
        source_model = UNetWithResNetEncoder(n_classes=n_classes).to(device)
        source_model.load_state_dict(torch.load(best_source_path, map_location=device))
        source_metrics = evaluate_model(source_model, target_test_loader,
                                        device, viz_dir, tag='source', logger=logger)
        with open(metrics_dir / 'source_metrics.json', 'w') as f:
            json.dump(source_metrics, f, indent=2)

        # Evaluate adapted model if exists
        if best_adapted_path.exists():
            adapted_metrics = evaluate_model(adapt_model, target_test_loader,
                                             device, viz_dir, tag='adapted', logger=logger)
            with open(metrics_dir / 'adapted_metrics.json', 'w') as f:
                json.dump(adapted_metrics, f, indent=2)
        else:
            adapted_metrics = None

        # Select final model (higher IoU on target test set)
        if adapted_metrics and adapted_metrics['iou'] >= source_metrics['iou']:
            final_model = adapt_model
            final_metrics = adapted_metrics
            success(logger, "Selected adapted model as final.")
        else:
            final_model = source_model
            final_metrics = source_metrics
            success(logger, "Selected source-only model as final.")

        # Save final model and metrics
        torch.save(final_model.state_dict(), final_model_path)
        with open(metrics_dir / 'final_metrics.json', 'w') as f:
            json.dump(final_metrics, f, indent=2)
    else:
        # If not evaluating, load final model if it exists, otherwise fallback
        if final_model_path.exists():
            final_model = UNetWithResNetEncoder(n_classes=n_classes).to(device)
            final_model.load_state_dict(torch.load(final_model_path, map_location=device))
        else:
            final_model = adapt_model if best_adapted_path.exists() else model
            torch.save(final_model.state_dict(), final_model_path)

    # ----- Step 4: Model conversion -----------------------------------------
    if args.mode in ('all', 'convert'):
        section(logger, "Step 4/5 • Export ONNX and TorchScript")
        input_shape = (img_h, img_w)
        onnx_path = model_dir / 'final_lane_model.onnx'
        torchscript_path = model_dir / 'final_lane_model.pt'
        convert_to_onnx(final_model, input_shape, onnx_path, device, logger=logger)
        convert_to_torchscript(final_model, input_shape, torchscript_path, device, logger=logger)

    # ----- Step 5: Generate HTML report -------------------------------------
    if args.mode in ('all', 'report'):
        section(logger, "Step 5/5 • Build HTML report")
        report_path = output_dir / 'results' / 'reports' / 'evaluation_report.html'
        generate_html_report(metrics_dir, viz_dir, report_path, logger=logger)

    success(logger, "Pipeline finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane Detection UDA Pipeline")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to configuration YAML file")
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'supervised', 'adaptation',
                                 'evaluate', 'convert', 'report'],
                        help="Which step to execute")
    args = parser.parse_args()
    main(args)