"""Multitask vision training script.

Usage:
    uv run python train.py --config configs/det_seg_depth.py
"""
import argparse
import importlib.util
import os
import time

import torch

import multitask_vision  # noqa: F401 — trigger registration
from multitask_vision.model import MultitaskVisionModel
from multitask_vision.data.multi_dataset import MultiDatasetLoader
from multitask_vision.registry import DATASETS


def load_config(config_path: str) -> dict:
    spec = importlib.util.spec_from_file_location('config', config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    return {k: getattr(cfg_module, k) for k in dir(cfg_module) if not k.startswith('_')}


def build_datasets(data_cfg: dict) -> list:
    """Build datasets from config using the DATASETS registry."""
    datasets = []
    for ds_cfg in data_cfg['datasets']:
        ds_cfg = dict(ds_cfg)  # copy to avoid mutating config
        ds_type = ds_cfg.pop('type')
        datasets.append(DATASETS.build(dict(type=ds_type, **ds_cfg)))
    return datasets


def train(config_path: str, work_dir: str = 'work_dirs'):
    cfg = load_config(config_path)

    device = torch.device(
        'mps' if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available()
        else 'cpu'
    )
    print(f'Using device: {device}')

    model = MultitaskVisionModel(
        blocks=cfg['model']['blocks'], losses=cfg['model']['losses'],
    ).to(device)
    print(f'Model blocks: {list(model.block_modules.keys())}')
    print(f'Topological order: {model.topo_order}')

    datasets = build_datasets(cfg['data'])
    for ds in datasets:
        print(f'Dataset: {ds.__class__.__name__}, size: {len(ds)}, tasks: {ds.tasks}')

    train_cfg = cfg['training']

    # Apply QAT if configured
    qat_cfg = train_cfg.get('qat')
    if qat_cfg:
        from multitask_vision.compression import apply_qat
        apply_qat(model, qat_cfg)
        print(f'QAT enabled: {qat_cfg}')

    loader = MultiDatasetLoader(
        datasets,
        batch_size=train_cfg['batch_size'],
        num_workers=train_cfg.get('num_workers', 4),
        sampling_strategy=train_cfg.get('sampling_strategy', 'proportional'),
    )

    opt_cfg = train_cfg['optimizer']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.get('lr', 1e-4),
        weight_decay=opt_cfg.get('weight_decay', 0.01),
    )

    max_iters = train_cfg['max_iters']
    log_interval = train_cfg.get('log_interval', 50)
    save_interval = train_cfg.get('save_interval', 5000)
    grad_clip = train_cfg.get('grad_clip')

    model.train()
    print(f'\nStarting training for {max_iters} iterations...\n')

    for step in range(1, max_iters + 1):
        t0 = time.time()
        batch = next(loader)

        inputs = batch['inputs'].to(device)
        data_samples = batch['data_samples']

        losses = model(inputs, data_samples, mode='loss')
        total_loss = sum(v for k, v in losses.items() if 'loss' in k)

        optimizer.zero_grad()
        total_loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), **grad_clip)
        optimizer.step()

        if step % log_interval == 0:
            loss_str = ', '.join(f'{k}: {v.item():.4f}' for k, v in losses.items())
            tasks = data_samples[0]['tasks']
            dt = time.time() - t0
            print(f'[{step}/{max_iters}] tasks={tasks} | {loss_str} | total={total_loss.item():.4f} | {dt:.2f}s')

        if step % save_interval == 0:
            os.makedirs(work_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(work_dir, f'step_{step}.pth'))

    os.makedirs(work_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(work_dir, 'latest.pth'))
    print('\nTraining complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--work-dir', type=str, default='work_dirs')
    args = parser.parse_args()
    train(args.config, args.work_dir)
