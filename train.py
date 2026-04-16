"""Multitask vision training script.

Usage:
    uv run python train.py --config configs/det_seg_depth.py
"""
import argparse
import time

import torch

import multitask_vision  # noqa: F401 — trigger registration
from multitask_vision.model import MultitaskVisionModel
from multitask_vision.data.multi_dataset import MultiDatasetLoader
from multitask_vision.registry import BLOCKS, LOSSES


def build_model_from_config(cfg: dict, device: torch.device) -> MultitaskVisionModel:
    model = MultitaskVisionModel(
        blocks=cfg['blocks'],
        losses=cfg['losses'],
    )
    return model.to(device)


def build_datasets(cfg: dict):
    """Build dataset instances from config dicts."""
    from multitask_vision.data.voc import VOCMultitaskDataset
    from multitask_vision.data.nyu_depth import NYUDepthDataset

    dataset_map = {
        'VOCMultitaskDataset': VOCMultitaskDataset,
        'NYUDepthDataset': NYUDepthDataset,
    }
    datasets = []
    for ds_cfg in cfg['datasets']:
        ds_type = ds_cfg.pop('type')
        datasets.append(dataset_map[ds_type](**ds_cfg))
        ds_cfg['type'] = ds_type  # restore
    return datasets


def train(config_path: str, work_dir: str = 'work_dirs'):
    # Load config as Python module
    import importlib.util
    spec = importlib.util.spec_from_file_location('config', config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    cfg = {k: getattr(cfg_module, k) for k in dir(cfg_module) if not k.startswith('_')}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Build model
    model = build_model_from_config(cfg['model'], device)
    print(f'Model blocks: {list(model.block_modules.keys())}')
    print(f'Topological order: {model.topo_order}')

    # Build datasets and loader
    datasets = build_datasets(cfg['data'])
    for ds in datasets:
        print(f'Dataset: {ds.__class__.__name__}, size: {len(ds)}, tasks: {ds.tasks}')

    loader = MultiDatasetLoader(
        datasets,
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['training'].get('num_workers', 4),
        sampling_strategy=cfg['training'].get('sampling_strategy', 'proportional'),
    )

    # Optimizer
    opt_cfg = cfg['training']['optimizer']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.get('lr', 1e-4),
        weight_decay=opt_cfg.get('weight_decay', 0.01),
    )

    max_iters = cfg['training']['max_iters']
    log_interval = cfg['training'].get('log_interval', 50)

    # Training loop
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
        optimizer.step()

        if step % log_interval == 0:
            loss_str = ', '.join(f'{k}: {v.item():.4f}' for k, v in losses.items())
            tasks = data_samples[0]['tasks']
            dt = time.time() - t0
            print(f'[{step}/{max_iters}] tasks={tasks} | {loss_str} | total={total_loss.item():.4f} | {dt:.2f}s')

        if step % cfg['training'].get('save_interval', 5000) == 0:
            import os
            os.makedirs(work_dir, exist_ok=True)
            save_path = os.path.join(work_dir, f'step_{step}.pth')
            torch.save(model.state_dict(), save_path)
            print(f'Saved checkpoint: {save_path}')

    print('\nTraining complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--work-dir', type=str, default='work_dirs')
    args = parser.parse_args()
    train(args.config, args.work_dir)
