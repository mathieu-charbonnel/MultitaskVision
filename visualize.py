"""Config-driven visualization of model predictions.

Discovers tasks from the model config and uses registered visualizers
per task. No task-specific logic in this file.

Usage:
    uv run python visualize.py --config configs/det_seg_depth_mmlab.py \
        --checkpoint work_dirs_mmlab/latest.pth --output vis_results/
"""
import argparse
import importlib.util
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

import multitask_vision  # noqa: F401
from multitask_vision.model import MultitaskVisionModel
from multitask_vision.visualization import denormalize, get_visualizer
from train import build_datasets  # uses DATASETS registry — no hardcoded names


def _discover_tasks(model_cfg: dict) -> list:
    return [blk['task'] for blk in model_cfg['blocks'] if blk.get('task')]


def _discover_head_names(model_cfg: dict) -> dict:
    return {blk['task']: blk['name'] for blk in model_cfg['blocks'] if blk.get('task')}


def run_visualization(
    config_path: str, checkpoint: str, output_dir: str,
    dataset_index: int = 0, num_samples: int = 8,
):
    os.makedirs(output_dir, exist_ok=True)

    spec = importlib.util.spec_from_file_location('config', config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    cfg = {k: getattr(cfg_module, k) for k in dir(cfg_module) if not k.startswith('_')}

    device = torch.device(
        'mps' if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available() else 'cpu'
    )

    model = MultitaskVisionModel(blocks=cfg['model']['blocks'], losses=cfg['model']['losses'])
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    tasks = _discover_tasks(cfg['model'])
    head_names = _discover_head_names(cfg['model'])
    num_tasks = len(tasks)
    print(f'Tasks: {tasks}')

    datasets = build_datasets(cfg['data'])
    ds = datasets[dataset_index]
    print(f'Dataset: {ds.__class__.__name__}, size: {len(ds)}, tasks: {ds.tasks}')

    for i in range(min(num_samples, len(ds))):
        sample_raw = ds[i]
        image = sample_raw['inputs'].unsqueeze(0).to(device)
        data_sample = sample_raw['data_samples']
        img_display = denormalize(sample_raw['inputs'])

        with torch.no_grad():
            outputs = model(image, [{'tasks': tasks}], mode='tensor')

        fig, axes = plt.subplots(2, num_tasks, figsize=(6 * num_tasks, 12))
        if num_tasks == 1:
            axes = axes.reshape(2, 1)
        fig.suptitle(f'Sample {i}', fontsize=16)

        for col, task in enumerate(tasks):
            pred = outputs.get(head_names[task])

            pred_vis = get_visualizer(task, 'pred')
            if pred is not None:
                pred_vis(axes[0, col], img_display, pred)
            else:
                axes[0, col].imshow(img_display)
                axes[0, col].set_title(f'{task} (no output)')
                axes[0, col].axis('off')

            gt_vis = get_visualizer(task, 'gt')
            gt_vis(axes[1, col], img_display, data_sample)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'sample_{i:03d}.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f'  Saved {save_path}')

    print(f'\nDone. Results in {output_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='vis_results')
    parser.add_argument('--dataset', type=int, default=0)
    parser.add_argument('--num-samples', type=int, default=8)
    args = parser.parse_args()
    run_visualization(args.config, args.checkpoint, args.output, args.dataset, args.num_samples)
