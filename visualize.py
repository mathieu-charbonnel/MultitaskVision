"""Visualize model predictions for all tasks.

Usage:
    uv run python visualize.py --config configs/det_seg_depth.py --checkpoint work_dirs/latest.pth --output vis_results/
"""
import argparse
import importlib.util
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from torchvision import transforms

import multitask_vision  # noqa: F401
from multitask_vision.model import MultitaskVisionModel
from multitask_vision.data.voc import VOCMultitaskDataset, VOC_CLASSES
from multitask_vision.data.nyu_depth import NYUDepthDataset

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor [3, H, W] to displayable numpy [H, W, 3]."""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * STD + MEAN
    return np.clip(img, 0, 1)


def visualize_detection(ax, img: np.ndarray, cls_scores, bbox_preds, threshold=0.3):
    """Draw detected bounding boxes on image."""
    ax.imshow(img)
    ax.set_title('Detection')

    # Use finest FPN level
    scores = cls_scores[0][0]  # [C, H, W]
    bboxes = bbox_preds[0][0]  # [4, H, W]

    h, w = img.shape[:2]
    score_map = scores.softmax(dim=0)
    max_scores, max_classes = score_map.max(dim=0)  # [H', W']

    # Find high-confidence locations
    sh, sw = max_scores.shape
    for y in range(sh):
        for x in range(sw):
            score = max_scores[y, x].item()
            cls_id = max_classes[y, x].item()
            if score > threshold and cls_id < len(VOC_CLASSES):
                # Map feature location to image coordinates
                cx = x / sw * w
                cy = y / sh * h
                bw, bh = w * 0.15, h * 0.15
                rect = patches.Rectangle(
                    (cx - bw/2, cy - bh/2), bw, bh,
                    linewidth=2, edgecolor='lime', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(cx - bw/2, cy - bh/2 - 4,
                        f'{VOC_CLASSES[cls_id]} {score:.2f}',
                        color='lime', fontsize=7, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5))
    ax.axis('off')


def visualize_segmentation(ax, img: np.ndarray, seg_pred: torch.Tensor):
    """Show segmentation prediction overlaid on image."""
    seg = seg_pred[0].argmax(dim=0).cpu().numpy()  # [H', W']
    seg_resized = np.array(
        __import__('PIL').Image.fromarray(seg.astype(np.uint8)).resize(
            (img.shape[1], img.shape[0]), __import__('PIL').Image.NEAREST
        )
    )
    ax.imshow(img)
    ax.imshow(seg_resized, alpha=0.5, cmap='tab20', vmin=0, vmax=20)
    ax.set_title('Segmentation')
    ax.axis('off')


def visualize_depth(ax, img: np.ndarray, depth_pred: torch.Tensor):
    """Show depth prediction as a heatmap."""
    depth = depth_pred[0, 0].cpu().numpy()  # [H', W']
    depth_resized = np.array(
        __import__('PIL').Image.fromarray(depth).resize(
            (img.shape[1], img.shape[0]), __import__('PIL').Image.BILINEAR
        )
    )
    ax.imshow(depth_resized, cmap='plasma')
    ax.set_title('Depth')
    ax.axis('off')


def visualize_ground_truth(axes, img, sample):
    """Show ground truth annotations."""
    # GT Detection
    axes[0].imshow(img)
    axes[0].set_title('GT Detection')
    if 'gt_bboxes' in sample:
        h, w = img.shape[:2]
        for bbox, label in zip(sample['gt_bboxes'], sample['gt_labels']):
            x1, y1, x2, y2 = bbox.numpy() * np.array([w, h, w, h])
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[0].add_patch(rect)
            cls_name = VOC_CLASSES[label] if label < len(VOC_CLASSES) else str(label.item())
            axes[0].text(x1, y1 - 4, cls_name, color='red', fontsize=7,
                         bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))
    axes[0].axis('off')

    # GT Segmentation
    if 'gt_seg_map' in sample:
        axes[1].imshow(img)
        seg = sample['gt_seg_map'].numpy()
        axes[1].imshow(seg, alpha=0.5, cmap='tab20', vmin=0, vmax=20)
        axes[1].set_title('GT Segmentation')
    else:
        axes[1].imshow(img)
        axes[1].set_title('GT Seg (N/A)')
    axes[1].axis('off')

    # GT Depth
    if 'gt_depth_map' in sample:
        depth = sample['gt_depth_map'].numpy()
        axes[2].imshow(depth, cmap='plasma')
        axes[2].set_title('GT Depth')
    else:
        axes[2].imshow(img)
        axes[2].set_title('GT Depth (N/A)')
    axes[2].axis('off')


def run_visualization(config_path: str, checkpoint: str, output_dir: str, num_samples: int = 8):
    os.makedirs(output_dir, exist_ok=True)

    # Load config
    spec = importlib.util.spec_from_file_location('config', config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    cfg = {k: getattr(cfg_module, k) for k in dir(cfg_module) if not k.startswith('_')}

    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available() else 'cpu')

    # Build model and load weights
    model = MultitaskVisionModel(blocks=cfg['model']['blocks'], losses=cfg['model']['losses'])
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # Load VOC dataset for detection + segmentation visualization
    voc_ds = VOCMultitaskDataset(data_root='data/VOCdevkit', split='val', img_size=512)

    print(f'Visualizing {num_samples} VOC samples (det + seg + depth)...')

    for i in range(min(num_samples, len(voc_ds))):
        sample_raw = voc_ds[i]
        image = sample_raw['inputs'].unsqueeze(0).to(device)
        data_sample = sample_raw['data_samples']

        img_display = denormalize(sample_raw['inputs'])

        with torch.no_grad():
            # Run all tasks
            all_tasks_sample = [{'tasks': ['detection', 'segmentation', 'depth']}]
            outputs = model(image, all_tasks_sample, mode='tensor')

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Sample {i}', fontsize=16)

        # Top row: predictions
        det_preds = outputs.get('det_head', None)
        if det_preds is not None:
            visualize_detection(axes[0, 0], img_display,
                                det_preds['cls_scores'], det_preds['bbox_preds'])
        else:
            axes[0, 0].imshow(img_display)
            axes[0, 0].set_title('Detection (no output)')
            axes[0, 0].axis('off')

        seg_pred = outputs.get('seg_head', None)
        if seg_pred is not None:
            visualize_segmentation(axes[0, 1], img_display, seg_pred)
        else:
            axes[0, 1].imshow(img_display)
            axes[0, 1].set_title('Segmentation (no output)')
            axes[0, 1].axis('off')

        depth_pred = outputs.get('depth_head', None)
        if depth_pred is not None:
            visualize_depth(axes[0, 2], img_display, depth_pred)
        else:
            axes[0, 2].imshow(img_display)
            axes[0, 2].set_title('Depth (no output)')
            axes[0, 2].axis('off')

        # Bottom row: ground truth
        visualize_ground_truth([axes[1, 0], axes[1, 1], axes[1, 2]],
                               img_display, data_sample)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'sample_{i:03d}.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f'  Saved {save_path}')

    print(f'\nDone. Results in {output_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/det_seg_depth.py')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='vis_results')
    parser.add_argument('--num-samples', type=int, default=8)
    args = parser.parse_args()
    run_visualization(args.config, args.checkpoint, args.output, args.num_samples)
