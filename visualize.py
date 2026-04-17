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


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.4) -> torch.Tensor:
    """Simple NMS. boxes: [N, 4] as (x1, y1, x2, y2), scores: [N]."""
    if len(boxes) == 0:
        return torch.zeros(0, dtype=torch.long)
    order = scores.argsort(descending=True)
    keep = []
    while len(order) > 0:
        i = order[0].item()
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = torch.max(boxes[i, 0], boxes[rest, 0])
        yy1 = torch.max(boxes[i, 1], boxes[rest, 1])
        xx2 = torch.min(boxes[i, 2], boxes[rest, 2])
        yy2 = torch.min(boxes[i, 3], boxes[rest, 3])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_rest - inter + 1e-6)
        order = rest[iou < iou_threshold]
    return torch.tensor(keep, dtype=torch.long)


def visualize_detection(ax, img: np.ndarray, cls_scores, bbox_preds,
                        score_threshold=0.15, nms_threshold=0.4, max_dets=20):
    """Decode FCOS-style predictions into boxes and draw them."""
    ax.imshow(img)
    ax.set_title('Detection')

    scores = cls_scores[0][0].cpu()  # [C, H, W]
    regs = bbox_preds[0][0].cpu()    # [4, H, W] — (l, t, r, b) in normalized coords

    C, fh, fw = scores.shape
    h, w = img.shape[:2]

    # Build grid of center locations (normalized 0-1)
    gy = (torch.arange(fh).float() + 0.5) / fh
    gx = (torch.arange(fw).float() + 0.5) / fw
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')  # [fh, fw]

    # Decode boxes: center - left/top, center + right/bottom
    l, t, r, b = regs[0], regs[1], regs[2], regs[3]
    x1 = (grid_x - l).clamp(0, 1) * w
    y1 = (grid_y - t).clamp(0, 1) * h
    x2 = (grid_x + r).clamp(0, 1) * w
    y2 = (grid_y + b).clamp(0, 1) * h

    # Per-class scores via softmax (class 0 = background)
    score_map = scores.softmax(dim=0)  # [C, fh, fw]

    # Collect candidate detections — skip class 0 (background)
    all_boxes = []
    all_scores = []
    all_classes = []

    for cls_id in range(C):
        cls_score = score_map[cls_id]  # [fh, fw]
        mask = cls_score > score_threshold
        if not mask.any():
            continue
        cls_boxes = torch.stack([x1[mask], y1[mask], x2[mask], y2[mask]], dim=1)
        cls_sc = cls_score[mask]
        all_boxes.append(cls_boxes)
        all_scores.append(cls_sc)
        all_classes.extend([cls_id] * len(cls_sc))

    if not all_boxes:
        ax.axis('off')
        return

    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_classes = torch.tensor(all_classes)

    # NMS
    keep = _nms(all_boxes, all_scores, nms_threshold)
    if len(keep) > max_dets:
        keep = keep[:max_dets]

    colors = plt.cm.Set2(np.linspace(0, 1, 20))
    for idx in keep:
        bx1, by1, bx2, by2 = all_boxes[idx].numpy()
        cls_id = all_classes[idx].item()
        score = all_scores[idx].item()
        color = colors[cls_id % len(colors)]

        rect = patches.Rectangle(
            (bx1, by1), bx2 - bx1, by2 - by1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        label = VOC_CLASSES[cls_id] if cls_id < len(VOC_CLASSES) else str(cls_id)
        ax.text(bx1, by1 - 4, f'{label} {score:.2f}',
                color='white', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor=color, alpha=0.8))
    ax.axis('off')


VOC_SEG_CLASSES = ['bg'] + VOC_CLASSES  # 0=background, 1-20=object classes

def visualize_segmentation(ax, img: np.ndarray, seg_pred: torch.Tensor):
    """Show segmentation prediction overlaid on image with class labels."""
    from PIL import Image as PILImage
    seg = seg_pred[0].argmax(dim=0).cpu().numpy()  # [H', W']
    seg_resized = np.array(
        PILImage.fromarray(seg.astype(np.uint8)).resize(
            (img.shape[1], img.shape[0]), PILImage.NEAREST
        )
    )
    ax.imshow(img)
    cmap = plt.cm.tab20
    ax.imshow(seg_resized, alpha=0.5, cmap=cmap, vmin=0, vmax=20)
    ax.set_title('Segmentation')

    # Add legend for classes present in prediction
    present = np.unique(seg_resized)
    legend_patches = []
    for cls_id in present:
        if cls_id == 0 or cls_id == 255:
            continue
        color = cmap(cls_id / 20.0)
        name = VOC_SEG_CLASSES[cls_id] if cls_id < len(VOC_SEG_CLASSES) else str(cls_id)
        legend_patches.append(patches.Patch(facecolor=color, label=name))
    if legend_patches:
        ax.legend(handles=legend_patches, loc='lower right', fontsize=6,
                  framealpha=0.7, handlelength=1, handleheight=1)
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
