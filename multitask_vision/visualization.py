"""Modular visualization functions, registered per task.

Each task registers a prediction visualizer and a GT visualizer.
The visualize script discovers tasks from the config and uses the
registered functions — no hardcoded task knowledge in the main loop.
"""
from typing import Any, Callable, Dict

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

# Registry: task_name -> (pred_visualizer, gt_visualizer)
_VIS_REGISTRY: Dict[str, Dict[str, Callable]] = {}


def register_visualizer(task: str):
    """Decorator to register a prediction visualizer for a task."""
    def decorator(fn):
        _VIS_REGISTRY.setdefault(task, {})['pred'] = fn
        return fn
    return decorator


def register_gt_visualizer(task: str):
    """Decorator to register a GT visualizer for a task."""
    def decorator(fn):
        _VIS_REGISTRY.setdefault(task, {})['gt'] = fn
        return fn
    return decorator


def get_visualizer(task: str, kind: str = 'pred') -> Callable:
    """Get a registered visualizer. Falls back to a no-op."""
    return _VIS_REGISTRY.get(task, {}).get(kind, _default_vis)


def get_registered_tasks():
    return list(_VIS_REGISTRY.keys())


def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * STD + MEAN
    return np.clip(img, 0, 1)


def _default_vis(ax, img, data, **kwargs):
    ax.imshow(img)
    ax.set_title('(no visualizer)')
    ax.axis('off')


# ─── Detection ──────────────────────────────────────────────────────

def _nms(boxes, scores, iou_threshold=0.4):
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


@register_visualizer('detection')
def vis_detection_pred(ax, img, predictions, score_threshold=0.15,
                       nms_threshold=0.4, max_dets=20, class_names=None):
    ax.imshow(img)
    ax.set_title('Detection')

    if isinstance(predictions, dict):
        cls_scores_list = predictions['cls_scores']
        bbox_preds_list = predictions['bbox_preds']
    elif isinstance(predictions, (tuple, list)):
        cls_scores_list = predictions[0]
        bbox_preds_list = predictions[1]
    else:
        ax.axis('off')
        return

    scores = cls_scores_list[0][0].cpu()
    regs = bbox_preds_list[0][0].cpu()
    C, fh, fw = scores.shape
    h, w = img.shape[:2]

    gy = (torch.arange(fh).float() + 0.5) / fh
    gx = (torch.arange(fw).float() + 0.5) / fw
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')

    l, t, r, b = regs[0], regs[1], regs[2], regs[3]
    if l.max() > 2.0:
        stride = h / fh
        cx_px, cy_px = grid_x * w, grid_y * h
        x1 = (cx_px - l * stride).clamp(0, w)
        y1 = (cy_px - t * stride).clamp(0, h)
        x2 = (cx_px + r * stride).clamp(0, w)
        y2 = (cy_px + b * stride).clamp(0, h)
    else:
        x1 = (grid_x - l).clamp(0, 1) * w
        y1 = (grid_y - t).clamp(0, 1) * h
        x2 = (grid_x + r).clamp(0, 1) * w
        y2 = (grid_y + b).clamp(0, 1) * h

    score_map = scores.sigmoid() if scores.min() < 0 else scores.softmax(dim=0)

    all_boxes, all_scores, all_classes = [], [], []
    for cls_id in range(C):
        mask = score_map[cls_id] > score_threshold
        if not mask.any():
            continue
        all_boxes.append(torch.stack([x1[mask], y1[mask], x2[mask], y2[mask]], dim=1))
        all_scores.append(score_map[cls_id][mask])
        all_classes.extend([cls_id] * mask.sum().item())

    if all_boxes:
        all_boxes = torch.cat(all_boxes)
        all_scores = torch.cat(all_scores)
        all_classes = torch.tensor(all_classes)
        keep = _nms(all_boxes, all_scores, nms_threshold)
        if len(keep) > max_dets:
            keep = keep[:max_dets]

        colors = plt.cm.Set2(np.linspace(0, 1, max(C, 2)))
        for idx in keep:
            bx1, by1, bx2, by2 = all_boxes[idx].numpy()
            cls_id = all_classes[idx].item()
            score = all_scores[idx].item()
            color = colors[cls_id % len(colors)]
            rect = patches.Rectangle((bx1, by1), bx2 - bx1, by2 - by1,
                                     linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            label = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
            ax.text(bx1, by1 - 4, f'{label} {score:.2f}', color='white', fontsize=8,
                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.15', facecolor=color, alpha=0.8))
    ax.axis('off')


@register_gt_visualizer('detection')
def vis_detection_gt(ax, img, gt, class_names=None):
    ax.imshow(img)
    ax.set_title('GT Detection')
    if 'gt_bboxes' in gt:
        h, w = img.shape[:2]
        for bbox, label in zip(gt['gt_bboxes'], gt['gt_labels']):
            x1, y1, x2, y2 = bbox.numpy() * np.array([w, h, w, h])
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            name = class_names[label] if class_names and label < len(class_names) else str(label.item())
            ax.text(x1, y1 - 4, name, color='red', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))
    ax.axis('off')


# ─── Segmentation ───────────────────────────────────────────────────

@register_visualizer('segmentation')
def vis_segmentation_pred(ax, img, predictions, class_names=None):
    from PIL import Image as PILImage
    ax.imshow(img)
    seg = predictions[0].argmax(dim=0).cpu().numpy()
    seg_resized = np.array(PILImage.fromarray(seg.astype(np.uint8)).resize(
        (img.shape[1], img.shape[0]), PILImage.NEAREST))
    cmap = plt.cm.tab20
    ax.imshow(seg_resized, alpha=0.5, cmap=cmap, vmin=0, vmax=20)
    ax.set_title('Segmentation')
    present = [c for c in np.unique(seg_resized) if c not in (0, 255)]
    if present and class_names:
        legend = [patches.Patch(facecolor=cmap(c / 20.0),
                                label=class_names[c] if c < len(class_names) else str(c))
                  for c in present]
        ax.legend(handles=legend, loc='lower right', fontsize=6, framealpha=0.7)
    ax.axis('off')


@register_gt_visualizer('segmentation')
def vis_segmentation_gt(ax, img, gt, class_names=None):
    ax.imshow(img)
    if 'gt_seg_map' in gt:
        seg = gt['gt_seg_map'].numpy()
        ax.imshow(seg, alpha=0.5, cmap='tab20', vmin=0, vmax=20)
        ax.set_title('GT Segmentation')
    else:
        ax.set_title('GT Seg (N/A)')
    ax.axis('off')


# ─── Depth ──────────────────────────────────────────────────────────

@register_visualizer('depth')
def vis_depth_pred(ax, img, predictions, **kwargs):
    from PIL import Image as PILImage
    depth = predictions[0, 0].cpu().numpy()
    depth_resized = np.array(PILImage.fromarray(depth).resize(
        (img.shape[1], img.shape[0]), PILImage.BILINEAR))
    ax.imshow(depth_resized, cmap='plasma')
    ax.set_title('Depth')
    ax.axis('off')


@register_gt_visualizer('depth')
def vis_depth_gt(ax, img, gt, **kwargs):
    if 'gt_depth_map' in gt:
        ax.imshow(gt['gt_depth_map'].numpy(), cmap='plasma')
        ax.set_title('GT Depth')
    else:
        ax.imshow(img)
        ax.set_title('GT Depth (N/A)')
    ax.axis('off')
