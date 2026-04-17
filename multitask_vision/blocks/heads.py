from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from multitask_vision.registry import BLOCKS


@BLOCKS.register_module()
class AnchorFreeDetHead(nn.Module):
    """FCOS-style anchor-free detection head.

    For each FPN level, predicts per-pixel class scores and distances
    to the four box edges (l, t, r, b). Positive locations are those
    falling inside a ground truth box.
    """

    def __init__(self, num_classes: int, in_channels: int, num_levels: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1),
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4, 1),
            nn.ReLU(inplace=True),  # distances are positive
        )

    def forward(self, fpn_features: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        cls_scores = [self.cls_head(f) for f in fpn_features]
        bbox_preds = [self.reg_head(f) for f in fpn_features]
        return {'cls_scores': cls_scores, 'bbox_preds': bbox_preds}

    def compute_loss(
        self, predictions: Dict[str, List[torch.Tensor]], targets: Dict
    ) -> Dict[str, torch.Tensor]:
        gt_bboxes_list = targets['gt_bboxes']  # list of [N_i, 4] per image (normalized)
        gt_labels_list = targets['gt_labels']  # list of [N_i] per image

        # Use only the finest FPN level
        cls_score = predictions['cls_scores'][0]  # [B, C, H, W]
        bbox_pred = predictions['bbox_preds'][0]  # [B, 4, H, W]

        B, C, fh, fw = cls_score.shape
        device = cls_score.device

        # Feature map grid coordinates (normalized 0-1)
        gy = (torch.arange(fh, device=device).float() + 0.5) / fh
        gx = (torch.arange(fw, device=device).float() + 0.5) / fw
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')  # [fh, fw]

        cls_targets = torch.zeros(B, fh, fw, dtype=torch.long, device=device)
        reg_targets = torch.zeros(B, 4, fh, fw, device=device)
        pos_mask = torch.zeros(B, fh, fw, dtype=torch.bool, device=device)

        for b in range(B):
            bboxes = gt_bboxes_list[b]  # [N, 4]
            labels = gt_labels_list[b]  # [N]
            if len(bboxes) == 0:
                continue

            for n in range(len(bboxes)):
                x1, y1, x2, y2 = bboxes[n]
                # Vectorized: mask of locations inside box
                inside = (grid_x >= x1) & (grid_x <= x2) & (grid_y >= y1) & (grid_y <= y2)
                cls_targets[b][inside] = labels[n].long() + 1
                pos_mask[b] |= inside
                # Regression targets: distances to edges
                reg_targets[b, 0][inside] = (grid_x[inside] - x1)
                reg_targets[b, 1][inside] = (grid_y[inside] - y1)
                reg_targets[b, 2][inside] = (x2 - grid_x[inside])
                reg_targets[b, 3][inside] = (y2 - grid_y[inside])

        num_pos = pos_mask.sum().clamp(min=1).float()
        loss_cls = F.cross_entropy(
            cls_score, cls_targets.clamp(0, C - 1), reduction='sum'
        ) / num_pos

        if pos_mask.any():
            pos_expanded = pos_mask.unsqueeze(1).expand_as(bbox_pred)
            loss_reg = F.l1_loss(bbox_pred[pos_expanded], reg_targets[pos_expanded])
        else:
            loss_reg = bbox_pred.sum() * 0.0

        return {'loss_det_cls': loss_cls, 'loss_det_reg': loss_reg}


@BLOCKS.register_module()
class FCNSegHead(nn.Module):
    """Fully convolutional segmentation head."""

    def __init__(self, num_classes: int, in_channels: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )
        self.num_classes = num_classes

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)  # [B, num_classes, H', W']

    def compute_loss(
        self, predictions: torch.Tensor, targets: Dict
    ) -> Dict[str, torch.Tensor]:
        gt_seg = targets['gt_seg_map']  # [B, H, W] long
        pred = F.interpolate(predictions, size=gt_seg.shape[1:], mode='bilinear', align_corners=False)
        loss = F.cross_entropy(pred, gt_seg, ignore_index=255)
        return {'loss_seg_ce': loss}


@BLOCKS.register_module()
class DenseDepthHead(nn.Module):
    """Dense depth prediction head."""

    def __init__(self, in_channels: int, max_depth: float = 10.0):
        super().__init__()
        self.max_depth = max_depth
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features) * self.max_depth  # [B, 1, H', W']

    def compute_loss(
        self, predictions: torch.Tensor, targets: Dict
    ) -> Dict[str, torch.Tensor]:
        gt_depth = targets['gt_depth_map']  # [B, H, W]
        if gt_depth.dim() == 3:
            gt_depth = gt_depth.unsqueeze(1)
        pred = F.interpolate(predictions, size=gt_depth.shape[2:], mode='bilinear', align_corners=False)
        # Masked L1: only compute where depth > 0
        valid = gt_depth > 0
        if valid.any():
            loss = F.l1_loss(pred[valid], gt_depth[valid])
        else:
            loss = pred.sum() * 0.0
        return {'loss_depth_l1': loss}
