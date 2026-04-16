from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from multitask_vision.registry import BLOCKS


@BLOCKS.register_module()
class AnchorFreeDetHead(nn.Module):
    """Simple anchor-free detection head (FCOS-style).

    Predicts per-pixel class scores and bounding box regression at each FPN level.
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
        )

    def forward(self, fpn_features: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        cls_scores = [self.cls_head(f) for f in fpn_features]
        bbox_preds = [self.reg_head(f) for f in fpn_features]
        return {'cls_scores': cls_scores, 'bbox_preds': bbox_preds}

    def compute_loss(
        self, predictions: Dict[str, List[torch.Tensor]], targets: Dict
    ) -> Dict[str, torch.Tensor]:
        """Simplified detection loss: focal-like cls + L1 reg on matched locations."""
        gt_labels_list = targets['gt_labels']  # list of [N_i] per image
        batch_size = len(gt_labels_list)

        # Use only the finest FPN level for a simple loss
        cls_score = predictions['cls_scores'][0]  # [B, C, H, W]
        bbox_pred = predictions['bbox_preds'][0]  # [B, 4, H, W]

        h, w = cls_score.shape[2:]
        device = cls_score.device

        # Create target maps: for simplicity, place GT class at center of image
        cls_target = torch.zeros(batch_size, h, w, dtype=torch.long, device=device)
        for i, labels in enumerate(gt_labels_list):
            if len(labels) > 0:
                cls_target[i, h // 2, w // 2] = labels[0].long() + 1  # +1 for bg=0

        # Classification loss (cross-entropy, treating 0 as background)
        cls_score_flat = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        cls_target_flat = cls_target.reshape(-1).clamp(0, self.num_classes - 1)
        loss_cls = F.cross_entropy(cls_score_flat, cls_target_flat)

        # Regression loss (dummy: L1 on all predictions toward zero)
        loss_reg = bbox_pred.abs().mean()

        return {'loss_det_cls': loss_cls, 'loss_det_reg': loss_reg * 0.1}


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
