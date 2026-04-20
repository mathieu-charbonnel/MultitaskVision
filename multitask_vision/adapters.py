"""Protocol adapters that bridge between the model graph and specific block APIs.

Each adapter handles two things:
1. call_forward: how to pass inputs to the block (tuple vs list vs tensor)
2. compute_loss: how to call the block's loss method and prepare GT data

The model graph is fully agnostic — it delegates all protocol specifics here.
"""
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import Registry

ADAPTERS = Registry('adapters', scope='multitask_vision')


class BaseAdapter:
    """Default adapter for native blocks."""

    def call_forward(self, module: nn.Module, block_inputs: list) -> Any:
        """Call module.forward with the resolved inputs."""
        if len(block_inputs) == 1:
            return module(block_inputs[0])
        return module(*block_inputs)

    def compute_loss(
        self, module: nn.Module, predictions: Any,
        gt: Dict, images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return module.compute_loss(predictions, gt)


@ADAPTERS.register_module()
class NativeAdapter(BaseAdapter):
    pass


@ADAPTERS.register_module()
class MMDetAdapter(BaseAdapter):
    """Adapter for mmdet modules (FPN, FCOS, RetinaNet, etc.).

    Forward: mmdet modules expect tuples of tensors.
    Loss: converts GT to InstanceData and calls loss_by_feat.
    """

    def call_forward(self, module: nn.Module, block_inputs: list) -> Any:
        if len(block_inputs) == 1:
            inp = block_inputs[0]
        else:
            inp = block_inputs

        # mmdet modules expect a tuple of tensors
        if isinstance(inp, dict):
            # backbone dict -> ordered tuple (layer1, layer2, ...)
            keys = sorted(k for k in inp.keys() if k.startswith('layer'))
            return module(tuple(inp[k] for k in keys))
        if isinstance(inp, list):
            return module(tuple(inp))
        return module(inp)

    def compute_loss(
        self, module: nn.Module, predictions: Any,
        gt: Dict, images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        import inspect
        from mmengine.structures import InstanceData

        batch_size = len(gt.get('gt_bboxes', []))
        img_h, img_w = images.shape[2], images.shape[3]

        batch_img_metas = [
            dict(img_shape=(img_h, img_w), pad_shape=(img_h, img_w),
                 scale_factor=(1.0, 1.0))
            for _ in range(batch_size)
        ]

        batch_gt_instances = []
        min_box_size = 4.0  # minimum box side in pixels to avoid near-zero centerness
        for i in range(batch_size):
            gt_instance = InstanceData()
            bboxes = gt['gt_bboxes'][i]
            if len(bboxes) > 0:
                bboxes_pixel = bboxes.clone()
                bboxes_pixel[:, [0, 2]] *= img_w
                bboxes_pixel[:, [1, 3]] *= img_h
                # Filter out degenerate boxes
                widths = bboxes_pixel[:, 2] - bboxes_pixel[:, 0]
                heights = bboxes_pixel[:, 3] - bboxes_pixel[:, 1]
                valid = (widths >= min_box_size) & (heights >= min_box_size)
                gt_instance.bboxes = bboxes_pixel[valid]
                gt_instance.labels = gt['gt_labels'][i][valid]
            else:
                gt_instance.bboxes = torch.zeros(0, 4, device=bboxes.device)
                gt_instance.labels = torch.zeros(0, dtype=torch.long, device=bboxes.device)
            batch_gt_instances.append(gt_instance)

        if isinstance(predictions, (tuple, list)):
            sig = inspect.signature(module.loss_by_feat)
            param_names = [
                p for p in sig.parameters
                if p not in ('batch_gt_instances', 'batch_img_metas',
                             'batch_gt_instances_ignore')
            ]
            kwargs = {name: val for name, val in zip(param_names, predictions)}
        elif isinstance(predictions, dict):
            kwargs = predictions
        else:
            raise ValueError(f"Unexpected predictions type: {type(predictions)}")

        return module.loss_by_feat(
            **kwargs,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
        )


@ADAPTERS.register_module()
class MMSegAdapter(BaseAdapter):
    """Adapter for mmseg modules.

    Forward: mmseg decode heads expect a list of tensors indexed by in_index.
    Loss: interpolates predictions to GT size and computes cross-entropy.
    """

    def call_forward(self, module: nn.Module, block_inputs: list) -> Any:
        if len(block_inputs) == 1:
            inp = block_inputs[0]
        else:
            inp = block_inputs

        # mmseg heads expect a list of tensors
        if isinstance(inp, torch.Tensor):
            return module([inp])
        if isinstance(inp, dict):
            keys = sorted(k for k in inp.keys() if k.startswith('layer'))
            return module([inp[k] for k in keys])
        return module(inp)

    def compute_loss(
        self, module: nn.Module, predictions: torch.Tensor,
        gt: Dict, images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        gt_seg = gt['gt_seg_map']
        pred = predictions
        if pred.shape[2:] != gt_seg.shape[1:]:
            pred = F.interpolate(
                pred, size=gt_seg.shape[1:], mode='bilinear', align_corners=False
            )
        loss = F.cross_entropy(pred, gt_seg, ignore_index=255)
        return {'loss_seg_ce': loss}


DEFAULT_ADAPTERS = {
    'native': NativeAdapter(),
    'mmdet': MMDetAdapter(),
    'mmseg': MMSegAdapter(),
}
