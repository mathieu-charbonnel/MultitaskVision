from typing import Dict

import torch
import torch.nn as nn

from multitask_vision.registry import LOSSES


@LOSSES.register_module()
class TaskLoss(nn.Module):
    """Generic task loss wrapper that applies a scalar weight to all sub-losses."""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, raw_losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v * self.weight for k, v in raw_losses.items()}
