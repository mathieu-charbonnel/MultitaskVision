from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from multitask_vision.registry import BLOCKS


@BLOCKS.register_module()
class FPNNeck(nn.Module):
    """Feature Pyramid Network.

    Expects a dict of features from the backbone (layer1..layer4).
    Returns a list of feature maps at each FPN level.
    """

    def __init__(self, in_channels: List[int], out_channels: int):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        for in_ch in in_channels:
            self.lateral_convs.append(nn.Conv2d(in_ch, out_channels, 1))
            self.output_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, features: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        feat_list = [features[f'layer{i}'] for i in range(1, 5)]

        laterals = [conv(f) for conv, f in zip(self.lateral_convs, feat_list)]

        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[2:], mode='nearest'
            )

        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        return outputs
