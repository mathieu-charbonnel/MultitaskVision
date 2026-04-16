from typing import Dict

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from multitask_vision.registry import BLOCKS


@BLOCKS.register_module()
class ResNetBackbone(nn.Module):
    """ResNet-50 backbone returning a dict of multi-scale features."""

    def __init__(self, depth: int = 50, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1   # stride 4,  256 ch
        self.layer2 = resnet.layer2   # stride 8,  512 ch
        self.layer3 = resnet.layer3   # stride 16, 1024 ch
        self.layer4 = resnet.layer4   # stride 32, 2048 ch

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(image)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return {'layer1': c1, 'layer2': c2, 'layer3': c3, 'layer4': c4}
