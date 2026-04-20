"""Verify that native PyTorch pruning works on our multitask model."""
import torch
import torch.nn.utils.prune as prune
import pytest

import multitask_vision
from multitask_vision.model import MultitaskVisionModel


def _make_model():
    return MultitaskVisionModel(
        blocks=[
            dict(name='backbone', type='ResNetBackbone',
                 args=dict(depth=50, pretrained=False), inputs=['image']),
            dict(name='neck', type='FPNNeck',
                 args=dict(in_channels=[256, 512, 1024, 2048], out_channels=256),
                 inputs=['backbone']),
            dict(name='det_head', type='AnchorFreeDetHead',
                 args=dict(num_classes=20, in_channels=256), inputs=['neck'],
                 task='detection'),
            dict(name='seg_head', type='FCNSegHead',
                 args=dict(num_classes=21, in_channels=2048),
                 inputs=['backbone.layer4'], task='segmentation'),
        ],
        losses=dict(
            detection=dict(type='TaskLoss', weight=1.0),
            segmentation=dict(type='TaskLoss', weight=1.0),
        ),
    )


def _make_batch():
    return (
        torch.randn(2, 3, 128, 128),
        [{'tasks': ['detection', 'segmentation'],
          'gt_bboxes': torch.tensor([[0.1, 0.1, 0.5, 0.5]]),
          'gt_labels': torch.tensor([3]),
          'gt_seg_map': torch.zeros(128, 128, dtype=torch.long),
          'gt_key_tasks': {'gt_bboxes': 'detection', 'gt_labels': 'detection',
                           'gt_seg_map': 'segmentation'}}] * 2,
    )


class TestL1UnstructuredPruning:
    def test_prune_single_layer(self):
        model = _make_model()
        conv = model.block_modules['backbone'].layer1[0].conv1
        prune.l1_unstructured(conv, name='weight', amount=0.3)
        assert hasattr(conv, 'weight_mask')
        sparsity = (conv.weight == 0).float().mean().item()
        assert sparsity >= 0.25  # at least ~30% pruned

    def test_forward_after_pruning(self):
        model = _make_model()
        conv = model.block_modules['backbone'].layer1[0].conv1
        prune.l1_unstructured(conv, name='weight', amount=0.5)
        imgs, ds = _make_batch()
        losses = model(imgs, ds, mode='loss')
        assert all(torch.isfinite(v) for v in losses.values())

    def test_backward_after_pruning(self):
        model = _make_model()
        conv = model.block_modules['backbone'].layer1[0].conv1
        prune.l1_unstructured(conv, name='weight', amount=0.5)
        imgs, ds = _make_batch()
        losses = model(imgs, ds, mode='loss')
        total = sum(losses.values())
        total.backward()
        # Pruned positions should have zero gradient
        assert conv.weight_orig.grad is not None


class TestGlobalPruning:
    def test_prune_all_convs(self):
        model = _make_model()
        params_to_prune = [
            (m, 'weight') for m in model.modules()
            if isinstance(m, torch.nn.Conv2d)
        ]
        prune.global_unstructured(
            params_to_prune, pruning_method=prune.L1Unstructured, amount=0.2,
        )
        total_zeros = sum(
            (m.weight == 0).sum().item() for m in model.modules()
            if isinstance(m, torch.nn.Conv2d) and hasattr(m, 'weight_mask')
        )
        total_params = sum(
            m.weight.numel() for m in model.modules()
            if isinstance(m, torch.nn.Conv2d) and hasattr(m, 'weight_mask')
        )
        sparsity = total_zeros / total_params
        assert 0.15 <= sparsity <= 0.25  # ~20% globally

    def test_forward_after_global_pruning(self):
        model = _make_model()
        params_to_prune = [
            (m, 'weight') for m in model.modules()
            if isinstance(m, torch.nn.Conv2d)
        ]
        prune.global_unstructured(
            params_to_prune, pruning_method=prune.L1Unstructured, amount=0.3,
        )
        imgs, ds = _make_batch()
        losses = model(imgs, ds, mode='loss')
        total = sum(losses.values())
        total.backward()
        assert all(torch.isfinite(v) for v in losses.values())


class TestStructuredPruning:
    def test_prune_output_channels(self):
        model = _make_model()
        conv = model.block_modules['backbone'].layer1[0].conv1
        out_channels_before = conv.weight.shape[0]
        prune.ln_structured(conv, name='weight', amount=0.25, n=1, dim=0)
        # Structured pruning zeros entire channels
        zero_channels = (conv.weight.abs().sum(dim=(1, 2, 3)) == 0).sum().item()
        assert zero_channels >= out_channels_before * 0.2


class TestPruningPermanence:
    def test_remove_reparametrization(self):
        model = _make_model()
        conv = model.block_modules['backbone'].layer1[0].conv1
        prune.l1_unstructured(conv, name='weight', amount=0.3)
        assert hasattr(conv, 'weight_mask')
        prune.remove(conv, 'weight')
        # After removal, the mask is baked into the weight permanently
        assert not hasattr(conv, 'weight_mask')
        assert not hasattr(conv, 'weight_orig')
        sparsity = (conv.weight == 0).float().mean().item()
        assert sparsity >= 0.25  # zeros are permanent now
