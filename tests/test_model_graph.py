import torch
import pytest

import multitask_vision  # trigger registration
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
            dict(name='depth_head', type='DenseDepthHead',
                 args=dict(in_channels=2048), inputs=['backbone.layer4'],
                 task='depth'),
        ],
        losses=dict(
            detection=dict(type='TaskLoss', weight=1.0),
            segmentation=dict(type='TaskLoss', weight=1.0),
            depth=dict(type='TaskLoss', weight=0.5),
        ),
    )


class TestTopologicalSort:
    def test_order_is_valid(self):
        model = _make_model()
        order = model.topo_order
        assert order.index('backbone') < order.index('neck')
        assert order.index('backbone') < order.index('seg_head')
        assert order.index('neck') < order.index('det_head')

    def test_all_blocks_present(self):
        model = _make_model()
        assert set(model.topo_order) == {'backbone', 'neck', 'det_head', 'seg_head', 'depth_head'}


class TestSelectiveForward:
    def test_detection_only(self):
        model = _make_model()
        images = torch.randn(2, 3, 128, 128)
        data_samples = [
            {'tasks': ['detection'],
             'gt_bboxes': torch.tensor([[0.1, 0.1, 0.5, 0.5]]),
             'gt_labels': torch.tensor([3])},
        ] * 2
        losses = model(images, data_samples, mode='loss')
        assert any('det' in k for k in losses)
        assert not any('seg' in k for k in losses)
        assert not any('depth' in k for k in losses)

    def test_segmentation_only(self):
        model = _make_model()
        images = torch.randn(2, 3, 128, 128)
        data_samples = [
            {'tasks': ['segmentation'],
             'gt_seg_map': torch.zeros(128, 128, dtype=torch.long)},
        ] * 2
        losses = model(images, data_samples, mode='loss')
        assert any('seg' in k for k in losses)
        assert not any('det' in k for k in losses)

    def test_depth_only(self):
        model = _make_model()
        images = torch.randn(2, 3, 128, 128)
        data_samples = [
            {'tasks': ['depth'],
             'gt_depth_map': torch.ones(128, 128) * 5.0},
        ] * 2
        losses = model(images, data_samples, mode='loss')
        assert any('depth' in k for k in losses)
        assert not any('det' in k for k in losses)

    def test_multi_task_batch(self):
        model = _make_model()
        images = torch.randn(2, 3, 128, 128)
        data_samples = [
            {'tasks': ['detection', 'segmentation'],
             'gt_bboxes': torch.tensor([[0.1, 0.1, 0.5, 0.5]]),
             'gt_labels': torch.tensor([3]),
             'gt_seg_map': torch.zeros(128, 128, dtype=torch.long)},
        ] * 2
        losses = model(images, data_samples, mode='loss')
        assert any('det' in k for k in losses)
        assert any('seg' in k for k in losses)
        assert not any('depth' in k for k in losses)

    def test_all_losses_are_tensors(self):
        model = _make_model()
        images = torch.randn(2, 3, 128, 128)
        data_samples = [
            {'tasks': ['detection', 'segmentation', 'depth'],
             'gt_bboxes': torch.tensor([[0.1, 0.1, 0.5, 0.5]]),
             'gt_labels': torch.tensor([3]),
             'gt_seg_map': torch.zeros(128, 128, dtype=torch.long),
             'gt_depth_map': torch.ones(128, 128) * 5.0},
        ] * 2
        losses = model(images, data_samples, mode='loss')
        for v in losses.values():
            assert isinstance(v, torch.Tensor)
            assert v.requires_grad


class TestBackwardPass:
    def test_gradient_flows(self):
        model = _make_model()
        images = torch.randn(2, 3, 128, 128)
        data_samples = [
            {'tasks': ['detection'],
             'gt_bboxes': torch.tensor([[0.1, 0.1, 0.5, 0.5]]),
             'gt_labels': torch.tensor([3])},
        ] * 2
        losses = model(images, data_samples, mode='loss')
        total = sum(losses.values())
        total.backward()
        # Backbone should get gradients from detection head
        assert model.block_modules['backbone'].layer1[0].conv1.weight.grad is not None
