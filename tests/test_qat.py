"""Verify that native PyTorch quantization-aware training works on our multitask model."""
import torch
import torch.ao.quantization as quant
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


class TestFakeQuantizeInsertion:
    def test_insert_observers_on_submodule(self):
        """Insert fake quantize observers on a backbone submodule."""
        model = _make_model()
        backbone = model.block_modules['backbone']

        backbone.qconfig = quant.QConfig(
            activation=quant.FakeQuantize.with_args(
                observer=quant.MovingAverageMinMaxObserver,
                dtype=torch.quint8,
            ),
            weight=quant.default_weight_fake_quant,
        )

        quant.prepare_qat(backbone, inplace=True)

        # Check that fake quantize modules were inserted
        fq_count = sum(
            1 for m in backbone.modules()
            if isinstance(m, torch.ao.quantization.FakeQuantize)
        )
        assert fq_count > 0, "No FakeQuantize modules inserted"

    def test_forward_with_fake_quantize(self):
        """Model forward still works after inserting fake quantize."""
        model = _make_model()
        backbone = model.block_modules['backbone']

        backbone.qconfig = quant.QConfig(
            activation=quant.FakeQuantize.with_args(
                observer=quant.MovingAverageMinMaxObserver,
                dtype=torch.quint8,
            ),
            weight=quant.default_weight_fake_quant,
        )
        quant.prepare_qat(backbone, inplace=True)

        imgs, ds = _make_batch()
        losses = model(imgs, ds, mode='loss')
        assert all(torch.isfinite(v) for v in losses.values())

    def test_backward_with_fake_quantize(self):
        """Gradients flow through fake quantize (STE)."""
        model = _make_model()
        backbone = model.block_modules['backbone']

        backbone.qconfig = quant.QConfig(
            activation=quant.FakeQuantize.with_args(
                observer=quant.MovingAverageMinMaxObserver,
                dtype=torch.quint8,
            ),
            weight=quant.default_weight_fake_quant,
        )
        quant.prepare_qat(backbone, inplace=True)

        imgs, ds = _make_batch()
        losses = model(imgs, ds, mode='loss')
        total = sum(losses.values())
        total.backward()

        # Check gradients flow through the quantized backbone
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in backbone.parameters()
        )
        assert has_grad, "No gradients in quantized backbone"


class TestQATTrainingLoop:
    def test_multi_step_qat(self):
        """Simulate a few QAT training steps."""
        model = _make_model()
        backbone = model.block_modules['backbone']

        backbone.qconfig = quant.QConfig(
            activation=quant.FakeQuantize.with_args(
                observer=quant.MovingAverageMinMaxObserver,
                dtype=torch.quint8,
            ),
            weight=quant.default_weight_fake_quant,
        )
        quant.prepare_qat(backbone, inplace=True)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        model.train()

        for step in range(3):
            imgs, ds = _make_batch()
            losses = model(imgs, ds, mode='loss')
            total = sum(losses.values())
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

        # Verify observer statistics are updated
        for m in backbone.modules():
            if isinstance(m, torch.ao.quantization.FakeQuantize):
                assert m.activation_post_process.min_val != float('inf'), \
                    "Observer stats not updated after training steps"
                break


class TestModuleDiscovery:
    def test_all_conv_layers_discoverable(self):
        """Verify all Conv2d layers are accessible for quantization."""
        model = _make_model()
        conv_layers = [
            (name, m) for name, m in model.named_modules()
            if isinstance(m, torch.nn.Conv2d)
        ]
        assert len(conv_layers) > 20  # ResNet50 has many convs

    def test_all_bn_layers_discoverable(self):
        model = _make_model()
        bn_layers = [
            (name, m) for name, m in model.named_modules()
            if isinstance(m, torch.nn.BatchNorm2d)
        ]
        assert len(bn_layers) > 10

    def test_named_modules_include_all_blocks(self):
        """All blocks are reachable via model.named_modules()."""
        model = _make_model()
        module_names = {name for name, _ in model.named_modules()}
        for block_name in model.block_modules:
            assert any(block_name in n for n in module_names), \
                f"Block '{block_name}' not discoverable via named_modules()"
