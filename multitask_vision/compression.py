"""Model compression utilities for training and deployment.

Provides config-driven pruning, quantization, and export.
Used by both train.py (QAT) and deploy.py (post-training compression + export).
"""
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.ao.quantization as quant


def apply_qat(model: nn.Module, qat_cfg: dict) -> None:
    """Prepare model for quantization-aware training.

    Inserts FakeQuantize observers on the specified modules.

    Args:
        model: the model to prepare
        qat_cfg: dict with keys:
            - targets: list of block names to quantize (or 'all')
            - observer: 'minmax' or 'histogram'
            - dtype: 'quint8' (default)
    """
    targets = qat_cfg.get('targets', 'all')
    observer_type = qat_cfg.get('observer', 'minmax')
    dtype = getattr(torch, qat_cfg.get('dtype', 'quint8'))

    observer_cls = {
        'minmax': quant.MovingAverageMinMaxObserver,
        'histogram': quant.HistogramObserver,
    }[observer_type]

    qconfig = quant.QConfig(
        activation=quant.FakeQuantize.with_args(observer=observer_cls, dtype=dtype),
        weight=quant.default_weight_fake_quant,
    )

    if targets == 'all':
        model.qconfig = qconfig
        quant.prepare_qat(model, inplace=True)
    else:
        for name in targets:
            if hasattr(model, 'block_modules') and name in model.block_modules:
                submodule = model.block_modules[name]
            else:
                submodule = dict(model.named_modules()).get(name)
            if submodule is None:
                raise ValueError(f"QAT target '{name}' not found in model")
            submodule.qconfig = qconfig
            quant.prepare_qat(submodule, inplace=True)


def apply_structured_pruning(model: nn.Module, pruning_cfg: dict) -> dict:
    """Apply structured (channel) pruning to conv layers.

    Args:
        model: the model to prune
        pruning_cfg: dict with keys:
            - amount: fraction of channels to prune (0.0 to 1.0)
            - targets: list of block names (or 'all')
            - norm: L1 or L2 norm for importance (default 1)

    Returns:
        dict with pruning statistics
    """
    amount = pruning_cfg.get('amount', 0.3)
    targets = pruning_cfg.get('targets', 'all')
    norm = pruning_cfg.get('norm', 1)

    modules_to_prune = []
    if targets == 'all':
        modules_to_prune = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    else:
        for name in targets:
            if hasattr(model, 'block_modules') and name in model.block_modules:
                block = model.block_modules[name]
            else:
                block = dict(model.named_modules()).get(name)
            if block:
                modules_to_prune.extend(m for m in block.modules() if isinstance(m, nn.Conv2d))

    total_channels_before = 0
    total_channels_pruned = 0

    for m in modules_to_prune:
        channels_before = m.weight.shape[0]
        total_channels_before += channels_before
        prune.ln_structured(m, name='weight', amount=amount, n=norm, dim=0)
        prune.remove(m, 'weight')
        alive = (m.weight.abs().sum(dim=(1, 2, 3)) > 0).sum().item()
        total_channels_pruned += (channels_before - alive)

    return {
        'layers_pruned': len(modules_to_prune),
        'total_channels_before': total_channels_before,
        'total_channels_pruned': total_channels_pruned,
        'sparsity': total_channels_pruned / max(total_channels_before, 1),
    }


def apply_global_unstructured_pruning(model: nn.Module, pruning_cfg: dict) -> dict:
    """Apply global unstructured (weight-level) pruning.

    Args:
        model: the model to prune
        pruning_cfg: dict with keys:
            - amount: fraction of weights to prune globally
            - targets: list of block names (or 'all')
    """
    amount = pruning_cfg.get('amount', 0.3)
    targets = pruning_cfg.get('targets', 'all')

    params_to_prune = []
    if targets == 'all':
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                params_to_prune.append((m, 'weight'))
    else:
        for name in targets:
            if hasattr(model, 'block_modules') and name in model.block_modules:
                block = model.block_modules[name]
            else:
                block = dict(model.named_modules()).get(name)
            if block:
                for m in block.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        params_to_prune.append((m, 'weight'))

    prune.global_unstructured(
        params_to_prune, pruning_method=prune.L1Unstructured, amount=amount,
    )

    # Make permanent
    for m, _ in params_to_prune:
        prune.remove(m, 'weight')

    total_params = sum(m.weight.numel() for m, _ in params_to_prune)
    total_zeros = sum((m.weight == 0).sum().item() for m, _ in params_to_prune)

    return {
        'layers_pruned': len(params_to_prune),
        'total_params': total_params,
        'total_zeros': total_zeros,
        'sparsity': total_zeros / max(total_params, 1),
    }


def convert_quantized(model: nn.Module) -> nn.Module:
    """Convert a QAT-prepared model to actual int8 quantized model."""
    model.eval()
    return quant.convert(model)


class _DictToTupleWrapper(nn.Module):
    """Wraps a module that returns a dict so it returns a tuple instead.

    TorchScript trace and ONNX export require tuple/tensor outputs.
    """
    def __init__(self, module: nn.Module, keys: List[str]):
        super().__init__()
        self.module = module
        self.keys = keys

    def forward(self, x):
        out = self.module(x)
        return tuple(out[k] for k in self.keys)


def export_torchscript(module: nn.Module, dummy_input: torch.Tensor, path: str) -> None:
    """Export a module to TorchScript via tracing.

    If the module returns a dict, wraps it to return a tuple.
    """
    module.eval()
    with torch.no_grad():
        test_out = module(dummy_input)

    if isinstance(test_out, dict):
        keys = list(test_out.keys())
        wrapper = _DictToTupleWrapper(module, keys)
        wrapper.eval()
        traced = torch.jit.trace(wrapper, dummy_input)
    else:
        traced = torch.jit.trace(module, dummy_input)

    traced.save(path)


def export_onnx(module: nn.Module, dummy_input: torch.Tensor, path: str,
                input_names: List[str] = None, output_names: List[str] = None) -> None:
    """Export a module to ONNX.

    If the module returns a dict, wraps it to return a tuple.
    """
    module.eval()
    with torch.no_grad():
        test_out = module(dummy_input)

    if isinstance(test_out, dict):
        keys = list(test_out.keys())
        wrapper = _DictToTupleWrapper(module, keys)
        wrapper.eval()
        out_names = output_names or keys
        torch.onnx.export(
            wrapper, dummy_input, path,
            input_names=input_names or ['input'],
            output_names=out_names,
            opset_version=13,
        )
    else:
        torch.onnx.export(
            module, dummy_input, path,
            input_names=input_names or ['input'],
            output_names=output_names or ['output'],
            opset_version=13,
        )
