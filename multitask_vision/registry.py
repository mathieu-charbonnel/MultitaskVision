"""Registries and module builder.

Blocks can come from our local BLOCKS registry or from mmlab registries
(mmdet, mmseg). The build_module function tries each in order and returns
the built module along with the detected protocol.
"""
from typing import Tuple

import torch.nn as nn
from mmengine.registry import Registry

BLOCKS = Registry('blocks', scope='multitask_vision')
LOSSES = Registry('losses', scope='multitask_vision')
DATASETS = Registry('datasets', scope='multitask_vision')

_mmlab_initialized = False


def _ensure_mmlab_registries():
    global _mmlab_initialized
    if _mmlab_initialized:
        return
    _mmlab_initialized = True
    try:
        import mmdet.models  # noqa: F401
    except ImportError:
        pass
    try:
        import mmseg.models  # noqa: F401
    except ImportError:
        pass


def build_module(type_str: str, args: dict) -> Tuple[nn.Module, str]:
    """Build a module by type name, searching registries in order.

    Returns (module, protocol) where protocol is 'native', 'mmdet', or 'mmseg'.
    """
    # Try our local registry first
    if BLOCKS.get(type_str) is not None:
        return BLOCKS.build(dict(type=type_str, **args)), 'native'

    # Try mmlab registries
    _ensure_mmlab_registries()

    for registry_import, protocol in [
        ('mmdet.registry', 'mmdet'),
        ('mmseg.registry', 'mmseg'),
    ]:
        try:
            import importlib
            reg_module = importlib.import_module(registry_import)
            registry = reg_module.MODELS
            if registry.get(type_str) is not None:
                module = registry.build(dict(type=type_str, **args))
                if hasattr(module, 'init_weights'):
                    module.init_weights()
                return module, protocol
        except ImportError:
            continue

    raise ValueError(
        f"Block type '{type_str}' not found in any registry "
        f"(multitask_vision.BLOCKS, mmdet.MODELS, mmseg.MODELS)"
    )
