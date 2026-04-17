# MultitaskVision

Config-driven multitask vision training framework. Define a model as a graph of blocks in a config file, train it on multiple datasets with different annotation types, and only compute losses for the tasks present in each batch.

Built on top of [OpenMMLab](https://github.com/open-mmlab) (mmengine) and PyTorch.

## Why

OpenMMLab provides excellent single-task toolboxes (MMDetection, MMSegmentation, etc.), but no built-in way to train a single shared backbone with multiple task heads simultaneously. This project fills that gap.

## How It Works

### Model as a DAG

The model architecture is defined as a directed acyclic graph of blocks in a Python config:

```python
model = dict(
    blocks=[
        dict(name='backbone', type='ResNetBackbone', inputs=['image']),
        dict(name='neck', type='FPNNeck', inputs=['backbone']),
        dict(name='det_head', type='AnchorFreeDetHead', inputs=['neck'], task='detection'),
        dict(name='seg_head', type='FCNSegHead', inputs=['backbone.layer4'], task='segmentation'),
        dict(name='depth_head', type='DenseDepthHead', inputs=['backbone.layer4'], task='depth'),
    ],
    ...
)
```

Each block declares its inputs by name (with dot-notation for sub-outputs like `backbone.layer4`). The framework resolves the graph, topologically sorts it, and executes blocks in the right order. Head blocks are skipped when their task has no annotations in the current batch.

### Multi-Dataset Training

Different datasets provide annotations for different tasks. Batches are homogeneous (one dataset per batch), and only the relevant losses are computed:

```python
data = dict(
    datasets=[
        dict(type='VOCMultitaskDataset', tasks=['detection', 'segmentation']),
        dict(type='NYUDepthDataset', tasks=['depth']),
    ],
)
```

A VOC batch computes detection + segmentation losses. A depth batch computes only the depth loss. The backbone receives gradients from all tasks across batches.

## Getting Started

### Install

```bash
uv sync
```

### Download Data

```bash
bash scripts/download_data.sh
```

This downloads PASCAL VOC 2012 and prepares depth data.

### Train

```bash
uv run python train.py --config configs/det_seg_depth.py
```

### Visualize

```bash
uv run python visualize.py --checkpoint work_dirs/latest.pth
```

Generates side-by-side comparisons of predictions vs ground truth for all tasks.

### Test

```bash
uv run pytest
```

## Project Structure

```
MultitaskVision/
├── configs/                    # Training configs
├── multitask_vision/
│   ├── model.py                # Graph-based multitask model (DAG builder)
│   ├── registry.py             # Block and loss registries
│   ├── losses.py               # Task loss wrappers
│   ├── blocks/
│   │   ├── backbones.py        # ResNet (torchvision)
│   │   ├── necks.py            # FPN
│   │   └── heads.py            # Detection, segmentation, depth heads
│   └── data/
│       ├── multi_dataset.py    # Multi-dataset loader with sampling strategies
│       ├── voc.py              # PASCAL VOC (detection + segmentation)
│       └── nyu_depth.py        # Depth dataset
├── train.py                    # Training script
├── visualize.py                # Visual validation
└── tests/
```

## Adding a New Task

1. Implement a head in `blocks/heads.py` with a `forward()` and `compute_loss()` method
2. Register it with `@BLOCKS.register_module()`
3. Add it to your config with `task='your_task'`
4. Add a dataset that provides the corresponding ground truth annotations
5. Add a loss wrapper in the config's `losses` dict

## Author

Mathieu Charbonnel
