# MultitaskVision

Config-driven multitask vision training framework. Define a model as a graph of blocks in a config file, train it on multiple datasets with different annotation types, and only compute losses for the tasks present in each batch.

Built on top of [OpenMMLab](https://github.com/open-mmlab) (mmengine, mmdet, mmseg) and PyTorch.

## Why

OpenMMLab provides excellent single-task toolboxes (MMDetection, MMSegmentation, etc.), but no built-in way to train a single shared backbone with multiple task heads simultaneously. This project fills that gap.

## How It Works

### Model as a DAG

The model architecture is defined as a directed acyclic graph of blocks in a Python config. Blocks can come from our own registry or directly from mmlab (mmdet, mmseg) — the framework auto-detects the source and adapts the calling convention via protocol adapters.

```python
model = dict(
    blocks=[
        dict(name='backbone', type='ResNetBackbone', inputs=['image']),
        dict(name='neck', type='FPN', inputs=['backbone']),           # mmdet's FPN
        dict(name='det_head', type='FCOSHead', inputs=['neck'],       # mmdet's FCOS
             task='detection', args=dict(num_classes=20, ...)),
        dict(name='seg_head', type='FCNHead', inputs=['backbone.layer4'],  # mmseg's FCN
             task='segmentation', args=dict(num_classes=21, ...)),
        dict(name='depth_head', type='DenseDepthHead', inputs=['backbone.layer4'],
             task='depth'),
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

### Deployment Pipeline

Trained models can be compressed and exported for embedded deployment:

```python
# configs/deploy.py
deploy = dict(
    pruning=dict(method='structured', amount=0.3, targets='all'),
    quantization=dict(method='static', observer='minmax'),
    export=dict(format='torchscript', blocks=['backbone']),
)
```

Supports structured/unstructured pruning, quantization-aware training (QAT), post-training quantization, and TorchScript/ONNX export. Compatible with native PyTorch pruning and quantization APIs.

**Important:** PyTorch pruning zeros out weights but does not physically remove channels or rebuild layers — the tensors keep their original dimensions. To get actual memory and compute savings on an embedded chip, the exported model must go through a **graph optimization** step (e.g. ONNX Runtime optimizer, TensorRT, or TFLite converter) which can strip zero channels and fuse operations. Quantization (int8), on the other hand, gives real 4x size reduction immediately.

## Getting Started

### Install

```bash
uv sync
```

### Download Data

```bash
bash scripts/download_data.sh
```

### Train

With native heads:
```bash
uv run python train.py --config configs/det_seg_depth_quick.py
```

With mmlab heads (FCOS + FCN):
```bash
uv run python train.py --config configs/det_seg_depth_mmlab.py
```

With quantization-aware training:
```bash
uv run python train.py --config configs/det_seg_depth_mmlab_qat.py
```

### Visualize

```bash
uv run python visualize.py --config configs/det_seg_depth_mmlab.py \
    --checkpoint work_dirs_mmlab/latest.pth
```

Discovers tasks from the config and generates side-by-side predictions vs ground truth for each task.

### Deploy

```bash
uv run python deploy.py --config configs/det_seg_depth_mmlab.py \
    --checkpoint work_dirs_mmlab/latest.pth \
    --deploy-config configs/deploy.py \
    --output deploy_output/
```

### Test

```bash
uv run pytest
```

## Architecture

```
MultitaskVision/
├── configs/                         # Training and deployment configs
│   ├── det_seg_depth_quick.py       # Native heads (lightweight)
│   ├── det_seg_depth_mmlab.py       # mmdet/mmseg heads
│   ├── det_seg_depth_mmlab_qat.py   # mmlab + QAT
│   └── deploy.py                    # Deployment pipeline config
├── multitask_vision/
│   ├── model.py                     # DAG executor (fully block-agnostic)
│   ├── registry.py                  # Block/loss/dataset registries + mmlab lookup
│   ├── adapters.py                  # Protocol adapters (native, mmdet, mmseg)
│   ├── compression.py               # Pruning, QAT, quantization, export utils
│   ├── visualization.py             # Registered per-task visualizers
│   ├── losses.py                    # Task loss wrappers
│   ├── blocks/
│   │   ├── backbones.py             # ResNet (torchvision)
│   │   ├── necks.py                 # FPN
│   │   └── heads.py                 # Detection, segmentation, depth heads
│   └── data/
│       ├── multi_dataset.py         # Multi-dataset loader with sampling strategies
│       ├── voc.py                   # PASCAL VOC (detection + segmentation)
│       └── nyu_depth.py             # Depth dataset
├── train.py                         # Training (config-driven, supports QAT)
├── visualize.py                     # Visual validation (config-driven)
├── deploy.py                        # Deployment pipeline (prune → quantize → export)
└── tests/
    ├── test_model_graph.py          # DAG construction, selective forward/loss
    ├── test_pruning.py              # PyTorch pruning compatibility
    ├── test_qat.py                  # Quantization-aware training compatibility
    └── test_model_compression.py    # Structured pruning, export
```

## Design Principles

- **model.py is fully block-agnostic** — no task names, no GT format knowledge, no mmlab references
- **Protocol adapters** handle the bridge between our graph and mmlab's APIs (forward conventions, loss computation, GT format conversion)
- **Registry-driven** — blocks, datasets, losses, and adapters are all registered and resolved from config
- **Configs reference mmlab blocks by name** (e.g. `type='FCOSHead'`) — no wrapper code needed

## Adding a New Task

1. Register a head with `@BLOCKS.register_module()` (or use an mmlab head by name)
2. Add it to your config with `task='your_task'`
3. Add a dataset that provides `gt_` prefixed keys and a `gt_key_tasks` mapping
4. Register a visualizer with `@register_visualizer('your_task')`

## Author

Mathieu Charbonnel
