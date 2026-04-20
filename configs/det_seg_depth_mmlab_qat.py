"""Same as det_seg_depth_mmlab.py but with quantization-aware training enabled.

The backbone and neck are quantized during training so the model learns
to be robust to int8 rounding. After training, use deploy.py with
method='qat_convert' to get the actual int8 model.
"""

# Inherit the base mmlab config
from configs.det_seg_depth_mmlab import model, data  # noqa: F401

training = dict(
    max_iters=2000,
    batch_size=2,
    num_workers=2,
    sampling_strategy='round_robin',
    log_interval=25,
    save_interval=1000,
    optimizer=dict(
        lr=1e-5,
        weight_decay=0.01,
    ),
    # QAT configuration
    qat=dict(
        targets=['backbone', 'neck'],  # which blocks to quantize ('all' or list)
        observer='minmax',              # 'minmax' or 'histogram'
        dtype='quint8',
    ),
)
