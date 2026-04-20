"""Deployment config: pruning + export.

Used with deploy.py to compress a trained checkpoint for embedded deployment.
Each step is optional — remove a section to skip it.
"""

deploy = dict(
    # Step 1: Pruning (optional)
    pruning=dict(
        method='structured',       # 'structured' (channel) or 'unstructured' (weight)
        amount=0.3,                # fraction to prune
        targets='all',             # 'all' or list of block names e.g. ['backbone', 'neck']
        norm=1,                    # L1 norm for importance ranking
    ),

    # Step 2: Quantization (optional)
    # quantization=dict(
    #     method='static',         # 'static', 'qat_convert' (if trained with QAT)
    #     observer='minmax',
    #     calibration_steps=50,
    #     img_size=256,
    # ),

    # Step 3: Export (optional)
    export=dict(
        format='torchscript',      # 'torchscript' or 'onnx'
        img_size=256,
        blocks=['backbone'],       # which blocks to export (or omit for all)
    ),
)
