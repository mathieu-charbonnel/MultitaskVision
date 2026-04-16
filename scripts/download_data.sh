#!/bin/bash
set -e

DATA_DIR="data"
mkdir -p "$DATA_DIR"

# ── PASCAL VOC 2012 ──────────────────────────────────────────────
echo "=== Downloading PASCAL VOC 2012 ==="
if [ ! -d "$DATA_DIR/VOCdevkit/VOC2012" ]; then
    cd "$DATA_DIR"
    curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    tar xf VOCtrainval_11-May-2012.tar
    rm VOCtrainval_11-May-2012.tar
    cd ..
    echo "VOC 2012 ready."
else
    echo "VOC 2012 already exists, skipping."
fi

# ── NYU Depth V2 (small subset) ─────────────────────────────────
echo "=== Preparing NYU Depth V2 (mini) ==="
NYU_DIR="$DATA_DIR/nyu_depth_v2/train"
if [ ! -d "$NYU_DIR/images" ]; then
    mkdir -p "$NYU_DIR/images" "$NYU_DIR/depths"
    python3 -c "
import numpy as np
from PIL import Image
import os

# Generate synthetic depth data for testing (replace with real NYU data for actual training)
np.random.seed(42)
for i in range(200):
    # Random RGB image
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    Image.fromarray(img).save(f'$NYU_DIR/images/{i:04d}.png')
    # Random depth map
    depth = np.random.uniform(0.5, 10.0, (480, 640)).astype(np.float32)
    np.save(f'$NYU_DIR/depths/{i:04d}.npy', depth)

print(f'Generated 200 synthetic depth samples in $NYU_DIR')
"
else
    echo "NYU depth data already exists, skipping."
fi

echo ""
echo "=== Data ready ==="
echo "VOC: $DATA_DIR/VOCdevkit/VOC2012"
echo "NYU: $DATA_DIR/nyu_depth_v2/train"
