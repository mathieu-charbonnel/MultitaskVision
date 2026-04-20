"""Config using mmlab blocks directly — no wrappers needed.

The graph executor auto-detects which registry each block type comes from
and adapts the forward/loss protocol accordingly.

- ResNetBackbone: our registry (torchvision, returns dict of features)
- FPN: mmdet registry (expects tuple, returns tuple)
- FCOSHead: mmdet registry (focal loss + IoU loss + centerness)
- FCNHead: mmseg registry (cross-entropy segmentation)
- DenseDepthHead: our registry (custom depth head)
"""

model = dict(
    blocks=[
        dict(
            name='backbone',
            type='ResNetBackbone',
            args=dict(depth=50, pretrained=True),
            inputs=['image'],
        ),
        dict(
            name='neck',
            type='FPN',
            args=dict(
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                num_outs=4,
            ),
            inputs=['backbone'],
        ),
        dict(
            name='det_head',
            type='FCOSHead',
            args=dict(
                num_classes=20,
                in_channels=256,
                feat_channels=256,
                stacked_convs=2,
                strides=[8, 16, 32, 64],
                regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 1e8)),
                loss_cls=dict(
                    type='FocalLoss', use_sigmoid=True,
                    gamma=2.0, alpha=0.25, loss_weight=1.0,
                ),
                loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                loss_centerness=dict(
                    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0,
                ),
            ),
            inputs=['neck'],
            task='detection',
        ),
        dict(
            name='seg_head',
            type='FCNHead',
            args=dict(
                in_channels=2048,
                channels=256,
                num_convs=2,
                num_classes=21,
                in_index=0,
                norm_cfg=dict(type='BN'),
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                ),
            ),
            inputs=['backbone.layer4'],
            task='segmentation',
        ),
        dict(
            name='depth_head',
            type='DenseDepthHead',
            args=dict(in_channels=2048, max_depth=10.0),
            inputs=['backbone.layer4'],
            task='depth',
        ),
    ],
    losses=dict(
        detection=dict(type='TaskLoss', weight=1.0),
        segmentation=dict(type='TaskLoss', weight=1.0),
        depth=dict(type='TaskLoss', weight=0.5),
    ),
)

data = dict(
    datasets=[
        dict(
            type='VOCMultitaskDataset',
            data_root='data/VOCdevkit',
            split='train',
            img_size=512,
        ),
        dict(
            type='NYUDepthDataset',
            data_root='data/nyu_depth_v2',
            split='train',
            img_size=512,
        ),
    ],
)

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
)
