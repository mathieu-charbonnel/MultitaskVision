"""Quick training config for validation on MPS. Smaller images, fewer iters."""

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
            type='FPNNeck',
            args=dict(in_channels=[256, 512, 1024, 2048], out_channels=256),
            inputs=['backbone'],
        ),
        dict(
            name='det_head',
            type='AnchorFreeDetHead',
            args=dict(num_classes=20, in_channels=256, num_levels=4),
            inputs=['neck'],
            task='detection',
        ),
        dict(
            name='seg_head',
            type='FCNSegHead',
            args=dict(num_classes=21, in_channels=2048),
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
            img_size=256,
        ),
        dict(
            type='NYUDepthDataset',
            data_root='data/nyu_depth_v2',
            split='train',
            img_size=256,
        ),
    ],
)

training = dict(
    max_iters=2000,
    batch_size=4,
    num_workers=2,
    sampling_strategy='round_robin',
    log_interval=25,
    save_interval=1000,
    optimizer=dict(
        lr=1e-4,
        weight_decay=0.01,
    ),
)
