# -*- coding: utf-8 -*-

_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

dataset_type = 'CocoDataset'

# MatPool
# data_root = '/mnt/dataset/VOC/coco/'
# Windows
data_root = 'D:/MyCode/Dataset/VOC/coco/'
# Linux
# data_root = '/home/taowenyin/MyCode/Dataset/voc2012/coco/'

classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tvmonitor')

model = dict(
    type='MTFire',
    backbone=dict(
        type='CMT',
        depth='base',
        out_indices=[0, 1, 2, 3],
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/mnt/mm_ws/checkpoints/m_cmt_base.pth',
        )
    ),
    neck=dict(
        type='FPN',
        in_channels=[76, 152, 304, 608],
        out_channels=76,
        start_level=0,
        num_outs=4,
        relu_before_extra_convs=True,
    ),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=20,
        in_channels=76,
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 1e8)),
        stacked_convs=4,
        feat_channels=256,
        strides=[4, 8, 16, 32],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(
            type='IoULoss',
            loss_weight=1.0
        ),
        loss_centerness=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0)
    ),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100
    )
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# img_norm_cfg = dict(
#     mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(256, 256), keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=8, # Batch Size
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        classes=classes,
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        classes=classes,
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        classes=classes,
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline)
)

runner = dict(type='EpochBasedRunner', max_epochs=12)

checkpoint_config = dict(create_symlink=False)

work_dir = '/mnt/mm_ws/voc_test/mtfire/base'
