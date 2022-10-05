_base_ = [
    '../_base_/datasets/coco_detection.py', '../_base_/default_runtime.py'
]

dataset_type = 'CocoDataset'

# Linux
# data_root = '/home/taowenyin/MyCode/Dataset/fire_coco/'
# MatPool
data_root = '/mnt/dataset/fire_coco/'
# Windows
# data_root = 'D:/MyCode/Dataset/VOC/coco/'
# MAC
# data_root = '/Users/taowenyin/Database/voc2012/coco/'

classes = ('fire',)

model = dict(
    type='DEMM',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    bbox_head=dict(
        type='DEMMHead',
        num_classes=80,
        in_channels=2048,
        mlp_mixer=dict(
            type='MLPMixer',
            encoder=dict(
                type='DemmMLPMixerEncoder'
            ),
            decoder=dict(
                type='DemmMLPMixerDecoder',
            )
        ),
        loss_cls=dict(
            type='CrossEntropyLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)
    ),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=1.),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=100)
)
