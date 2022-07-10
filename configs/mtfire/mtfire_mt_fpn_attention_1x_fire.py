_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

dataset_type = 'CocoDataset'
# data_root = '/home/taowenyin/MyCode/Dataset/voc2012/coco/'
data_root = 'D:/MyCode/Dataset/voc2007/coco'

classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tvmonitor')

model = dict(
    type='MTFire',
    backbone=dict(
        type='MTNet',
        num_classes=1000,
        embed_dims=[46, 92, 184, 368],
        stem_channel=16,
        fc_dim=1280,
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[3.6, 3.6, 3.6, 3.6],
        qkv_bias=True,
        qk_scale=None,
        representation_size=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        hybrid_backbone=None,
        norm_layer=None,
        depths=[2, 2, 10, 2],
        qk_ratio=1,
        sr_ratios=[8, 4, 2, 1],
        dp=0.1
    ),
    neck=dict(

    ),
    bbox_head=dict(

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

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        classes=classes,
        img_prefix=data_root + 'train2017/'),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        classes=classes,
        img_prefix=data_root + 'val2017/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        classes=classes,
        img_prefix=data_root + 'val2017/'))

runner = dict(type='EpochBasedRunner', max_epochs=20)

checkpoint_config = dict(create_symlink=False)