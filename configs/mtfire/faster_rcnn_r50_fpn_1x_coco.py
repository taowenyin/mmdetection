_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
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
    roi_head=dict(
        bbox_head=dict(num_classes=1)))

train_pipeline = [
    dict(type='Resize', img_scale=(256, 256), keep_ratio=False),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        transforms=[
            dict(type='Resize', img_scale=(256, 256), keep_ratio=False),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=6,
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

runner = dict(type='EpochBasedRunner', max_epochs=12)

load_from = '/mnt/mm_ws/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

checkpoint_config = dict(create_symlink=False)

work_dir = '/mnt/mm_ws/fire_detection/fasterrcnn'
