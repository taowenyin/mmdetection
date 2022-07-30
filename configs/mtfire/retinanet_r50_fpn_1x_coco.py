_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

dataset_type = 'CocoDataset'

# Linux
# data_root = '/home/taowenyin/MyCode/Dataset/fire_coco/'
# MatPool
data_root = '/mnt/dataset/fire_coco/'
# Windows
# data_root = 'D:/MyCode/Dataset/voc2007/coco/'
# MAC
# data_root = '/Users/taowenyin/Database/voc2012/coco/'

classes = ('fire',)

model = dict(
    bbox_head=dict(
        num_classes=1,
    ),
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
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
        img_prefix=data_root + 'val2017/')
)

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

load_from = './checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'

checkpoint_config = dict(create_symlink=False)

work_dir = './fire_detection/retinanet'