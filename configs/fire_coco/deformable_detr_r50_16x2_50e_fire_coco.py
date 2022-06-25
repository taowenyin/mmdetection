_base_ = '../deformable_detr/deformable_detr_r50_16x2_50e_coco.py'

# 定义类别个数
model = dict(
    bbox_head=dict(
        num_classes=1))

# 修改数据集相关设置
dataset_type = 'CocoDataset'
classes = ('fire',)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        img_prefix='/mnt/Dataset/Fire_COCO/train2017/',
        classes=classes,
        ann_file='/mnt/Dataset/Fire_COCO/annotations/instances_train2017.json'),
    val=dict(
        img_prefix='/mnt/Dataset/Fire_COCO/val2017/',
        classes=classes,
        ann_file='/mnt/Dataset/Fire_COCO/annotations/instances_val2017.json'),
    test=dict(
        img_prefix='/mnt/Dataset/Fire_COCO/val2017/',
        classes=classes,
        ann_file='/mnt/Dataset/Fire_COCO/annotations/instances_val2017.json'))

# 每5个EPOCH保存一个模型参数
checkpoint_config = dict(interval=5)
# 每训练5个数据保存一个Log数据
log_config = dict(interval=5)
load_from = 'checkpoints/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth'
# 设置最大的EPOCH数量
runner = dict(type='EpochBasedRunner', max_epochs=50)