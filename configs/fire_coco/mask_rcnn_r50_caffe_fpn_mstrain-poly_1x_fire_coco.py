_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# 修改数据集相关设置
dataset_type = 'CocoDataset'
classes = ('fire',)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        img_prefix='data/fire_coco/train2017/',
        classes=classes,
        ann_file='data/fire_coco/annotations/instances_train2017.json'),
    val=dict(
        img_prefix='data/fire_coco/train2017/',
        classes=classes,
        ann_file='data/fire_coco/annotations/instances_val2017.json'),
    test=dict(
        img_prefix='data/fire_coco/val2017/',
        classes=classes,
        ann_file='data/fire_coco/annotations/instances_test2017.json'))

# 我们可以使用预训练的 Mask R-CNN 来获取更好的性能
load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'