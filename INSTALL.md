```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -U openmim
mim install mmcv-full
pip install -v -e .
pip install timm
```

# Anchor-Base方法

## Faster-RCNN

### 训练指令

```bash
python tools/train.py ./configs/mtfire/faster_rcnn_r50_fpn_1x_coco.py
python tools/train.py ./configs/voc_test/faster_rcnn_r50_fpn_1x_coco.py
```

### 模型复杂度指令

```bash
python tools/analysis_tools/get_flops.py ./configs/mtfire/faster_rcnn_r50_fpn_1x_coco.py --shape 800 1280
```

### 模型FPS指令

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py ./configs/mtfire/faster_rcnn_r50_fpn_1x_coco.py ./checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --launcher pytorch
```

## RetinNet

### 训练指令

```bash
python tools/train.py ./configs/mtfire/retinanet_r50_fpn_1x_coco.py
```

### 模型复杂度指令

```bash
python tools/analysis_tools/get_flops.py ./configs/mtfire/retinanet_r50_fpn_1x_coco.py --shape 800 1280
```

### 模型FPS指令

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py ./configs/mtfire/retinanet_r50_fpn_1x_coco.py ./checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth --launcher pytorch
```

## YoloV3

### 训练指令

```bash
python tools/train.py ./configs/mtfire/yolov3_mobilenetv2_320_300e_coco.py
```

### 模型复杂度指令

```bash
python tools/analysis_tools/get_flops.py ./configs/mtfire/yolov3_mobilenetv2_320_300e_coco.py --shape 800 1280
```

### 模型FPS指令

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py ./configs/mtfire/yolov3_mobilenetv2_320_300e_coco.py ./checkpoints/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth --launcher pytorch
```

# Anchor-Free方法

## FCOS

### 训练指令

```bash
python tools/train.py ./configs/mtfire/fcos_r50_caffe_fpn_gn-head_1x_coco.py
python tools/train.py ./configs/voc_test/fcos_r50_caffe_fpn_gn-head_1x_coco.py
python tools/train.py ./configs/voc_test/mobilenet_fcos_r50_caffe_fpn_gn-head_1x_coco.py
```

### 模型复杂度指令

```bash
python tools/analysis_tools/get_flops.py ./configs/mtfire/fcos_r50_caffe_fpn_gn-head_1x_coco.py --shape 800 1280
```

### 模型FPS指令

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py ./configs/mtfire/fcos_r50_caffe_fpn_gn-head_1x_coco.py ./checkpoints/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth --launcher pytorch
```

## ATSS

### 训练指令

```bash
python tools/train.py ./configs/mtfire/atss_r50_fpn_1x_coco.py
```

### 模型复杂度指令

```bash
python tools/analysis_tools/get_flops.py ./configs/mtfire/atss_r50_fpn_1x_coco.py --shape 800 1280
```

### 模型FPS指令

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py ./configs/mtfire/atss_r50_fpn_1x_coco.py ./checkpoints/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth --launcher pytorch
```

## AutoAssign

### 训练指令

```bash
python tools/train.py ./configs/mtfire/autoassign_r50_fpn_8x2_1x_coco.py
```

### 模型复杂度指令

```bash
python tools/analysis_tools/get_flops.py ./configs/mtfire/autoassign_r50_fpn_8x2_1x_coco.py --shape 800 1280
```

### 模型FPS指令

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py ./configs/mtfire/autoassign_r50_fpn_8x2_1x_coco.py ./checkpoints/auto_assign_r50_fpn_1x_coco_20210413_115540-5e17991f.pth --launcher pytorch
```

# Transform

## DeformableDetr

### 训练指令

```bash
python tools/train.py ./configs/mtfire/deformable_detr_r50_16x2_50e_coco.py
```

### 模型复杂度指令

```bash
python tools/analysis_tools/get_flops.py ./configs/mtfire/deformable_detr_r50_16x2_50e_coco.py --shape 800 1280
```

### 模型FPS指令

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py ./configs/mtfire/deformable_detr_r50_16x2_50e_coco.py ./checkpoints/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth --launcher pytorch
```

# 自定义模型

## MTFire

### 训练指令

#### Fire

```bash
python tools/train.py ./configs/mtfire/mtfire_cmt_fcoshead_fpn_attention_1x_fire.py
python tools/train.py ./configs/mtfire/mtfire_cmt_atss_fpn_attention_1x_fire.py
python ./tools/dist_train.sh ./configs/mtfire/mtfire_cmt_atss_fpn_attention_1x_fire.py
```

```bash
bash ./tools/dist_train.sh ./configs/mtfire/mtfire_cmt_atss_fpn_attention_1x_fire.py 4
```

#### VOC

```bash
python tools/train.py ./configs/voc_test/mtfire_cmt_base_fpn_attention_1x_fire.py
python tools/train.py ./configs/voc_test/mtfire_cmt_small_fpn_attention_1x_fire.py
python tools/train.py ./configs/voc_test/mtfire_cmt_xs_fpn_attention_1x_fire.py
python tools/train.py ./configs/voc_test/mtfire_cmt_tiny_fpn_attention_1x_fire.py
```

### 模型复杂度指令

#### Fire

```bash
python tools/analysis_tools/get_flops.py ./configs/mtfire/mtfire_cmt_atss_fpn_attention_1x_fire.py --shape 256 256
```

#### VOC

```bash
python tools/analysis_tools/get_flops.py ./configs/voc_test/mtfire_cmt_base_fpn_attention_1x_fire.py --shape 256 256
python tools/analysis_tools/get_flops.py ./configs/voc_test/mtfire_cmt_small_fpn_attention_1x_fire.py --shape 224 224
python tools/analysis_tools/get_flops.py ./configs/voc_test/mtfire_cmt_xs_fpn_attention_1x_fire.py --shape 192 192
python tools/analysis_tools/get_flops.py ./configs/voc_test/mtfire_cmt_tiny_fpn_attention_1x_fire.py --shape 160 160
```

### 模型FPS指令

#### Fire

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py ./configs/mtfire/mtfire_cmt_atss_fpn_attention_1x_fire.py ./fire_detection/mtfire/atss/epoch_12.pth --launcher pytorch
```

#### VOC

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py ./configs/voc_test/mtfire_cmt_base_fpn_attention_1x_fire.py ./voc_test/mtfire/base/epoch_12.pth --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py ./configs/voc_test/mtfire_cmt_small_fpn_attention_1x_fire.py ./voc_test/mtfire/small/epoch_12.pth --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py ./configs/voc_test/mtfire_cmt_xs_fpn_attention_1x_fire.py ./voc_test/mtfire/xs/epoch_12.pth --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py ./configs/voc_test/mtfire_cmt_tiny_fpn_attention_1x_fire.py ./voc_test/mtfire/tiny/epoch_12.pth --launcher pytorch
```

# 模型间比较

## AP

```bash
python tools/analysis_tools/analyze_logs.py plot_curve ./fire_detection/fasterrcnn/20220730_132346.log.json ./fire_detection/retinanet/20220729_154416.log.json ./fire_detection/fcos/20220719_034237.log.json ./fire_detection/autoassign/20220729_150647.log.json ./fire_detection/atss/20220729_144919.log.json ./fire_detection/yolov3/20220730_132105.log.json --legend Faster-RCNN RetinaNet FCOS AutoAssign ATSS YOLOv3 --keys bbox_mAP
```