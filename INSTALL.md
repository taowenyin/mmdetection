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

## Faster-RCNN训练指令

```bash
python tools/train.py ./configs/mtfire/faster_rcnn_r50_fpn_1x_coco.py
python tools/train.py ./configs/voc_test/faster_rcnn_r50_fpn_1x_coco.py
```

## RetinNet训练指令

```bash
python tools/train.py ./configs/mtfire/retinanet_r50_fpn_1x_coco.py
```

## YoloV3训练指令

```bash
python tools/train.py ./configs/mtfire/yolov3_mobilenetv2_320_300e_coco.py
```

# Anchor-Free方法

## FCOS训练指令

```bash
python tools/train.py ./configs/mtfire/fcos_r50_caffe_fpn_gn-head_1x_coco.py
python tools/train.py ./configs/voc_test/fcos_r50_caffe_fpn_gn-head_1x_coco.py
python tools/train.py ./configs/voc_test/mobilenet_fcos_r50_caffe_fpn_gn-head_1x_coco.py
```

## ATSS训练指令

```bash
python tools/train.py ./configs/mtfire/atss_r50_fpn_1x_coco.py
```

## AutoAssign训练指令

```bash
python tools/train.py ./configs/mtfire/autoassign_r50_fpn_8x2_1x_coco.py
```

# 自定义方法

MTFire训练指令

```bash
python tools/train.py ./configs/mtfire/mtfire_mt_fpn_attention_1x_fire.py
python tools/train.py ./configs/voc_test/mtfire_mt_fpn_attention_1x_fire.py
```

