FCOS训练指令

```bash
python tools/train.py ./configs/mtfire/fcos_r50_caffe_fpn_gn-head_1x_coco.py --work-dir=./fire_detection/
```

Faster-RCNN训练指令

```bash
python tools/train.py ./configs/mtfire/faster_rcnn_r50_fpn_1x_coco.py --work-dir=./fire_detection/
```

MTFire训练指令

```bash
python tools/train.py ./configs/mtfire/mtfire_mt_fpn_attention_1x_fire.py --work-dir=./fire_detection/
```