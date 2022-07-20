```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -U openmim
mim install mmcv-full
pip install -v -e .
pip install timm
```

FCOS训练指令

```bash
python tools/train.py ./configs/mtfire/fcos_r50_caffe_fpn_gn-head_1x_coco.py
```

Faster-RCNN训练指令

```bash
python tools/train.py ./configs/mtfire/faster_rcnn_r50_fpn_1x_coco.py
```

MTFire训练指令

```bash
python tools/train.py ./configs/mtfire/mtfire_mt_fpn_attention_1x_fire.py
```