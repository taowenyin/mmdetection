import torch.nn as nn

from mmcv.runner import BaseModule
from functools import partial
from ..builder import BACKBONES


@BACKBONES.register_module()
class MTNet(BaseModule):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=None,
                 stem_channel=16, fc_dim=1280, num_heads=None, mlp_ratios=None,
                 qkv_bias=True, qk_scale=None, representation_size=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None,
                 depths=None, qk_ratio=1, sr_ratios=None, dp=0.1,
                 init_cfg=None):
        super(MTNet, self).__init__(init_cfg)

        if sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]
        if depths is None:
            depths = [2, 2, 10, 2]
        if mlp_ratios is None:
            mlp_ratios = [3.6, 3.6, 3.6, 3.6]
        if num_heads is None:
            num_heads = [1, 2, 4, 8]
        self.num_heads = num_heads
        if embed_dims is None:
            embed_dims = [46, 92, 184, 368]

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dims[-1]
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.stem_conv1 = nn.Conv2d(in_chans, stem_channel, kernel_size=3, stride=2, padding=1, bias=True)
        self.stem_relu1 = nn.GELU()
        self.stem_norm1 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.stem_conv2 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu2 = nn.GELU()
        self.stem_norm2 = nn.BatchNorm2d(stem_channel, eps=1e-5)

        self.stem_conv3 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu3 = nn.GELU()
        self.stem_norm3 = nn.BatchNorm2d(stem_channel, eps=1e-5)


    def forward(self, x):
        print('')

        return x
