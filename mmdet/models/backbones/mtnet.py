import torch.nn as nn

from mmcv.runner import BaseModule
from functools import partial
from timm.models.layers import to_2tuple
from ..builder import BACKBONES


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super(PatchEmbed, self).__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


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

        self.patch_embed_a = PatchEmbed(
            img_size=img_size // 2, patch_size=2, in_chans=stem_channel, embed_dim=embed_dims[0])
        self.patch_embed_b = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed_c = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed_d = PatchEmbed(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        self.relative_pos_a = nn.Parameter(torch.randn(
            num_heads[0], self.patch_embed_a.num_patches,
            self.patch_embed_a.num_patches // sr_ratios[0] // sr_ratios[0]))
        self.relative_pos_b = nn.Parameter(torch.randn(
            num_heads[1], self.patch_embed_b.num_patches,
            self.patch_embed_b.num_patches // sr_ratios[1] // sr_ratios[1]))
        self.relative_pos_c = nn.Parameter(torch.randn(
            num_heads[2], self.patch_embed_c.num_patches,
            self.patch_embed_c.num_patches // sr_ratios[2] // sr_ratios[2]))
        self.relative_pos_d = nn.Parameter(torch.randn(
            num_heads[3], self.patch_embed_d.num_patches,
            self.patch_embed_d.num_patches // sr_ratios[3] // sr_ratios[3]))

    def forward(self, x):
        print('')

        return x
