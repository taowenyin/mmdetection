import torch.nn as nn
import torch

from collections import OrderedDict
from mmcv.runner import BaseModule
from functools import partial
from timm.models.layers import to_2tuple, DropPath, trunc_normal_
from torch.nn.modules.batchnorm import _BatchNorm
# from ..builder import BACKBONES


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

        # [B, C, H, W] -> [B, embed_dim, H / 2, W / 2] -> [B, embed_dim, H * W / 4] -> [B, H * W / 4, embed_dim]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, qk_ratio=1, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, qk_ratio=qk_ratio, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, H, W, relative_pos):
        B, N, C = x.shape
        # [B, N, C] -> [B, C, H, W]
        cnn_feat = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat
        # [B, C, H, W] -> [B, C, N] -> [B, N, C]
        x = x.flatten(2).permute(0, 2, 1)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, relative_pos))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., qk_ratio=1, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qk_dim = dim // qk_ratio

        self.q = nn.Linear(dim, self.qk_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.qk_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        # Exactly same as PVTv1
        if self.sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim, bias=True),
                nn.BatchNorm2d(dim, eps=1e-5),
            )

    def forward(self, x, H, W, relative_pos):
        B, N, C = x.shape

        # [B, N, C] -> [B, N, qk_dim] -> [B, N, num_heads, qk_dim // num_heads] ->
        # [B, num_heads, N, qk_dim // num_heads]
        q = self.q(x).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            # [B, N, C] -> [B, C, H, W]
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            # [B, C, H, W] -> [B, C, H / sr_ratio, W / sr_ratio] ->
            # [B, C, (H / sr_ratio) * (W / sr_ratio)] -> [B, (H / sr_ratio) * (W / sr_ratio), C]
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            # [B, (H / sr_ratio) * (W / sr_ratio), C] -> [B, (H / sr_ratio) * (W / sr_ratio), qk_dim] ->
            # [B, (H / sr_ratio) * (W / sr_ratio), num_heads, qk_dim / num_heads] ->
            # [B, num_heads, (H / sr_ratio) * (W / sr_ratio), qk_dim / num_heads]
            k = self.k(x_).reshape(B, -1, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
            # [B, (H / sr_ratio) * (W / sr_ratio), C] -> [B, (H / sr_ratio) * (W / sr_ratio), C] ->
            # [B, (H / sr_ratio) * (W / sr_ratio), num_heads, C / num_heads] ->
            # [B, num_heads, (H / sr_ratio) * (W / sr_ratio), C / num_heads]
            v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            k = self.k(x).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 标准计算Self-Attention的方法
        # [B, num_heads, N, (H / sr_ratio) * (W / sr_ratio)]
        attn = (q @ k.transpose(-2, -1)) * self.scale + relative_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # [B, num_heads, N, (H / sr_ratio) * (W / sr_ratio)] -> [B, num_heads, N, C / num_heads] ->
        # [B, N, num_heads, C / num_heads] -> [B, N, C]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
        )
        self.proj = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.proj_act = nn.GELU()
        self.proj_bn = nn.BatchNorm2d(hidden_features, eps=1e-5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_features, eps=1e-5),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj(x) + x
        x = self.proj_act(x)
        x = self.proj_bn(x)
        x = self.conv2(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.drop(x)
        return x


# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


# @BACKBONES.register_module()
class CMT(BaseModule):
    """
    MTFire的Backbone

    Args:
        img_size: 输入图像的大小
        in_chans: 输入图像的通道数
        num_classes: 分类数
        embed_dims: Patch Embed的维度
        stem_channel: Stem的阶段的通道数
        fc_dim: 全连接层的维度
        num_heads: MSA的数量
        mlp_ratios:
        qkv_bias:
        qk_scale:
        representation_size:
        drop_rate:
        attn_drop_rate:
        drop_path_rate:
        hybrid_backbone:
        norm_layer:
        depths: 表示每个Stage的Block的数量
        qk_ratio:
        sr_ratios:
        dp:
        frozen_stages: 冻结层的表示
        init_cfg:
    """

    def __init__(self,
                 img_size=224,
                 in_chans=3,
                 num_classes=20,
                 embed_dims=None,
                 stem_channel=16,
                 fc_dim=1280,
                 num_heads=None,
                 mlp_ratios=None,
                 qkv_bias=True,
                 qk_scale=None,
                 representation_size=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 hybrid_backbone=None,
                 norm_layer=None,
                 norm_eval=True,
                 depths=None,
                 qk_ratio=1,
                 sr_ratios=None,
                 stages=4,
                 dp=0.1,
                 frozen_stages=1,
                 init_cfg=None):
        super(CMT, self).__init__(init_cfg)

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

        self.frozen_stages = frozen_stages
        self.stages = stages
        self.norm_eval=norm_eval,

        self.stem = nn.Sequential(
            # Stem1
            nn.Conv2d(in_chans, stem_channel, kernel_size=3, stride=2, padding=1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(stem_channel, eps=1e-5),
            # Stem2
            nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(stem_channel, eps=1e-5),
            # Stem3
            nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(stem_channel, eps=1e-5)
        )

        self.patch_embed = []
        for index in range(0, self.stages):
            if index == 0:
                pe = PatchEmbed(img_size=img_size // (2 ** (index + 1)), patch_size=2,
                                in_chans=stem_channel, embed_dim=embed_dims[index])
            else:
                pe = PatchEmbed(img_size=img_size // (2 ** (index + 1)), patch_size=2,
                                in_chans=embed_dims[index - 1], embed_dim=embed_dims[index])

            self.patch_embed.append(pe)

        self.relative_pos = []
        for index in range(0, self.stages):
            rp = nn.Parameter(
                torch.randn(
                    num_heads[index],
                    self.patch_embed[index].num_patches,
                    self.patch_embed[index].num_patches // sr_ratios[index] // sr_ratios[index]
                )
            )

            self.relative_pos.append(rp)

        # 构建[0, drop_path_rate]之间sum(depths)个数字的序列
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.blocks = []
        for index in range(0, self.stages):
            b = nn.ModuleList([
                Block(
                    dim=embed_dims[index], num_heads=num_heads[index], mlp_ratio=mlp_ratios[index],
                    qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i], norm_layer=norm_layer, qk_ratio=qk_ratio,
                    sr_ratio=sr_ratios[index])
                for i in range(depths[index])
            ])
            cur += depths[index]
            block_name = f'block{index + 1}'
            self.add_module(block_name, b)
            self.blocks.append(block_name)

        # 冻结相关层
        self._freeze_stages()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            # 把Stem冻结
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False

        for index in range(1, self.frozen_stages + 1):
            pe = self.patch_embed[index - 1]
            block = self.blocks[index - 1]
            rp = self.relative_pos[index - 1]
            pe.eval()
            block.eval()

            for param in pe.parameters():
                param.requires_grad = False
            for param in block.parameters():
                param.requires_grad = False
            rp.requires_grad = False

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, Attention):
                m.update_temperature()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        # [B, C, H, W]
        B = x.shape[0]

        # [B, C, H, W] -> [B, stem_channel, H / 2, W / 2]
        x = self.stem(x)

        # FPN的输入特征
        outs = []
        for index in range(0, self.stages):
            # [B, (H / 2 / Patch_Size^{index + 1}) * (W / 2 / Patch_Size^{index + 1}), embed_dims]
            x, (H, W) = self.patch_embed[index](x)
            for i, blk_name in enumerate(self.blocks[index]):
                blk = getattr(self, blk_name)
                x = blk(x, H, W, self.relative_pos[index])

            # [B, (H / 2 / Patch_Size^{index + 1}) * (W / 2 / Patch_Size^{index + 1}), embed_dims] ->
            # [B, embed_dims, (H / 2 / Patch_Size^{index + 1}), (W / 2 / Patch_Size^{index + 1})]
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            # 添加FPN特征
            outs.append(x)

        # 输出多尺度特征
        return outs

    def forward(self, x):
        outs = self.forward_features(x)
        outs = tuple(outs)

        return outs

    def train(self, mode=True):
        super(CMT, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


if __name__ == '__main__':
    # B C H W
    input = torch.rand(20, 3, 224, 224)

    model = CMT()
    output = model(input)

    print(model)
