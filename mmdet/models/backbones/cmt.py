import torch.nn as nn
import torch

from collections import OrderedDict
from mmcv.runner import BaseModule
from functools import partial
from timm.models.layers import to_2tuple, DropPath, trunc_normal_
from torch.nn.modules.batchnorm import _BatchNorm

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


@BACKBONES.register_module()
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
        frozen_stages: 冻结的层
        norm_eval: 确保归一化层冻结
        attn_drop_rate:
        drop_path_rate:
        hybrid_backbone:
        norm_layer:
        depths: 表示每个Stage的Block的数量
        qk_ratio:
        sr_ratios:
        dp:
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
                 frozen_stages=1,
                 norm_eval=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 hybrid_backbone=None,
                 norm_layer=None,
                 depths=None,
                 qk_ratio=1,
                 sr_ratios=None,
                 dp=0.1,
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
        self.norm_eval = norm_eval

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

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.blocks_a = nn.ModuleList([
            Block(
                dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        cur += depths[0]
        self.blocks_b = nn.ModuleList([
            Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        cur += depths[1]
        self.blocks_c = nn.ModuleList([
            Block(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        cur += depths[2]
        self.blocks_d = nn.ModuleList([
            Block(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, qk_ratio=qk_ratio, sr_ratio=sr_ratios[3])
            for i in range(depths[3])])

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                # todo embed_dims ?
                ('fc', nn.Linear(embed_dims[3], representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        self.apply(self._init_weights)

        # 冻结网络
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
            for m in [self.stem_conv1, self.stem_relu1, self.stem_norm1]:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

            for m in [self.stem_conv2, self.stem_relu2, self.stem_norm2]:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

            for m in [self.stem_conv3, self.stem_relu3, self.stem_norm3]:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

        for stage, block in enumerate([self.blocks_a, self.blocks_b,
                                       self.blocks_c, self.blocks_d]):
            if stage < self.frozen_stages:
                block.eval()
                for param in block.parameters():
                    param.requires_grad = False
        for stage, patch_embed in enumerate([self.patch_embed_a, self.patch_embed_b,
                                             self.patch_embed_c, self.patch_embed_d]):
            if stage < self.frozen_stages:
                patch_embed.eval()
                for param in patch_embed.parameters():
                    param.requires_grad = False

        # for stage, relative_pos in enumerate([self.relative_pos_a, self.relative_pos_b,
        #                                      self.relative_pos_c, self.relative_pos_d]):
        #     if stage < self.frozen_stages:
        #         relative_pos.requires_grad = False

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, Attention):
                m.update_temperature()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        # [B, C, H, W]
        B = x.shape[0]
        # [B, C, H, W] -> [B, stem_channel, H / 2, W / 2]
        x = self.stem_conv1(x)
        x = self.stem_relu1(x)
        x = self.stem_norm1(x)

        # [B, stem_channel, H / 2, W / 2] -> [B, stem_channel, H / 2, W / 2]
        x = self.stem_conv2(x)
        x = self.stem_relu2(x)
        x = self.stem_norm2(x)

        # [B, stem_channel, H / 2, W / 2] -> [B, stem_channel, H / 2, W / 2]
        x = self.stem_conv3(x)
        x = self.stem_relu3(x)
        x = self.stem_norm3(x)

        # FPN的输入特征
        outs = []

        # [B, stem_channel, H / 2, W / 2] -> [B, (H / 2 / Patch_Size) * (H / 2 / Patch_Size), embed_dims]
        x, (H, W) = self.patch_embed_a(x)
        for i, blk in enumerate(self.blocks_a):
            x = blk(x, H, W, self.relative_pos_a)
        # [B, (H / 2 / Patch_Size^{1}) * (W / 2 / Patch_Size^{1}), embed_dims] ->
        # [B, embed_dims, (H / 2 / Patch_Size^{1}), (W / 2 / Patch_Size^{1})]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # 添加FPN特征
        outs.append(x)

        # [B, (H / 2 / Patch_Size^{2}) * (W / 2 / Patch_Size^{2}), embed_dims]
        x, (H, W) = self.patch_embed_b(x)
        for i, blk in enumerate(self.blocks_b):
            x = blk(x, H, W, self.relative_pos_b)
        # [B, (H / 2 / Patch_Size^{2}) * (W / 2 / Patch_Size^{2}), embed_dims] ->
        # [B, embed_dims, (H / 2 / Patch_Size^{2}), (W / 2 / Patch_Size^{2})]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # 添加FPN特征
        outs.append(x)

        # [B, (H / 2 / Patch_Size^{3}) * (W / 2 / Patch_Size^{3}), embed_dims]
        x, (H, W) = self.patch_embed_c(x)
        for i, blk in enumerate(self.blocks_c):
            x = blk(x, H, W, self.relative_pos_c)
        # [B, (H / 2 / Patch_Size^{3}) * (W / 2 / Patch_Size^{3}), embed_dims] ->
        # [B, embed_dims, (H / 2 / Patch_Size^{3}), (W / 2 / Patch_Size^{3})]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # 添加FPN特征
        outs.append(x)

        # [B, (H / 2 / Patch_Size^{4}) * (W / 2 / Patch_Size^{4}), embed_dims]
        x, (H, W) = self.patch_embed_d(x)
        for i, blk in enumerate(self.blocks_d):
            x = blk(x, H, W, self.relative_pos_d)
        # [B, (H / 2 / Patch_Size^{4}) * (W / 2 / Patch_Size^{4}), embed_dims] ->
        # [B, embed_dims, (H / 2 / Patch_Size^{4}), (W / 2 / Patch_Size^{4})]
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
    model = CMT(frozen_stages=2)

    pretrain = torch.load('../../../checkpoints/cmt_tiny_mm_wo_rp.pth', map_location=torch.device('cpu'))
    # 载入与训练参数
    model.load_state_dict(pretrain['state_dict'])

    output = model(input)

    print('')