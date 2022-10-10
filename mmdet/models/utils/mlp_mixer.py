import torch
import torch.nn as nn
import copy

from mmcv.runner.base_module import BaseModule, ModuleList
from mmdet.models.utils.builder import (MLPMIXER, MLPMIXER_LAYER_SEQUENCE, MLPMIXER_LAYER,
                                        build_mlpmixer_layer_sequence, build_mlpmixer_layer)
from mmcv.cnn import xavier_init, build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed


arch_zoo = {
    **dict.fromkeys(
        ['s', 'small'], {
            'embed_dims': 512,
            'num_layers': 8,
            'tokens_mlp_dims': 256,
            'channels_mlp_dims': 2048,
        }),
    **dict.fromkeys(
        ['b', 'base'], {
            'embed_dims': 768,
            'num_layers': 12,
            'tokens_mlp_dims': 384,
            'channels_mlp_dims': 3072,
        }),
    **dict.fromkeys(
        ['l', 'large'], {
            'embed_dims': 1024,
            'num_layers': 24,
            'tokens_mlp_dims': 512,
            'channels_mlp_dims': 4096,
        }),
}


@MLPMIXER.register_module()
class MlpMixer(BaseModule):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 encoder=None,
                 decoder=None,
                 init_cfg=None):
        super(MlpMixer, self).__init__(init_cfg=init_cfg)

        # if isinstance(arch, str):
        #     arch = arch.lower()
        #     assert arch in set(self.arch_zoo), \
        #         f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
        #     self.arch_settings = self.arch_zoo[arch]
        # else:
        #     essential_keys = {
        #         'embed_dims', 'num_layers', 'tokens_mlp_dims',
        #         'channels_mlp_dims'
        #     }
        #     assert isinstance(arch, dict) and set(arch) == essential_keys, \
        #         f'Custom arch needs a dict with keys {essential_keys}'
        #     self.arch_settings = arch
        #
        # self.tokens_mlp_dims = self.arch_settings['tokens_mlp_dims']
        # self.channels_mlp_dims = self.arch_settings['channels_mlp_dims']
        # self.embed_dims = self.arch_settings['embed_dims']

        self.encoder = build_mlpmixer_layer_sequence(encoder)
        self.decoder = build_mlpmixer_layer_sequence(decoder)

        self.embed_dims = self.encoder.embed_dims
        _patch_cfg = dict(
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
        )

        # 创建Patch Embed对象
        self.patch_embed = PatchEmbed(**_patch_cfg)

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, x, query_embed):
        bs, c, h, w = x.shape
        # [bs, c, h, w] -> [h*w, bs, c]
        x = x.view(bs, c, -1).permute(2, 0, 1)
        # [num_query, dim] -> [num_query, bs, dim]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        memory = self.encoder(x)
        target = torch.zeros_like(query_embed)
        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(query=target, memory=memory)

        out_dec = out_dec.transpose(1, 2)
        memory = memory.permute(1, 2, 0).reshape(bs, c, h, w)

        return out_dec, memory


@MLPMIXER_LAYER_SEQUENCE.register_module()
class MlpMixerLayerSequence(BaseModule):
    def __init__(self, mlpmixerlayers=None, num_layers=None, init_cfg=None):
        super(MlpMixerLayerSequence, self).__init__(init_cfg)

        if isinstance(mlpmixerlayers, dict):
            mlpmixerlayers = [
                copy.deepcopy(mlpmixerlayers) for _ in range(num_layers)
            ]
        else:
            assert isinstance(mlpmixerlayers, list) and len(mlpmixerlayers) == num_layers
        self.num_layers = num_layers
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(build_mlpmixer_layer(mlpmixerlayers[i]))
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm

    def forward(self, x):
        return None


@MLPMIXER_LAYER.register_module()
class BaseMlpMixerLayer(BaseModule):
    def __init__(self,
                 num_tokens,
                 embed_dims,
                 tokens_mlp_dims,
                 channels_mlp_dims,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(BaseMlpMixerLayer, self).__init__(init_cfg)

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.token_mix = FFN(
            embed_dims=num_tokens,
            feedforward_channels=tokens_mlp_dims,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=False)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)
        self.channel_mix = FFN(
            embed_dims=embed_dims,
            feedforward_channels=channels_mlp_dims,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def init_weights(self):
        super(BaseMlpMixerLayer, self).init_weights()
        for m in self.token_mix.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)
        for m in self.channel_mix.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        out = self.norm1(x).transpose(1, 2)
        x = x + self.token_mix(out).transpose(1, 2)
        x = self.channel_mix(self.norm2(x), identity=x)

        return x


@MLPMIXER_LAYER.register_module()
class DemmMlpMixerEncoderLayer(BaseMlpMixerLayer):
    def __init__(self):
        super(DemmMlpMixerEncoderLayer, self).__init__()

    def forward(self):
        return None


@MLPMIXER_LAYER.register_module()
class DemmMlpMixerDecoderLayer(BaseMlpMixerLayer):
    def __init__(self):
        super(DemmMlpMixerDecoderLayer, self).__init__()

    def forward(self):
        return None



@MLPMIXER_LAYER_SEQUENCE.register_module()
class DemmMlpMixerEncoder(MlpMixerLayerSequence):
    def __init__(self, *args, **kwargs):
        super(DemmMlpMixerEncoder, self).__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        x = super(DemmMlpMixerEncoder, self).forward(*args, **kwargs)

        return x


@MLPMIXER_LAYER_SEQUENCE.register_module()
class DemmMlpMixerDecoder(MlpMixerLayerSequence):
    def __init__(self):
        super(DemmMlpMixerDecoder, self).__init__()

    def forward(self):
        return None


@MLPMIXER.register_module()
class DemmMlpMixer(MlpMixer):
    def __init__(self):
        super(DemmMlpMixer, self).__init__()

    def forward(self):
        return None

