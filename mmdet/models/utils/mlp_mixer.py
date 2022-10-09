import torch
import copy

from mmcv.runner.base_module import BaseModule, ModuleList
from mmdet.models.utils.builder import (MLPMIXER, MLPMIXER_LAYER_SEQUENCE, MLPMIXER_LAYER,
                                        build_mlpmixer_layer_sequence, build_mlpmixer_layer)
from mmcv.cnn import xavier_init


@MLPMIXER.register_module()
class MlpMixer(BaseModule):

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

    def __init__(self, arch='base', encoder=None, decoder=None, init_cfg=None):
        super(MlpMixer, self).__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'tokens_mlp_dims',
                'channels_mlp_dims'
            }
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.tokens_mlp_dims = self.arch_settings['tokens_mlp_dims']
        self.channels_mlp_dims = self.arch_settings['channels_mlp_dims']

        self.encoder = build_mlpmixer_layer_sequence(encoder)
        self.decoder = build_mlpmixer_layer_sequence(decoder)
        self.embed_dims = self.encoder.embed_dims

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

    def forward(self):
        return None


@MLPMIXER_LAYER.register_module()
class BaseMlpMixerLayer(BaseModule):
    def __init__(self,
                 mixer_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        super(BaseMlpMixerLayer, self).__init__(init_cfg)

    def forward(self):
        return None


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

