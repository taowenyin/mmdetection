from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import MLPMIXER, MLPMIXER_LAYER_SEQUENCE, build_mlpmixer_layer_sequence
from mmcv.cnn import xavier_init

@MLPMIXER.register_module()
class MlpMixer(BaseModule):
    def __init__(self, encoder=None, decoder=None, init_cfg=None):
        super(MlpMixer, self).__init__(init_cfg=init_cfg)

        self.encoder = build_mlpmixer_layer_sequence(encoder)
        self.decoder = build_mlpmixer_layer_sequence(decoder)
        self.embed_dims = self.encoder.embed_dims

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self):
        return None



@MLPMIXER_LAYER_SEQUENCE.register_module()
class MlpMixerLayerSequence(BaseModule):
    def __init__(self):
        super(MlpMixerLayerSequence, self).__init__()


@MLPMIXER_LAYER_SEQUENCE.register_module()
class DemmMlpMixerEncoder(MlpMixerLayerSequence):
    def __init__(self):
        super(DemmMlpMixerEncoder, self).__init__()


@MLPMIXER_LAYER_SEQUENCE.register_module()
class DemmMlpMixerDecoder(MlpMixerLayerSequence):
    def __init__(self):
        super(DemmMlpMixerEncoder, self).__init__()


@MLPMIXER.register_module()
class DemmMlpMixer(MlpMixer):
    def __init__(self):
        super(DemmMlpMixer, self).__init__()

    def forward(self):
        return None

