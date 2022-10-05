import warnings

import torch

from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class DEMM(SingleStageDetector):

    def __init__(self,
                 backbone,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(DEMM, self).__init__(backbone, None, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)

    # 重载`forward_dummy`函数，因为前向的bbox_head过程需要图像信息
    def forward_dummy(self, img):
        """用于计算网络的FLOPs.

        参考 `mmdetection/tools/analysis_tools/get_flops.py`
        """
        warnings.warn('Warning! MultiheadAttention in DETR does not '
                      'support flops computation! Do not use the '
                      'results in your papers!')

        batch_size, _, height, width = img.shape
        dummy_img_metas = [
            dict(
                batch_input_shape=(height, width),
                img_shape=(height, width, 3)) for _ in range(batch_size)
        ]
        x = self.extract_feat(img)
        outs = self.bbox_head(x, dummy_img_metas)
        return outs
