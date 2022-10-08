import torch
import torch.nn as nn

from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN
from mmdet.core import build_assigner, build_sampler
from mmdet.models.utils import build_mlpmixer
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
from mmcv.runner import force_fp32


@HEADS.register_module()
class DEMMHead(AnchorFreeHead):
    """
    Args:
        num_classes (int): 分类的类别数
        in_channels (int): 输入特征图的通道数
        num_query (int): MLP-Mixer的Query数
        num_reg_fcs (int, optional): FFN的层数。默认为2
        mlp_mixer (obj:`mmcv.ConfigDict`|dict): MLP-Mixer的配置
        sync_cls_avg_factor：默认为False
        loss_cls (obj:`mmcv.ConfigDict`|dict): 分类损失。默认为CrossEntropyLoss
        loss_bbox (obj:`mmcv.ConfigDict`|dict): BBox的回归损失。默认为L1Loss
        loss_iou (obj:`mmcv.ConfigDict`|dict): IOU的回归损失。默认为GIoULoss
        tran_cfg (obj:`mmcv.ConfigDict`|dict): MLP-Mixer的训练配置
        test_cfg (obj:`mmcv.ConfigDict`|dict): MLP-Mixer的测试配置
        init_cfg (dict or list[dict], optional): 初始化配置
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=100,
                 num_reg_fcs=2,
                 mlp_mixer=None,
                 sync_cls_avg_factor=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)

        if class_weight is not None and (self.__class__ is DEMMHead):
            assert isinstance(class_weight, float), f'class_weight的类型应该位float，但现在是{type(class_weight)}'
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), f'bg_cls_weight的类型应该位float，但现在是{type(bg_cls_weight)}'
            class_weight = torch.ones(num_classes + 1) * class_weight
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, '当设置了train_cfg，那么应该包含assigner属性'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                '分类损失的权重loss_cls应该和assigner中的分类损失权重相同'
            assert loss_bbox['loss_weight'] == assigner['reg_cost']['weight'], \
                'BBox的L1回归损失权重loss_bbox应该和assigner中的回归损失权重相同'
            assert loss_iou['loss_weight'] == assigner['iou_cost']['weight'], \
                'IOU损失权重loss_iou应该和assigner中的IOU损失权重相同.'
            self.assigner = build_assigner(assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.act_cfg = mlp_mixer.get('act_cfg',
                                     dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)

        self.mlp_mixer = build_mlpmixer(mlp_mixer)
        self.embed_dims = self.mlp_mixer.embed_dims

    def _init_layers(self):
        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)
        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        self.reg_ffn = FFN(
            self.embed_dims,
            self.embed_dims,
            self.num_reg_fcs,
            self.act_cfg,
            dropout=0.0,
            add_residual=False)
        self.fc_reg = Linear(self.embed_dims, 4)
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):

        return None

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores_list,
             all_bbox_preds_list,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None):

        return None






