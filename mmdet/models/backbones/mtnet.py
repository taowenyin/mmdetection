import torch.nn as nn

from ..builder import BACKBONES


@BACKBONES.register_module()
class MTNet(nn.Module):
    def __init__(self):
        print('')

    def forward(self, x):
        print('')

        return x