from .builder import OPTIMIZER_BUILDERS
from torch.optim import Optimizer


'''
https://github.com/davda54/sam
'''


@OPTIMIZER_BUILDERS.register_module()
class SAM(Optimizer):
    def __init__(self, params, lr=0.1, momentum=0.9):
        super().__init__(params, {lr: lr, momentum: momentum})
