import torch

from ..optimizers.sam import SAM
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class SamOptimizerHooK(Hook):
    def __init__(self):
        print('xxx')

    def after_train_iter(self, runner):
        if isinstance(runner.optimizer, SAM):
            loss = runner.outputs['loss']
            loss.backward()

            runner.optimizer.second_step()

            print('')


        print('')