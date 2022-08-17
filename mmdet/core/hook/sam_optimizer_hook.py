import torch

from typing import Optional
from ..optimizers.sam import SAM
from mmcv.runner.hooks import HOOKS, OptimizerHook


@HOOKS.register_module()
class SamOptimizerHooK(OptimizerHook):
    def __init__(self, grad_clip: Optional[dict] = None):
        super().__init__(grad_clip)

    def after_train_iter(self, runner):
        if isinstance(runner.optimizer, SAM):
            if self.detect_anomalous_params:
                self.detect_anomalous_parameters(runner.outputs['loss'], runner)

            # SAM
            runner.outputs['loss'].backward(retain_graph=True)
            runner.optimizer.first_step(zero_grad=True)

            runner.outputs['loss'].backward()
            runner.optimizer.second_step(zero_grad=True)
        else:
            runner.optimizer.zero_grad()
            if self.detect_anomalous_params:
                self.detect_anomalous_parameters(runner.outputs['loss'], runner)
            runner.outputs['loss'].backward()

            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                             runner.outputs['num_samples'])
            runner.optimizer.step()
