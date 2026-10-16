import math

import torch


__notice__ = 'Modified for UNT-STLLM from Ranger with Gradient Centralization; see THIRD_PARTY.md and licenses/Apache-2.0.txt.'


class Ranger(torch.optim.Optimizer):
    def __init__(self, params, lr=.003, weight_decay=.0001):
        if lr <= 0 or weight_decay < 0:
            raise ValueError('Invalid optimizer hyperparameters')
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, betas=(.95, .999),
                                     eps=1e-5, alpha=.5, k=6))
        self.rectification = [[None, None, None] for _ in range(10)]

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.data.float()
                if gradient.is_sparse:
                    raise RuntimeError('Sparse gradients are not supported')
                current = parameter.data.float()
                state = self.state[parameter]
                if not state:
                    state.update(step=0, exp_avg=torch.zeros_like(current), exp_avg_sq=torch.zeros_like(current),
                                 slow_buffer=parameter.data.clone())
                state['exp_avg'] = state['exp_avg'].type_as(current)
                state['exp_avg_sq'] = state['exp_avg_sq'].type_as(current)
                if gradient.dim() > 1:
                    gradient.add_(-gradient.mean(dim=tuple(range(1, gradient.dim())), keepdim=True))
                state['step'] += 1
                first, second = state['exp_avg'], state['exp_avg_sq']
                second.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                first.mul_(beta1).add_(gradient, alpha=1 - beta1)
                cache = self.rectification[state['step'] % 10]
                if cache[0] == state['step']:
                    effective, step_size = cache[1:]
                else:
                    power = beta2 ** state['step']
                    maximum = 2 / (1 - beta2) - 1
                    effective = maximum - 2 * state['step'] * power / (1 - power)
                    if effective > 5:
                        step_size = math.sqrt((1 - power) * (effective - 4) / (maximum - 4)
                                              * (effective - 2) / effective * maximum / (maximum - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1 / (1 - beta1 ** state['step'])
                    cache[:] = [state['step'], effective, step_size]
                if group['weight_decay']:
                    current.add_(current, alpha=-group['weight_decay'] * group['lr'])
                if effective > 5:
                    denominator = second.sqrt().add_(group['eps'])
                    current.addcdiv_(first, denominator, value=-step_size * group['lr'])
                else:
                    current.add_(first, alpha=-step_size * group['lr'])
                parameter.data.copy_(current)
                if state['step'] % group['k'] == 0:
                    state['slow_buffer'].add_(parameter.data - state['slow_buffer'], alpha=group['alpha'])
                    parameter.data.copy_(state['slow_buffer'])
        return None
