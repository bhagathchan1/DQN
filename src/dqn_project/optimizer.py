"""Centered RMSProp with epsilon inside sqrt, as in the released Nature DQN code."""
import torch


class NatureRMSprop(torch.optim.Optimizer):
    def __init__(self, params, lr=0.00025, decay=0.95, eps=0.01):
        super().__init__(params, dict(lr=lr, decay=decay, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                if not state:
                    state['mean'] = torch.zeros_like(parameter)
                    state['square'] = torch.zeros_like(parameter)
                decay = group['decay']
                state['mean'].mul_(decay).add_(gradient, alpha=1-decay)
                state['square'].mul_(decay).addcmul_(gradient, gradient, value=1-decay)
                denominator = (state['square'] - state['mean'].square()).clamp_min_(0).add_(group['eps']).sqrt_()
                parameter.addcdiv_(gradient, denominator, value=-group['lr'])
        return loss
