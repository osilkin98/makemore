from typing import List, Iterable
import torch
from torch.nn import Parameter
from torch import optim


# -----------------------------------------------------------------------------
# Base Optimizer Class

class Optimizer:
    def __init__(self):
        self.params = []

    def zero_grad(self, set_to_none=True):
        # we respect `set_to_none` for pytorch compatability
        for p in self.params:
            if set_to_none:
                p.grad = None
            else:
                p.grad.data = torch.zeros_like(p.data)

    def step(self):
        raise NotImplementedError()

    @torch.no_grad()
    def lr_norms(self) -> List[torch.Tensor]:
        raise NotImplementedError()

# # -----------------------------------------------------------------------------
# # Stochastic Gradient Descent

# class SGD(Optimizer):
#   def __init__(self, params: List[Parameter], lr=1e-1):
#     super().__init__()
#     self.params = [p for p in params]
#     self.lr = lr

#   @torch.no_grad()
#   def step(self):
#     # very simple optimization step
#     for p in self.params:
#       p.data = p.data - self.lr * p.grad.data

# # -----------------------------------------------------------------------------
# # SGD With Momentum

# class SGDWithMomentum(Optimizer):
#   def __init__(self, params: Iterable[Parameter], lr = 1e-1, momentum = 0.9):
#     super().__init__()
#     self.params = [p for p in params]
#     self.velocity = [torch.zeros_like(p) for p in self.params]
#     self.lr = lr
#     self.momentum = momentum

#   @torch.no_grad()
#   def step(self):
#     for i, p in enumerate(self.params):
#       self.velocity[i] = self.momentum * self.velocity[i] - self.lr * p.grad.data
#       p.data = p.data + self.velocity[i]

# # -----------------------------------------------------------------------------
# # Nesterov Accelerated Gradient

# class NAG(Optimizer):
#   def __init__(self, params: Iterable[Parameter], lr = 0.001, momentum = 0.9):
#     self.params = [p for p in params]
#     self.velocity = [None for _ in self.params]
#     self.lr = lr
#     self.momentum = momentum

#   @torch.no_grad()
#   def step(self):
#     for i, p in enumerate(self.params):
#       if self.velocity[i] is None:
#         self.velocity[i] = p.grad.data.clone().detach()
#       else:
#         self.velocity[i] = self.velocity[i] * self.momentum + p.grad.data
#       p.data = p.data - self.lr * (p.grad.data + self.velocity[i] * self.momentum)

# -----------------------------------------------------------------------------
# Adagrad, based on: https://jmlr.org/papers/v12/duchi11a.html

class Adagrad(Optimizer):
    def __init__(self, params: Iterable[Parameter], lr = 1e-2, eps = 1e-10):
        super().__init__()
        self.params = [p for p in params]
        self.grad_noise = [torch.zeros_like(p) for p in self.params]
        self.lr = lr
        self.eps = eps

    @torch.no_grad()
    def step(self):
        for i, p in enumerate(self.params):
            # update gradient noise
            self.grad_noise[i] += p.grad.data ** 2
            scaled_lr = self.lr * ((torch.sqrt(self.grad_noise[i]) + self.eps) ** -1)
            p.data = p.data - scaled_lr * p.grad.data

    @torch.no_grad()
    def lr_norms(self) -> List[torch.Tensor]:
        # collect all of the learning rates
        norms = []
        for gn in self.grad_noise:
            norms.append((self.lr * ((torch.sqrt(gn) + self.eps)**-1)).norm(p=2))
        return norms

# -----------------------------------------------------------------------------
# RMSProp, based on: https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

class RMSProp(Optimizer):
    def __init__(self, params: List[Parameter], lr=1e-2, alpha = 0.99, eps = 1e-8, momentum: float = 0):
        super().__init__()
        self.params = [p for p in params]
        self.variance = [torch.zeros_like(p) for p in self.params] 
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.velocity = [None for _ in self.params]

    @torch.no_grad()
    def step(self):
        # very simple optimization step
        for i, p in enumerate(self.params):
            grad = p.grad.data
            # compute the uncentered variance
            self.variance[i] = self.alpha * self.variance[i] + (1 - self.alpha) * (grad ** 2)

            if self.momentum > 0:
                # calculation with momentum
                if self.velocity[i] is None:
                    self.velocity[i] = torch.zeros_like(p)

                # here we scale the learning rate with the momentum + rescaled gradient
                # scaled_grad = p.grad.data * ((torch.sqrt(self.variance[i]) + self.eps) ** -1)
                # self.velocity[i] = self.momentum * self.velocity[i] + scaled_grad * (1 - self.momentum)

                grad_term = grad / (torch.sqrt(self.variance[i]) + self.eps)
                self.velocity[i] = self.momentum * self.velocity[i] + grad_term

                # now the learning rate gets distributed to both the velocity + scaled gradient
                p.data = p.data - self.lr * self.velocity[i]
            else:
                # simple calculation without momentum
                # step 1: scale the learning rate
                scaled_lr = self.lr * ((torch.sqrt(self.variance[i]) + self.eps) ** -1)

                # step 2: update!
                p.data = p.data - scaled_lr * p.grad.data

    @torch.no_grad()
    def lr_norms(self) -> List[torch.Tensor]:
        norms = []
        for var in self.variance:
            lr = self.lr * ((torch.sqrt(var) + self.eps) ** -1) 
            norms.append(lr.norm(p=2))
        return norms

# -----------------------------------------------------------------------------
# Adam optimizer, based on: https://arxiv.org/pdf/1412.6980

class Adam(Optimizer):
    def __init__(self, params: List[Parameter], lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.params = [p for p in params]
        self.momentum = [torch.zeros_like(p) for p in self.params]
        self.variance = [torch.zeros_like(p) for p in self.params]
        
        # beta values + correction accumulators
        self.b1, self.b2 = betas
        self.b1_accum, self.b2_accum = 1, 1


    @torch.no_grad()
    def step(self):
        # first update the correction accumulators
        self.b1_accum *= self.b1
        self.b2_accum *= self.b2
        for i, p in enumerate(self.params):
            # update moments
            self.momentum[i] = self.b1 * self.momentum[i] + (1 - self.b1) * p.grad.data
            self.variance[i] = self.b2 * self.variance[i] + (1 - self.b2) * (p.grad.data ** 2)

            # correct for bias towards zero
            momentum_c = self.momentum[i] / (1 - self.b1_accum)
            variance_c = self.variance[i] / (1 - self.b2_accum)

            # scale the learning rate & update
            lr = self.lr * ((torch.sqrt(variance_c) + self.eps) ** -1)
            p.data = p.data - lr * momentum_c

    @torch.no_grad()
    def lr_norms(self) -> List[torch.Tensor]:
        assert self.b2_accum != 1, "lr_norms cannot be called before making a step with Adam"
        norms = []
        for var in self.variance:
            corrected = var / (1 - self.b2_accum)
            lr = self.lr * ((torch.sqrt(corrected) + self.eps) ** -1)
            norms.append(lr.norm(p=2))
        return norms

# -----------------------------------------------------------------------------
# AdamW optimizer, based on: https://arxiv.org/pdf/1711.05101

class AdamW(Optimizer):
    def __init__(self, params: List[Parameter], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.params = [p for p in params]
        self.momentum = [None for _ in self.params]
        self.variance = [None for _ in self.params]
        
        # beta values + correction accumulators
        self.b1, self.b2 = betas
        self.b1_accum, self.b2_accum = 1, 1

    @torch.no_grad()
    def step(self):
        # first update the correction accumulators
        self.b1_accum *= self.b1
        self.b2_accum *= self.b2
        for i, p in enumerate(self.params):
            # perform decay
            p.data = p.data - self.lr * self.weight_decay * p.data

            if self.momentum[i] is None:
                self.momentum[i] = torch.zeros_like(p.data)
                self.variance[i] = torch.zeros_like(p.data)


            # update moments
            self.momentum[i] = self.b1 * self.momentum[i] + (1 - self.b1) * p.grad.data
            self.variance[i] = self.b2 * self.variance[i] + (1 - self.b2) * (p.grad.data ** 2)

            # correct for bias towards zero
            momentum_c = self.momentum[i] / (1 - self.b1_accum)
            variance_c = self.variance[i] / (1 - self.b2_accum)

            # scale the learning rate & update
            lr = self.lr * ((torch.sqrt(variance_c) + self.eps) ** -1)
            p.data = p.data - lr * momentum_c

    @torch.no_grad()
    def lr_norms(self) -> List[torch.Tensor]:
        assert self.b2_accum != 1, "lr_norms cannot be called before making a step with Adam"
        norms = []
        for var in self.variance:
            corrected = var / (1 - self.b2_accum)
            lr = self.lr * ((torch.sqrt(corrected) + self.eps) ** -1)
            norms.append(lr.norm(p=2))
        return norms


# -----------------------------------------------------------------------------
# SGD with momentum and nesterov's accelerated gradient per pytorch implementation

class SGD(Optimizer):
    def __init__(self, params: List[Parameter], lr=1e-1, momentum: float = 0.0, nesterov = False):
        super().__init__()
        self.params = [p for p in params]
        self.velocity = [None for _ in self.params] 
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov

    @torch.no_grad()
    def step(self):
        # very simple optimization step
        for i, p in enumerate(self.params):
            grad = p.grad.data 
            if self.momentum > 0:
                # will only activate when non-zero momentum was provided
                if self.velocity[i] is None:
                    self.velocity[i] = grad.clone().detach()
                else:
                    self.velocity[i] = self.momentum * self.velocity[i] + grad
                grad = self.velocity[i]
                if self.nesterov:
                    grad = grad + self.velocity[i] * self.momentum
                else:
                    grad = self.velocity[i]
            p.data = p.data - self.lr * grad


def get_optimizer(name: str, use_torch: bool) -> Optimizer | optim.Optimizer:
    # for debugging against PyTorch optimizers
    table = {
        'sgd': (SGD, optim.SGD),
        'adagrad': (Adagrad, optim.Adagrad),
        'rmsprop': (RMSProp, optim.RMSprop),
        'adam': (Adam, optim.Adam),
        'adamw': (AdamW, optim.AdamW)
    }
    if name not in table:
        raise ValueError(f'invalid optimizer {name}! Must be one of: {table.keys()}')
    _optim, _torch_optim = table[name]
    return _torch_optim if use_torch else _optim