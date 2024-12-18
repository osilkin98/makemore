"""
you give this script some words (one per line) and it will generate more things like it.
uses super state of the art Transformer AI tech
this code is intended to be super hackable. tune it to your needs.

Changes from minGPT:
- I removed the from_pretrained function where we init with GPT2 weights
- I removed dropout layers because the models we train here are small,
  it's not necessary to understand at this stage and at this scale.
- I removed weight decay and all of the complexity around what parameters are
  and are not weight decayed. I don't believe this should make a massive
  difference at the scale that we operate on here.
"""

import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Parameter
from typing import Iterable
from torch import optim

# -----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    block_size: int = None # length of the input sequences of integers
    vocab_size: int = None # the input integers are in range [0 .. vocab_size -1]
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4

# -----------------------------------------------------------------------------
# Transformer Language Model (*exactly* as used in GPT-2)

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class Transformer(nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

# -----------------------------------------------------------------------------
# Bag of Words (BoW) language model

class CausalBoW(nn.Module):
    """
    Causal bag of words. Averages the preceding elements and looks suspiciously like
    a CausalAttention module you'd find in a transformer, for no apparent reason at all ;)
    """
    def __init__(self, config):
        super().__init__()

        # used to mask out vectors and preserve autoregressive property
        self.block_size = config.block_size
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, n_embd

        # do the weighted average of all preceeding token features
        att = torch.zeros((B, T, T), device=x.device)
        att = att.masked_fill(self.bias[:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ x # (B, T, T) x (B, T, C) -> (B, T, C)

        return y

class BoWBlock(nn.Module):
    """ collects BoW features and adds an MLP """

    def __init__(self, config):
        super().__init__()

        # Causal BoW module
        self.cbow = CausalBoW(config)
        # MLP assembler
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, config.n_embd2),
            c_proj  = nn.Linear(config.n_embd2, config.n_embd),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(F.tanh(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.cbow(x)
        x = x + self.mlpf(x)
        return x

class BoW(nn.Module):
    """
    takes the previous block_size tokens, encodes them with a lookup table,
    also encodes their positions with lookup table, then averages all of those
    embeddings up and uses that to predict the next token.
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        # token embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # position embedding
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        # context block
        self.context_block = BoWBlock(config)
        # language model head decoder layer
        self.lm_head = nn.Linear(config.n_embd, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):

        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the token and position embedding layers
        tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos) # position embeddings of shape (1, t, n_embd)
        # add and run through the decoder MLP
        x = tok_emb + pos_emb
        # run the bag of words context module
        x = self.context_block(x)
        # decode to next token probability
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

# -----------------------------------------------------------------------------
"""
Recurrent Neural Net language model: either a vanilla RNN recurrence or a GRU.
Did not implement an LSTM because its API is a bit more annoying as it has
both a hidden state and a cell state, but it's very similar to GRU and in
practice works just as well.
"""

class RNNCell(nn.Module):
    """
    the job of a 'Cell' is to:
    take input at current time step x_{t} and the hidden state at the
    previous time step h_{t-1} and return the resulting hidden state
    h_{t} at the current timestep
    """
    def __init__(self, config):
        super().__init__()
        self.xh_to_h = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
        xh = torch.cat([xt, hprev], dim=1)
        ht = F.tanh(self.xh_to_h(xh))
        return ht

class GRUCell(nn.Module):
    """
    same job as RNN cell, but a bit more complicated recurrence formula
    that makes the GRU more expressive and easier to optimize.
    """
    def __init__(self, config):
        super().__init__()
        # input, forget, output, gate
        self.xh_to_z = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_r = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)
        self.xh_to_hbar = nn.Linear(config.n_embd + config.n_embd2, config.n_embd2)

    def forward(self, xt, hprev):
        # first use the reset gate to wipe some channels of the hidden state to zero
        xh = torch.cat([xt, hprev], dim=1)
        r = F.sigmoid(self.xh_to_r(xh))
        hprev_reset = r * hprev
        # calculate the candidate new hidden state hbar
        xhr = torch.cat([xt, hprev_reset], dim=1)
        hbar = F.tanh(self.xh_to_hbar(xhr))
        # calculate the switch gate that determines if each channel should be updated at all
        z = F.sigmoid(self.xh_to_z(xh))
        # blend the previous hidden state and the new candidate hidden state
        ht = (1 - z) * hprev + z * hbar
        return ht

class RNN(nn.Module):

    def __init__(self, config, cell_type):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.start = nn.Parameter(torch.zeros(1, config.n_embd2)) # the starting hidden state
        self.wte = nn.Embedding(config.vocab_size, config.n_embd) # token embeddings table
        if cell_type == 'rnn':
            self.cell = RNNCell(config)
        elif cell_type == 'gru':
            self.cell = GRUCell(config)
        self.lm_head = nn.Linear(config.n_embd2, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()

        # embed all the integers up front and all at once for efficiency
        emb = self.wte(idx) # (b, t, n_embd)

        # sequentially iterate over the inputs and update the RNN state each tick
        hprev = self.start.expand((b, -1)) # expand out the batch dimension
        hiddens = []
        for i in range(t):
            xt = emb[:, i, :] # (b, n_embd)
            ht = self.cell(xt, hprev) # (b, n_embd2)
            hprev = ht
            hiddens.append(ht)

        # decode the outputs
        hidden = torch.stack(hiddens, 1) # (b, t, n_embd2)
        logits = self.lm_head(hidden)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

# -----------------------------------------------------------------------------
# MLP language model

class MLP(nn.Module):
    """
    takes the previous block_size tokens, encodes them with a lookup table,
    concatenates the vectors and predicts the next token with an MLP.

    Reference:
    Bengio et al. 2003 https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd) # token embeddings table
        # +1 in the line above for a special <BLANK> token that gets inserted if encoding a token
        # before the beginning of the input sequence
        self.mlp = nn.Sequential(
            nn.Linear(self.block_size * config.n_embd, config.n_embd2),
            nn.Tanh(),
            nn.Linear(config.n_embd2, self.vocab_size)
        )

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):

        # gather the word embeddings of the previous 3 words
        embs = []
        for k in range(self.block_size):
            tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocab_size # special <BLANK> token
            embs.append(tok_emb)

        # concat all of the embeddings together and pass through an MLP
        x = torch.cat(embs, -1) # (b, t, n_embd * block_size)
        logits = self.mlp(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

# -----------------------------------------------------------------------------
# Bigram language model

class Bigram(nn.Module):
    """
    Bigram Language Model 'neural net', simply a lookup table of logits for the
    next character given a previous character.
    """

    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        self.logits = nn.Parameter(torch.zeros((n, n)))

    def get_block_size(self):
        return 1 # this model only needs one previous character to predict the next

    def forward(self, idx, targets=None):

         # 'forward pass', lol
        logits = self.logits[idx]

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

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
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
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
            # perform decay
            p.data = p.data - self.lr * self.weight_decay * p.data

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
# -----------------------------------------------------------------------------
# helper functions for evaluating and sampling from the model

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def print_samples(num=10):
    """ samples from the model and pretty prints the decoded samples """
    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
    top_k = args.top_k if args.top_k != -1 else None
    steps = train_dataset.get_output_length() - 1 # -1 because we already start with <START> token (index 0)
    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')
    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        # get the i'th row of sampled integers, as python list
        row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token
        # token 0 is the <STOP> token, so we crop the output sequence at that point
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)
        # separately track samples that we have and have not seen before
        if train_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif test_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)
    print('-'*80)
    for lst, desc in [(train_samples, 'in train'), (test_samples, 'in test'), (new_samples, 'new')]:
        print(f"{len(lst)} samples that are {desc}:")
        for word in lst:
            print(word)
    print('-'*80)

@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss

@torch.no_grad()
def gradnorm(model: nn.Module) -> float:
    """
    Given a PyTorch model, computes the average of the gradnorm across all parameters.
    """
    grad_norms = []
    for p in model.parameters():
      grad_norms.append(p.grad.norm())
    return grad_norms


# -----------------------------------------------------------------------------
# helper functions for creating the training and test Datasets that emit words

class CharDataset(Dataset):

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)}
        self.itos = {i:s for s,i in self.stoi.items()} # inverse mapping

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1 # all the possible characters and special 0 token

    def get_output_length(self):
        return self.max_word_length + 1 # <START> token followed by words

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = ''.join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1 # index -1 will mask the loss at the inactive locations
        return x, y

def create_datasets(input_file):

    # preprocessing of the input text file
    with open(input_file, 'r') as f:
        data = f.read()
    words = data.splitlines()
    words = [w.strip() for w in words] # get rid of any leading or trailing white space
    words = [w for w in words if w] # get rid of any empty strings
    chars = sorted(list(set(''.join(words)))) # all the possible characters
    max_word_length = max(len(w) for w in words)
    print(f"number of examples in the dataset: {len(words)}")
    print(f"max word length: {max_word_length}")
    print(f"number of unique characters in the vocabulary: {len(chars)}")
    print("vocabulary:")
    print(''.join(chars))

    # partition the input data into a training and the test set
    test_set_size = min(1000, int(len(words) * 0.1)) # 10% of the training set, or up to 1000 examples
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]
    print(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")

    # wrap in dataset objects
    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)

    return train_dataset, test_dataset

class InfiniteDataLoader:
    """
    this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader?
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration: # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch


def get_optimizer(name: str, use_torch: bool):
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

# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser(description="Make More")
    # system/input/output
    parser.add_argument('--input-file', '-i', type=str, default='names.txt', help="input file with things one per line")
    parser.add_argument('--work-dir', '-o', type=str, default='out', help="output working directory")
    parser.add_argument('--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
    parser.add_argument('--sample-only', action='store_true', help="just sample from the model and quit, don't train")
    parser.add_argument('--num-workers', '-n', type=int, default=4, help="number of data workers for both train/test")
    parser.add_argument('--max-steps', type=int, default=-1, help="max number of optimization steps to run for, or -1 for infinite.")
    parser.add_argument('--device', type=str, default='cpu', help="device to use for compute, examples: cpu|cuda|cuda:2|mps")
    parser.add_argument('--seed', type=int, default=3407, help="seed")
    # sampling
    parser.add_argument('--top-k', type=int, default=-1, help="top-k for sampling, -1 means no top-k")
    # model
    parser.add_argument('--type', type=str, default='transformer', help="model class type to use, bigram|mlp|rnn|gru|bow|transformer")
    parser.add_argument('--n-layer', type=int, default=4, help="number of layers")
    parser.add_argument('--n-head', type=int, default=4, help="number of heads (in a transformer)")
    parser.add_argument('--n-embd', type=int, default=64, help="number of feature channels in the model")
    parser.add_argument('--n-embd2', type=int, default=64, help="number of feature channels elsewhere in the model")
    # optimization
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--use-torch-optim', action='store_true', default=False, help="Use the PyTorch optimizer implementations instead of our own.")
    parser.add_argument('--batch-size', '-b', type=int, default=32, help="batch size during optimization")
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4, help="learning rate")
    parser.add_argument('--momentum', '-m', type=float, default=0.0, help="momentum")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01, help="weight decay")
    parser.add_argument('--nesterov', action='store_true', default=False)
    parser.add_argument('--alpha', type=float, default=0.99, help='The amount by which RMSProp decays its uncentered variance')
    parser.add_argument('--beta1', type=float, default=0.9, help="Exponential decay term for the first moment estimation in Adam & AdamW")
    parser.add_argument('--beta2', type=float, default=0.999, help="Exponential decay term for the second moment estimation in Adam & AdamW")
    args = parser.parse_args()
    print(vars(args))

    # system inits
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    # init datasets
    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    print(f"dataset determined that: {vocab_size=}, {block_size=}")

    # init model
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd, n_embd2=args.n_embd2)
    if args.type == 'transformer':
        model = Transformer(config)
    elif args.type == 'bigram':
        model = Bigram(config)
    elif args.type == 'mlp':
        model = MLP(config)
    elif args.type == 'rnn':
        model = RNN(config, cell_type='rnn')
    elif args.type == 'gru':
        model = RNN(config, cell_type='gru')
    elif args.type == 'bow':
        model = BoW(config)
    else:
        raise ValueError(f'model type {args.type} is not recognized')
    model.to(args.device)
    print(f"model #params: {sum(p.numel() for p in model.parameters())}")
    if args.resume or args.sample_only: # note: if we sample-only then we also assume we are resuming
        print("resuming from existing model in the workdir")
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'model.pt')))
    if args.sample_only:
        print_samples(num=50)
        sys.exit()

    # init optimizer
    Optim = get_optimizer(args.optimizer, args.use_torch_optim)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)
    match args.optimizer:
        case "sgd":
            optimizer = Optim(params=model.parameters(), lr=args.learning_rate, momentum=args.momentum, nesterov=args.nesterov)
        case "adagrad":
            optimizer = Optim(params=model.parameters(), lr=args.learning_rate)
        case "rmsprop":
            optimizer = Optim(params=model.parameters(), lr=args.learning_rate, alpha=args.alpha, momentum=args.momentum)
        case "adam":
            optimizer = Optim(params=model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
        case "adamw":
            optimizer = Optim(params=model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        case _:
            raise ValueError(f'invalid optimizer selected: {args.optimizer}')

    # init dataloader
    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

    # training loop
    best_loss = None
    step = 0
    while True:

        t0 = time.time()

        # get the next batch, ship to device, and unpack it to input and target
        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch]
        X, Y = batch

        # feed into the model
        logits, loss = model(X, Y)

        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # wait for all CUDA work on the GPU to finish then calculate iteration time taken
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        # logging
        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

        # evaluate the model
        if step > 0 and step % 500 == 0:
            train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10)
            test_loss  = evaluate(model, test_dataset,  batch_size=100, max_batches=10)
            gradnorms = gradnorm(model)

            # calculate gradnorm statistics
            avg_gradnorm = 0 if not gradnorms else sum(gradnorms) / len(gradnorms)
            max_gradnorm = max(gradnorms)
            min_gradnorm = min(gradnorms)
            total_gradnorm = sum(gradnorms)

            writer.add_scalar("Loss/train", train_loss, step)
            writer.add_scalar("Loss/test", test_loss, step)
            writer.add_scalar("Gradnorm/average", avg_gradnorm, step)
            writer.add_scalar("Gradnorm/max", max_gradnorm, step)
            writer.add_scalar("Gradnorm/min", min_gradnorm, step)
            writer.add_scalar("Gradnorm/total", total_gradnorm, step)

            # only our optimizers emit the lr_norm. The torch optimizers do not.
            if args.optimizer in ["adagrad", "rmsprop", "adam", "adamw"] and not args.use_torch_optim:
                lr_norms = optimizer.lr_norms()

                # calculate learning rate statistics
                lrnorm_avg = 0 if not lr_norms else sum(lr_norms) / len(lr_norms)
                lrnorm_max = max(lr_norms)
                lrnorm_min = min(lr_norms)
                lrnorm_total = sum(lr_norms)

                writer.add_scalar("LRNorm/average", lrnorm_avg, step)
                writer.add_scalar("LRNorm/max", lrnorm_max, step)
                writer.add_scalar("LRNorm/min", lrnorm_min, step)
                writer.add_scalar("LRNorm/total", lrnorm_total, step)


            writer.flush()
            print(f"step {step} train loss: {train_loss} test loss: {test_loss}")
            # save the model to disk if it has improved
            if best_loss is None or test_loss < best_loss:
                out_path = os.path.join(args.work_dir, "model.pt")
                print(f"test loss {test_loss} is the best so far, saving model to {out_path}")
                torch.save(model.state_dict(), out_path)
                best_loss = test_loss

        # sample from the model
        if step > 0 and step % 200 == 0:
            print_samples(num=10)

        step += 1
        # termination conditions
        if args.max_steps >= 0 and step >= args.max_steps:
            break
