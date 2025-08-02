from __future__ import annotations

import torch
from torch import Tensor, tensor, is_tensor
from torch.nn import LSTM, Module, ModuleList

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# "memory lstm"
# as described in section 3.3, it is just two lstm layers with identity skip connection

class MemoryLSTM(Module):
    def __init__(
        self,
        dim,
        dim_inner = None
    ):
        super().__init__()
        dim_inner = default(dim_inner, dim)

        self.lstm1 = LSTM(dim, dim_inner, batch_first = True)
        self.lstm2 = LSTM(dim_inner, dim, batch_first = True)

    def forward(
        self,
        feats,
        hiddens: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:

        # handle previous hiddens

        hidden1 = hidden2 = None

        if exists(hiddens):
            hidden1, hidden2 = hiddens

        # store residual

        residual = feats

        # the two lstms

        inner, next_hidden1 = self.lstm1(feats, hidden1)

        out, next_hidden2 = self.lstm2(inner, hidden2)

        # add the residual

        out_with_residual = out + residual

        # pass out the next hiddens as tuple of two tensors

        next_hiddens = (next_hidden1, next_hidden2)

        return out_with_residual, next_hiddens

# classes

class HSTasNet(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

    def forward(
        self,
        audio
    ):
        return audio
