from __future__ import annotations
from functools import partial

import torch
from torch import Tensor, tensor, is_tensor, cat
from torch.nn import LSTM, Module, ModuleList

# constants

LSTM = partial(LSTM, batch_first = True)

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

        self.lstm1 = LSTM(dim, dim_inner)
        self.lstm2 = LSTM(dim_inner, dim)

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
        dim,
        dim_inner = None,
        small = False
    ):
        super().__init__()

        self.small = small

        # they do a single layer of lstm in their "small" variant

        rnn_klass = LSTM if small else MemoryLSTM

        self.pre_spec_branch = rnn_klass(dim, dim_inner)
        self.post_spec_branch = rnn_klass(dim, dim_inner)

        dim_fusion_branch_input = dim * (2 if not small else 1)

        self.fusion_branch = rnn_klass(dim_fusion_branch_input, dim_inner)

        self.pre_waveform_branch = rnn_klass(dim, dim_inner)
        self.post_waveform_branch = rnn_klass(dim, dim_inner)

    @property
    def num_parameters(self):
        return sum([p.numel() for p in self.parameters()])

    def forward(
        self,
        spec,
        waveform,
        hiddens = None # handle properly later
    ):

        spec_residual, waveform_residual = spec, waveform

        spec, _ = self.pre_spec_branch(spec)

        waveform, _ = self.pre_waveform_branch(waveform)

        # if small, they just sum the two branches

        if self.small:
            fusion_input = spec + waveform
        else:
            fusion_input = cat((spec, waveform), dim = -1)

        # fusing

        fused, _ = self.fusion_branch(fusion_input)

        # split if not small, handle small next week

        fused_spec, fused_waveform = fused.chunk(2, dim = -1)

        # residual from encoded

        spec = fused_spec + spec_residual

        waveform = fused_waveform + waveform_residual

        # layer for both branches

        spec, _ = self.post_spec_branch(spec)

        waveform, _ = self.post_waveform_branch(waveform)

        return spec, waveform
