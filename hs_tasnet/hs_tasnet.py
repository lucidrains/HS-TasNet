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

# classes

class HSTasNet(Module):
    def __init__(
        self,
        dim = 500,      # they have 500 hidden units for the network, with 1000 at fusion (concat from both representation branches)
        small = False   # params cut in half by 1 layer lstm vs 2, fusion uses summed representation
    ):
        super().__init__()

        # they do a single layer of lstm in their "small" variant

        self.small = small
        lstm_num_layers = 1 if small else 2

        # lstms

        self.pre_spec_branch = LSTM(dim, dim, lstm_num_layers)
        self.post_spec_branch = LSTM(dim, dim, lstm_num_layers)

        dim_fusion = dim * (2 if not small else 1)

        self.fusion_branch = LSTM(dim_fusion, dim_fusion, lstm_num_layers)

        self.pre_waveform_branch = LSTM(dim, dim, lstm_num_layers)
        self.post_waveform_branch = LSTM(dim, dim, lstm_num_layers)

    @property
    def num_parameters(self):
        return sum([p.numel() for p in self.parameters()])

    def forward(
        self,
        spec,
        waveform,
        hiddens = None
    ):
        # handle previous hiddens

        hiddens = default(hiddens, (None,) * 5)

        (
            pre_spec_hidden,
            pre_waveform_hidden,
            fusion_hidden,
            post_spec_hidden,
            post_waveform_hidden
        ) = hiddens

        # residuals

        spec_residual, waveform_residual = spec, waveform

        spec, next_pre_spec_hidden = self.pre_spec_branch(spec, pre_spec_hidden)

        waveform, next_pre_waveform_hidden = self.pre_waveform_branch(waveform, pre_waveform_hidden)

        # if small, they just sum the two branches

        if self.small:
            fusion_input = spec + waveform
        else:
            fusion_input = cat((spec, waveform), dim = -1)

        # fusing

        fused, next_fusion_hidden = self.fusion_branch(fusion_input, fusion_hidden)

        # split if not small, handle small next week

        if self.small:
            fused_spec, fused_waveform = fused, fused
        else:
            fused_spec, fused_waveform = fused.chunk(2, dim = -1)

        # residual from encoded

        spec = fused_spec + spec_residual

        waveform = fused_waveform + waveform_residual

        # layer for both branches

        spec, next_post_spec_hidden = self.post_spec_branch(spec, post_spec_hidden)

        waveform, next_post_waveform_hidden = self.post_waveform_branch(waveform, post_waveform_hidden)

        # outputs

        outputs = (spec, waveform)

        lstm_hiddens = (
            next_pre_spec_hidden,
            next_pre_waveform_hidden,
            next_fusion_hidden,
            next_post_spec_hidden,
            next_post_waveform_hidden
        )

        return outputs, lstm_hiddens
