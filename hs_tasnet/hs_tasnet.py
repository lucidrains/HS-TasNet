from __future__ import annotations
from functools import partial, wraps

import torch
import torch.nn.functional as F
from torch import nn, Tensor, tensor, is_tensor, cat, stft, istft, hann_window, view_as_real, view_as_complex
from torch.nn import LSTM, Module, ModuleList

from einx import add, multiply
from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

# ein tensor notation:

# b - batch
# t - sources
# n - length (audio or embed)
# d - dimension / channels
# s - stereo [2]
# c - complex [2]

# constants

LSTM = partial(LSTM, batch_first = True)

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# residual

def residual(fn):

    @wraps(fn)
    def decorated(t, *args, **kwargs):
        out, hidden = fn(t, *args, **kwargs)
        return t + out, hidden

    return decorated

# classes

class HSTasNet(Module):
    def __init__(
        self,
        dim = 500,          # they have 500 hidden units for the network, with 1000 at fusion (concat from both representation branches)
        small = False,      # params cut in half by 1 layer lstm vs 2, fusion uses summed representation
        stereo = False,
        num_basis = 1024,
        segment_len = 1024,
        overlap_len = 512,
        n_fft = 1024,
        num_sources = 4,    # drums, bass, vocals, other
    ):
        super().__init__()
        audio_channels = 2 if stereo else 1

        self.audio_channels = audio_channels
        self.segment_len = segment_len
        self.num_sources = num_sources

        # spec branch encoder stft hparams

        self.stft_kwargs = dict(
            n_fft = n_fft,
            win_length = segment_len,
            hop_length = overlap_len,
        )

        spec_dim_input = (n_fft // 2 + 1) * 2 * audio_channels

        self.spec_encode = nn.Sequential(
            Rearrange('(b s) f n c -> b n (s f c)', s = audio_channels),
            nn.Linear(spec_dim_input, dim)
        )

        self.to_spec_mask = nn.Sequential(
            nn.Linear(dim, spec_dim_input * num_sources),
            Rearrange('b n (s f c t) -> (b s) f n c t', c = 2, s = audio_channels, t = num_sources)
        )

        # waveform branch encoder

        self.stereo = stereo

        self.conv_encode = nn.Conv1d(audio_channels, num_basis * 2, segment_len, stride = overlap_len, padding = overlap_len)

        self.basis_to_embed = nn.Sequential(
            nn.Conv1d(num_basis, dim, 1),
            Rearrange('b c l -> b l c')
        )

        self.to_waveform_masks = nn.Sequential(
            nn.Linear(dim, num_sources * num_basis, bias = False),
            Rearrange('... (t basis) -> ... basis t', t = num_sources)
        )

        self.conv_decode = nn.ConvTranspose1d(num_basis, audio_channels, segment_len, stride = overlap_len, padding = overlap_len)

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
        audio,
        hiddens = None,
        targets = None
    ):
        batch, device = audio.shape[0], audio.device

        if exists(targets):
            assert targets.shape == (batch, self.num_sources, *audio.shape[1:])

        # handle audio shapes

        audio_is_squeezed = audio.ndim == 2 # input audio is (batch, length) shape, make sure output is correspondingly squeezed

        if audio_is_squeezed: # (b l) -> (b c l)
            audio = rearrange(audio, 'b l -> b 1 l')

        assert not (self.stereo and audio.shape[1] != 2), 'audio channels must be 2 if training stereo'

        # handle spec encoding

        spec_audio_input = rearrange(audio, 'b s ... -> (b s) ...')

        stft_window = hann_window(self.segment_len, device = device)

        complex_spec = stft(spec_audio_input, window = stft_window, **self.stft_kwargs, return_complex = True)

        real_spec = view_as_real(complex_spec)

        spec = self.spec_encode(real_spec)

        # handle encoding as detailed in original tasnet
        # to keep non-negative, they do a glu with relu on main branch

        to_relu, to_sigmoid = self.conv_encode(audio).chunk(2, dim = 1)

        basis = to_relu.relu() * to_sigmoid.sigmoid() # non-negative basis (1024)

        # basis to waveform embed for mask estimation
        # paper mentions linear for any mismatched dimensions

        waveform = self.basis_to_embed(basis)

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

        spec, next_pre_spec_hidden = residual(self.pre_spec_branch)(spec, pre_spec_hidden)

        waveform, next_pre_waveform_hidden = residual(self.pre_waveform_branch)(waveform, pre_waveform_hidden)

        # if small, they just sum the two branches

        if self.small:
            fusion_input = spec + waveform
        else:
            fusion_input = cat((spec, waveform), dim = -1)

        # fusing

        fused, next_fusion_hidden = residual(self.fusion_branch)(fusion_input, fusion_hidden)

        # split if not small, handle small next week

        if self.small:
            fused_spec, fused_waveform = fused, fused
        else:
            fused_spec, fused_waveform = fused.chunk(2, dim = -1)

        # residual from encoded

        spec = fused_spec + spec_residual

        waveform = fused_waveform + waveform_residual

        # layer for both branches

        spec, next_post_spec_hidden = residual(self.post_spec_branch)(spec, post_spec_hidden)

        waveform, next_post_waveform_hidden = residual(self.post_waveform_branch)(waveform, post_waveform_hidden)

        # spec mask

        spec_mask = self.to_spec_mask(spec).softmax(dim = -1)

        real_spec_per_source = multiply('b ..., b ... t -> (b t) ...', real_spec, spec_mask)

        complex_spec_per_source = view_as_complex(real_spec_per_source)

        recon_audio_from_spec = istft(complex_spec_per_source, window = stft_window, **self.stft_kwargs, return_complex = False)

        recon_audio_from_spec = rearrange(recon_audio_from_spec, '(b s t) ... -> b t s ...', b = batch, s = self.audio_channels)

        # waveform mask

        waveform_mask = self.to_waveform_masks(waveform).softmax(dim = -1)

        basis_per_source = multiply('b basis n, b n basis t -> (b t) basis n', basis, waveform_mask)

        recon_audio_from_waveform = self.conv_decode(basis_per_source)

        recon_audio_from_waveform = rearrange(recon_audio_from_waveform, '(b t) ... -> b t ...', b = batch)

        # recon audio

        recon_audio = recon_audio_from_spec + recon_audio_from_waveform

        # take care of l1 loss if target is passed in

        if audio_is_squeezed:
            recon_audio = rearrange(recon_audio, 'b s 1 n -> b s n')

        if exists(targets):
            recon_loss = F.l1_loss(recon_audio, targets) # they claim a simple l1 loss is better than all the complicated stuff of past
            return recon_loss

        # outputs

        lstm_hiddens = (
            next_pre_spec_hidden,
            next_pre_waveform_hidden,
            next_fusion_hidden,
            next_post_spec_hidden,
            next_post_waveform_hidden
        )

        return recon_audio, lstm_hiddens
