from __future__ import annotations

import pickle
from pathlib import Path
from functools import partial, wraps

import torch
import torch.nn.functional as F
from torch.fft import irfft
from torch import nn, compiler, Tensor, tensor, is_tensor, cat, stft, hann_window, view_as_real, view_as_complex
from torch.nn import LSTM, GRU, Module, ModuleList

from numpy import ndarray

from einx import add, multiply, divide
from einops import rearrange, repeat, reduce, pack, unpack
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
GRU = partial(GRU, batch_first = True)

# disable complex related functions from torch compile

(
    view_as_real,
    view_as_complex
) = tuple(compiler.disable()(fn) for fn in (
    view_as_real,
    view_as_complex
))

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def identity(t):
    return t

# residual

def residual(fn):

    @wraps(fn)
    def decorated(t, *args, **kwargs):
        out, hidden = fn(t, *args, **kwargs)
        return t + out, hidden

    return decorated

# fft related

class STFT(Module):
    """
    need this custom module to address an issue with certain window and no centering in istft in pytorch - https://github.com/pytorch/pytorch/issues/91309
    this solution was retailored from the working solution used at vocos https://github.com/gemelo-ai/vocos/blob/03c4fcbb321e4b04dd9b5091486eedabf1dc9de0/vocos/spectral_ops.py#L7
    """

    def __init__(
        self,
        n_fft,
        hop_length,
        win_length,
        eps = 1e-11
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        window = hann_window(win_length)
        self.register_buffer('window', window)

        self.eps = eps

    @compiler.disable()
    def inverse(self, spec):
        n_fft, hop_length, win_length, window = self.n_fft, self.hop_length, self.win_length, self.window

        batch, freqs, frames = spec.shape

        # inverse FFT

        ifft = irfft(spec, n_fft, dim = 1, norm = 'backward')

        ifft = multiply('b w f, w', ifft, window)

        # overlap and add

        output_size = (frames - 1) * hop_length + win_length

        y = F.fold(
            ifft,
            output_size = (1, output_size),
            kernel_size = (1, win_length),
            stride = (1, hop_length),
        )[:, 0, 0]

        # window envelope

        window_sq = repeat(window.square(), 'w -> 1 w t', t = frames)

        window_envelope = F.fold(
            window_sq,
            output_size = (1, output_size),
            kernel_size = (1, win_length),
            stride = (1, hop_length)
        )

        window_envelope = rearrange(window_envelope, '1 1 1 n -> n')

        # normalize out

        return divide('b n, n', y, window_envelope.clamp(min = self.eps))

    @compiler.disable()
    def forward(self, audio):

        stft = torch.stft(
            audio,
            n_fft = self.n_fft,
            win_length = self.win_length,
            hop_length = self.hop_length,
            center = False,
            window = self.window,
            return_complex = True
        )

        return stft

# classes

class HSTasNet(Module):
    def __init__(
        self,
        *,
        dim = 500,          # they have 500 hidden units for the network, with 1000 at fusion (concat from both representation branches)
        small = False,      # params cut in half by 1 layer lstm vs 2, fusion uses summed representation
        stereo = False,
        num_basis = 1024,
        segment_len = 1024,
        overlap_len = 512,
        n_fft = 1024,
        num_sources = 4,    # drums, bass, vocals, other
        torch_compile = False,
        use_gru = False
    ):
        super().__init__()

        # auto-saving config

        _locals = locals()
        _locals.pop('self', None)
        _locals.pop('__class__', None)
        self._config = pickle.dumps(_locals)

        # hyperparameters

        audio_channels = 2 if stereo else 1

        self.audio_channels = audio_channels
        self.num_sources = num_sources

        assert overlap_len < segment_len

        self.segment_len = segment_len
        self.overlap_len = overlap_len

        assert divisible_by(segment_len, 2)
        self.causal_pad = segment_len // 2

        # spec branch encoder stft hparams

        self.stft = STFT(
            n_fft = n_fft,
            win_length = segment_len,
            hop_length = overlap_len
        )

        spec_dim_input = (n_fft // 2 + 1) * 2 * audio_channels

        self.spec_encode = nn.Sequential(
            Rearrange('(b s) f n ... -> b n (s f ...)', s = audio_channels),
            nn.Linear(spec_dim_input, dim)
        )

        self.to_spec_mask = nn.Sequential(
            nn.Linear(dim, spec_dim_input * num_sources),
            Rearrange('b n (s f c t) -> (b s) f n c t', c = 2, s = audio_channels, t = num_sources)
        )

        # waveform branch encoder

        self.stereo = stereo

        self.conv_encode = nn.Conv1d(audio_channels, num_basis * 2, segment_len, stride = overlap_len)

        self.basis_to_embed = nn.Sequential(
            nn.Conv1d(num_basis, dim, 1),
            Rearrange('b c l -> b l c')
        )

        self.to_waveform_masks = nn.Sequential(
            nn.Linear(dim, num_sources * num_basis, bias = False),
            Rearrange('... (t basis) -> ... basis t', t = num_sources)
        )

        self.conv_decode = nn.ConvTranspose1d(num_basis, audio_channels, segment_len, stride = overlap_len)

        # they do a single layer of lstm in their "small" variant

        self.small = small
        lstm_num_layers = 1 if small else 2

        # rnn

        rnn_klass = LSTM if not use_gru else GRU

        self.pre_spec_branch = rnn_klass(dim, dim, lstm_num_layers)
        self.post_spec_branch = rnn_klass(dim, dim, lstm_num_layers)

        dim_fusion = dim * (2 if not small else 1)

        self.fusion_branch = rnn_klass(dim_fusion, dim_fusion, lstm_num_layers)

        self.pre_waveform_branch = rnn_klass(dim, dim, lstm_num_layers)
        self.post_waveform_branch = rnn_klass(dim, dim, lstm_num_layers)

        # torch compile forward

        if torch_compile:
            self.forward = torch.compile(self.forward)

    # saving and loading

    def save(self, path, overwrite = True):
        path = Path(path)
        assert overwrite or not path.exists()

        pkg = dict(
            model = self.state_dict(),
            config = self._config,
        )

        torch.save(pkg, str(path))

    def load(self, path, strict = True):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path), map_location = 'cpu')

        self.load_state_dict(pkg['model'], strict = strict)

    @classmethod
    def init_and_load_from(cls, path, strict = True):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')

        assert 'config' in pkg, 'model configs were not found in this saved checkpoint'

        config = pickle.loads(pkg['config'])
        model = cls(**config)

        self.load(model, path, strict = strict)
        return model

    # returns a function that closures the hiddens and past audio chunk for integration with sounddevice audio callback

    def init_stream_fn(
        self,
        device = None,
        return_reduced_sources: list[int] | None = None,
    ):
        chunk_len = self.overlap_len
        self.eval()

        past_audio = torch.zeros((self.audio_channels, self.overlap_len), device = device)
        hiddens = None

        @torch.inference_mode()
        def fn(audio_chunk: ndarray | Tensor):
            assert audio_chunk.shape[-1] == chunk_len

            nonlocal hiddens
            nonlocal past_audio

            is_numpy_input = isinstance(audio_chunk, ndarray)

            if is_numpy_input:
                audio_chunk = torch.from_numpy(audio_chunk)

            squeezed_audio_channel = audio_chunk.ndim == 1

            if squeezed_audio_channel:
                audio_chunk = rearrange(audio_chunk, '... -> 1 ...')

            if exists(device):
                audio_chunk = audio_chunk.to(device)

            # add past audio chunk

            full_chunk = cat((past_audio, audio_chunk), dim = -1)

            full_chunk = rearrange(full_chunk, '... -> 1 ...')

            # forward chunk with past overlap through model

            transformed, hiddens = self.forward(full_chunk, hiddens = hiddens, return_reduced_sources = return_reduced_sources)

            transformed = rearrange(transformed, '1 ... -> ...')

            if squeezed_audio_channel:
                transformed = rearrange(transformed, '... 1 n -> ... n')

            if is_numpy_input:
                transformed = transformed.cpu().numpy()

            # save next overlap chunk for next timestep

            past_audio = audio_chunk

            return transformed[..., -chunk_len:]

        return fn

    @property
    def num_parameters(self):
        return sum([p.numel() for p in self.parameters()])

    def forward(
        self,
        audio,
        hiddens = None,
        targets = None,
        return_reduced_sources: list[int] | None = None,
        auto_causal_pad = None
    ):
        auto_causal_pad = default(auto_causal_pad, self.training)

        batch, audio_len, device = audio.shape[0], audio.shape[-1], audio.device

        assert divisible_by(audio_len, self.segment_len)

        maybe_residual = residual if not self.small else identity

        if exists(targets):
            assert targets.shape == (batch, self.num_sources, *audio.shape[1:])

        # handle audio shapes

        audio_is_squeezed = audio.ndim == 2 # input audio is (batch, length) shape, make sure output is correspondingly squeezed

        if audio_is_squeezed: # (b l) -> (b c l)
            audio = rearrange(audio, 'b l -> b 1 l')

        assert not (self.stereo and audio.shape[1] != 2), 'audio channels must be 2 if training stereo'

        # pad the audio manually on the left side for causal, and set stft center False

        if auto_causal_pad:
            audio = F.pad(audio, (self.causal_pad, 0), value = 0.)

        # handle spec encoding

        spec_audio_input = rearrange(audio, 'b s ... -> (b s) ...')

        complex_spec = self.stft(spec_audio_input)

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

        spec, next_pre_spec_hidden = maybe_residual(self.pre_spec_branch)(spec, pre_spec_hidden)

        waveform, next_pre_waveform_hidden = maybe_residual(self.pre_waveform_branch)(waveform, pre_waveform_hidden)

        # if small, they just sum the two branches

        if self.small:
            fusion_input = spec + waveform
        else:
            fusion_input = cat((spec, waveform), dim = -1)

        # fusing

        fused, next_fusion_hidden = maybe_residual(self.fusion_branch)(fusion_input, fusion_hidden)

        # split if not small, handle small next week

        if self.small:
            fused_spec, fused_waveform = fused, fused
        else:
            fused_spec, fused_waveform = fused.chunk(2, dim = -1)

        # residual from encoded

        spec = fused_spec + spec_residual

        waveform = fused_waveform + waveform_residual

        # layer for both branches

        spec, next_post_spec_hidden = maybe_residual(self.post_spec_branch)(spec, post_spec_hidden)

        waveform, next_post_waveform_hidden = maybe_residual(self.post_waveform_branch)(waveform, post_waveform_hidden)

        # spec mask

        spec_mask = self.to_spec_mask(spec).softmax(dim = -1)

        real_spec_per_source = multiply('b ..., b ... t -> (b t) ...', real_spec, spec_mask)

        complex_spec_per_source = view_as_complex(real_spec_per_source.contiguous())

        recon_audio_from_spec = self.stft.inverse(complex_spec_per_source)

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

        # excise out the causal padding

        if auto_causal_pad:
            recon_audio = recon_audio[..., self.causal_pad:]

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

        # take care of selecting out and reducing the sources if a list of source indices are passed in `return_reduced_sources`
        # this is for eventual direct integration with e2e sounddevice audio callback streams

        if exists(return_reduced_sources):
            return_reduced_sources = tensor(return_reduced_sources, device = device)

            sources = recon_audio.index_select(1, return_reduced_sources)
            recon_audio = reduce(sources, 'b s ... -> b ...', 'sum')

        return recon_audio, lstm_hiddens
