import pytest
param = pytest.mark.parametrize

import torch

@param('small', (False, True))
@param('stereo', (False, True))
def test_model(
    small,
    stereo
):
    from hs_tasnet.hs_tasnet import HSTasNet
    model = HSTasNet(512, small = small, stereo = stereo)

    audio = torch.randn(1, 2 if stereo else 1, 352800)

    spec = torch.randn(1, 688, 512)

    (spec_out, waveform_out), hiddens1 = model(audio, spec)
    (spec_out, waveform_out), hiddens2 = model(audio, spec, hiddens = hiddens1)
    (spec_out, waveform_out), hiddens3 = model(audio, spec, hiddens = hiddens2)

    assert spec.shape == spec_out.shape
