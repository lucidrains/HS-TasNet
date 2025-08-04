import pytest
param = pytest.mark.parametrize

import torch

@param('small', (False, True))
def test_model(
    small
):
    from hs_tasnet.hs_tasnet import HSTasNet
    model = HSTasNet(512, 1024, small = small)

    spec = torch.randn(1, 256, 512)
    waveform = torch.randn(1, 256, 512)

    (spec_out, waveform_out), hiddens1 = model(spec, waveform)
    (spec_out, waveform_out), hiddens2 = model(spec, waveform, hiddens = hiddens1)
    (spec_out, waveform_out), hiddens3 = model(spec, waveform, hiddens = hiddens2)

    assert spec.shape == spec_out.shape
    assert waveform.shape == waveform_out.shape
