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

    shape = (2, 1024 * 12) if stereo else (1024 * 12,)

    audio = torch.randn(3, *shape)
    targets = torch.rand(3, 4, *shape)

    loss = model(audio, targets = targets)
    loss.backward()

    # after much training

    pred1, hiddens1 = model(audio)
    pred2, hiddens1 = model(audio, hiddens = hiddens1)
