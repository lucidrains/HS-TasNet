import pytest

import torch
from hs_tasnet.hs_tasnet import HSTasNet

def test_model():
    model = HSTasNet(512)

    audio = torch.randn(1, 8192 * 2)
    transformed = model(audio)

    assert audio.shape == transformed.shape
