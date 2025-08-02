import pytest

import torch

def test_memory_lstm():
    from hs_tasnet.hs_tasnet import MemoryLSTM

    memory_lstm = MemoryLSTM(512, 1024)

    feats = torch.randn(3, 1024, 512)

    out1, hiddens1 = memory_lstm(feats)
    out2, hiddens2 = memory_lstm(feats, hiddens1)

    assert out1.shape == out2.shape == feats.shape

def test_model():
    from hs_tasnet.hs_tasnet import HSTasNet
    model = HSTasNet(512)

    audio = torch.randn(1, 8192 * 2)
    transformed = model(audio)

    assert audio.shape == transformed.shape
