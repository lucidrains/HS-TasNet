import torch
from torch.nn import Module, ModuleList

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

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
