from torch.nn import Module, ModuleList
from minGRU_pytorch import minGRU

# helper functions

def exists(v):
    return v is not None

# classes

class minGRUWrapper(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers
    ):
        super().__init__()
        assert input_size == hidden_size
        self.layers = ModuleList([minGRU(input_size) for _ in range(num_layers)])

    def forward(
        self,
        x,
        hiddens = None
    ):
        if not exists(hiddens):
            hiddens = (None,) * len(self.layers)

        next_hiddens = []
        out = x

        for layer, h in zip(self.layers, hiddens):
            out, next_h = layer(out, prev_hidden = h, return_next_prev_hidden = True)
            next_hiddens.append(next_h)

        return out, tuple(next_hiddens)
