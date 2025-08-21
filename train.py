# /// script
# dependencies = [
#   "fire",
#   "HS-TasNet"
# ]
# ///

# model

import fire

import musdb
from hs_tasnet import HSTasNet, Trainer

@fire.Fire
def train(
    small = False
):

    model = HSTasNet(
        small = small
    )

    # the musdb dataset

    mus = musdb.DB(download = True)

    # trainer

    from hs_tasnet import Trainer

    trainer = Trainer(
        model,
        dataset = mus,
        batch_size = 4,
        max_steps = 50_000,
    )

    trainer()

# fire cli
# --small for small model

train()
