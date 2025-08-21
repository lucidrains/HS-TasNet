# /// script
# dependencies = [
#   "HS-TasNet"
# ]
# ///

# model

from hs_tasnet import HSTasNet

model = HSTasNet()

# the musdb dataset

import musdb
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
