# /// script
# dependencies = [
#   "fire",
#   "HS-TasNet",
#   "wandb"
# ]
# ///

# model

import fire

import musdb
from hs_tasnet import HSTasNet, Trainer

@fire.Fire
def train(
    small = False,
    batch_size = 4,
    max_steps = 50_000,
    max_epochs = 20,
    use_wandb = False,
    wandb_project = 'HS-TasNet',
    wandb_run_name = None,
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
        batch_size = batch_size,
        max_steps = max_steps,
        max_epochs = max_epochs,
        use_wandb = use_wandb,
        experiment_project = wandb_project,
        experiment_run_name = wandb_run_name
    )

    trainer()

# fire cli
# --small for small model

train()
