# /// script
# dependencies = [
#   "fire",
#   "HS-TasNet",
#   "wandb"
# ]
# ///

# model

from shutil import rmtree
import fire

import musdb
from hs_tasnet import HSTasNet, Trainer

@fire.Fire
def train(
    small = False,
    stereo = False,
    batch_size = 16,
    max_steps = 50_000,
    max_epochs = 20,
    use_wandb = False,
    wandb_project = 'HS-TasNet',
    wandb_run_name = None,
    split_dataset_for_eval = True,
    split_dataset_eval_frac = 0.05,
    clear_folders = False,
    full_train_dataset = False,
    full_train_dataset_root = "./full-musdb-dataset"
):

    model = HSTasNet(
        small = small,
        stereo = stereo
    )

    # the musdb dataset

    if full_train_dataset:
        musdb_kwargs = dict(root = full_train_dataset_root)
    else:
        musdb_kwargs = dict(download = True)

    mus = musdb.DB(**musdb_kwargs)

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
        experiment_run_name = wandb_run_name,
        random_split_dataset_for_eval_frac = 0. if not split_dataset_for_eval else split_dataset_eval_frac
    )

    if clear_folders:
        trainer.clear_folders()

    trainer()

# fire cli
# --small for small model

train()
