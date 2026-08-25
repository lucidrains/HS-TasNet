# /// script
# dependencies = [
#   "fire",
#   "HS-TasNet>=0.2.1",
#   "wandb"
# ]
# ///

# model

from shutil import rmtree
import fire

import musdb
from hs_tasnet import HSTasNet, Trainer

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
    decay_lr_steps = 3,
    early_stop_steps = 10,
    use_full_musdb_dataset = False,
    full_musdb_dataset_root = "./full-musdb-dataset",
    dataset_max_seconds = 7.,       # random segment length in seconds, as in the demucs v2 training pipeline used by the paper
    max_grad_norm = 1.
):

    model = HSTasNet(
        small = small,
        stereo = stereo
    )

    # trainer

    from hs_tasnet import Trainer

    trainer = Trainer(
        model,
        dataset = None,                                     # add your own
        concat_musdb_dataset = True,                        # whether to concat the musdb dataset
        use_full_musdb_dataset = use_full_musdb_dataset,    # whether to use sample musdb or full
        full_musdb_dataset_root = full_musdb_dataset_root,
        batch_size = batch_size,
        max_steps = max_steps,
        max_epochs = max_epochs,
        dataset_max_seconds = dataset_max_seconds,
        max_grad_norm = max_grad_norm,
        use_wandb = use_wandb,
        experiment_project = wandb_project,
        experiment_run_name = wandb_run_name,
        random_split_dataset_for_eval_frac = 0. if not split_dataset_for_eval else split_dataset_eval_frac,
        decay_lr_if_not_improved_steps = decay_lr_steps,
        early_stop_if_not_improved_steps = early_stop_steps
    )

    if clear_folders:
        trainer.clear_folders()

    trainer()

# fire cli
# --small for small model

if __name__ == '__main__':
    fire.Fire(train)
