# /// script
# dependencies = [
#   "HS-TasNet",
#   "termcolor"
# ]
# ///

import torch
from torch import stack, sin, pi, linspace
from torch.utils.data import Dataset

from hs_tasnet import HSTasNet, Trainer

# dataset

class DummySineDataset(Dataset):
    def __init__(
        self,
        sample_rate = 44_100,
        duration = 1.0,
        num_sources = 4,
        freqs = (220., 440., 880., 1760.)
    ):
        super().__init__()
        num_samples = int(sample_rate * duration)

        t = linspace(0, duration, num_samples)

        targets = []

        for i in range(num_sources):
            freq = freqs[i % len(freqs)]
            sine = sin(2 * pi * freq * t)

            target = stack((sine, sine), dim = 0)
            targets.append(target)

        self.targets = stack(targets, dim = 0)
        self.audio = self.targets.sum(dim = 0)

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return self.audio, self.targets

# main

if __name__ == '__main__':

    model = HSTasNet(
        small = True,
        stereo = True,
        dim = 128,
        num_basis = 256,
        segment_len = 1024,
        overlap_len = 512,
        num_sources = 4
    )

    dataset = DummySineDataset(duration = 1.0)

    trainer = Trainer(
        model = model,
        dataset = dataset,
        eval_dataset = dataset,
        concat_musdb_dataset = False,
        use_full_musdb_dataset = False,
        batch_size = 2,
        max_epochs = 300,
        learning_rate = 1e-3,
        decay_lr_if_not_improved_steps = 300,
        early_stop_if_not_improved_steps = 300,
        early_stop_if_sdr_greater_than = 15.0,
        augment_gain = False,
        augment_channel_swap = False,
        augment_remix = False,
        checkpoint_folder = './toy_checkpoints',
        eval_results_folder = './toy_eval',
        use_wandb = False,
        use_ema = False,
        cpu = False
    )

    try:
        from termcolor import cprint
        cprint('\n[INFO] For an overfitted single-sample sine wave experiment, a capable source separation model should rapidly achieve an SI-SDR >15.0 dB.\n', 'cyan')
    except ImportError:
        print('\n[INFO] For an overfitted single-sample sine wave experiment, a capable source separation model should rapidly achieve an SI-SDR >15.0 dB.\n')

    trainer()
