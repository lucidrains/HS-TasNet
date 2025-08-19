from __future__ import annotations
from functools import partial
from random import random
from pathlib import Path

import torch
from torch import cat, stack, tensor
from torch.nn import Module
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import numpy as np

import matplotlib.pyplot as plt

from accelerate import Accelerator

from hs_tasnet.hs_tasnet import HSTasNet

from ema_pytorch import EMA

from einops import rearrange, reduce

from musdb import DB as MusDB

# constants

pad_sequence = partial(pad_sequence, batch_first = True)

# functions

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def satisfy_prob(prob):
    return random() < prob

def rand_range(shape, min, max, device = None):
    rand = torch.rand(shape, device = device)
    return rand * (max - min) + min

def compose(*fns):

    def inner(x):
        for fn in fns:
            x = fn(x)
        return x

    return inner

def db_to_amplitude(db):
    return 10 ** (db / 20.)

def not_improved_last_n_steps(losses, steps):
    if len(losses) <= steps:
        return False

    last_n_losses = losses[-(steps + 1):]

    return (last_n_losses[1:] <= last_n_losses[:-1]).all().item()

# dataset collation

def default_collate_fn(
    data: list[tuple[Tensor, Tensor]]
):
    audios, targets = tuple(zip(*data))

    audio_lens = tuple(t.shape[-1] for t in audios)

    return pad_sequence(audios), pad_sequence(targets), tensor(audio_lens)

# dataset related

# wrap the musdb MultiTrack if detected

class MusDBDataset(Dataset):
    def __init__(self, musdb_data: MultiTrack):
        self.musdb_data = musdb_data

    def __len__(self):
        return len(self.musdb_data)

    def __getitem__(self, idx):
        sample = self.musdb_data[idx]

        audio = rearrange(sample.audio, 'n s -> s n')

        targets = rearrange(sample.stems[1:], 't n s -> t s n') # the first one is the entire mixture

        audio = audio.astype(np.float32)

        targets = targets.astype(np.float32)

        return audio, targets

# transforms

class CastTorch(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio, targets = self.dataset[idx]

        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)

        return audio, targets

class StereoToMonoDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio, targets = self.dataset[idx]

        if audio.ndim == 2:
            audio = reduce(audio, 's n -> n', 'mean')

        if targets.ndim == 3:
            targets = reduce(targets, 't s n -> t n', 'mean')

        return audio, targets

# augmentations

class GainAugmentation(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        *,
        prob = 0.5,
        db_range = (-3., 10.),
        clip = False,
        clip_range = (-1., 1.),
        recon_audio_from_targets = False
    ):
        self.dataset = dataset

        self.prob = prob

        self.db_range = db_range
        self.clip = clip
        self.clip_range = clip_range

        self.recon_audio_from_targets = recon_audio_from_targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio, targets = self.dataset[idx]

        if not satisfy_prob(self.prob):
            return audio, targets

        db_min, db_max = self.db_range

        rand_db_gain = rand_range((), db_min, db_max)
        rand_scale = db_to_amplitude(rand_db_gain)

        audio = audio * rand_scale
        targets = targets * rand_scale

        if self.clip:
            clip_min, clip_max = self.clip_range
            targets = targets.clamp(min = clip_min, max = clip_max)

            if self.recon_audio_from_targets:
                audio = reduce(targets, 't s n -> s n', 'sum')
            else:
                audio = audio.clamp(min = clip_min, max = clip_max)

        return audio, targets

class ChannelSwapAugmentation(Dataset):
    def __init__(self, dataset: Dataset, *, prob = 0.5):
        self.dataset = dataset
        self.prob = prob

    def __len__(self, len):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio, targets = self.dataset[idx]

        if not satisfy_prob(self.prob):
            return audio, targets

        if audio.ndim == 2:
            audio = audio.flip(dims = (0,))

        if targets.ndim == 3:
            targets = targets.flip(dims = (1,))

        return audio, targets

def augment_remix_fn(
    inp: tuple[Tensor, Tensor, Tensor],
    frac_augment = 0.5
):
    """
    this is the effective augmentation used in the separation field that generates synthetic data by combining different tracks across different sources
    """

    audio, targets, audio_lens = inp

    batch_size, device = audio.shape[0], audio.device
    num_augment = int(frac_augment * batch_size)

    if num_augment == 0:
        return inp

    num_sources = targets.shape[1]

    # get indices

    source_arange = torch.arange(num_sources, device = device)
    batch_randperm = torch.randint(0, batch_size, (num_augment, num_sources), device = device)

    # pick out the new targets and ...

    remixed_targets = targets[batch_randperm, source_arange]

    # compose new source from them. take the minimum of the audio lens

    remixed_audio = reduce(remixed_targets, 'b t ... -> b ...', 'sum')
    remixed_audio_lens = reduce(audio_lens[batch_randperm], 'b t -> b', 'min')

    # concat onto the input

    audio = cat((audio[-num_augment:], remixed_audio))
    targets = cat((targets[-num_augment:], remixed_targets))
    audio_lens = cat((audio_lens[-num_augment:], remixed_audio_lens))

    return audio, targets, audio_lens

# classes

class Trainer(Module):
    def __init__(
        self,
        model: HSTasNet,
        dataset: Dataset | MusDB,
        eval_dataset: Dataset | None = None,
        optim_klass = Adam,
        batch_size = 128,
        grad_accum_every = 1,
        learning_rate = 3e-4,
        max_epochs = 10,
        max_steps = None,
        accelerate_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
        cpu = False,
        use_ema = True,
        ema_decay = 0.995,
        ema_kwargs: dict = dict(),
        checkpoint_every = 1,
        checkpoint_folder = './checkpoints',
        decay_lr_factor = 0.5,
        decay_lr_if_not_improved_steps = 3,    # decay learning rate if validation loss does not improve for this amount of epochs
        early_stop_if_not_improved_steps = 10, # they do early stopping if 10 evals without improved loss
        use_wandb = False,
        experiment_project = 'HS-TasNet',
        experiment_run_name = None,
        experiment_hparams: dict = dict(),
        augment_gain = True,
        augment_channel_swap = True,
        augment_remix = True,
        augment_remix_frac = 0.5
    ):
        super().__init__()

        # have the trainer detect if the model is stereo and handle the data accordingly

        self.model_is_stereo = model.stereo

        # epochs

        self.max_epochs = max_epochs

        self.max_steps = max_steps

        # saving

        self.checkpoint_every = checkpoint_every

        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(parents = True, exist_ok = True)

        # optimizer

        optimizer = optim_klass(
            model.parameters(),
            lr = learning_rate,
            **optimizer_kwargs
        )

        # dataset

        if isinstance(dataset, MusDB):
            # give musdb special treatment
            dataset = MusDBDataset(dataset)

        # torch from this point on

        dataset = CastTorch(dataset)

        # handle model is not stereo but data is stereo

        if not self.model_is_stereo:
            dataset = StereoToMonoDataset(dataset)

        # augmentations

        if augment_gain:
            dataset = GainAugmentation(dataset)

        collate_fn = default_collate_fn

        if augment_remix:
            collate_fn = compose(collate_fn, partial(augment_remix_fn, frac_augment = augment_remix_frac))

        # dataloader

        dataloader = DataLoader(dataset, batch_size = batch_size, drop_last = True, shuffle = True, collate_fn = collate_fn)

        eval_dataloader = None
        if exists(eval_dataset):
            eval_dataloader = DataLoader(eval_dataset, batch_size = batch_size, drop_last = True, shuffle = True, collate_fn = collate_fn)

        # hf accelerate

        self.accelerator = Accelerator(
            cpu = cpu,
            gradient_accumulation_steps = grad_accum_every,
            **accelerate_kwargs
        )

        # maybe experiment tracker

        if use_wandb:

            try:
                import wandb
            except ImportError:
                print(f'install `wandb` first for experiment tracking')

            self.accelerator.init_trackers(experiment_project, config = experiment_hparams)

            if exists(experiment_run_name):
                wandb_tracker = self.accelerator.trackers[0]
                wandb_tracker.run.name = experiment_run_name

        # decay lr logic

        scheduler = StepLR(optimizer, 1, gamma = decay_lr_factor)

        self.decay_lr_if_not_improved_steps = decay_lr_if_not_improved_steps

        # setup ema on main process

        self.use_ema = use_ema

        if use_ema:
            self.ema_model = EMA(model, beta = ema_decay, **ema_kwargs)

        # preparing

        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.dataloader
        ) = self.accelerator.prepare(
            model,
            optimizer,
            scheduler,
            dataloader
        )

        # has eval

        self.needs_eval = exists(eval_dataloader)

        # early stopping

        assert early_stop_if_not_improved_steps >= 2
        self.early_stop_steps = early_stop_if_not_improved_steps

        # prepare eval

        if self.needs_eval:
            self.eval_dataloader = self.accelerator.prepare(eval_dataloader)

        # step

        self.register_buffer('step', tensor(0))

    @property
    def device(self):
        return self.accelerator.device

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def print(self, *args):
        return self.accelerator.print(*args)

    def log(self, **data):
        return self.accelerator.log(data, step = self.step.item())

    def forward(self):

        exceeds_max_step = False
        past_eval_losses = [] # for learning rate decay and early stopping detection

        for epoch in range(self.max_epochs):

            self.model.train()

            # training steps

            for audio, targets, audio_lens in self.dataloader:

                with self.accelerator.accumulate(self.model):

                    loss = self.model(
                        audio,
                        targets = targets,
                        audio_lens = audio_lens,
                        auto_curtail_length_to_multiple = True
                    )

                    self.accelerator.backward(loss)

                    self.print(f'[{epoch}] loss: {loss.item():.3f}')

                    self.log(loss = loss)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # max steps

                self.step.add_(1)

                exceeds_max_step = exists(self.max_steps) and self.step.item() >= self.max_steps

                if exceeds_max_step:
                    break

            # update ema

            self.wait()

            if self.use_ema and self.is_main:
                self.ema_model.update()

            # maybe eval

            self.wait()

            if self.needs_eval:

                # evaluation at the end of each epoch

                eval_losses = []

                for eval_audio, eval_targets, eval_audio_lens in self.eval_dataloader:

                    with torch.no_grad():
                        self.model.eval()

                        eval_loss = self.model(
                            audio,
                            targets = targets,
                            audio_lens = eval_audio_lens,
                            auto_curtail_length_to_multiple = True
                        )

                        eval_losses.append(eval_loss)

                    avg_eval_loss = stack(eval_losses).mean()
                    past_eval_losses.append(avg_eval_loss)

                self.print(f'[{epoch}] eval loss: {avg_eval_loss.item():.3f}')

                self.log(
                    valid_loss = avg_eval_loss
                )

            # maybe save

            self.wait()

            if (
                divisible_by(epoch + 1, self.checkpoint_every) and
                self.is_main
            ):
                checkpoint_index = (epoch + 1) // self.checkpoint_every
                self.unwrapped_model.save(self.checkpoint_folder / f'hs-tasnet.ckpt.{checkpoint_index}.pt')

                if self.use_ema:
                    self.ema_model.ema_model.save(self.checkpoint_folder /f'hs_tasnet.ema.ckpt.{checkpoint_index}.pt') # save ema

            # determine lr decay and early stopping based on eval

            if self.needs_eval:
                # stack validation losses for all epochs

                last_n_eval_losses = stack(past_eval_losses)

                # decay lr if criteria met

                if not_improved_last_n_steps(last_n_eval_losses, self.decay_lr_if_not_improved_steps):
                    self.scheduler.step()

                # early stop if criteria met

                if not_improved_last_n_steps(last_n_eval_losses, self.early_stop_steps):
                    self.print(f'early stopping at epoch {epoch} since last three eval losses have not improved: {last_n_eval_losses}')
                    break

            if exceeds_max_step:
                break

        # cleanup accelerator

        self.accelerator.end_training()

        self.print(f'training complete')
