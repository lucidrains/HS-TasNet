from __future__ import annotations
from pathlib import Path

import torch
from torch import stack, tensor
from torch.nn import Module
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

import numpy as np

from accelerate import Accelerator

from hs_tasnet.hs_tasnet import HSTasNet

from ema_pytorch import EMA

from einops import rearrange

from musdb import DB as MusDB

# functions

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def not_improved_last_n_steps(losses, steps):
    if len(losses) <= steps:
        return False

    last_n_losses = losses[-(steps + 1):]

    return (last_n_losses[1:] <= last_n_losses[:-1]).all().item()

# wrap the musdb MultiTrack if detected

class MusDBDataset(Dataset):
    def __init__(
        self,
        musdb_data: MultiTrack
    ):
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
    ):
        super().__init__()

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

        # dataloader

        dataloader = DataLoader(dataset, batch_size = batch_size, drop_last = True, shuffle = True)

        eval_dataloader = None
        if exists(eval_dataset):
            eval_dataloader = DataLoader(eval_dataset, batch_size = batch_size, drop_last = True, shuffle = True)

        # hf accelerate

        self.accelerator = Accelerator(
            cpu = cpu,
            gradient_accumulation_steps = grad_accum_every,
            **accelerate_kwargs
        )

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

            for audio, targets in self.dataloader:

                with self.accelerator.accumulate(self.model):

                    loss = self.model(
                        audio,
                        targets = targets,
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

                for eval_audio, eval_targets in self.eval_dataloader:

                    self.model.eval()

                    with torch.no_grad():
                        eval_loss = self.model(
                            audio,
                            targets = targets,
                            auto_curtail_length_to_multiple = True
                        )

                        eval_losses.append(eval_loss)

                    avg_eval_loss = stack(eval_losses).mean()
                    past_eval_losses.append(avg_eval_loss)

                self.print(f'[{epoch}] eval loss: {avg_eval_loss.item():.3f}')

                self.log(valid_loss = avg_eval_loss)

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

        self.print(f'training complete')
