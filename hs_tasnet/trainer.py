from __future__ import annotations
from pathlib import Path

import torch
from torch import stack
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator

from hs_tasnet.hs_tasnet import HSTasNet

# functions

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

# classes

class Trainer(Module):
    def __init__(
        self,
        model: HSTasNet,
        dataset: Dataset,
        eval_dataset: Dataset | None = None,
        optim_klass = Adam,
        batch_size = 128,
        learning_rate = 3e-4,
        max_epochs = 10,
        accelerate_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
        cpu = False,
        checkpoint_every = 1,
        checkpoint_folder = './checkpoints',
        early_stop_if_not_improved_steps = 3 # they do early stopping if 3 evals without improved loss
    ):
        super().__init__()

        # epochs

        self.max_epochs = max_epochs

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

        # data

        dataloader = DataLoader(dataset, batch_size = batch_size, drop_last = True, shuffle = True)

        eval_dataloader = None
        if exists(eval_dataset):
            eval_dataloader = DataLoader(eval_dataset, batch_size = batch_size, drop_last = True, shuffle = True)

        # hf accelerate

        self.accelerator = Accelerator(
            cpu = cpu,
            **accelerate_kwargs
        )

        # preparing

        (
            self.model,
            self.optimizer,
            self.dataloader
        ) = self.accelerator.prepare(
            model,
            optimizer,
            dataloader
        )

        # has eval

        self.needs_eval = exists(eval_dataloader)

        assert early_stop_if_not_improved_steps >= 2
        self.early_stop_steps = early_stop_if_not_improved_steps

        # prepare eval

        if self.needs_eval:
            self.eval_dataloader = self.accelerator.prepare(eval_dataloader)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    def print(self, *args):
        return self.accelerator.print(*args)

    def forward(self):

        past_eval_losses = [] # for early stopping detection

        for epoch in range(self.max_epochs):

            # training steps

            for audio, targets in self.dataloader:
                loss = self.model(audio, targets = targets)

                self.print(f'[{epoch}] loss: {loss.item():.3f}')

                self.accelerator.backward(loss)

                self.optimizer.step()
                self.optimizer.zero_grad()

            if not self.needs_eval:
                continue

            self.accelerator.wait_for_everyone()

            # evaluation at the end of each epoch

            eval_losses = []

            for eval_audio, eval_targets in self.eval_dataloader:

                self.model.eval()

                with torch.no_grad():
                    eval_loss = self.model(audio, targets = targets)
                    eval_losses.append(eval_loss)

                avg_eval_loss = stack(eval_losses).mean()
                past_eval_losses.append(avg_eval_loss)

            self.print(f'[{epoch}] eval loss: {avg_eval_loss.item():.3f}')

            # maybe save

            if (
                divisible_by(epoch + 1, self.checkpoint_every) and
                self.accelerator.is_main_process
            ):
                checkpoint_index = (epoch + 1) // self.checkpoint_every
                self.unwrapped_model.save(self.checkpoint_folder / f'hs-tasnet.ckpt.{checkpoint_index}.pt')

            self.accelerator.wait_for_everyone()

            # early stop if criteria met

            last_n_eval_losses = stack(past_eval_losses[-self.early_stop_steps:])

            if (
                len(last_n_eval_losses) > self.early_stop_steps and
                (last_n_eval_losses[1:] >= last_n_eval_losses[:-1]).all() # losses have not improved
            ):
                self.print(f'early stopping at epoch {epoch} since last three eval losses have not improved: {last_n_eval_losses}')
                break
