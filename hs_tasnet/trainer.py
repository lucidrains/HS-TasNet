from __future__ import annotations

import torch
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator

from hs_tasnet.hs_tasnet import HSTasNet

# functions

def exists(v):
    return v is not None

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
        cpu = False
    ):
        super().__init__()

        # epochs

        self.max_epochs = max_epochs

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

        if exists(eval_dataset):
            self.eval_dataloader = self.accelerator.prepare(eval_dataloader)

    @property
    def device(self):
        return self.accelerator.device

    def print(self, *args):
        return self.accelerator.print(*args)

    def forward(self):

        for _ in range(self.max_epochs):

            for audio, targets in self.dataloader:
                loss = self.model(audio, targets = targets)

                self.print(f'loss: {loss.item():.3f}')

                self.accelerator.backward(loss)

                self.optimizer.step()
                self.optimizer.zero_grad()
