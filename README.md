<img src="./fig1.png" width="350px"></img>

## HS-TasNet

Implementation of [HS-TasNet](https://arxiv.org/abs/2402.17701), "Real-time Low-latency Music Source Separation using Hybrid Spectrogram-TasNet", proposed by the research team at L-Acoustics

## Install

```bash
$ pip install HS-TasNet
```

## Usage

```python
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
    batch_size = 2,
    max_steps = 2,
    cpu = True,
)

trainer()

# after much training
# inferencing

model.sounddevice_stream(
    duration_seconds = 2,
    return_reduced_sources = [0, 2]
)

# or from the exponentially smoothed model (in the trainer)

trainer.ema_model.sounddevice_stream(...)

# or you can load from a specific checkpoint

model.load('./checkpoints/path.to.desired.ckpt.pt')
model.sounddevice_stream(...)

```

## Training script

First make sure dependencies are there by running

```shell
$ sh install.sh
```

Then make sure `uv` is installed

```shell
$ pip install uv
```

Finally, and make sure the loss goes down

```shell
$ uv run train.py
```

For distributed training, you just need to run `accelerate config` first, courtesy of [`accelerate` from 🤗](https://huggingface.co/docs/accelerate/en/index) but single machine is fine too

## Experiment tracking

To enable online experiment monitoring / tracking, you need to have `wandb` installed and logged in

```shell
$ pip install wandb && wandb login
```

Then

```shell
$ uv run train.py --use-wandb
```

## Sponsors

This open sourced work is sponsored by [Sweet Spot](https://github.com/sweetspotsoundsystem)

## Citations

```bibtex
@misc{venkatesh2024realtimelowlatencymusicsource,
    title    = {Real-time Low-latency Music Source Separation using Hybrid Spectrogram-TasNet}, 
    author   = {Satvik Venkatesh and Arthur Benilov and Philip Coleman and Frederic Roskam},
    year     = {2024},
    eprint   = {2402.17701},
    archivePrefix = {arXiv},
    primaryClass = {eess.AS},
    url      = {https://arxiv.org/abs/2402.17701}, 
}
```
