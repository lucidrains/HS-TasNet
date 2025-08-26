<img src="./fig1.png" width="350px"></img>

## HS-TasNet

Implementation of [HS-TasNet](https://arxiv.org/abs/2402.17701), "Real-time Low-latency Music Source Separation using Hybrid Spectrogram-TasNet", proposed by the research team at L-Acoustics

## Install

```bash
$ pip install HS-TasNet
```

## Usage

```python
import torch
from hs_tasnet import HSTasNet

model = HSTasNet()

audio = torch.randn(1, 2, 204800) # ~5 seconds of stereo

separated_audios, _ = model(audio)

assert separated_audios.shape == (1, 4, 2, 204800) # second dimension is the separated tracks
```

With the `Trainer`

```python
# model

from hs_tasnet import HSTasNet, Trainer

model = HSTasNet()

# trainer

trainer = Trainer(
    model,
    dataset = None,               # add your in-house Dataset
    concat_musdb_dataset = True,  # concat the musdb dataset automatically
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

# to load an HS-TasNet from any of the saved checkpoints, without having to save its hyperparameters, just run

model = HSTasNet.init_and_load_from('./checkpoints/path.to.desired.ckpt.pt')

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

Finally run the following to train a newly initialized model on a small subset of MusDB, and make sure the loss goes down

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

To wipe the previous checkpoints and evaluated results, append `--clear-folders`

## Test

```shell
$ uv pip install '.[test]' --system
```

Then

```shell
$ pytest tests
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
