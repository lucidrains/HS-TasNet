<img src="./fig1.png" width="350px"></img>

## HS-TasNet (wip)

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
