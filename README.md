<img src="./fig1.png" width="350px"></img>

## HS-TasNet (wip)

Implementation of [HS-TasNet](https://arxiv.org/abs/2402.17701), "Real-time Low-latency Music Source Separation using Hybrid Spectrogram-TasNet", proposed by the research team at L-Acoustics

## Install

```bash
$ pip install HS-TasNet
```

## Usage

```python
import torch

from hs_tasnet.hs_tasnet import HSTasNet

model = HSTasNet()

print(model.num_parameters) # 40325881 ~ 41M in paper

small_model = HSTasNet(small = True)

print(small_model.num_parameters) # 18297881 ~ 16M in paper
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
