# SCDM-HTD

## Overview

This repository provides the implementation of the paper:

**[SCDM-HTD: Spectral-Wise Conditional Diffusion Model for Hyperspectral Target Detection](https://arxiv.org/abs/2610.32768)**

## Requirements

- Python 3.10.14
- PyTorch 2.1.1
- NVIDIA GeForce RTX 3090 GPU

## Dataset

The default dataset used in this implementation is:

- `abu-airport-4`

## Usage

Run `Main.py` directly to perform testing on the provided dataset.

To apply the method to a new dataset:

1. Update the dataset name accordingly.
2. Set the status to `train`.
3. Adjust the loss weight parameter as needed to achieve better performance.

## Citation

If you find this code useful in your research, please consider citing our paper:

```bibtex
@article{shen2026scdm,
  title={SCDM-HTD: Spectral-Wise Conditional Diffusion Model for Hyperspectral Target Detection},
  author={Shen, Dunbin and Kong, Wenfeng and Liu, Jianjun and Ma, Xiaorui and Wang, Hongyu},
  journal={arXiv preprint arXiv:2610.32768},
  year={2026}
}
```
