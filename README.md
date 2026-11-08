# SCDM-HTD

## Overview
This is the official implementation of the paper "SCDM-HTD: Spectral-Wise Conditional Diffusion
Model for Hyperspectral Target Detection" https://arxiv.org/abs/2610.32768

## Requirements
Python 3.10.14  

torch 2.1.1

NVIDIA GeForce RTX 3090 GPU

## Dataset
abu-airport-4

## Run
You can run "Main.py" directly to perform testing on the given dataset. To use a new dataset, simply update the dataset name and change the status to 'train', and can tune the loss weight parameter for better performance.

## Citation
If you use this code, please cite:
```bibtex
@article{shen2026scdm,
  title={SCDM-HTD: Spectral-Wise Conditional Diffusion Model for Hyperspectral Target Detection},
  author={Shen, Dunbin and Kong, Wenfeng and Liu, Jianjun and Ma, Xiaorui and Wang, Hongyu},
  journal={arXiv preprint arXiv:2610.32768},
  year={2026}
}
```
