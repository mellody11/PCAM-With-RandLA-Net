# PCAM With RandLA-Net

This repository contains a PyTorch implementation of PCAM using RandLA-Net as backbone.

Only support S3DIS and ScanNetV2.

## Preparation

### Dataset  preparation

As for S3DIS refers to [here](https://github.com/mellody11/RandLA-Net-Pytorch-New).

As for ScanNetV2 directly run the `train_Scannet.py`. If it doesn't work, please refer to [MPRM](https://github.com/plusmultiply/mprm) to prepare your dataset.

### CPP tools preparation

Refers to [here](https://github.com/mellody11/RandLA-Net-Pytorch-New).

## Train a model

```
  	python train_S3DIS.py
or
  	python train_Scannet.py
```

## Results

We will report both the quantitative results(both S3DIS and Scannet) and qualitative results(Only S3DIS)
