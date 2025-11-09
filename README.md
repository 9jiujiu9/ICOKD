
# Inter-class Enhanced Feature Correlation for Adaptive Online Knowledge Distillation

This project provides source code for our Inter-class Enhanced Feature Correlation for Adaptive Online Knowledge Distillation (ICOKD).


## Installation

### Requirements

Ubuntu 18.04 LTS

Python 3.7 ([Anaconda](https://www.anaconda.com/) is recommended)

CUDA 11.1

PyTorch 1.7.0


## CIFAR-100 dataset
### Dataset
CIFAR-100 : [download](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

### Training two baseline networks
```
python main_cifar.py --arch resnet32 --number-net 2 --feat-dim 64 --alpha 0 --beta 0
```
More commands for training various architectures can be found in `scripts/train_cifar_baseline.sh`

### Training two networks
```
python main_cifar.py --arch resnet32 --feat-dim 64  --number-net 2 

More commands for training various architectures can be found in `scripts/train_cifar.sh`


## Supervised Learning on ImageNet dataset

### Dataset preparation

- Download the ImageNet dataset to YOUR_IMAGENET_PATH and move validation images to labeled subfolders
    - The [script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) may be helpful.

```
### Training two networks
```
python main_imagenet.py --dataset imagenet --arch resnet18  --number-net 2  --feat-dim 512
```
More commands for training various architectures can be found in `scripts/train_other.sh`
