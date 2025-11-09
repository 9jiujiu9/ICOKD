#cifar10
python main_cifar.py --dataset cifar10 --arch resnet32 --number-net 2 --feat-dim 64  --alpha 0 --beta 0
python main_cifar.py --dataset cifar10 --arch resnet56 --number-net 2 --feat-dim 64  --alpha 0 --beta 0 
python main_cifar.py --dataset cifar10 --arch wrn_16_2 --number-net 2 --feat-dim 128  --alpha 0 --beta 0
python main_cifar.py --dataset cifar10 --arch wrn_40_2 --number-net 2 --feat-dim 128  --alpha 0 --beta 0
python main_cifar.py --dataset cifar10 --arch ShuffleNetV2_1x --number-net 2 --feat-dim 464 --alpha 0 --beta 0
python main_cifar.py --dataset cifar10 --arch vgg16 --number-net 2 --feat-dim 512 --alpha 0 --beta 0
python main_cifar.py --dataset cifar10 --arch efficientnet_b0 --number-net 2 --feat-dim 1280 --alpha 0 --beta 0

#cifar100
python main_cifar.py --dataset cifar100 --arch resnet32 --number-net 2 --feat-dim 64  --alpha 0 --beta 0
python main_cifar.py --dataset cifar100 --arch resnet56 --number-net 2 --feat-dim 64  --alpha 0 --beta 0
python main_cifar.py --dataset cifar100 --arch wrn_16_2 --number-net 2 --feat-dim 128  --alpha 0 --beta 0
python main_cifar.py --dataset cifar100 --arch wrn_40_2 --number-net 2 --feat-dim 128  --alpha 0 --beta 0 
python main_cifar.py --dataset cifar100 --arch ShuffleNetV2_1x --number-net 2 --feat-dim 464 --alpha 0 --beta 0
python main_cifar.py --dataset cifar100 --arch vgg16 --number-net 2 --feat-dim=512 --alpha 0 --beta 0
python main_cifar.py --dataset cifar100 --arch efficientnet_b0 --number-net 2 --feat-dim 1280 --alpha 0 --beta 0