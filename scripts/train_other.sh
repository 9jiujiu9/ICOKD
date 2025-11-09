#tinyimagenet
python main_tinyimagenet.py --dataset tinyimagenet --arch resnet18  --number-net 2 --feat-dim 512
python main_tinyimagenet.py --dataset tinyimagenet --arch wrn-28-4  --number-net 2 --feat-dim 256
python main_tinyimagenet.py --dataset tinyimagenet --arch vgg16  --number-net 2 --feat-dim 512
python main_tinyimagenet.py --dataset tinyimagenet --arch vgg19  --number-net 2 --feat-dim 512

#cars
python main_cars.py --dataset cars --arch resnet18  --number-net 2 --feat-dim 512
python main_cars.py --dataset cars --arch resnet34  --number-net 2 --feat-dim 512
python main_cars.py --dataset cars --arch ShuffleNetV2_1x  --number-net 2 --feat-dim 464

#imagenet
python main_imagenet.py --dataset imagenet --arch resnet18  --number-net 2 --feat-dim 512
python main_imagenet.py --dataset imagenet --arch resnet34  --number-net 2 --feat-dim 512
