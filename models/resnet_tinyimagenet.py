import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import random
import numpy as np


# __all__ = ['CIFAR_ResNet18', 'CIFAR_ResNet18_dks', 'CIFAR_ResNet18_byot',
#             'CIFAR_ResNet34', 'CIFAR_ResNet34_dks', 'CIFAR_ResNet34_byot',
#             'CIFAR_ResNet50', 'CIFAR_ResNet50_dks', 'CIFAR_ResNet50_byot',
#             'CIFAR_ResNet101', 'CIFAR_ResNet101_dks', 'CIFAR_ResNet101_byot',
#             'manifold_mixup_CIFAR_ResNet18', 'manifold_mixup_CIFAR_ResNet50']
__all__ = ['resnet18', 'resnet34']


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        if stride != 1 or inplanes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride
        if stride != 1 or inplanes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


            
class CIFAR_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100, number_net=4):
        super(CIFAR_ResNet, self).__init__()
        # self.network_channels = [64 * block.expansion, 128 * block.expansion, 256 * block.expansion, 512 * block.expansion]
        self.number_net = number_net
        self.num_classes = num_classes
        self.inplanes = 64
        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        
        fix_planes = self.inplanes
        for i in range(self.number_net):
            self.inplanes = fix_planes
            setattr(self, 'layer3_' + str(i), self._make_layer(block, 256, layers[2], stride=2))
            setattr(self, 'layer4_' + str(i), self._make_layer(block, 512, layers[3], stride=2))
            setattr(self, 'classifier_' + str(i), nn.Linear(512 * block.expansion, self.num_classes))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)

        logits = []
        embedding = []
        input = x
        f = []
        for i in range(self.number_net):
            f.append(input)
            x = getattr(self, 'layer3_' + str(i))(input)
            f.append(x)
            x = getattr(self, 'layer4_' + str(i))(x)
            f.append(x)
            x = self.avgpool(x)
            
            x = x.view(x.size(0), -1)
            embedding.append(x)
            x = getattr(self, 'classifier_' + str(i))(x)
            logits.append(x)

        return logits, embedding, f

def resnet18(num_classes, number_net):
    return CIFAR_ResNet(PreActBlock, [2,2,2,2], num_classes=num_classes, number_net=number_net)




def resnet34(num_classes, number_net):
    return CIFAR_ResNet(PreActBlock, [3,4,6,3], num_classes=num_classes, number_net=number_net)




if __name__ == '__main__':
    net = CIFAR_ResNet18(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    import sys
    sys.path.append('..')
    from utils import cal_param_size, cal_multi_adds
    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (2, 3, 32, 32)) / 1e6))