import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['resnet32', 'resnet56', 'resnet110']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
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
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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


class ResNet(nn.Module):

    def __init__(self, block, layers, number_net=4, num_classes=100, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.number_net = number_net

        self.dilation = 1
        self.inplanes = 16
        self.number_net = number_net
        self.num_classes = num_classes

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64 * block.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        embedding = x
        x = self.classifier(x)

        return x, embedding

    # Allow for accessing forward method in a inherited class
    forward = _forward


# class ResNet_n(nn.Module):
#     def __init__(self, block, layers, num_classes=100, number_net=2):
#         super(ResNet_n, self).__init__()
#         self.number_net = number_net
#
#         self.module_list = nn.ModuleList([])
#         for i in range(number_net):
#             self.module_list.append(ResNet(num_classes=num_classes,
#                                            block=block, layers=layers))
#
#     def forward(self, x):
#         logits = []
#         embeddings = []
#         for i in range(self.number_net):
#             log, emb = self.module_list[i](x)
#             logits.append(log)
#             embeddings.append(emb)
#         return logits, embeddings


class ResNet_b(nn.Module):
    def __init__(self, block, layers, num_classes=100, number_net=4, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_b, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.number_net = number_net

        self.dilation = 1
        self.inplanes = 16
        self.number_net = number_net
        self.num_classes = num_classes

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        fix_planes = self.inplanes
        for i in range(self.number_net):
            self.inplanes = fix_planes
            setattr(self, 'layer2_' + str(i), self._make_layer(block, 32, layers[1], stride=2,
                                                               dilate=replace_stride_with_dilation[1]))
            setattr(self, 'layer3_' + str(i), self._make_layer(block, 64, layers[2], stride=2,
                                                               dilate=replace_stride_with_dilation[2]))
            setattr(self, 'classifier_' + str(i), nn.Linear(64 * block.expansion, self.num_classes))

        self.conv = nn.Conv2d(1200, 100, kernel_size=1, stride=1, padding=1,
                              bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.control_v1 = nn.Linear(16, 3)
        self.bn_v1 = nn.BatchNorm1d(3)
        self.avgpool_c = nn.AvgPool2d(16)

        # self.cbam1 = CBAM(16 * block.expansion)
        # self.cbam2 = CBAM(32 * block.expansion)
        # self.cbam3 = CBAM(48 * block.expansion)

        

        # self.sa = SA(48 * block.expansion)

        self.classfier1 = nn.Linear(16, num_classes)
        self.classfier2 = nn.Linear(32, num_classes)
        self.classfier3 = nn.Linear(64, num_classes)

        self.classfier_f = nn.Linear(64, num_classes)

        self.lin = nn.Linear(300, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward(self, input):
        f = []
        x_t = []
        c=[]
        x = self.conv1(input)
        # i = self.avgpool(x)  # B×16×1×1
        # fea = i.view(i.size(0), -1)  # B×16
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x1 = x
        # x_1 = self.ca1(x)
        # c_1=x_1.view(x_1.size(0), -1)
        # c.append(c_1)
        # x1 = self.avgpool(x)  # B×16×1×1
        # f1 = x1.view(x1.size(0), -1)  # B×16
        # f11 = self.sa1(f1, f1, f1)
        # f11 = f11.view(f11.size(0), -1)

        logits = []
        embedding = []
        t_l = []
        input = x
        for i in range(self.number_net):
            f.append(x1)
            x = getattr(self, 'layer2_' + str(i))(input)
            f.append(x)
            # x_2 = self.ca2(x)
            # c_2 = x_2.view(x_2.size(0), -1)
            # c.append(c_2)
            # x2 = self.avgpool(x)  # B×32×1×1
            # f2 = x2.view(x2.size(0), -1)  # B×32
            # f22 = self.sa2(f2, f2, f2)
            # f22 = f11.view(f22.size(0), -1)
            # f.append(f2)
            x = getattr(self, 'layer3_' + str(i))(x)
            f.append(x)
            # x_3 = self.ca3(x)
            # c_3 = x_3.view(x_3.size(0), -1)
            x3 = self.avgpool(x)  # B×64×1×1
            f3 = x3.view(x3.size(0), -1)  # B×64
            # f33 = self.sa3(f3, f3, f3)
            # f33 = f11.view(f33.size(0), -1)
            # f.append(f3)

            embedding.append(f3)
            x = getattr(self, 'classifier_' + str(i))(f3)
            logits.append(x)

            # f_t = torch.cat([f1, f2], 1)
            # f_t = torch.cat([f_t, f3], 1)
            # f_ca = self.ca4(f_t.unsqueeze(-1).unsqueeze(-1))
            # f_ca = f_ca.view(f_ca.size(0), -1)
            # t_l.append(f_ca)
            
            # ff_t = self.sa(f3, f3, f3)
            # ff_t = ff_t.view(ff_t.size(0), -1)
            # ff_t = self.classfier_f(ff_t)
            # x_t.append(ff_t)


            # x_c = self.control_v1(fea)  # B×3
            # x_c = self.bn_v1(x_c)  # B×3
            # x_c = F.relu(x_c)  # B×3
            # w = F.softmax(x_c, dim=1)  # B×3
            #
            # x_1 = self.classfier1(f1)
            # x_2 = self.classfier2(f2)
            # x_3 = self.classfier3(f3)
            #
            # w_1 = w[:, 0].repeat(x_1.size()[1], 1).transpose(0, 1)
            # w_2 = w[:, 1].repeat(x_2.size()[1], 1).transpose(0, 1)
            # w_3 = w[:, 2].repeat(x_3.size()[1], 1).transpose(0, 1)
            #
            # f_1 = w_1 * x_1
            # f_2 = w_2 * x_2
            # f_3 = w_3 * x_3
            # # f=torch.cat([f,w_1*f1],1)
            # f_t = torch.cat([f_1, f_2], 1)
            # f_t = torch.cat([f_t, f_3], 1)
            #
            # ft = self.lin(f_t)
            # t_l.append(ft)
            #
            # if i == 0:
            #     x_t.append(f_t)
            #     # x_totalfea=f_t
            # else:
            #     x_t.append(f_t)
        #         x_totalfea=torch.cat([x_totalfea,f_t],1)
        #
        # # x = nn.BatchNorm2d(x_totalfea.size(0))
        # x=x_totalfea.unsqueeze(-1).unsqueeze(-1)
        # x=self.conv(x)
        # x = self.avgpool(x)
        # feature_logit = x.view(x.size(0), -1)
        # # x=torch.reshape(x_totalfea,[1,60,1,1])
        # # x = F.adaptive_avg_pool2d(F.relu(x),1)
        # x= self.avgpool(x)
        # # x=torch.reshape(x,[1,64])
        # # x=x_totalfea.transpose(0,1)
        # feature_logit = getattr(self, 'classifier_' + str(i))(x)
        # feature_logit=1

        return logits, embedding, f

    # Allow for accessing forward method in a inherited class
    forward = _forward


def resnet32(num_classes, number_net):
    arch = ResNet_b
    return arch(BasicBlock, [5, 5, 5], num_classes=num_classes, number_net=number_net)


def resnet56(num_classes, number_net):
    arch = ResNet_b
    return arch(BasicBlock, [9, 9, 9], num_classes=num_classes, number_net=number_net)


def resnet110(num_classes, number_net):
    arch = ResNet_b
    return arch(Bottleneck, [12, 12, 12], num_classes=num_classes, number_net=number_net)


if __name__ == '__main__':
    net = resnet32(num_classes=100, number_net=4)
    for n, p in net.named_parameters():
        print(n)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    from utils import cal_param_size, cal_multi_adds

    print('Params: %.2fM, Multi-adds: %.3fM'
          % (cal_param_size(net) / 1e6, cal_multi_adds(net, (2, 3, 32, 32)) / 1e6))

