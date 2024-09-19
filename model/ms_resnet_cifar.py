import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, surrogate,functional
from spikingjelly.activation_based.neuron import ParametricLIFNode, LIFNode,IFNode


surrogate_function = surrogate.ATan()


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('SpikingBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SpikingBasicBlock")

        self.conv1 = layer.SeqToANNContainer(
            conv3x3(inplanes, planes, stride),
            norm_layer(planes)
        )
        self.sn1 = IFNode(detach_reset=True, surrogate_function=surrogate_function, backend='torch')

        self.conv2 = layer.SeqToANNContainer(
            conv3x3(planes, planes),
            norm_layer(planes)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn2 = IFNode(detach_reset=True, surrogate_function=surrogate_function, backend='torch')

    def forward(self, x):
        identity = x

        out = self.conv1(self.sn1(x))

        out = self.conv2(self.sn2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = layer.SeqToANNContainer(
            conv1x1(inplanes, width),
            norm_layer(width)
        )
        self.sn1 = IFNode(detach_reset=True, surrogate_function=surrogate_function, backend='torch')

        self.conv2 = layer.SeqToANNContainer(
            conv3x3(width, width, stride, groups, dilation),
            norm_layer(width)
        )
        self.sn2 = IFNode(detach_reset=True, surrogate_function=surrogate_function, backend='torch')

        self.conv3 = layer.SeqToANNContainer(
            conv1x1(width, planes * self.expansion),
            norm_layer(planes * self.expansion)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn3 = IFNode(detach_reset=True, surrogate_function=surrogate_function, backend='torch')

    def forward(self, x):
        identity = x

        out = self.conv1(self.sn1(x))

        out = self.conv2(self.sn2(out))

        out = self.conv3(self.sn3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out


class MSResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T=4, T_c_config=None):
        super(MSResNet, self).__init__()
        self.T = T
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)

        self.sn1 = IFNode(detach_reset=True, surrogate_function=surrogate_function, backend='torch')
        self.maxpool = layer.SeqToANNContainer(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                IFNode(detach_reset=True, surrogate_function=surrogate_function, backend='torch'),
                layer.SeqToANNContainer(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
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

    def _forward_impl(self, x, return_feature=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x.unsqueeze_(0)
        x = x.repeat(self.T, 1, 1, 1, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.sn1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc(x)
        return x.mean(dim=0)

    def forward(self, x, return_feature=False):
        return self._forward_impl(x, return_feature)


def ms_resnet(block, layers, **kwargs):
    model = MSResNet(block, layers, **kwargs)
    return model


def ms_resnet18(**kwargs):
    return ms_resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ms_resnet34(**kwargs):
    return ms_resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ms_resnet104(**kwargs):
    return ms_resnet(BasicBlock, [3, 8, 32, 8], **kwargs)



class BasicBlock_qs(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,T_c =None):
        super(BasicBlock_qs, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('SpikingBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SpikingBasicBlock")

        self.conv1 = layer.SeqToANNContainer(
            conv3x3(inplanes, planes, stride),
            norm_layer(planes)
        )
        self.sn1 = IFNode(detach_reset=True, surrogate_function=surrogate_function, backend='torch')

        self.conv2 = layer.SeqToANNContainer(
            conv3x3(planes, planes),
            norm_layer(planes)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn2 = IFNode(detach_reset=True, surrogate_function=surrogate_function, backend='torch')

        self.T_c = T_c

    def forward(self, x): 
        T = x.shape[0] if not self.T_c else self.T_c 
        identity = x  
        out = self.conv1(self.sn1(x).mean(0,keepdim=True)) 
        out = self.conv2(self.sn2(out.expand(T,-1,-1,-1,-1)).mean(0,keepdim=True))
        if self.downsample is not None:
            identity = self.downsample(x)
            out = out.expand(T,-1,-1,-1,-1) + identity.expand(T,-1,-1,-1,-1)
        else:
            out = out.expand(T,-1,-1,-1,-1) + identity.mean(0,keepdim=True).expand(T,-1,-1,-1,-1)   
        return out


class Bottleneck_qs(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,T_c =None):
        super(Bottleneck_qs, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = layer.SeqToANNContainer(
            conv1x1(inplanes, width),
            norm_layer(width)
        )
        self.sn1 = IFNode(detach_reset=True, surrogate_function=surrogate_function, backend='torch')

        self.conv2 = layer.SeqToANNContainer(
            conv3x3(width, width, stride, groups, dilation),
            norm_layer(width)
        )
        self.sn2 = IFNode(detach_reset=True, surrogate_function=surrogate_function, backend='torch')

        self.conv3 = layer.SeqToANNContainer(
            conv1x1(width, planes * self.expansion),
            norm_layer(planes * self.expansion)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn3 = IFNode(detach_reset=True, surrogate_function=surrogate_function, backend='torch')
        self.T_c = T_c

    def forward(self, x):
        T = x.shape[0] if not self.T_c else self.T_c 
        identity = x
        out = self.conv1(self.sn1(x).mean(0,keepdim=True))

        out = self.conv2(self.sn2(out.expand(T,-1,-1,-1,-1)).mean(0,keepdim=True))

        out = self.conv3(self.sn3(out.expand(T,-1,-1,-1,-1)).mean(0,keepdim=True))

        if self.downsample is not None:
            identity = self.downsample(x)
            out = out.expand(T,-1,-1,-1,-1) + identity.expand(T,-1,-1,-1,-1)
        else:
            out = out.expand(T,-1,-1,-1,-1) + identity.mean(0,keepdim=True).expand(T,-1,-1,-1,-1)           

        return out

class downsample_module(nn.Module):
    def __init__(self,inplanes,planes,block_expansion,stride,norm_layer):
        super(downsample_module, self).__init__()
        self.sn = IFNode(detach_reset=True, surrogate_function=surrogate_function, backend='torch')
        self.Conv_BN = layer.SeqToANNContainer(
                    conv1x1(inplanes, planes * block_expansion, stride),
                    norm_layer(planes * block_expansion))
        
    def forward(self,x):
        return self.Conv_BN(self.sn(x).mean(0,keepdim=True))

class MSResNet_qs(nn.Module):

    def __init__(self, block, layers, num_classes=100, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T=4, T_c_config=[None,None,None,None]):
        super(MSResNet_qs, self).__init__()
        self.T = T
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)

        self.sn1 = IFNode(detach_reset=True, surrogate_function=surrogate_function, backend='torch')
        self.maxpool = layer.SeqToANNContainer(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64, layers[0],T_c=T_c_config[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],T_c=T_c_config[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],T_c=T_c_config[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],T_c=T_c_config[3])
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,T_c=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = downsample_module(self.inplanes,planes,block.expansion,stride,norm_layer)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,T_c=T_c))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,T_c=T_c))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x.unsqueeze_(0)
        x = x.repeat(self.T, 1, 1, 1, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.sn1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc(x)
        return x.mean(dim=0)

    def forward(self, x):
        return self._forward_impl(x)




def ms_resnet_qs(block, layers, **kwargs):
    model = MSResNet_qs(block, layers, **kwargs)
    return model


def ms_resnet18_qs(**kwargs):
    return ms_resnet_qs(BasicBlock_qs, [2, 2, 2, 2], **kwargs)


def ms_resnet34_qs(**kwargs):
    return ms_resnet_qs(BasicBlock_qs, [3, 4, 6, 3], **kwargs)


def ms_resnet104_qs(**kwargs):
    return ms_resnet_qs(BasicBlock_qs, [3, 8, 32, 8], **kwargs)


if __name__ == '__main__':
    model = ms_resnet18_qs(T_c_config=[4,2,4,2])
    x = torch.randn(32,3,32,32)
    y = model(x)
    functional.set_step_mode(model,'m')
    print(y.shape)