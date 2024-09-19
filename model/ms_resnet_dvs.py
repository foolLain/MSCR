import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, surrogate
from spikingjelly.activation_based.neuron import LIFNode,IFNode
import numpy as np

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
                 base_width=64, dilation=1, norm_layer=None,T_c=None,T=None):
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
        self.sn1 = LIFNode(tau=1.5, detach_reset=True, backend='torch')

        self.conv2 = layer.SeqToANNContainer(
            conv3x3(planes, planes),
            norm_layer(planes)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn2 = LIFNode(tau=1.5, detach_reset=True, backend='torch')

    def forward(self, x):
        identity = x

        out = self.conv1(self.sn1(x))

        out = self.conv2(self.sn2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity

        return out





class MSResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T=4,T_c_config=None ):
        super(MSResNet, self).__init__()
        self.T = T
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = layer.SeqToANNContainer(nn.Conv2d(2, self.inplanes, kernel_size=3, stride=2, padding=1,
                               bias=False))
        self.bn1 = layer.SeqToANNContainer(norm_layer(self.inplanes))

        self.sn1 = LIFNode(tau=1.5, detach_reset=True, backend='torch')
        self.maxpool = layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

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
                LIFNode(tau=1.5, detach_reset=True, backend='torch'),
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
        x = x.permute(1, 0, 2, 3, 4)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
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

def ms_resnet20(**kwargs):
    return ms_resnet(BasicBlock, [3, 3, 3], **kwargs)

class BasicBlock_qs(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,T_c =None,T=None):
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
        self.sn1 = LIFNode(tau=1.5, detach_reset=True, backend='torch')

        self.conv2 = layer.SeqToANNContainer(
            conv3x3(planes, planes),
            norm_layer(planes)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn2 = LIFNode(tau=1.5, detach_reset=True, backend='torch')
        self.T  = T
        # self.time_compression0 = nn.Linear(self.T,1)
        # self.time_compression1 = nn.Linear(T_c,1)
        # self.time_recovery0 = nn.Linear(1,T_c)
        # self.time_recovery1 = nn.Linear(1,self.T)
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

        # out = self.conv1(self.time_compression0(self.sn1(x).permute(1,2,3,4,0)).permute(4,0,1,2,3))

        # out = self.conv2(self.time_compression1(self.sn2(self.time_recovery0(out.permute(1,2,3,4,0)).permute(4,0,1,2,3)).permute(1,2,3,4,0)).permute(4,0,1,2,3))

        # if self.downsample is not None:
        #     identity = self.downsample(x)
        #     out = self.time_recovery1(out.permute(1,2,3,4,0)).permute(4,0,1,2,3) + identity
        # else:
        #     out = self.time_recovery1(out.permute(1,2,3,4,0)).permute(4,0,1,2,3) + identity

        return out


class downsample_module(nn.Module):
    def __init__(self,inplanes,planes,block_expansion,stride,norm_layer):
        super(downsample_module, self).__init__()
        self.sn = LIFNode(tau=1.5, detach_reset=True, backend='torch')
        self.Conv_BN = layer.SeqToANNContainer(
                    conv1x1(inplanes, planes * block_expansion, stride),
                    norm_layer(planes * block_expansion))
    def forward(self,x):
        return self.Conv_BN(self.sn(x).mean(0,keepdim=True))

class MSResNet_qs(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T=4,T_c_config=[None,None,None,None] ):
        super(MSResNet_qs, self).__init__()
        self.T = T
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = layer.SeqToANNContainer(nn.Conv2d(2, self.inplanes, kernel_size=3, stride=2, padding=1,
                               bias=False))
        self.bn1 = layer.SeqToANNContainer(norm_layer(self.inplanes))

        self.sn1 = LIFNode(tau=1.5, detach_reset=True, backend='torch')
        self.maxpool = layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


        self.layer1 = self._make_layer(block, 16, layers[0],T_c=T_c_config[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],T_c=T_c_config[1])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],T_c=T_c_config[2])

        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,T_c=None):
        if T_c==None or T_c == 0:
            block = BasicBlock
        else:
            block = BasicBlock_qs
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
                            self.base_width, previous_dilation, norm_layer,T_c=T_c,T=self.T))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,T_c=T_c,T=self.T))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = x.permute(1, 0, 2, 3, 4)  # T,B,C,H,W\
        # print(x.max(),x.min())
        x = x.mean(0,keepdim=True).expand(x.shape[0],-1,-1,-1,-1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)  
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
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


def ms_resnet20_qs(**kwargs):
    return ms_resnet_qs(BasicBlock_qs, [3, 3, 3], **kwargs)






if __name__ == '__main__':
    x = torch.randn(4,16,2,128,128)
    # model = ms_resnet20(T=16)
    # y = model(x)
    model = ms_resnet20_qs(T=16,T_c_config=[16,16,16])
    y = model(x)
    print("test ok")