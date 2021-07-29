### dropout has been removed in this code. original code had dropout#####
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import random


import math



class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

    
    
class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)
        
def mixup_data(x, y, lam):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
   
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_b = y[index]
    return mixed_x, y_b, lam

    
class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, stride = 1):
        flatten = True
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, stride)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2)
        # global average pooling and linear
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = nChannels[3]
        self.outdim = nChannels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    
    def forward(self, x, target= None, mixup=False, mixup_hidden=True, mixup_alpha=None , lam = 0.4):
        if target is not None: 
            if mixup_hidden:
                layer_mix = random.randint(0,3)
            elif mixup:
                layer_mix = 0
            else:
                layer_mix = None   

            out = x

            target_a = target_b  = target

            if layer_mix == 0:
                out, target_b , lam = mixup_data(out, target, lam=lam)

            out = self.conv1(out)
            out = self.block1(out)


            if layer_mix == 1:
                out, target_b , lam  = mixup_data(out, target, lam=lam)

            out = self.block2(out)

            if layer_mix == 2:
                out, target_b , lam = mixup_data(out, target, lam=lam)


            out = self.block3(out)
            if  layer_mix == 3:
                out, target_b , lam = mixup_data(out, target, lam=lam)

            out = self.relu(self.bn1(out))

            return out, target_b
        else: 
            out = x
            out = self.conv1(out)
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            return out
        
                  
def create_model():
    return WideResNet(depth=28, widen_factor=10, stride = 1)
