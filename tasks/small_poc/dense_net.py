"""
This file contains the Dense ResNet Architecture.
References - 
    1. https://arxiv.org/pdf/1608.06993.pdf
    2. https://github.com/kuangliu/pytorch-cifar/blob/master/models/densenet.py
"""

import math
from typing import List

import torch
import torch.nn as nn
from torch.nn import init

from golu.activation_utils import get_activation_function


def _weights_init(m):
    """This is a custom weight initialization method.

    Args:
        m (_type_): Layers of the network.
    """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class Bottleneck(nn.Module):
    """This is the implementation of the Bottleneck Block.

    Args:
        nn (_type_): Extends the nn.Module class from PyTorch.
    """
    def __init__(self, activation: str, in_planes: int, growth_rate: int):
        """Constructor of the Bottleneck Block.

        Args:
            activation (str): The activation function used to train the network.
            in_planes (int): Number of input channels.
            growth_rate (int): Factor by which the number of channels are increased.
        """
        super(Bottleneck, self).__init__()

        if activation in ['FlexiV5', 'FlexiV6', 'FlexiV7', 'FlexiV9', 'SwishPar']:
            dim_net = (0, 2, 3)
            requires_grad = [False, True, True]
        else:
            dim_net = (1, 2, 3)
            requires_grad = [False, True, True]

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.activation_function_net1 = get_activation_function(activation)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.activation_function_net2 = get_activation_function(activation)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Corresponds to the forward pass of the Bottleneck Block.

        Args:
            x (torch.Tensor): The input tensor passed through the network.

        Returns:
            torch.Tensor: Output tensor from this block.
        """
        out = self.bn1(x)
        out = self.activation_function_net1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.activation_function_net2(out)
        out = self.conv2(out)
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    """This is the implementation of the Transition Block.

    Args:
        nn (_type_): Extends the nn.Module class from PyTorch.
    """
    def __init__(self, activation: str, in_planes: int, out_planes: int):
        """Constructor of the Transition Block.

        Args:
            activation (str): The activation function used to train the network.
            in_planes (int): Number of input channels.
            out_planes (int): Number of output channels.
        """
        super(Transition, self).__init__()

        if activation in ['FlexiV5', 'FlexiV6', 'FlexiV7', 'FlexiV9', 'SwishPar']:
            dim_net = (0, 2, 3)
            requires_grad = [False, True, True]
        else:
            dim_net = (1, 2, 3)
            requires_grad = [False, True, True]

        self.bn = nn.BatchNorm2d(in_planes)
        self.activation_function_net = get_activation_function(activation)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.avgpool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Corresponds to the forward pass of Transition Block.

        Args:
            x (torch.Tensor): The input tensor passed through the network.

        Returns:
            torch.Tensor: Output tensor from this block.
        """
        out = self.bn(x)
        out = self.activation_function_net(out)
        out = self.conv(out)
        out = self.avgpool(out)
        return out


class DenseNet(nn.Module):
    """This is the implementation of the DenseNet Architecture.

    Args:
        nn (_type_): Extends the nn.Module class from PyTorch.
    """
    def __init__(self, nblocks: List, growth_rate: int, reduction: float, activation: str, num_classes: int):
        """Constructor of the Dense ResNet Architecture.

        Args:
            nblocks (List): Number of blocks in DenseNet.
            growth_rate (int): Controls number of feature maps in the blocks.
            reduction (float): Fraction by which number of feature maps are reduced or retained.
            activation (str): The activation to be used for training the architecture.
            num_classes (int): The number of classes in the task.
        """
        super(DenseNet, self).__init__()

        if activation in ['FlexiV5', 'FlexiV6', 'FlexiV7', 'FlexiV9', 'SwishPar']:
            dim_net = (0, 2, 3)
            requires_grad = [False, True, True]
        else:
            dim_net = (1, 2, 3)
            requires_grad = [False, True, True]

        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(activation=activation, in_planes=num_planes, nblock=nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(activation=activation, in_planes=num_planes, out_planes=out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(activation=activation, in_planes=num_planes, nblock=nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(activation=activation, in_planes=num_planes, out_planes=out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(activation=activation, in_planes=num_planes, nblock=nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(activation=activation, in_planes=num_planes, out_planes=out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(activation=activation, in_planes=num_planes, nblock=nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.activation_function_net = get_activation_function(activation)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(num_planes, num_classes)
        
        self.apply(_weights_init)

    def _make_dense_layers(self, activation: str, in_planes: int, nblock: int) -> nn.Sequential:
        """This function is used to stack multiple Dense Blocks.

        Args:
            activation (str): The activation function used for training the network.
            in_planes (int): Numer of input channels.
            nblock (int): Number of Bottlenecks to be stacked together.

        Returns:
            nn.Sequential: Returns a stack of Dense Blocks.
        """
        layers = []
        for _ in range(nblock):
            layers.append(Bottleneck(activation, in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Corresponds to the forward pass of the DenseNet Architecture.

        Args:
            x (torch.Tensor): The input tensor passed through the network.

        Returns:
            torch.Tensor: Predictions from the network.
        """
        out = self.conv1(x)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        out = self.trans3(out)
        out = self.dense4(out)
        out = self.bn(out)
        out = self.activation_function_net(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def densenet_40(activation: str, num_classes: int) -> DenseNet:
    """This function creates a DenseNet-40 model with growth_rate=12 and reduction=0.5.

    Args:
        activation (str): The activation to be used for training the architecture.
        num_classes (int): The number of classes in the task.

    Returns:
        DenseNet: Returns a DenseNet-40 model.
    """
    return DenseNet(
        nblocks=[12, 12, 12],
        growth_rate=12,
        reduction=0.5,
        activation=activation,
        num_classes=num_classes
    )
