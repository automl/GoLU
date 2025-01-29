"""
This file contains the ResNet Architecture.
References - 
    1. https://arxiv.org/pdf/1512.03385.pdf
    2. https://arxiv.org/pdf/1604.04112v4.pdf
    3. https://github.com/akamaster/pytorch_resnet_cifar10/tree/master
"""

from typing import List
import torch
from torch import nn
from torch.nn import init

from golu.activation_utils import get_activation_function


def _weights_init(m):
    """This is a custom weight initialization method.

    Args:
        m (_type_): Layers of the network.
    """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class BasicBlock(nn.Module):
    """This is the implementation of the Basic ResNet Block.

    Args:
        nn (_type_): Extends the nn.Module class from PyTorch.
    """
    expansion = 1


    def __init__(self, activation: str, in_channels: int,
                 out_channels: int, stride: int = 1):
        """Constructor of the Basic ResNet Block.

        Args:
            activation (str): The activation to be used for training the network.
            in_channels (int): The number of input channels to a Conv Layer.
            out_channels (int): The number of output channels from a Conv Layer.
            stride (int, optional): The stride which should be passed in a Conv Layer.
            Defaults to 1.
        """
        super(BasicBlock, self).__init__()

        if activation in ['FlexiV5', 'FlexiV6', 'FlexiV7', 'FlexiV9', 'SwishPar']:
            dim_net = (0, 2, 3)
            requires_grad = [False, True, True]
        else:
            dim_net = (1, 2, 3)
            requires_grad = [False, True, True]

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation_function_net1 = get_activation_function(activation)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, self.expansion * out_channels,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * out_channels)
            )

        self.activation_function_net2 = get_activation_function(activation)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Corresponds to the forward pass of the Basic ResNet Block.

        Args:
            x (torch.Tensor): The input tensor passed through the network.

        Returns:
            torch.Tensor: Output of the network.
        """
        out = self.activation_function_net1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation_function_net2(out)
        return out


class ResNet(nn.Module):
    """This is the implementation of the ResNet Architecture.

    Args:
        nn (_type_): Extends the nn.Module class from PyTorch.
    """
    def __init__(self, num_blocks: List, num_classes: int, activation: str):
        """Constructor of the ResNet Architecture.

        Args:
            num_blocks (List): Number of blocks in ResNet.
            num_classes (int): The number of classes in the task.
            activation (str): The activation to be used for training the architecture.
        """
        super(ResNet, self).__init__()

        if activation in ['FlexiV5', 'FlexiV6', 'FlexiV7', 'FlexiV9']:
            dim_net = (0, 2, 3)
            requires_grad = [False, True, True]
        else:
            dim_net = (1, 2, 3)
            requires_grad = [False, True, True]

        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation_function_net = get_activation_function(activation)
        self.layer1 = self._make_layer(activation=activation, out_channels=16,
                                       num_blocks=num_blocks[0], stride=1)
        self.layer2 = self._make_layer(activation=activation, out_channels=32,
                                       num_blocks=num_blocks[1], stride=2)
        self.layer3 = self._make_layer(activation=activation, out_channels=64,
                                       num_blocks=num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)


    def _make_layer(self, activation: str, out_channels: int,
                    num_blocks: int, stride: int) -> nn.Sequential:
        """This function is used to stack multiple Basic Blocks together.

        Args:
            activation (str): The activation to be used for training the architecture.
            out_channels (int): The number of output channels from a Conv Layer.
            num_blocks (int): The number of Basic Blocks.
            stride (int): The stride in convolution of a Basic Block.

        Returns:
            nn.Sequential: a stack of Basic ResNet Blocks.
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(activation, self.in_channels, out_channels, stride))
            self.in_channels = out_channels * BasicBlock.expansion

        return nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Corresponds to the forward pass of the ResNet Architecture.

        Args:
            x (torch.Tensor): The input tensor passed through the network.

        Returns:
            torch.Tensor: Predictions from the network.
        """
        out = self.activation_function_net(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(activation: str, num_classes: int) -> ResNet:
    """This function creates a ResNet20 model.

    Args:
        activation (str): The activation to be used for training the architecture.
        num_classes (int): The number of classes in the task.

    Returns:
        ResNet: Returns a ResNet model.
    """
    return ResNet(num_blocks=[3, 3, 3], num_classes=num_classes, activation=activation)


def resnet32(activation: str, num_classes: int) -> ResNet:
    """This function creates a ResNet32 model.

    Args:
        activation (str): The activation to be used for training the architecture.
        num_classes (int): The number of classes in the task.

    Returns:
        ResNet: Returns a ResNet model.
    """
    return ResNet(num_blocks=[5, 5, 5], num_classes=num_classes, activation=activation)


def resnet44(activation: str, num_classes: int) -> ResNet:
    """This function creates a ResNet44 model.

    Args:
        activation (str): The activation to be used for training the architecture.
        num_classes (int): The number of classes in the task.

    Returns:
        ResNet: Returns a ResNet model.
    """
    return ResNet(num_blocks=[7, 7, 7], num_classes=num_classes, activation=activation)


def resnet56(activation: str, num_classes: int) -> ResNet:
    """This function creates a ResNet56 model.

    Args:
        activation (str): The activation to be used for training the architecture.
        num_classes (int): The number of classes in the task.

    Returns:
        ResNet: Returns a ResNet model.
    """
    return ResNet(num_blocks=[9, 9, 9], num_classes=num_classes, activation=activation)


def resnet110(activation: str, num_classes: int) -> ResNet:
    """This function creates a ResNet110 model.

    Args:
        activation (str): The activation to be used for training the architecture.
        num_classes (int): The number of classes in the task.

    Returns:
        ResNet: Returns a ResNet model.
    """
    return ResNet(num_blocks=[18, 18, 18], num_classes=num_classes, activation=activation)
