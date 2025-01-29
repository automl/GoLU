"""
This file contains the Wide ResNet Architecture.
References - 
    1. https://arxiv.org/pdf/1605.07146.pdf
    2. https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py
"""

import numpy as np
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
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)


def conv3x3(in_channels: int, out_channels: int, stride: int=1) -> nn.Conv2d:
    """This function creates a Conv2d instance.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride to be applied to Convolution. Defaults to 1.

    Returns:
        nn.Conv2d: An instance of the nn.Conv2d class.
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


class wide_basic(nn.Module):
    """This is the implementation of the Wide Basic ResNet Block.

    Args:
        nn (_type_): Extends the nn.Module class from PyTorch.
    """
    def __init__(self, activation, in_channels, out_channels, stride=1):
        """Constructor of the wide_basic Block.

        Args:
            activation (str): The activation to be used for training the network.
            in_channels (int): The number of input channels to a Conv Layer.
            out_channels (int): The number of output channels from a Conv Layer.
            stride (int, optional): The stride which should be passed in a Conv Layer. Defaults to 1.
        """
        
        if activation in ['FlexiV5', 'FlexiV6', 'FlexiV7', 'FlexiV9', 'SwishPar']:
            dim_net = (0, 2, 3)
            requires_grad = [False, True, True]
        else:
            dim_net = (1, 2, 3)
            requires_grad = [False, True, True]
        
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.activation_function_net1 = get_activation_function(activation)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation_function_net2 = get_activation_function(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Corresponds to the forward pass of the Wide Basic Architecture.

        Args:
            x (torch.Tensor): The input tensor passed through the network.

        Returns:
            torch.Tensor: Output from the network.
        """
        out = self.bn1(x)
        out = self.activation_function_net1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.activation_function_net2(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class Wide_ResNet(nn.Module):
    """This is the implementation of the Wide_ResNet Architecture.

    Args:
        nn (_type_): Extends the nn.Module class from PyTorch.
    """
    def __init__(
        self, activation: str, num_classes: int, depth: int, widen_factor: int
    ):
        """Constructor of the Wide ResNet Architecture.

        Args:
            activation (str): The activation to be used for training the architecture.
            num_classes (int): The number of classes in the task.
            depth (int): The depth of the network.
            widen_factor (int): The width of the network.
        """
        super(Wide_ResNet, self).__init__()

        if activation in ['FlexiV5', 'FlexiV6', 'FlexiV7', 'FlexiV9', 'SwishPar']:
            dim_net = (0, 2, 3)
            requires_grad = [False, True, True]
        else:
            dim_net = (1, 2, 3)
            requires_grad = [False, True, True]

        self.in_channels = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(
            activation=activation, out_channels=nStages[1], num_blocks=n, stride=1
        )
        self.layer2 = self._wide_layer(
            activation=activation, out_channels=nStages[2], num_blocks=n, stride=2
        )
        self.layer3 = self._wide_layer(
            activation=activation, out_channels=nStages[3], num_blocks=n, stride=2
        )
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.activation_function_net = get_activation_function(activation)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(nStages[3], num_classes)

        self.apply(_weights_init)

    def _wide_layer(
        self, activation: str, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """This function is used to stack multiple wide_basic Blocks together.

        Args:
            activation (str): The activation to be used for training the architecture.
            out_channels (int): The number of output channels from a Conv Layer.
            num_blocks (int): The number of wide_basic blocks.
            stride (int): The stride to be applied to convolutions.

        Returns:
            nn.Sequential: A stack of wide_basic blocks.
        """
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(
                wide_basic(
                    activation=activation,
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=stride
                )
            )
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Corresponds to the forward pass of the Wide ResNet Architecture.

        Args:
            x (torch.Tensor): The input tensor passed through the network.

        Returns:
            torch.Tensor: Predictions from the network.
        """
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn1(out)
        out = self.activation_function_net(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# Dropout not added - the original paper did not run this version with dropout
# Other versions have dropout in them
def wideresnet_28_2(activation: str, num_classes: int) -> Wide_ResNet:
    """This function creates a Wide_ResNet model with depth=28 and width=2.

    Args:
        activation (str): The activation to be used for training the architecture.
        num_classes (int): The number of classes in the task.

    Returns:
        Wide_ResNet: Returns a Wide_ResNet model.
    """
    return Wide_ResNet(
        activation=activation,
        num_classes=num_classes,
        depth=28,
        widen_factor=2
    )
