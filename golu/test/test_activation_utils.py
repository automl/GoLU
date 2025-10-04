"""
This file contains unittests to test the utilities for getting and changing an activation
"""

import unittest
import torch.nn as nn
from torch.nn import Sigmoid, Tanh, ReLU, Softplus, LeakyReLU, PReLU, ELU, SELU, CELU, GELU, SiLU, Mish
from golu.golu_activation import GoLU
from golu.activation_utils import get_activation_function, replace_activation_by_torch_module, \
    replace_activation_by_name, update_golu_parameters


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = nn.Conv2d(1, 10, kernel_size=5)
        self.activation = ReLU()
        self.fc = nn.Linear(10, 1)
        self.activation2 = Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.activation2(x)
        return x


class GoLUNet(nn.Module):
    def __init__(self):
        super(GoLUNet, self).__init__()
        self.linear = nn.Linear(10, 10)
        self.activation = GoLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x


class TestActivationUtils(unittest.TestCase):
    
    def test_get_activation_function_valid(self):
        activations = [
            "Sigmoid", "Tanh", "ReLU", "Softplus", "LeakyReLU", "PReLU", "ELU", "SELU", "CELU", "GELU", "GELU_tanh",
            "Swish", "Mish", "GoLU"
        ]
        expected_types = [
            Sigmoid, Tanh, ReLU, Softplus, LeakyReLU, PReLU, ELU, SELU, CELU, GELU, GELU, SiLU, Mish, GoLU
        ]
        for activation_name, expected_type in zip(activations, expected_types):
            with self.subTest(activation=activation_name):
                activation_function = get_activation_function(activation=activation_name)
                self.assertIsInstance(activation_function, expected_type)
                if activation_name == "GELU_tanh":
                    self.assertEqual(activation_function.approximate, "tanh")

    def test_get_activation_function_invalid(self):
        invalid_activations = ["", "UnknownActivation", "Relu"]  # Case-sensitive
        for activation_name in invalid_activations:
            with self.subTest(activation=activation_name):
                with self.assertRaises(ValueError) as context:
                    get_activation_function(activation=activation_name)
                self.assertIn(
                    f"The activation named {activation_name} doesn't exists!",
                    str(context.exception),
                )

    def test_replace_activation_by_torch_module(self):
        net = SimpleNet()
        self.assertIsInstance(net.activation, ReLU)
        self.assertIsInstance(net.activation2, Sigmoid)
        new_net = replace_activation_by_torch_module(net, ReLU, "Tanh")
        self.assertIsInstance(new_net.activation, Tanh)
        self.assertIsInstance(new_net.activation2, Sigmoid)

    def test_replace_activation_by_torch_module_sigmoid_to_golu(self):
        net = SimpleNet()
        self.assertIsInstance(net.activation2, Sigmoid)
        new_net = replace_activation_by_torch_module(net, Sigmoid, "GoLU")
        self.assertIsInstance(new_net.activation2, GoLU)
        self.assertIsInstance(new_net.activation, ReLU)

    def test_replace_activation_by_torch_module_nested(self):
        class NestedNet(nn.Module):
            def __init__(self):
                super(NestedNet, self).__init__()
                self.layer1 = SimpleNet()
                self.layer2 = ReLU()

            def forward(self, x):
                x = self.layer1(x)
                x = self.layer2(x)
                return x

        net = NestedNet()
        self.assertIsInstance(net.layer1.activation, ReLU)
        self.assertIsInstance(net.layer2, ReLU)
        new_net = replace_activation_by_torch_module(net, ReLU, "ELU")
        self.assertIsInstance(new_net.layer1.activation, ELU)
        self.assertIsInstance(new_net.layer2, ELU)

    def test_replace_activation_by_name(self):
        net = SimpleNet()
        self.assertIsInstance(net.activation2, Sigmoid)
        new_net = replace_activation_by_name(net, "activation2", "Mish")
        self.assertIsInstance(new_net.activation2, Mish)
        self.assertIsInstance(new_net.activation, ReLU)

    def test_replace_activation_by_name_nested(self):
        class NestedNet(nn.Module):
            def __init__(self):
                super(NestedNet, self).__init__()
                self.layer1 = SimpleNet()
                self.activation = ReLU()

            def forward(self, x):
                x = self.layer1(x)
                x = self.activation(x)
                return x

        net = NestedNet()
        self.assertIsInstance(net.layer1.activation2, Sigmoid)
        self.assertIsInstance(net.activation, ReLU)
        new_net = replace_activation_by_name(net, "activation", "ELU")
        self.assertIsInstance(new_net.activation, ELU)
        new_net = replace_activation_by_name(new_net, "activation2", "Mish")
        self.assertIsInstance(new_net.layer1.activation2, Mish)

    def test_update_golu_parameters(self):
        net = GoLUNet()
        net.activation.alpha = 1.0
        net.activation.beta = 1.0
        net.activation.gamma = 1.0
        new_net = update_golu_parameters(net, new_alpha=2.0, new_beta=3.0, new_gamma=4.0)
        self.assertEqual(new_net.activation.alpha, 2.0)
        self.assertEqual(new_net.activation.beta, 3.0)
        self.assertEqual(new_net.activation.gamma, 4.0)

    def test_update_golu_parameters_nested(self):
        class NestedGoLUNet(nn.Module):
            def __init__(self):
                super(NestedGoLUNet, self).__init__()
                self.layer1 = GoLUNet()
                self.activation = GoLU()

            def forward(self, x):
                x = self.layer1(x)
                x = self.activation(x)
                return x

        net = NestedGoLUNet()
        net.layer1.activation.alpha = 1.0
        net.layer1.activation.beta = 1.0
        net.layer1.activation.gamma = 1.0
        net.activation.alpha = 1.0
        net.activation.beta = 1.0
        net.activation.gamma = 1.0
        new_net = update_golu_parameters(net, new_alpha=2.0, new_beta=3.0, new_gamma=4.0)
        self.assertEqual(new_net.layer1.activation.alpha, 2.0)
        self.assertEqual(new_net.layer1.activation.beta, 3.0)
        self.assertEqual(new_net.layer1.activation.gamma, 4.0)
        self.assertEqual(new_net.activation.alpha, 2.0)
        self.assertEqual(new_net.activation.beta, 3.0)
        self.assertEqual(new_net.activation.gamma, 4.0)


if __name__ == "__main__":
    unittest.main()
