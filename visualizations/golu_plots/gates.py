"""
This file contains the gates for some of the self-gated activation functions
"""


import torch
import numpy as np


def gompertz(x: np.array, alpha=1, beta=1, gamma=1) -> np.array:
    return alpha * np.exp(-beta * np.exp(-gamma * x))

def gompertz_backward_x(x: np.array, alpha=1, beta=1, gamma=1) -> np.array:
    return alpha * beta * gamma * np.exp(-beta * np.exp(-gamma * x)) * np.exp(-gamma * x)


def golu_act(x: np.array, alpha=1, beta=1, gamma=1) -> np.array:
    return x * alpha * np.exp(-beta * np.exp(-gamma * x))

def golu_act_torch(x: torch.Tensor, alpha=1, beta=1, gamma=1) -> torch.Tensor:
    return x * alpha * torch.exp(-beta * torch.exp(-gamma * x))

def golu_backward_x(x: np.array, alpha=1, beta=1, gamma=1) -> np.array:
    return alpha * np.exp(-beta * np.exp(-gamma * x)) * (1 + x * beta * gamma * np.exp(-gamma * x))


def golu_backward_alpha(alpha: np.array, x=1, beta=1, gamma=1) -> np.array:
    return np.ones(alpha.shape) * x * np.exp(-beta * np.exp(-gamma * x))


def golu_backward_beta(beta: np.array, alpha=1, x=1, gamma=1) -> np.array:
    return -x * alpha * np.exp(-beta * np.exp(-gamma * x)) * np.exp(-gamma * x)


def golu_backward_gamma(gamma: np.array, alpha=1, beta=1, x=1) -> np.array:
    return x**2 * alpha * beta * np.exp(-beta * np.exp(-gamma * x)) * np.exp(-gamma * x) 


def gelu(x: np.array) -> np.array:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def swish(x: np.array) -> np.array:
    return x / (1 + np.exp(-x))


def relu(x: np.array) -> np.array:
    return np.where(x > 0, x, 0)


def leakyrelu(x: np.array) -> np.array:
    return np.where(x > 0, x, 0.01 * x)


def elu(x: np.array) -> np.array:
    return np.where(x > 0, x, 1.0 * (np.exp(x) - 1))

def mish(x: np.array) -> np.array:
    return x * np.tanh(np.log1p(np.exp(x)))


def gelu_backward_x(x: np.array) -> np.array:
    return (1 + np.exp(-1.702 * x))**-1 * (1 + 1.702 * x * np.exp(-1.702 * x) * (1 + np.exp(-1.702 * x))**-1)


def swish_backward_x(x: np.array) -> np.array:
    return (1 + np.exp(-x))**-1 * (1 + x * np.exp(-x) * (1 + np.exp(-x))**-1)


def relu_backward_x(x: np.array) -> np.array:
    return np.where(x > 0, 1, 0)


def leakyrelu_backward_x(x: np.array) -> np.array:
    return np.where(x > 0, 1, 0.01)


def elu_backward_x(x: np.array) -> np.array:
    return np.where(x > 0, 1, 0.1 * np.exp(x))


def mish_backward_x(x: np.array) -> np.array:
    softplus = np.log1p(np.exp(x))
    tanh_softplus = np.tanh(softplus)
    sigmoid_x = 1 / (1 + np.exp(-x))
    sech2_softplus = 1 - tanh_softplus**2
    return tanh_softplus + x * sech2_softplus * sigmoid_x


def gaussian_cdf(x: np.array) -> np.array:
    return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def sigmoid(x: np.array) -> np.array:
    return 1 / (1 + np.exp(-x))


def tanh(x: np.array) -> np.array:
    return np.tanh(x)

def tanhsoftplus(x: np.array) -> np.array:
    return np.tanh(np.log1p(np.exp(x)))


def relu_gate(x: np.array) -> np.array:
    return np.where(x > 0, 1, 0)
