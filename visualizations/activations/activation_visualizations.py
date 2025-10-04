"""
This file generates the visualizations for all the activations.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from golu.activation_utils import get_activation_function
from tasks.utils.utils import create_directory

# Declare some global variables
ACTIVATIONS = [
    'Sigmoid', 'Tanh', 'Softplus', 'Softsign',  # Primitive Activations 
    'ReLU', 'LeakyReLU', 'PReLU', 'ELU', 'SELU', 'CELU',  # Step Activations
    'GELU', 'Swish', 'Mish', 'GoLU'  # Gated Activations
]
ROOT_PATH = './visualizations/activations/plots'

# Creates directory if it doesn't exist
create_directory(
    path=ROOT_PATH
)

def plot_activation_and_gradient(
    x: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    activation: str
) -> None:
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.grid(True, color='#afbab2', linewidth=1.5)
    plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
    plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
    plt.plot(x, y, color='teal')
    plt.title(f'{activation} - Forward Pass')
    plt.xlabel('Input Neuron Value')
    plt.ylabel('Output Neuron Value')
    
    plt.subplot(1, 2, 2)
    plt.grid(True, color='#afbab2', linewidth=1.5)
    plt.axhline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
    plt.axvline(0, color='#A9A9A9', linestyle='dashed', linewidth=1.5)
    plt.plot(x, g, color='red')
    plt.title(f'{activation} - Backward Pass')
    plt.legend()
    plt.xlabel('Input Neuron Value')
    plt.ylabel('Derivative Value')
    
    plt.savefig(f'{ROOT_PATH}/{activation}.png')
    plt.close()

for activation in ACTIVATIONS:
    print(f'Running Activation - {activation}')
    activation_function = get_activation_function(activation=activation).to('cuda')
    if activation == "GoLU":
        activation = "GoLU"
    x = torch.linspace(-5, 5, 100, device='cuda', requires_grad=True)
    y = activation_function(x)
    # Compute first derivative
    g = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=False)[0]
    # Convert tensors to numpy arrays for plotting
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    g_np = g.detach().cpu().numpy()
    # Plot the activation function and its first derivative
    plot_activation_and_gradient(x=x_np, y=y_np, g=g_np, activation=activation)
