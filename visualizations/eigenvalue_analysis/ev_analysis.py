"""
This file does the eigenvalue analysis of the output and gradients of various activation functions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from golu.activation_utils import get_activation_function

# Define parameter grids
mean_values = torch.linspace(-2, 2, steps=5).tolist()     # Means: [-2, -1, 0, 1, 2]
std_values = torch.linspace(0.5, 2.5, steps=5).tolist()   # Std Devs: [0.5, 1.0, 1.5, 2.0, 2.5]

# Activation functions to consider
activations = ["GELU", "Swish", "Mish", "GoLU"]

torch.manual_seed(42)

# Initialize dictionaries to store max eigenvalues
max_eigenvalues = {act: np.zeros((len(mean_values), len(std_values))) for act in activations}
max_eigenvalues_grad = {act: np.zeros((len(mean_values), len(std_values))) for act in activations}

# Process Gaussian Distribution for varying means and std devs
for i, mean in enumerate(mean_values):
    for j, std in enumerate(std_values):
        # Generate base tensor from Gaussian distribution
        base_tensor = torch.normal(mean=mean, std=std, size=(1000, 1000), device='cuda')
        for act_name in activations:
            print(f"Mean - {mean:.1f} | Std Dev - {std:.1f} | Activation - {act_name}")
            activation_fn = get_activation_function(act_name).to('cuda')
            # Create a new tensor for each activation with requires_grad=True
            tensor = base_tensor.clone().detach().requires_grad_(True)
            # Apply activation function
            transformed_tensor = activation_fn(tensor)
            # Compute eigenvalues
            transformed_matrix = transformed_tensor.detach().cpu().numpy()
            eigenvalues = np.linalg.eigvals(transformed_matrix)
            # Get the maximum absolute eigenvalue
            max_eigenvalue = np.max(np.abs(eigenvalues))
            # Store the max eigenvalue
            max_eigenvalues[act_name][i, j] = max_eigenvalue

            # Compute gradients
            total_output = transformed_tensor.sum()
            total_output.backward()
            gradients = tensor.grad
            gradients_matrix = gradients.detach().cpu().numpy()
            grad_eigenvalues = np.linalg.eigvals(gradients_matrix)
            # Max absolute gradient (eigenvalue of the Jacobian)
            max_grad_eigenvalue = np.max(np.abs(grad_eigenvalues))
            # Store the max gradient eigenvalue
            max_eigenvalues_grad[act_name][i, j] = max_grad_eigenvalue

# Plotting heatmaps (same as before)
import matplotlib.pyplot as plt

# Compute global min and max for consistent color scaling
global_min = min([max_eigenvalues[act_name].min() for act_name in activations])
global_max = max([max_eigenvalues[act_name].max() for act_name in activations])

global_min_grad = min([max_eigenvalues_grad[act_name].min() for act_name in activations])
global_max_grad = max([max_eigenvalues_grad[act_name].max() for act_name in activations])

fig, axes = plt.subplots(nrows=2, ncols=len(activations), figsize=(20, 8))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# First row: Max eigenvalues of the transformed tensor
for idx, act_name in enumerate(activations):
    ax = axes[0, idx]
    # Get the max eigenvalue matrix
    max_eigen_matrix = max_eigenvalues[act_name]
    # Plot heatmap with consistent color scale
    im = ax.imshow(
        max_eigen_matrix,
        origin='lower',
        aspect='auto',
        cmap='coolwarm',
        extent=[std_values[0], std_values[-1], mean_values[0], mean_values[-1]],
        vmin=global_min,  # Set global min for color scale
        vmax=global_max   # Set global max for color scale
    )
    # Set title
    act_name_display = "GoLU" if act_name == "GoLU" else act_name
    ax.set_title(f"{act_name_display} - Output Max Eigenvalue", fontsize=12)
    # Set axis labels
    ax.set_xlabel("Std Dev", fontsize=10)
    ax.set_ylabel("Mean", fontsize=10)
    # Set ticks
    ax.set_xticks(std_values)
    ax.set_yticks(mean_values)
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Output Max Eigenvalue', rotation=270, labelpad=15)

    # Annotate the cells with max eigenvalue up to 2 decimal places
    x_start, x_end = std_values[0], std_values[-1]
    y_start, y_end = mean_values[0], mean_values[-1]
    num_x = len(std_values)
    num_y = len(mean_values)
    cell_width_x = (x_end - x_start) / num_x
    cell_width_y = (y_end - y_start) / num_y
    x_centers = x_start + (np.arange(num_x) + 0.5) * cell_width_x
    y_centers = y_start + (np.arange(num_y) + 0.5) * cell_width_y

    for m in range(num_y):
        for n in range(num_x):
            value = max_eigen_matrix[m, n]
            x = x_centers[n]
            y = y_centers[m]
            ax.text(x, y, f'{value:.2f}', ha='center', va='center', color='black', fontsize=8, fontweight='bold')

# Second row: Max gradients (eigenvalues of the Jacobian)
for idx, act_name in enumerate(activations):
    ax = axes[1, idx]
    # Get the max gradient eigenvalue matrix
    max_grad_matrix = max_eigenvalues_grad[act_name]
    # Plot heatmap with consistent color scale
    im = ax.imshow(
        max_grad_matrix,
        origin='lower',
        aspect='auto',
        cmap='coolwarm',
        extent=[std_values[0], std_values[-1], mean_values[0], mean_values[-1]],
        vmin=global_min_grad,  # Set global min for color scale
        vmax=global_max_grad   # Set global max for color scale
    )
    # Set title
    act_name_display = "GoLU" if act_name == "GoLU" else act_name
    ax.set_title(f"{act_name_display} - Gradient Max Eigenvalue", fontsize=12)
    # Set axis labels
    ax.set_xlabel("Std Dev", fontsize=10)
    ax.set_ylabel("Mean", fontsize=10)
    # Set ticks
    ax.set_xticks(std_values)
    ax.set_yticks(mean_values)
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Gradient Max Eigenvalue', rotation=270, labelpad=15)

    # Annotate the cells with max gradient eigenvalue up to 2 decimal places
    x_start, x_end = std_values[0], std_values[-1]
    y_start, y_end = mean_values[0], mean_values[-1]
    num_x = len(std_values)
    num_y = len(mean_values)
    cell_width_x = (x_end - x_start) / num_x
    cell_width_y = (y_end - y_start) / num_y
    x_centers = x_start + (np.arange(num_x) + 0.5) * cell_width_x
    y_centers = y_start + (np.arange(num_y) + 0.5) * cell_width_y

    for m in range(num_y):
        for n in range(num_x):
            value = max_grad_matrix[m, n]
            x = x_centers[n]
            y = y_centers[m]
            ax.text(x, y, f'{value:.2f}', ha='center', va='center', color='black', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('./visualizations/eigenvalue_analysis/max_eigenvalue_heatmaps.png')
