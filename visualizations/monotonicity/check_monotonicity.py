"""
This file shows the benefits of a sparser negative space of GoLU over other activation functions
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.patches as mpatches
import pandas as pd
from golu.activation_utils import get_activation_function

# Step 1: Load an image and normalize it
image_path = './visualizations/monotonicity/cat.jpg'  # Replace with your image path
image = Image.open(image_path).convert('RGB')  # Ensure RGB format
image_np = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, H, W)
image_tensor = image_tensor.to('cuda')

# Step 2: Define a 5x5 convolution and batch normalization
conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, padding=2).to('cuda')
batch_norm = nn.BatchNorm2d(num_features=1).to('cuda')

# Initialize weights for reproducibility
torch.manual_seed(5)
conv.weight.data.normal_(mean=0.0, std=0.02)
conv.bias.data.zero_()
batch_norm.weight.data.fill_(1)
batch_norm.bias.data.zero_()

# Pass through convolution and batch norm
with torch.no_grad():
    conv_output = conv(image_tensor)
    pre_activation = batch_norm(conv_output)

# Step 3: Get activation functions
relu = get_activation_function('ReLU').to('cuda')
gelu = get_activation_function('GELU').to('cuda')
swish = get_activation_function('Swish').to('cuda')
mish = get_activation_function('Mish').to('cuda')
golu = get_activation_function('GoLUCUDA').to('cuda')

# Apply activation functions
relu_output = relu(pre_activation)
gelu_output = gelu(pre_activation)
swish_output = swish(pre_activation)
mish_output = mish(pre_activation)
golu_output = golu(pre_activation)

# Step 4: Visualize activation maps
def plot_pre_activation(ax, activation_map, title):
    activation_map = activation_map.squeeze(0).cpu().numpy()  # Shape: (C, H, W)
    activation_map = activation_map[0]  # Since C=1, get the 2D map of shape (H, W)

    # Plot the activation map
    im = ax.imshow(activation_map, cmap='viridis')
    ax.set_title(title, fontsize=25)
    ax.axis('off')

    # Create the overlay
    overlay = np.zeros((*activation_map.shape, 4))  # Shape: (H, W, 4)

    # Define bins and colors
    bins = [
        (-2, -1),
        (-1, 0)
    ]
    colors = [
        [0, 0, 1, 1],  # Blue with alpha=0.5
        [1, 0, 0, 1],  # Red with alpha=0.5
    ]

    # For legends, store patches
    patches = []

    for bin_range, color in zip(bins, colors):
        mask = (activation_map > bin_range[0]) & (activation_map <= bin_range[1])
        overlay[mask] = color
        # For legend
        patch = mpatches.Patch(color=color[:3], label=f"{bin_range[0]} to {bin_range[1]}")
        patches.append(patch)

    # Overlay the masks
    ax.imshow(overlay)

    # Add legend
    ax.legend(handles=patches, loc='lower right', fontsize=16)

def plot_activation(ax, activation_map, title):
    activation_map = activation_map.squeeze(0).cpu().numpy()  # Shape: (C, H, W)
    activation_map = activation_map[0]  # Since C=1, get the 2D map of shape (H, W)
    im = ax.imshow(activation_map, cmap='viridis')
    ax.set_title(title, fontsize=25)
    ax.axis('off')

    # Create the overlay
    overlay = np.zeros((*activation_map.shape, 4))  # Shape: (H, W, 4)

    # Define bins and colors
    bins = [
        (-0.1, -0.05),
        (-0.05, 0)
    ]
    colors = [
        [0, 0, 1, 1],  # Blue with alpha=0.5
        [1, 0, 0, 1],  # Red with alpha=0.5
    ]

    # For legends, store patches
    patches = []

    for bin_range, color in zip(bins, colors):
        mask = (activation_map > bin_range[0]) & (activation_map <= bin_range[1])
        overlay[mask] = color
        # For legend
        patch = mpatches.Patch(color=color[:3], label=f"{bin_range[0]} to {bin_range[1]}")
        patches.append(patch)

    # Overlay the masks
    ax.imshow(overlay)

    # Add legend
    ax.legend(handles=patches, loc='lower right', fontsize=16)

# Create a figure with 1 row and 7 columns
fig, axes = plt.subplots(1, 7, figsize=(35, 6))

# Plot the original image
axes[0].imshow(image_np)
axes[0].set_title('Original Image', fontsize=25)
axes[0].axis('off')

# Plot the pre-activation with overlays and legends
plot_pre_activation(axes[1], pre_activation, 'Normalized Pre-Activation')

# Plot the activation maps with overlays
plot_activation(axes[2], relu_output, 'ReLU')
plot_activation(axes[3], gelu_output, 'GELU')
plot_activation(axes[4], swish_output, 'Swish')
plot_activation(axes[5], mish_output, 'Mish')
plot_activation(axes[6], golu_output, 'GoLU')

plt.tight_layout()
plt.savefig('./visualizations/monotonicity/plot_maps.jpg')

# Collect activation values
activation_values = {}

activation_values['Normalized Pre-Activation'] = pre_activation.squeeze(0).cpu().numpy()[0].flatten()
activation_values['GELU'] = gelu_output.squeeze(0).cpu().numpy()[0].flatten()
activation_values['Swish'] = swish_output.squeeze(0).cpu().numpy()[0].flatten()
activation_values['Mish'] = mish_output.squeeze(0).cpu().numpy()[0].flatten()
activation_values['GoLU'] = golu_output.squeeze(0).cpu().numpy()[0].flatten()

for key in activation_values:
    activation_values[key] = np.clip(activation_values[key], -2, 2)

# Create DataFrames for plotting
df_pre_activation = pd.DataFrame({
    'Activation Value': activation_values['Normalized Pre-Activation']
})

data_activations = []
for name in ['GELU', 'Swish', 'Mish', 'GoLU']:
    values = activation_values[name]
    data_activations.extend(zip(values, [name]*len(values)))

df_activations = pd.DataFrame(data_activations, columns=['Activation Value', 'Activation Function'])

# Create a figure with 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(20, 6))

# Plot the distribution of the normalized pre-activation map
axes[0].hist(
    df_pre_activation['Activation Value'],
    bins=100,
    density=True,
    histtype='step',  # No fill
    color='blue',
    alpha=0  # Hide the histogram bars
)

# Plot KDE
from scipy.stats import gaussian_kde
x_vals = np.linspace(df_pre_activation['Activation Value'].min(), df_pre_activation['Activation Value'].max(), 1000)
kde = gaussian_kde(df_pre_activation['Activation Value'])
axes[0].plot(x_vals, kde(x_vals), color='blue')

axes[0].set_title('Normalized Pre-Activation Distribution', fontsize=22)
axes[0].set_xlabel('Feature Value', fontsize=20)
axes[0].set_ylabel('Density', fontsize=20)
axes[0].set_ylim(0, 0.8)
axes[0].grid(True)

# Plot the distributions of the activations
for name, color in zip(['GELU', 'Swish', 'Mish', 'GoLU'], ['#8BD41E', '#3AA5D6', '#7921B8', '#FF5733']):
    subset = df_activations[df_activations['Activation Function'] == name]
    axes[1].hist(
        subset['Activation Value'],
        bins=100,
        density=True,
        histtype='step',
        color=color,
        alpha=0  # Hide the histogram bars
    )
    kde = gaussian_kde(subset['Activation Value'])
    axes[1].plot(x_vals, kde(x_vals), label=name, color=color)

focus_range = (-0.75, 1.75)  # Replace with your desired range
axes[1].set_xlim(focus_range)
axes[1].set_title('Activation Distributions', fontsize=22)
axes[1].set_xlabel('Activated Feature Value', fontsize=20)
axes[1].set_ylabel('Density', fontsize=20)
axes[1].set_ylim(0, 4)
axes[1].legend(fontsize=17)
axes[1].grid(True)

plt.tight_layout()
plt.savefig('./visualizations/monotonicity/activation_distributions.jpg')
plt.close()
