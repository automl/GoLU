"""
This file generates the visualizations for all the activations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import torchvision
from torchvision import transforms
from golu.activation_utils import get_activation_function
from tasks.utils.utils import create_directory, set_seed

# Declare some global variables
ACTIVATIONS = [
    'ReLU', 'LeakyReLU', 'PReLU', 'ELU', 'SELU', 'GELU', 'Swish', 'Mish', 'GoLU'
]
CIFAR_10_LABELS = {
    0: 'Airplane', 1: 'Automobile', 2: 'Bird', 3: 'Cat', 4: 'Deer',
    5: 'Dog', 6: 'Frog', 7: 'Horse', 8: 'Ship', 9: 'Truck'
}
BATCH_SIZE = 4
ROOT_PATH = './visualizations/theoretical_experiments/plots'

# Creates directory if it doesn't exits
create_directory(
    path=ROOT_PATH
)

# Set the seed
set_seed(
    seed=42, device_type='cpu'
)

# The following function converts the image to numpy
def imageshow(img):
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))

# The following function produces a kde plot
def plot_distribution(tensor, ax, color):
    tensor = tensor.detach().numpy().flatten()
    sns.kdeplot(tensor, ax=ax, lw=2, color=color)

# The following function computes the mean and variance of a tensor
def compute_mean_and_variance(tensor):
    return torch.mean(tensor), torch.var(tensor)


train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
)

# Get the 4 random images from CIFAR-10
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Define the dataframe to store the results
mean_df = pd.DataFrame(
    columns=[f'Image {idx + 1}' for idx in range(BATCH_SIZE)], index=ACTIVATIONS
)
variance_df = pd.DataFrame(
    columns=[f'Image {idx + 1}' for idx in range(BATCH_SIZE)], index=ACTIVATIONS
)

conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
bn = nn.BatchNorm2d(1)
conv = conv.to('cuda')
bn = bn.to('cuda')

for activation in ACTIVATIONS:
    print(f'Processing {activation} Activation Function')
    activation_function = (get_activation_function(activation=activation)).to('cuda')
    for idx in range(BATCH_SIZE):
        image = (images[idx].unsqueeze(0)).to('cuda')
        with torch.no_grad():
            conv_out = conv(image)
            bn_out = bn(conv_out)
            activated_out = activation_function(bn_out)

        mean, variance = compute_mean_and_variance(activated_out)
        mean_df.at[activation, f'Image {idx + 1}'] = mean.item()
        variance_df.at[activation, f'Image {idx + 1}'] = variance.item()

mean_df.to_csv(f'{ROOT_PATH}/mean.csv', index=True)
variance_df.to_csv(f'{ROOT_PATH}/variance.csv', index=True)

# Plot Variance
plt.figure(figsize=(10, 6))
plt.gca().set_facecolor('#ebf7ff')
plt.grid(True, color='white', linewidth=1.5)
plt.axhline(variance_df.loc['GoLU', 'Image 1'], color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axhline(variance_df.loc['GoLU', 'Image 2'], color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axhline(variance_df.loc['GoLU', 'Image 3'], color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.axhline(variance_df.loc['GoLU', 'Image 4'], color='#A9A9A9', linestyle='dashed', linewidth=1.5)
plt.plot(variance_df.index, variance_df['Image 1'], label=CIFAR_10_LABELS[int(labels[0])])
plt.plot(variance_df.index, variance_df['Image 2'], label=CIFAR_10_LABELS[int(labels[1])])
plt.plot(variance_df.index, variance_df['Image 3'], label=CIFAR_10_LABELS[int(labels[2])])
plt.plot(variance_df.index, variance_df['Image 4'], label=CIFAR_10_LABELS[int(labels[3])])
plt.title('Variance for Different Activations')
plt.xlabel('Activation Function')
plt.ylabel('Variance')
plt.xticks(rotation=45)
plt.legend()
plt.savefig(f"{ROOT_PATH}/variance_plot.png", bbox_inches='tight')


# New plot: Display 4 images in a 2x2 grid with their correct labels
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

for idx in range(BATCH_SIZE):
    ax = axs[idx // 2, idx % 2]
    ax.imshow(imageshow(images[idx]))
    ax.set_title(f'{CIFAR_10_LABELS[int(labels[idx])]}', fontsize=20)
    ax.axis('off')

# Adjust the layout and save the plot
plt.tight_layout()
plt.savefig(f"{ROOT_PATH}/selected_images.png", bbox_inches='tight')
