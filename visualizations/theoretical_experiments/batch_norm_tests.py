"""
This file checks the properties of batch norm
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import torchvision
from torchvision import transforms
from tasks.utils.utils import create_directory, set_seed

# Declare some global variables
CIFAR_10_LABELS = {
    0: 'Airplane', 1: 'Automobile', 2: 'Bird', 3: 'Cat', 4: 'Deer', 5: 'Dog', 6: 'Frog', 7: 'Horse', 8: 'Ship', 9: 'Truck'
}
BATCH_SIZE = 3
ROOT_PATH = './visualizations/theoretical_experiments/plots'

# Creates directory if it doesn't exist
create_directory(path=ROOT_PATH)

# Set the seed
set_seed(seed=42, device_type='cpu')

# Function to convert the image to numpy
def imageshow(img):
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))

# Function to produce a KDE plot
def plot_distribution(tensor, ax, color):
    tensor = tensor.detach().numpy().flatten()
    sns.kdeplot(tensor, ax=ax, lw=2, color=color)

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
)

# Get 4 random images from CIFAR-10
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Define the layers
conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
bn = nn.BatchNorm2d(1)
conv = conv.to('cuda')
bn = bn.to('cuda')

# Create the plot with 4 rows and 3 columns
fig, axs = plt.subplots(BATCH_SIZE, 3, figsize=(15, 12))

for idx in range(BATCH_SIZE):
    image = (images[idx].unsqueeze(0)).to('cuda')
    with torch.no_grad():
        conv_out = conv(image)
        bn_out = bn(conv_out)
    
    # Show the image
    axs[idx, 0].imshow(imageshow(image.squeeze(0).cpu()))
    axs[idx, 0].set_title(f'Image {idx + 1} - {CIFAR_10_LABELS[int(labels[idx])]}', fontsize=20)
    axs[idx, 0].axis('off')
    
    # Plot the input probability distribution to BatchNorm
    plot_distribution(conv_out.cpu(), axs[idx, 1], color='orange')
    if idx == 0:
        axs[idx, 1].set_title('Input Probability Distribution', fontsize=20)
    
    # Plot the output probability distribution from BatchNorm
    plot_distribution(bn_out.cpu(), axs[idx, 2], color='teal')
    if idx == 0:
        axs[idx, 2].set_title('Output Probability Distribution', fontsize=20)

plt.subplots_adjust(wspace=0.0)

# Adjust the layout and save the plot
plt.tight_layout()
plt.savefig(f"{ROOT_PATH}/batchnorm_prob_distributions.png", bbox_inches='tight')
