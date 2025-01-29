"""
This file generates the loss landscapes with random scaled perturbations to the learned weights of ResNet-20 CIFAR-10
"""

import torch
import torch.nn as nn
import torchvision
import numpy as np
import plotly.graph_objects as go
import imageio
import os
import json
import pickle
from plotly.subplots import make_subplots
from datetime import datetime
import cv2

# Define a function to load the saved models
def load_model(model_path):
    model = torch.load(model_path)
    return model

# Function to compute the loss landscape for a given model
def compute_loss_landscape(model, testloader, criterion, alpha, beta, d1, d2):
    # Filter out the parameters belonging to the activation functions
    theta_0 = np.concatenate([p.data.cpu().numpy().flatten() for name, p in model.named_parameters() if 'activation' not in name.lower()])
    loss_values = np.zeros((len(alpha), len(beta)))

    with torch.no_grad():
        for i, a in enumerate(alpha):
            for j, b in enumerate(beta):
                theta = theta_0 + a * d1 + b * d2
                index = 0
                for name, p in model.named_parameters():
                    if 'activation' in name.lower():
                        continue
                    size = np.prod(p.size())
                    p.data = torch.tensor(theta[index:index+size].reshape(p.size()), dtype=torch.float32).to('cuda')
                    index += size
                total_loss = 0.0
                for data in testloader:
                    inputs, labels = data
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                loss_values[i, j] = total_loss / len(testloader)

    return loss_values

# Paths for saved d1, d2, and loss landscapes
d_path = './visualizations/loss_landscapes/directions.pkl'
loss_landscapes_path = './visualizations/loss_landscapes/loss_landscapes.json'

# Define paths to the saved models
model_paths = {
    'ReLU': './results/cifar_10/resnet20/ReLU/model_1.pth',
    'GELU': './results/cifar_10/resnet20/GELU/model_1.pth',
    'Swish': './results/cifar_10/resnet20/Swish/model_1.pth',
    'GoLU': './results/cifar_10/resnet20/GoLUCUDA/model_1.pth'
}

if not os.path.exists(loss_landscapes_path):
    testset = torchvision.datasets.CIFAR10(root='./data/cifar_10', train=False, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)
    criterion = nn.CrossEntropyLoss().to('cuda')

# Generate or load random directions for projection
if os.path.exists(d_path):
    with open(d_path, 'rb') as f:
        d1, d2 = pickle.load(f)
    print('Loaded the directions')
else:
    np.random.seed(0)
    dummy_model = (load_model(model_paths['ReLU'])).to('cuda')
    theta_0 = np.concatenate([p.data.cpu().numpy().flatten() for name, p in dummy_model.named_parameters() if 'activation' not in name.lower()])
    d1 = np.random.randn(theta_0.size)
    d2 = np.random.randn(theta_0.size)
    d1 /= np.linalg.norm(d1)
    d2 /= np.linalg.norm(d2)
    with open(d_path, 'wb') as f:
        pickle.dump((d1, d2), f)

# Create grid
alpha = np.linspace(-1, 1, 50)
beta = np.linspace(-1, 1, 50)

# Load existing loss landscapes if available
if os.path.exists(loss_landscapes_path):
    with open(loss_landscapes_path, 'r') as f:
        loss_landscapes = json.load(f)
    print('Loaded the loss landscapes')
else:
    loss_landscapes = {}

# Compute loss landscapes for each activation function
for activation_fn, model_path in model_paths.items():
    if activation_fn not in loss_landscapes:
        print(f'{datetime.now()} - Starting Activation - {activation_fn}')
        model = (load_model(model_path)).to('cuda')
        loss_landscapes[activation_fn] = compute_loss_landscape(model, testloader, criterion, alpha, beta, d1, d2).tolist()  # Convert to list for JSON serialization
        print(f'{datetime.now()} - Completed Activation - {activation_fn}')
        with open(loss_landscapes_path, 'w') as f:
            json.dump(loss_landscapes, f)

# Convert lists back to numpy arrays for plotting
for activation_fn in loss_landscapes:
    loss_landscapes[activation_fn] = np.array(loss_landscapes[activation_fn])

# Create a subplot with individual 3D plots for each activation function
fig = make_subplots(
    rows=1, cols=4,
    specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
    column_widths=[0.2, 0.2, 0.2, 0.2],  # Adjust widths to reduce extra space
    subplot_titles=list(model_paths.keys())
)

activations = ['ReLU', 'GELU', 'Swish', 'GoLU']

loss_landscapes = {key: loss_landscapes[key] for key in activations if key in loss_landscapes}

# Add each activation function's loss landscape to the subplot
for idx, (activation_fn, loss_values) in enumerate(loss_landscapes.items()):
    A, B = np.meshgrid(alpha, beta)
    fig.add_trace(go.Surface(z=loss_values, x=A, y=B, colorscale='Inferno', showscale=(idx == 3),
                             contours=dict(
                                 x=dict(show=True, start=-1, end=1, size=0.1),
                                 y=dict(show=True, start=-1, end=1, size=0.1),
                                 z=dict(show=True, start=np.min(loss_values), end=np.max(loss_values), size=0.1)
                             )), row=1, col=idx+1)

# Update layout with axis labels for 3D plots
for i in range(1, 5):
    scene_id = f'scene{i}' if i > 1 else 'scene'
    fig.update_layout({scene_id: dict(
        xaxis_title='α',
        yaxis_title='β',
        zaxis_title='Loss'
    )})

# Update the font size of plot titles
counter = 0
for annotation in fig['layout']['annotations']:
    if counter < 4:
        annotation['font'] = dict(size=50)
    counter += 1

fig.update_layout(height=600, width=2400)

# Save the plot as an image
fig.write_image('./visualizations/loss_landscapes/loss_landscape_plot.png')

# Show the interactive plot
fig.show()

# Create directory for frames if it does not exist
frame_dir = './visualizations/loss_landscapes/frames'
os.makedirs(frame_dir, exist_ok=True)

# Function to update the camera view for a given subplot
def update_camera(fig, row, col, angle):
    camera = dict(
        eye=dict(x=np.cos(np.radians(angle)), y=np.sin(np.radians(angle)), z=1.25)
    )
    scene_id = f'scene{col}' if col > 1 else 'scene'
    fig.update_layout({scene_id: dict(camera=camera)})

# Generate video frames and save them individually
frames = []
for angle in range(0, 360, 5):
    for col in range(1, 5):  # Iterate over the 4 subplots
        update_camera(fig, 1, col, angle)
    frame_path = os.path.join(frame_dir, f'frame_{angle}.png')
    fig.write_image(frame_path)
    frame = imageio.imread(frame_path)
    frames.append(frame)

# Save the GIF
gif_path = './visualizations/loss_landscapes/loss_landscape_rotation.gif'
imageio.mimsave(gif_path, frames, fps=5)

# Save the MP4 video
mp4_path = './visualizations/loss_landscapes/loss_landscape_rotation.mp4'
frame_size = (frames[0].shape[1], frames[0].shape[0])
out = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, frame_size)

for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

out.release()
