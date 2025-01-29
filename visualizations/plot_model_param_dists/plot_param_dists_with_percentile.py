"""
This file generates the plots for the parameter distributions across various models trained on d/f datasets.
"""

import os
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torchvision.models.segmentation.deeplabv3 import _deeplabv3_resnet
from torchvision.models import resnet50

from tasks.gpt.model import GPTConfig, GPT, LayerNorm
from golu.golu_cuda_activation import GoLUCUDA
from golu.activation_utils import replace_activation_by_torch_module

# Activation functions
activations = [
    'GELU',
    'Swish',
    'Mish',
    'GoLUCUDA'
]
display_activations = {'GoLUCUDA': 'GoLU'}  # Mapping for display names
colors = {
    'GELU': '#77c944',
    'Swish': '#ba45d1',
    'Mish': '#28d1a4',
    'GoLUCUDA': '#e61e32',
}

# Model paths
model_dirs = {
    'ResNet50 - ImageNet-1K': './results/imagenet_1k/resnet50',
    'ViT-B/32 - ImageNet-1K': './results/imagenet_1k/vit_b_32',
    'DeepLabV3-RN50 - MS-COCO': './results/coco/deeplabv3_resnet50_0.01',
    'GPT2-S - Open Web Text': './results/open_web_text/gpt2s'
}

# We know these values already from past runs of the code. Lower limit is max of 1 percentile across 4 activations
# while upper limit is min of 99 percentile across 4 activations
param_clip_ranges = {
    'ResNet50 - ImageNet-1K': (-0.04006460253, 0.04600124154),
    'ViT-B/32 - ImageNet-1K': (-0.06291432269, 0.06303090453),
    'DeepLabV3-RN50 - MS-COCO': (-0.02460635519, 0.02569680847),
    'GPT2-S - Open Web Text': (-0.04864466935, 0.04995486204)
}

# Prepare subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
model_axes = [
    ('ResNet50 - ImageNet-1K', axs[0, 0]),
    ('ViT-B/32 - ImageNet-1K', axs[0, 1]),
    ('DeepLabV3-RN50 - MS-COCO', axs[1, 0]),
    ('GPT2-S - Open Web Text', axs[1, 1])
]

histogram_bins = {
    'ResNet50 - ImageNet-1K': 2000,
    'ViT-B/32 - ImageNet-1K': 2000,
    'DeepLabV3-RN50 - MS-COCO': 2000,
    'GPT2-S - Open Web Text': 20000
}

# Key to access model parameters in the checkpoint
param_key = 'model'

seeds = [1, 2, 3]

param_count_dict = {
    'Architecture': [],
    'Activation': [],
    'Percentile1': [],
    'Percentile99': [],
    'NumberParams': [],
    'Variance': []
}

# Edit these variables as per requirement
# We generate the plot as follows,
#   1. We set clip_range and clip_on_percentile to False. This gives us the 1 percentile and 99 percentile
#   2. Next, for every model and 4 activations i.e. GELU, Swish, Mish and GoLU, we get the max of 1 percentile and min 
#      of 99 percentile and fix this range in which we keep parameters only in this range - check dict param_clip_ranges
#      for more details
#   3. Finally, we decide if we need to focus on a specific range
clip_range = True
clip_on_percentile = False
focus_on_range = True

for model_name, ax in model_axes:
    print(f"Processing model: {model_name}")
    model_dir = model_dirs[model_name]
    
    for activation in activations:
        
        param_values = []
        
        for seed in seeds:
            
            checkpoint_path = os.path.join(model_dir, activation, f'checkpoint_{seed}.pth')
            
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint not found: {checkpoint_path}")
                continue
            
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cuda')
            
            if model_name == "ResNet50 - ImageNet-1K":
                model = torchvision.models.get_model("resnet50", weights=None, num_classes=1000)
                model = replace_activation_by_torch_module(model, nn.ReLU, activation)
                model.to('cuda')
                model.load_state_dict(checkpoint["model"])
                
            elif model_name == "ViT-B/32 - ImageNet-1K":
                model = torchvision.models.get_model("vit_b_32", weights=None, num_classes=1000)
                model = replace_activation_by_torch_module(model, nn.GELU, activation)
                model.to('cuda')
                model.load_state_dict(checkpoint["model"])
                
            elif model_name == "DeepLabV3-RN50 - MS-COCO":
                backbone = resnet50(weights=None, replace_stride_with_dilation=[False, True, True])
                backbone = replace_activation_by_torch_module(
                    module=backbone, old_activation=nn.ReLU, new_activation=activation
                )
                backbone = nn.SyncBatchNorm.convert_sync_batchnorm(backbone)
                model = _deeplabv3_resnet(backbone=backbone, num_classes=21, aux=True)    
                model = replace_activation_by_torch_module(
                    module=model, old_activation=nn.ReLU, new_activation=activation
                )
                model.to('cuda')
                model.load_state_dict(checkpoint["model"])
                
            elif model_name == "GPT2-S - Open Web Text":
                model_args = dict(
                    n_layer=12, n_head=12, n_embd=768, block_size=1024, bias=False,
                    vocab_size=None, dropout=0.0
                )
                checkpoint_model_args = checkpoint['model_args']
                for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                    model_args[k] = checkpoint_model_args[k]
                gptconf = GPTConfig(**model_args)
                model = GPT(gptconf)
                replace_activation_by_torch_module(model, nn.GELU, activation)
                state_dict = checkpoint['model']
                unwanted_prefix = '_orig_mod.'
                for k,v in list(state_dict.items()):
                    if k.startswith(unwanted_prefix):
                        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                model.to('cuda')
                model.load_state_dict(state_dict)
                
            for name, module in model.named_modules():
                if isinstance(
                    module, (nn.LayerNorm, LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.GELU, nn.SiLU, nn.Mish, GoLUCUDA)
                ) or name == "transformer.wte" or name == "transformer.wpe":  # We exclude embedding layers from GPT
                    continue
                else:
                    if hasattr(module, 'weight') and module.weight is not None:
                        param_values.append(module.weight.data.cpu().numpy().flatten())
                        if hasattr(module, 'bias') and module.bias is not None:
                            param_values.append(module.bias.data.cpu().numpy().flatten())
        
        param_values = np.concatenate(param_values)
        p1 = np.percentile(param_values, 1)
        p99 = np.percentile(param_values, 99)
        if clip_range:
            if clip_on_percentile:
                mask = (param_values >= p1) & (param_values <= p99)
            else:
                mask = (param_values >= param_clip_ranges[model_name][0]) & (param_values <= param_clip_ranges[model_name][1])
            param_values = param_values[mask]
        
        
        # Plot histogram with density normalization
        ax.hist(
            param_values,
            bins=histogram_bins[model_name],
            density=True,
            histtype='step',
            label=f'{display_activations.get(activation, activation)}',
            color=colors[activation],
        )
        if focus_on_range:
            focus_range = (-0.065, 0.065)  # Replace with your desired range
            ax.set_xlim(focus_range)
        
        param_count_dict['Architecture'].append(model_name)
        param_count_dict['Activation'].append(activation)
        param_count_dict['Percentile1'].append(p1)
        param_count_dict['Percentile99'].append(p99)
        param_count_dict['NumberParams'].append(len(param_values))
        param_count_dict['Variance'].append(np.var(param_values))
        
        # Free memory
        del checkpoint
        del model
        del param_values
        torch.cuda.empty_cache()

    # Set plot titles and labels
    ax.set_title(model_name, fontsize=18)
    ax.set_xlabel('Parameter Values', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.legend(fontsize=14)

# Adjust layout and display the plot
plt.tight_layout()
plt.savefig('./visualizations/plot_model_param_dists/param_dists.png')

df = pd.DataFrame(param_count_dict)
df.to_csv('./visualizations/plot_model_param_dists/param_count.csv', index=False)
