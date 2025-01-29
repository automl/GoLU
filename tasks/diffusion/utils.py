"""
The following code has been taken from - https://github.com/KimRass/DDPM/blob/main/utils.py
"""

# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1
    # https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddpm.html

import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from datetime import timedelta
from time import time
from PIL import Image
from pathlib import Path
from collections import OrderedDict
import random
import numpy as np
import os
import re


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    return device


def get_grad_scaler(device):
    return torch.amp.GradScaler() if 'cuda' in device else None


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def create_dir(x):
    x = Path(x)
    if x.is_dir():
        x.mkdir(parents=True, exist_ok=True)
    else:
        x.parent.mkdir(parents=True, exist_ok=True)


def save_image(image, save_path):
    create_dir(save_path)
    _to_pil(image).save(str(save_path), quality=100)


def get_elapsed_time(start_time):
    return timedelta(seconds=round(time() - start_time))


def denorm(x, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    return TF.normalize(
        x, mean=-(np.array(mean) / np.array(std)), std=(1 / np.array(std)),
    )


def image_to_grid(image, n_cols):
    tensor = image.clone().detach().cpu()
    tensor = denorm(tensor)
    grid = make_grid(tensor, nrow=n_cols, padding=1, pad_value=1)
    grid.clamp_(0, 1)
    grid = TF.to_pil_image(grid)
    return grid


def modify_state_dict(state_dict, pattern=r"^module.|^_orig_mod."):
    new_state_dict = OrderedDict()
    for old_key, value in state_dict.items():
        new_key = re.sub(pattern=pattern, repl="", string=old_key)
        new_state_dict[new_key] = value
    return new_state_dict


def print_n_params(model):
    n_params = 0
    n_trainable_params = 0
    for p in model.parameters():
        n = p.numel()
        n_params += n
        if p.requires_grad:
            n_trainable_params += n
    print(f"[ # OF PARAMS: {n_params:,} ][ # OF TRAINABLE PARAMS: {n_trainable_params:,} ]")
