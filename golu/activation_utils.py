"""
The following file contains utils for calling different activations
"""

from typing import Union
from torch.nn import Sigmoid, Tanh, ReLU, Softplus, Softsign, LeakyReLU, PReLU, ELU, SELU, CELU, GELU, SiLU, Mish
from golu.golu_cuda_activation import GoLUCUDA


def get_activation_function(
    activation: str = ''
) -> Union[
        Sigmoid, Tanh, ReLU, Softplus, Softsign, LeakyReLU, PReLU, ELU, SELU, CELU, GELU, SiLU, Mish, GoLUCUDA
    ]:
    """
    This is a helper function that helps to get any activation function from torch.
    
    Example Usage - 

        1. To get ReLU
            
            activation_function = get_activation_function(activation='ReLU')

    Args:
        activation (str, optional): The name of the activation to use. Defaults to ''. Choices include 'Sigmoid', \
            'Tanh', 'ReLU', 'Softplus', 'LeakyReLU', 'PReLU', 'ELU', 'SELU', 'CELU', 'GELU', 'GELU_tanh', \
                'Swish', 'Mish', 'GoLUCUDA'

    Raises:
        ValueError: If the activation string doesn't match any of the known activations, it raises a ValueError.

    Returns:
        Union[
            Sigmoid, Tanh, ReLU, Softplus, LeakyReLU, PReLU, ELU, SELU, CELU, GELU, SiLU, Mish, GoLUCUDA\
                ]: Get activation
    """
    
    # S-Shaped Functions -----------------------
    
    if activation == 'Sigmoid':
        return Sigmoid()
    
    elif activation == 'Tanh':
        return Tanh()
    
    elif activation == 'Softsign':
        return Softsign()
    
    # Step-wise activations --------------------
    
    elif activation == 'Softplus':
        return Softplus()
    
    elif activation == 'ReLU':
        return ReLU()
    
    elif activation == 'LeakyReLU':
        return LeakyReLU()
    
    elif activation == 'PReLU':
        return PReLU()
    
    elif activation == 'ELU':
        return ELU()
    
    elif activation == 'SELU':
        return SELU()
    
    elif activation == 'CELU':
        return CELU()
    
    # Self-gated activations -------------------
    
    elif activation == 'GELU':
        return GELU()

    elif activation == 'GELU_tanh':
        return GELU(approximate='tanh')
    
    elif activation == 'Swish':
        return SiLU()
    
    elif activation == 'Mish':
        return Mish()
    
    elif activation == 'GoLUCUDA':
        return GoLUCUDA()
    
    else:
        raise ValueError(f"The activation named {activation} doesn't exists!")


def replace_activation_by_torch_module(
    module, old_activation, new_activation
):
    for name, child in module.named_children():
        if isinstance(child, old_activation):
            setattr(module, name, get_activation_function(activation=new_activation))
        else:
            # Recurse into child modules
            replace_activation_by_torch_module(child, old_activation, new_activation)
    return module


def replace_activation_by_name(
    module, attr_name, new_activation
):
    for name, child in module.named_children():
        if name == attr_name:
            setattr(module, name, get_activation_function(activation=new_activation))
        else:
            replace_activation_by_name(child, attr_name, new_activation)
    return module


def update_golu_parameters(
    module, new_alpha=1.0, new_beta=1.0, new_gamma=1.0
):
    for name, child in module.named_children():
        if isinstance(child, GoLUCUDA):
            child.alpha = new_alpha
            child.beta = new_beta
            child.gamma = new_gamma
        else:
            # Recurse into child modules
            update_golu_parameters(child, new_alpha, new_beta, new_gamma)
    return module
