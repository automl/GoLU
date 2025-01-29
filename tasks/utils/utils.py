"""
This file contains all the utility functions required to train architectures
"""

import os
import logging
import datetime
import json
import random
import numpy as np
from typing import Any, List, Union, Tuple, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, LinearLR, PolynomialLR, \
    MultiStepLR


# Instantiating some global variables ----------------------------------------------------------------------------------
def str_to_bool(
    value: str
) -> bool:
    """
    This function converts boolean strings to boolean

    Args:
        value (str): True or False

    Returns:
        bool: True or False
    """
    if value in ['true', 'True', True]:
        return True
    else:
        return False


def set_seed(
    seed: int,
    device_type: str,
    ddp: bool = False,
    master_process: bool = False
) -> None:
    """
    The seed to set for reproducibility

    Args:
        seed (int): The seed value to set
        device_type (str): The device on which the model is trained
        ddp (bool, optional): Flag which tells if this is ddp training is done. Defaults to False.
        master_process (bool, optional): Flag which tells if this is the master process in ddp. Defaults to False.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_type == 'cuda':
        torch.cuda.manual_seed(seed)
        if torch.cuda.device_count() > 1:
            torch.cuda.manual_seed_all(seed)
    elif device_type == 'xpu':
        torch.xpu.manual_seed(seed)
        if torch.xpu.device_count() > 1:
            torch.xpu.manual_seed_all(seed)
            
    if (ddp and master_process) or not ddp:
        logging.info('Successfully set the seed!')


def create_directory(
    path: str,
    ddp: bool = False,
    master_process: bool = False
) -> None:
    """
    This function creates the required directories

    Args:
        path (str): The path to the directory to create
        ddp (bool, optional): Flag which tells if this is ddp training is done. Defaults to False.
        master_process (bool, optional): Flag which tells if this is the master process in ddp. Defaults to False.
    """
    os.makedirs(path, exist_ok=True)
    if (ddp and master_process) or not ddp:
        logging.info('Successfully created the directories!')


def num_parameters(
    model: Any,
) -> None:
    """
    This function computes the number of trainable parameters in the PyTorch model

    Args:
        model (Any): The instance of the model
    """
    total_parameters = 0
    trainable_parameters = 0
    for param in model.parameters():
        total_parameters += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    logging.info(f'# of Total Parameters - {total_parameters / 1000000:.4f} Million')
    logging.info(f'# of Trainable Parameters - {trainable_parameters / 1000000:.4f} Million')


def get_optimizer(
    model: Any,
    optimizer: str,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
    dampening: float,
    betas: List,
    eps: float,
    alpha: float,
    fused: Optional[bool]=None,
    model_parameters: Any=[],
    ddp: bool = False,
    master_process: bool = False
) -> Union[SGD, Adam, AdamW, RMSprop]:
    """
    This function creates the optimizer instance

    Args:
        model (Any): Instance of the model
        optimizer (str): Name of the optimizer
        learning_rate (float): Learning Rate to use for training
        weight_decay (float): Weight Decay to use for training
        momentum (float): Momentum to use in SGD or RMSProp
        dampening (float): Dampening factor in SGD
        betas (List): List of betas for Adam and AdamW
        eps (float): Epsilon factor for Adam, AdamW and RMSProp
        alpha (float): Rho factor for RMSProp
        fused (bool): Fused AdamW
        model_parameters (Any): The dictionary that holds the model parameters
        ddp (bool, optional): Flag which tells if this is ddp training is done. Defaults to False.
        master_process (bool, optional): Flag which tells if this is the master process in ddp. Defaults to False.

    Returns:
        Union[SGD, Adam, AdamW, RMSprop]: The optimizer to be used
    """
    if optimizer == 'sgdm':
        if (ddp and master_process) or not ddp:
            logging.info(
                f'Using SGD with lr={learning_rate}, momentum={momentum}, weight_decay={weight_decay}, dampening={dampening}'
            )
        if model_parameters:
            return SGD(
                model_parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay, dampening=dampening
            )
        else:
            return SGD(
                model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay, dampening=dampening
            )

    elif optimizer == 'sgdnm':
        if (ddp and master_process) or not ddp:
            logging.info(
                f'Using SGD with lr={learning_rate}, momentum={momentum}, weight_decay={weight_decay}, nestrov=True, dampening={dampening}'
            )
        if model_parameters:
            return SGD(
                model_parameters, lr=learning_rate, nesterov=True, momentum=momentum, weight_decay=weight_decay,
                dampening=dampening
            )
        else:
            return SGD(
                model.parameters(), lr=learning_rate, nesterov=True, momentum=momentum, weight_decay=weight_decay,
                dampening=dampening
            )

    elif optimizer == 'adam':
        if (ddp and master_process) or not ddp:
            logging.info(
                f'Using Adam with lr={learning_rate}, weight_decay={weight_decay}, betas={str(betas)}, eps={eps}'
            )
        if model_parameters:
            return Adam(
                model_parameters, lr=learning_rate, weight_decay=weight_decay, betas=betas, eps=eps
            )
        else:
                return Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas, eps=eps
            )

    elif optimizer == 'adam_w':
        if (ddp and master_process) or not ddp:
            logging.info(
                f'Using AdamW with lr={learning_rate}, weight_decay={weight_decay}, betas={str(betas)}, eps={eps}, fused={fused}'
            )
        if model_parameters:
            return AdamW(
                model_parameters, lr=learning_rate, weight_decay=weight_decay, betas=betas, eps=eps, fused=fused
            )
        else:
            return AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas, eps=eps, fused=fused
            )

    elif optimizer == 'rmsprop':
        if (ddp and master_process) or not ddp:
            logging.info(
                f'Using RMSProp with lr={learning_rate}, weight_decay={weight_decay}, momentum={momentum}, alpha={alpha}, eps={eps}'
            )
        if model_parameters:
            return RMSprop(
                model_parameters, lr=learning_rate, weight_decay=weight_decay, momentum=momentum, alpha=alpha, eps=eps
            )
        else:
            return RMSprop(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, alpha=alpha, eps=eps
            )

    else:
        if (ddp and master_process) or not ddp:
            logging.info(
                f'Using SGD with lr={learning_rate}'
            )
        if model_parameters:
            return SGD(model_parameters, lr=learning_rate)
        else:
            return SGD(model.parameters(), lr=learning_rate)


def get_scheduler(
    scheduler: str,
    optimizer: Any,
    num_epochs: int,
    eta_min: float,
    T_0: int,
    T_mult: int,
    step_size: int,
    gamma: float,
    milestones: list,
    power: float,
    lr_warmup_epochs: int,
    iters_per_epoch: int = 0,
    ddp: bool = False,
    master_process: bool = False
) -> Union[CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, MultiStepLR, PolynomialLR, None]:
    """
    This function creates the scheduler instance

    Args:
        scheduler (str): The name of the scheduler to use
        optimizer (Any): Instance of the configured optimizer
        num_epochs (int): Total number of epochs in training
        eta_min (float): Minimum LR. Used in CosineAnnealingLR
        T_0 (int): Number of epochs until which an LR cycle runs. Used in CosineAnnealingWarmRestarts
        T_mult (int): Multiplicative factor to increase or decrease the length of the cycle. Used in \
            CosineAnnealingWarmRestarts
        step_size (int): The step size to be used in StepLR
        gamma (float): The gamma value used in StepLR or MultistepLR
        milestones (list): The epoch milestones at which learning rate is updated. Used for MultiStepLR
        power (float): the power value used in PloynomialLR
        lr_warmup_epochs (int): The number of epochs until which learning rate warmup is done. Used in \
            ConsineAnnealingLR and PloynomialLR
        iters_per_epoch (int): Used if training is done per iteration
        ddp (bool, optional): Flag which tells if this is ddp training is done. Defaults to False.
        master_process (bool, optional): Flag which tells if this is the master process in ddp. Defaults to False.

    Returns:
        Union[CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, MultiStepLR, PolynomialLR, None]: \
            The scheduler to use
    """
    if scheduler == 'cosine':
        if iters_per_epoch == 0:
            if (ddp and master_process) or not ddp:
                logging.info(f'Using CosineAnnealingLR with T_max={num_epochs - lr_warmup_epochs} and eta_min={eta_min}')
            return CosineAnnealingLR(optimizer, T_max=num_epochs - lr_warmup_epochs, eta_min=eta_min)
        else:
            T_max = iters_per_epoch * (num_epochs - lr_warmup_epochs)
            if (ddp and master_process) or not ddp:
                logging.info(f'Using CosineAnnealingLR with T_max={T_max} and eta_min={eta_min}')
            return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    elif scheduler == 'warm':
        if (ddp and master_process) or not ddp:
            logging.info(f'Using CosineAnnealingWarmRestarts with T_0={T_0}, T_mult={T_mult} and eta_min={eta_min}')
        return CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)

    elif scheduler == 'step':
        if (ddp and master_process) or not ddp:
            logging.info(f'Using StepLR with step_size={step_size} and gamma={gamma}')
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler == 'multistep':
        if (ddp and master_process) or not ddp:
            logging.info(f'Using MultiStepLR with milestones={milestones} and gamma={gamma}')
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    elif scheduler == 'polynomial':
        if iters_per_epoch == 0:
            if (ddp and master_process) or not ddp:
                logging.info(f'Using PolynomialLR with total_iters={num_epochs - lr_warmup_epochs} and power={power}')
            return PolynomialLR(optimizer, total_iters=num_epochs - lr_warmup_epochs, power=power)
        else:
            total_iters = iters_per_epoch * (num_epochs - lr_warmup_epochs)
            if (ddp and master_process) or not ddp:
                logging.info(f'Using PolynomialLR with total_iters={total_iters} and power={power}')
            return PolynomialLR(optimizer, total_iters=total_iters, power=power)
    
    else:
        if (ddp and master_process) or not ddp:
            logging.info('No Learning Rate Scheduler used!')
        return None


def get_warmup_scheduler(
    lr_warmup: str,
    optimizer: Any,
    lr_warmup_epochs: int,
    lr_warmup_decay: float,
    linear_end_factor: float,
    ddp: bool = False,
    master_process: bool = False
) -> Union[LinearLR, None]:
    """
    This function creates the scheduler used for learning rate warmup

    Args:
        lr_warmup (str): The method used for learning rate warmup
        optimizer (Any): Instance of the configured optimizer
        lr_warmup_epochs (int): Number of epochs for which learning rate warmup is done
        lr_warmup_decay (float): The decay factor for learning rate warmup
        linear_end_factor (float): The end_factor for Linear warmup.
        ddp (bool, optional): Flag which tells if this is ddp training is done. Defaults to False.
        master_process (bool, optional): Flag which tells if this is the master process in ddp. Defaults to False.

    Returns:
        Union[LinearLR, None]: The learning rate warmup scheduler to use
    """
    if lr_warmup == 'linear':
        if (ddp and master_process) or not ddp:
            logging.info(
                f'Using LinearLR warmup scheduler with lr_warmup_epochs={lr_warmup_epochs}, lr_warmup_decay={lr_warmup_decay} and linear_end_factor={linear_end_factor}'
            )
        return LinearLR(
                optimizer, start_factor=lr_warmup_decay, end_factor=linear_end_factor, total_iters=lr_warmup_epochs
        )
    else:
        if (ddp and master_process) or not ddp:
            logging.info(
                f'No LR warmer is used!'
            )
        return None


def get_loss(
    training_loss: str,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
    ddp: bool = False,
    master_process: bool = False
) -> Union[torch.nn.CrossEntropyLoss]:
    """
    This functions creates the training loss

    Args:
        training_loss (str): Contains the name of the training loss to use
        label_smoothing (float): The label smoothing value for loss
        ignore_index (int): The index to ignore during training and inferencing
        ddp (bool, optional): Flag which tells if this is ddp training is done. Defaults to False.
        master_process (bool, optional): Flag which tells if this is the master process in ddp. Defaults to False.

    Returns:
        Union[torch.nn.CrossEntropyLoss]: The training loss
    """
    if training_loss == 'cross_entropy':
        if (ddp and master_process) or not ddp:
            logging.info(f'Using the Cross Entropy loss with label_smoothing={label_smoothing}')
        return torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=ignore_index)
    elif training_loss == 'bce_with_logits_loss':
        if (ddp and master_process) or not ddp:
            logging.info(f'Using the BCEWithLogits loss')
        return torch.nn.BCEWithLogitsLoss()
    elif training_loss == 'mse':
        if (ddp and master_process) or not ddp:
            logging.info(f'Using the MSE loss')
        return torch.nn.MSELoss()
    elif training_loss == 'nll':
        if (ddp and master_process) or not ddp:
            logging.info(f'Using the NLL loss')
        return torch.nn.NLLLoss()


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    Taken from torchvision.
    """
    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def save_results(
    results_dict: dict,
    path: str
) -> None:
    """
    This functions stores results at a particular path

    Args:
        results_dict (dict): Contains the performance metric
        path (str): The path where the results are stored
    """
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(results_dict, outfile)
    logging.info(f'Successfully stored file at {path}')


def setup_ddp(
    backend: str,
    device_type: str,
    seed: int,
    port: int,
    timeout: int
) -> Tuple[int, int, str, bool]:
    """
    This method sets up DDP

    Args:
        backend (str): The backend to use for DDP Training
        device_type (str): The type of device, cuda or xpu
        seed (int): Gives the seed number of the task
        port (int): The port using which DDP communication happens between GPUs
        timeout (int): Timeout in seconds

    Returns:
        Tuple[int, int, str, bool]: Rank, World Size, Device and if this is the master process
    """
    # Setup the commpunication address and port ------------------------------------------------------------------------
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port + seed)
    
    # Get the rank and world_size --------------------------------------------------------------------------------------
    rank = int(os.environ.get('RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', -1))
    
    if rank == -1:
        rank = int(os.environ.get('PMI_RANK', -1))
        world_size = int(os.environ.get('PMI_SIZE', -1))
    
    if rank == -1:
        rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', -1))
        world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', -1))

    if rank == -1:
        raise ValueError('Rank could not be determined.')
    
    logging.info(f'RANK - {rank} | WORLD_SIZE - {world_size}')
    
    # Initialize the process group -------------------------------------------------------------------------------------
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=timeout)
    )
    
    # Set the device ---------------------------------------------------------------------------------------------------
    device = f'{device_type}:{rank}'
    if device_type == 'cuda':
        torch.cuda.set_device(device)
    elif device_type == 'xpu':
        torch.xpu.set_device(device)
    
    # Set the master_process -------------------------------------------------------------------------------------------
    master_process = rank == 0 # This process will do logging, checkpointing etc.
    
    return rank, world_size, device, master_process


def load_json(
    path:str
):
    """
    This function loads JSON files

    Args:
        path (str): The path to the JSON file
    """
    with open(path, 'r') as file:
        data = json.load(file)
    return data


def load_instances(
    path: str,
    weights_only: bool=False
):
    """
    This function loads any saved instance from torch, like model, optimizer, scheduler or sampler

    Args:
        path (str): Path to load the instance from
        weights_only (bool): Flag to load only weights
    """
    return torch.load(path, weights_only=weights_only)


def save_instances(
    instance: Any,
    path: str
):
    """
    This function saves any instance from torch, like model, optimizer, scheduler or sampler

    Args:
        instance (Any): The object of the model, optimizer, scheduler or sampler
        path (str): The path where the instance is stored
    """
    if isinstance(instance, DistributedDataParallel):
        torch.save(instance.module, path)
    else:
        torch.save(instance, path)
    logging.info(f'Saved instance at - {path}')


def set_worker_sharing_strategy(
    worker_id: int
) -> None:
    """
    This is the worker_init_fn for multi-CPU data loader in pytorch

    Args:
        worker_id (int): The ID of the worker being used to load data in parallel
    """
    torch.multiprocessing.set_sharing_strategy('file_system')
