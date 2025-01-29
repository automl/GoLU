"""
This file consists of all the plots across various tasks
"""

import logging
import matplotlib.pyplot as plt


def save_classification_large_plots(
    model_name: str,
    activation: str,
    seed: int,
    opt_trace: dict,
    path: str
) -> None:
    """
    This function creates and saves the plots required for image classification

    Args:
        model_name (str): Name of the model to train
        activation (str): Name of the activation to use
        seed (int): Seed number
        opt_trace (dict): Optimization Trace
        path (str): Path to store results
    """
    fig = plt.figure(figsize=(10, 12))
    fig.suptitle(model_name + ' - ' + activation + ' - Seed ' + str(seed) + ' Plots')
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(opt_trace['epochs'], opt_trace['train_loss'])
    ax1.plot(opt_trace['epochs'], opt_trace['test_loss'])
    ax1.set_title('Loss Plot')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(['Training Loss', 'Test Loss'])

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(opt_trace['epochs'], opt_trace['train_top1_acc'])
    ax2.plot(opt_trace['epochs'], opt_trace['test_top1_acc'])
    ax2.set_title('Top-1 Accuracy Plot')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Top-1 Accuracy (in %age)')
    ax2.legend(['Training Top-1 Accuracy', 'Test Top-1 Accuracy'])

    fig.savefig(path)
    logging.info(f'Saved plots at {path}')


def save_segmentation_plots(
    model_name: str,
    activation: str,
    seed: int,
    opt_trace: dict,
    path: str
) -> None:
    """
    This function creates and saves the plots required for semantic segmentation

    Args:
        model_name (str): Name of the model to train
        activation (str): Name of the activation to use
        seed (int): Seed number
        opt_trace (dict): Optimization Trace
        path (str): Path to store results
    """
    fig = plt.figure(figsize=(10, 12))
    fig.suptitle(model_name + ' - ' + activation + ' - Seed ' + str(seed) + ' Plots')
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(opt_trace['epochs'], opt_trace['train_loss'])
    ax1.plot(opt_trace['epochs'], opt_trace['test_loss'])
    ax1.set_title('Loss Plot')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(['Training Loss', 'Test Loss'])

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(opt_trace['epochs'], opt_trace['train_miou'])
    ax2.plot(opt_trace['epochs'], opt_trace['test_miou'])
    ax2.set_title('Mean Intersection over Union Plot')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('mIoU')
    ax2.legend(['Training IoU', 'Test IoU'])

    fig.savefig(path)
    logging.info(f'Saved plots at {path}')
    

def save_detection_plots(
    model_name: str,
    activation: str,
    seed: int,
    opt_trace: dict,
    path: str
) -> None:
    """
    This function creates and saves the plots required for object detection

    Args:
        model_name (str): Name of the model to train
        activation (str): Name of the activation to use
        seed (int): Seed number
        opt_trace (dict): Optimization Trace
        path (str): Path to store results
    """
    fig = plt.figure(figsize=(10, 12))
    fig.suptitle(model_name + ' - ' + activation + ' - Seed ' + str(seed) + ' Plots')
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(opt_trace['epochs'], opt_trace['train_loss'])
    ax1.set_title('Loss Plot')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(['Training Loss'])

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(opt_trace['epochs'], opt_trace['test_box_map'])
    ax2.set_title('Box MAP Plot')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Box MAP')
    ax2.legend(['Test Box MAP'])

    fig.savefig(path)
    logging.info(f'Saved plots at {path}')


def save_instance_segmentation_plots(
    model_name: str,
    activation: str,
    seed: int,
    opt_trace: dict,
    path: str
) -> None:
    """
    This function creates and saves the plots required for instance segmentation

    Args:
        model_name (str): Name of the model to train
        activation (str): Name of the activation to use
        seed (int): Seed number
        opt_trace (dict): Optimization Trace
        path (str): Path to store results
    """
    fig = plt.figure(figsize=(10, 12))
    fig.suptitle(model_name + ' - ' + activation + ' - Seed ' + str(seed) + ' Plots')
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(opt_trace['epochs'], opt_trace['train_loss'])
    ax1.set_title('Loss Plot')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(['Training Loss', 'Test Loss'])

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(opt_trace['epochs'], opt_trace['test_box_map'])
    ax2.plot(opt_trace['epochs'], opt_trace['test_mask_map'])
    ax2.set_title('Box MAP Plot')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Box MAP')
    ax2.legend(['Test Box MAP', 'Test Mask MAP'])

    fig.savefig(path)
    logging.info(f'Saved plots at {path}')


def save_diffusion_plots(
    model_name: str,
    activation: str,
    seed: int,
    opt_trace: dict,
    path: str
) -> None:
    """
    This function creates and saves the plots required for diffusion

    Args:
        model_name (str): Name of the model to train
        activation (str): Name of the activation to use
        seed (int): Seed number
        opt_trace (dict): Optimization Trace
        path (str): Path to store results
    """
    fig = plt.figure(figsize=(10, 12))
    fig.suptitle(model_name + ' - ' + activation + ' - Seed ' + str(seed) + ' Plots')
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(opt_trace['epoch'], opt_trace['train_loss'])
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(['Training Loss'])

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(opt_trace['epoch'], opt_trace['test_loss'])
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend(['Test Loss'])

    fig.savefig(path)
    logging.info(f'Saved plots at {path}')


def save_gpt_plots(
    model_name: str,
    activation: str,
    seed: int,
    opt_trace: dict,
    path: str
) -> None:
    """
    This function creates and saves the plots required for GPT-2

    Args:
        model_name (str): Name of the model to train
        activation (str): Name of the activation to use
        seed (int): Seed number
        opt_trace (dict): Optimization Trace
        path (str): Path to store results
    """
    fig = plt.figure(figsize=(10, 12))
    fig.suptitle(model_name + ' - ' + activation + ' - Seed ' + str(seed) + ' Plots')
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(opt_trace['iteration'], opt_trace['train_loss'])
    ax1.plot(opt_trace['iteration'], opt_trace['test_loss'])
    ax1.set_title('Loss Plot')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.legend(['Training Loss', 'Test Loss'])

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(opt_trace['iteration'], opt_trace['train_acc'])
    ax2.plot(opt_trace['iteration'], opt_trace['test_acc'])
    ax2.set_title('Accuracy Plot')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Accuracy')
    ax2.legend(['Training Accuracy', 'Test Accuracy'])

    fig.savefig(path)
    logging.info(f'Saved plots at {path}')

