"""
This file contains the code to train small models on CIFAR-10.
"""

import ast
import os
import argparse
import logging
import datetime
import time
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

from tasks.small_poc.resnet import resnet20, resnet32, resnet44, resnet56, resnet110
from tasks.small_poc.wide_resnet import wideresnet_28_2
from tasks.small_poc.dense_net import densenet_40


number_of_epochs = {
    'resnet20': 164,
    'resnet32': 164,
    'resnet44': 164,
    'resnet56': 164,
    'resnet110': 164,
    'wideresnet_28_2': 200,
    'densenet_40': 300
}

batch_size = {
    'resnet20': 128,
    'resnet32': 128,
    'resnet44': 128,
    'resnet56': 128,
    'resnet110': 128,
    'wideresnet_28_2': 128,
    'densenet_40': 64
}

lr = {
    'resnet20': 0.1,
    'resnet32': 0.1,
    'resnet44': 0.1,
    'resnet56': 0.1,
    'resnet110': 0.01,
    'wideresnet_28_2': 0.1,
    'densenet_40': 0.1
}

weight_decay = {
    'resnet20': 0.0001,
    'resnet32': 0.0001,
    'resnet44': 0.0001,
    'resnet56': 0.0001,
    'resnet110': 0.0001,
    'wideresnet_28_2': 0.0005,
    'densenet_40': 0.0001
}

momentum = {
    'resnet20': 0.9,
    'resnet32': 0.9,
    'resnet44': 0.9,
    'resnet56': 0.9,
    'resnet110': 0.9,
    'wideresnet_28_2': 0.9,
    'densenet_40': 0.9
}

nesterov = {
    'resnet20': False,
    'resnet32': False,
    'resnet44': False,
    'resnet56': False,
    'resnet110': False,
    'wideresnet_28_2': True,
    'densenet_40': True
}

milestones = {
    'resnet20': [81, 122],
    'resnet32': [81, 122],
    'resnet44': [81, 122],
    'resnet56': [81, 122],
    'resnet110': [2, 81, 122],
    'wideresnet_28_2': [60, 120, 160],
    'densenet_40': [150, 225]
}

milestone_factors = {
    'resnet20': [0.1, 0.1],
    'resnet32': [0.1, 0.1],
    'resnet44': [0.1, 0.1],
    'resnet56': [0.1, 0.1],
    'resnet110': [10, 0.1, 0.1],
    'wideresnet_28_2': [0.2, 0.2, 0.2],
    'densenet_40': [0.1, 0.1]
}


def main(
        model_name: str,
        seed: int,
        activation: str,
        save_results_str: str,
        device: str,
):
        
    np.random.seed(seed)
    torch.manual_seed(seed)

    logging.info(f'Training Model - {model_name} | Activation - {activation} | Seed - {seed}')
    logging.info(f'Training starts at - {datetime.datetime.now()}')

    # Instantiating data augmentations --------------------------------------------------------
    aug_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    aug_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    logging.info(f'Created augmentations!')

    # Load the dataset ------------------------------------------------------------------------
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=aug_train, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=aug_test, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size[model_name], shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size[model_name], shuffle=False)
    logging.info(f'Created datasets and loaders!')

    # Instantiate the model -------------------------------------------------------------------
    model = eval(model_name)(
        num_classes=len(train_dataset.classes), activation=activation
    )
    model.to(device)
    logging.info(f'The model looks like - \n{model}')

    # Instantiating Training Criterion --------------------------------------------------------
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Instantiate Optimizer -------------------------------------------------------------------
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr[model_name],
        nesterov=nesterov[model_name],
        momentum=momentum[model_name],
        weight_decay=weight_decay[model_name]
    )
    logging.info(f'Created optimizer!')
        
    # Train the model -------------------------------------------------------------------------
    # Loop through the epochs

    epoch_train_loss = []
    epoch_train_acc = []
    epoch_test_loss = []
    epoch_test_acc = []

    train_time = 0
    inference_time = 0

    results = dict()
    
    num_epochs = number_of_epochs[model_name]
    milestone = milestones[model_name]
    milestone_factor = milestone_factors[model_name]
    
    for epoch in range(num_epochs):
        logging.info('#' * 80)
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}]')

        if epoch + 1 in milestone:
            for param_group in optimizer.param_groups:
                idx = milestone.index(epoch + 1)
                param_group['lr'] *= milestone_factor[idx]
                logging.info(f'Adjusted learning rate by {milestone_factor[idx]} at epoch {epoch + 1}')

        train_loss = 0
        train_acc = 0
        test_loss = 0
        test_acc = 0

        train_sum_loss = 0
        train_sum_acc = 0
        test_sum_loss = 0
        test_sum_acc = 0

        train_cnt_loss = 0
        train_cnt_acc = 0
        test_cnt_loss = 0
        test_cnt_acc = 0

        # Loop through the training data batches ----------------------------------------------
        model.train()
        train_time_begin = time.time()
        for _, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == labels) / len(labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate the average loss and accuracy for all batches run so far
            n = images.size(0)
            train_sum_loss += loss.item() * n
            train_cnt_loss += n
            train_loss = train_sum_loss / train_cnt_loss
            train_sum_acc += acc.item() * n
            train_cnt_acc += n
            train_acc = train_sum_acc / train_cnt_acc

        train_time += time.time() - train_time_begin

        epoch_train_loss.append(train_loss)
        epoch_train_acc.append(train_acc)
        logging.info('Train Loss: %.4f | Train Acc: %.4f', train_loss, train_acc * 100)

        # Compute the test stats --------------------------------------------------------------
        torch.cuda.empty_cache()
        model.eval()
        inference_time_begin = time.time()
        with torch.no_grad():
            for _, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                preds = torch.argmax(logits, dim=1)
                acc = torch.sum(preds == labels) / len(labels)

                # Calculate the average loss and accuracy for all batches run so far
                n = images.size(0)
                test_sum_loss += loss.item() * n
                test_cnt_loss += n
                test_loss = test_sum_loss / test_cnt_loss
                test_sum_acc += acc.item() * n
                test_cnt_acc += n
                test_acc = test_sum_acc / test_cnt_acc
        inference_time += time.time() - inference_time_begin
        
        epoch_test_loss.append(test_loss)
        epoch_test_acc.append(test_acc)
        logging.info('Test Loss: %.4f | Test Acc: %.4f', test_loss, test_acc * 100)
        logging.info(f'Training Time: {train_time / 60:.4f} Mins | Inference Time: {inference_time / 60:.4f} Mins')

    results['train_loss'] = epoch_train_loss[-1]
    results['train_acc'] = epoch_train_acc[-1] * 100
    results['test_loss'] = epoch_test_loss[-1]
    results['test_acc'] = epoch_test_acc[-1] * 100
    results['train_time'] = train_time / 60
    results['inference_time'] = inference_time / 60

    logging.info('Average Training Time / Epoch: %.2f Mins', train_time / (60 * num_epochs))

    # Plot the Losses and Accuracies for Training and Test ------------------------------
    results_save_dir = os.path.join(save_results_str, model_name, activation)

    if not os.path.exists(results_save_dir):
        os.makedirs(results_save_dir)

    x = range(1, num_epochs + 1)
    fig = plt.figure(figsize=(10, 12))
    fig.suptitle(model_name + ' - ' + activation + ' - Seed ' + str(seed) + ' Plots')
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x, epoch_train_loss)
    ax1.plot(x, epoch_test_loss)
    ax1.set_title('Avg Loss Plot')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(['Training Loss', 'Test Loss'])

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(x, [acc * 100 for acc in epoch_train_acc])
    ax2.plot(x, [acc * 100 for acc in epoch_test_acc])
    ax2.set_title('Avg Accuracy Plot')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (in %age)')
    ax2.legend(['Training Accuracy', 'Test Accuracy'])

    final_results_str = os.path.join(results_save_dir, f'plot_{seed}') + '.png'
    fig.savefig(final_results_str)
    logging.info('Saved plots in %s folder', results_save_dir)

    results_json_path = os.path.join(save_results_str, model_name,
                                        activation, f'results_{seed}.json')
    with open(results_json_path, "w", encoding="utf-8") as outfile:
        json.dump(results, outfile)
    logging.info('Saved results!')

    # Save the result trace from the training and testing set ---------------------------------
    trace_results = {
        'train_loss': epoch_train_loss,
        'train_acc': epoch_train_acc,
        'test_loss': epoch_test_loss,
        'test_acc': epoch_test_acc
    }
    trace_json_path = os.path.join(save_results_str, model_name,
                                    activation, f'opt_trace_{seed}.json')
    with open(trace_json_path, "w", encoding="utf-8") as outfile:
        json.dump(trace_results, outfile)
    logging.info('Saved trace results!')

    # Save the trained model ------------------------------------------------------------------
    model_save_path = os.path.join(save_results_str, model_name, activation, f'model_{seed}.pth')
    torch.save(model, model_save_path)
    logging.info('Saved the model!\n\n')

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':

    cmdline_parser = argparse.ArgumentParser('Train Models')
    cmdline_parser.add_argument('-m', '--model',
                                default='resnet20',
                                choices=[
                                    'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                    'wideresnet_28_2', 'densenet_40'],
                                help='Name of model to train',
                                type=str)
    cmdline_parser.add_argument('-s', '--seed',
                                default=0,
                                help='Seed to fix randomization',
                                type=int)
    cmdline_parser.add_argument('-a', '--activation',
                                default="GoLU",
                                help='Activation function',
                                type=str)
    cmdline_parser.add_argument('-r', '--results_path',
                                default='./results',
                                help='Path to store all the results',
                                type=str)

    args, unknowns = cmdline_parser.parse_known_args()
    logging.basicConfig(level='INFO')

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')
        
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        raise ValueError('Cannot train model without GPU as it uses GoLU CUDA Kernel!')

    main(
        model_name=args.model,
        seed=args.seed,
        activation=args.activation,
        save_results_str=args.results_path,
        device=device,
    )
