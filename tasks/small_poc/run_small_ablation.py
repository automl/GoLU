import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import pandas as pd
import gc
import os
import datetime

from golu.activation_utils import get_activation_function

def parse_args():
    parser = argparse.ArgumentParser(description='Ablation Study on MNIST/CIFAR-10 with various activations.')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                        help='Dataset to use: mnist or cifar10')
    args = parser.parse_args()
    return args

def get_dataset_and_transforms(dataset_name):
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        input_dim = 28 * 28
        num_classes = 10
        
    elif dataset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        
        input_dim = 3 * 32 * 32
        num_classes = 10
        
    return train_dataset, test_dataset, input_dim, num_classes

class MLP(nn.Module):
    def __init__(self, num_blocks, activation_name, input_dim, num_classes):
        super(MLP, self).__init__()
        layers = []
        in_features = input_dim

        for i in range(num_blocks):
            if i == 0:
                layers.append(nn.Linear(in_features, 512))
            else:
                layers.append(nn.Linear(512, 512))
            layers.append(nn.LayerNorm(512))
            layers.append(get_activation_function(activation_name))

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.feature_extractor(x)
        return self.classifier(x)

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    l2_norm = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max logits
            correct += pred.eq(target.view_as(pred)).sum().item()
        # Compute L2 norm of all weights
        for param in model.parameters():
            l2_norm += torch.norm(param).item() ** 2
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    l2_norm = l2_norm ** 0.5
    return test_loss, accuracy, l2_norm

def main():
    args = parse_args()
    dataset_name = args.dataset

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset and determine input_dim/num_classes
    train_dataset, test_dataset, input_dim, num_classes = get_dataset_and_transforms(dataset_name)

    # Define parameters for ablation
    activations = ["ReLU", "GELU", "Swish", "Mish", "GoLU"]
    seeds = [1, 2, 3]
    ablations = ['blocks_vs_lr', 'blocks_vs_epochs', 'blocks_vs_batch_size']

    num_blocks_list = [1, 2, 3, 4]
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    epochs_list = [5, 6, 7, 8]
    batch_sizes = [32, 64, 128, 256]

    # Initialize list to store results
    results = []

    for activation_name in activations:
        print(f'{datetime.datetime.now()} - Starting Activation - {activation_name}')

        for seed in seeds:
            print(f'{datetime.datetime.now()} - Starting Seed - {seed}')
            # Set random seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            for ablation in ablations:
                print(f'{datetime.datetime.now()} - Ablation - {ablation}')
                
                if ablation == 'blocks_vs_lr':
                    fixed_epochs = 8
                    fixed_batch_size = 128
                    for num_blocks in num_blocks_list:
                        for lr in learning_rates:
                            print(f'{datetime.datetime.now()} - Num Blocks - {num_blocks} | Learning Rate - {lr}')
                            # Prepare data loaders
                            train_loader = torch.utils.data.DataLoader(
                                train_dataset, batch_size=fixed_batch_size, shuffle=True)
                            test_loader = torch.utils.data.DataLoader(
                                test_dataset, batch_size=fixed_batch_size, shuffle=False)
                            
                            # Build model
                            model = MLP(num_blocks, activation_name, input_dim, num_classes).to(device)
                            # Define optimizer and criterion
                            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                            criterion = nn.CrossEntropyLoss()
                            # Train the model
                            for epoch in range(fixed_epochs):
                                train(model, device, train_loader, optimizer, criterion)
                            # Test the model
                            test_loss, accuracy, l2_norm = test(model, device, test_loader, criterion)
                            # Store results
                            results.append({
                                'activation': activation_name,
                                'seed': seed,
                                'ablation': ablation,
                                'num_blocks': num_blocks,
                                'learning_rate': lr,
                                'epochs': fixed_epochs,
                                'batch_size': fixed_batch_size,
                                'test_loss': test_loss,
                                'test_accuracy': accuracy,
                                'l2_norm': l2_norm
                            })
                            # Free up memory
                            del model
                            del optimizer
                            del train_loader
                            del test_loader
                            torch.cuda.empty_cache()
                            gc.collect()
                
                elif ablation == 'blocks_vs_epochs':
                    fixed_learning_rate = 0.01
                    fixed_batch_size = 128
                    for num_blocks in num_blocks_list:
                        for epochs in epochs_list:
                            print(f'{datetime.datetime.now()} - Num Blocks - {num_blocks} | Epochs - {epochs}')
                            # Prepare data loaders
                            train_loader = torch.utils.data.DataLoader(
                                train_dataset, batch_size=fixed_batch_size, shuffle=True)
                            test_loader = torch.utils.data.DataLoader(
                                test_dataset, batch_size=fixed_batch_size, shuffle=False)
                            
                            # Build model
                            model = MLP(num_blocks, activation_name, input_dim, num_classes).to(device)
                            # Define optimizer and criterion
                            optimizer = optim.SGD(model.parameters(), lr=fixed_learning_rate, momentum=0.9)
                            criterion = nn.CrossEntropyLoss()
                            # Train the model
                            for epoch in range(epochs):
                                train(model, device, train_loader, optimizer, criterion)
                            # Test the model
                            test_loss, accuracy, l2_norm = test(model, device, test_loader, criterion)
                            # Store results
                            results.append({
                                'activation': activation_name,
                                'seed': seed,
                                'ablation': ablation,
                                'num_blocks': num_blocks,
                                'learning_rate': fixed_learning_rate,
                                'epochs': epochs,
                                'batch_size': fixed_batch_size,
                                'test_loss': test_loss,
                                'test_accuracy': accuracy,
                                'l2_norm': l2_norm
                            })
                            # Free up memory
                            del model
                            del optimizer
                            del train_loader
                            del test_loader
                            torch.cuda.empty_cache()
                            gc.collect()
                
                elif ablation == 'blocks_vs_batch_size':
                    fixed_learning_rate = 0.01
                    fixed_epochs = 8
                    for num_blocks in num_blocks_list:
                        for bs in batch_sizes:
                            print(f'{datetime.datetime.now()} - Num Blocks - {num_blocks} | Batch Size - {bs}')
                            # Prepare data loaders
                            train_loader = torch.utils.data.DataLoader(
                                train_dataset, batch_size=bs, shuffle=True)
                            test_loader = torch.utils.data.DataLoader(
                                test_dataset, batch_size=bs, shuffle=False)
                            
                            # Build model
                            model = MLP(num_blocks, activation_name, input_dim, num_classes).to(device)
                            # Define optimizer and criterion
                            optimizer = optim.SGD(model.parameters(), lr=fixed_learning_rate, momentum=0.9)
                            criterion = nn.CrossEntropyLoss()
                            # Train the model
                            for epoch in range(fixed_epochs):
                                train(model, device, train_loader, optimizer, criterion)
                            # Test the model
                            test_loss, accuracy, l2_norm = test(model, device, test_loader, criterion)
                            # Store results
                            results.append({
                                'activation': activation_name,
                                'seed': seed,
                                'ablation': ablation,
                                'num_blocks': num_blocks,
                                'learning_rate': fixed_learning_rate,
                                'epochs': fixed_epochs,
                                'batch_size': bs,
                                'test_loss': test_loss,
                                'test_accuracy': accuracy,
                                'l2_norm': l2_norm
                            })
                            # Free up memory
                            del model
                            del optimizer
                            del train_loader
                            del test_loader
                            torch.cuda.empty_cache()
                            gc.collect()

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    if not os.path.exists("./results"):
        os.makedirs("./results")

    # Save results to CSV
    df.to_csv(f'./results/{dataset_name}_ablation_results.csv', index=False)


if __name__ == '__main__':
    main()
