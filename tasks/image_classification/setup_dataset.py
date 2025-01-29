"""
This file contains the code to set up dataset before training a model
"""

import argparse
import logging
from torchvision import transforms
from torchvision import datasets
from tasks.utils.utils import create_directory


if __name__ == '__main__':
    
    cmdline_parser = argparse.ArgumentParser('Set up ImageNet-1K Dataset')
    
    cmdline_parser.add_argument('-dp', '--dataset_path',
                                default='./data',
                                help='Path where the dataset is stored',
                                type=str)
    
    args, unknowns = cmdline_parser.parse_known_args()
    logging.basicConfig(level=logging.INFO)
    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')
    
    dataset_name = 'imagenet_1k'
    dataset_path = f'{args.dataset_path}/{dataset_name}'
    create_directory(
        path=dataset_path
    )
    logging.info(dataset_path)
    
    # The following piece of code unzips the files and extracts the images for training
    train_dataset = datasets.ImageNet(
        root=dataset_path, split='train', transform=transforms.Compose([transforms.ToTensor()])
    )
    test_dataset = datasets.ImageNet(
        root=dataset_path, split='val', transform=transforms.Compose([transforms.ToTensor()])
    )
    logging.info('Dataset Extracted!')
