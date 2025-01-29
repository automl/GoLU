"""
This file helps to merge all the results for different architectures trained.
"""

import json
import numpy as np
import pandas as pd
from mlxtend.file_io import find_files


# Keeping the Architectures and Activations for which merged results are to be generated ----------
datasets_types = ['imagenet_1k']
architectures = [
    'resnet18',
    'resnet34',
    'wide_resnet50_2',
    'resnet50',
    'densenet121',
    'vit_b_16_aa_in',
    'vit_b_16',
    'vit_b_32',
    'efficientnet_b0',
]
activations = [
    'LeakyReLU', 'ReLU', 'GELU', 'ELU', 'Swish', 'Mish', 'GoLUCUDA',
]

excel_path = './results/img_cls_large.xlsx'

# Generating Merged Results in the json format ----------------------------------------------------
all_files = find_files(substring=".json", path="./results/imagenet_1k", recursive=True)
all_files = [file for file in all_files if 'merged' not in file]
all_files = [file for file in all_files if 'opt' not in file]
all_files = [file.replace('\\', '/') for file in all_files]
print(f'File path looks like this - {all_files[0]}')
print(f'Total number of files - {len(all_files)}')

# Dataframe to store the results
columns = [
    'Dataset', 'Architecture', 'Activation',
    'Best Test Top-1 Acc', 'Avg Train Loss', 'Avg Train Top-1 Acc', 'Avg Test Loss', 'Avg Test Top-1 Acc', 'Std Train Loss',
    'Std Train Top-1 Acc', 'Std Test Loss', 'Std Test Top-1 Acc', 'Avg Training Time (Mins)', 'Avg Inference Time (Mins)',
    'Std Training Time (Mins)', 'Std Inference Time (Mins)'
]
df = pd.DataFrame(columns=columns)
counter = 0

for dataset in datasets_types:
    for architecture in architectures:
        for activation in activations:
            
            print(f'Generating Merged JSON for - {dataset} | {architecture} | {activation}')
            
            # Defining some variables that accumulate the results
            TRAIN_LOSS = []
            TRAIN_TOP_1 = []
            TEST_LOSS = []
            TEST_TOP_1 = []
            TRAIN_TIME = []
            INFERENCE_TIME = []
            
            best_test_top1_acc = 0
            avg_train_loss = 0
            avg_train_top1_acc = 0
            avg_test_loss = 0
            avg_test_top1_acc = 0
            std_train_loss = 0
            std_train_top1_acc = 0
            std_test_loss = 0
            std_test_top1_acc = 0
            avg_train_time = 0
            avg_inference_time = 0
            std_train_time = 0
            std_inference_time = 0
            
            # Choosing the files which have the architecture and activation name
            result_files = [
                file for file in all_files if dataset in file and architecture in file and activation in file
            ]
            
            if result_files:
                
                for path in result_files:

                    # Read the JSON File
                    with open(path, 'r', encoding="utf-8") as file:
                        data = json.load(file)

                    # Accumulate the results
                    TRAIN_LOSS.append(data['train_loss'])
                    TRAIN_TOP_1.append(data['train_top1_acc'])
                    TEST_LOSS.append(data['test_loss'])
                    TEST_TOP_1.append(data['test_top1_acc'])
                    TRAIN_TIME.append(data['train_time'])
                    INFERENCE_TIME.append(data['inference_time'])
                
                best_test_top1_acc = np.max(TEST_TOP_1)
                avg_train_loss = np.mean(TRAIN_LOSS)
                avg_train_top1_acc = np.mean(TRAIN_TOP_1)
                avg_test_loss = np.mean(TEST_LOSS)
                avg_test_top1_acc = np.mean(TEST_TOP_1)
                std_train_loss = np.std(TRAIN_LOSS)
                std_train_top1_acc = np.std(TRAIN_TOP_1)
                std_test_loss = np.std(TEST_LOSS)
                std_test_top1_acc = np.std(TEST_TOP_1)
                avg_train_time = np.mean(TRAIN_TIME)
                avg_inference_time = np.mean(INFERENCE_TIME)
                std_train_time = np.std(TRAIN_TIME)
                std_inference_time = np.std(INFERENCE_TIME)

                results = {
                    'best_test_top1_acc' : best_test_top1_acc,
                    'avg_train_loss' : avg_train_loss,
                    'avg_train_top1_acc' : avg_train_top1_acc,
                    'avg_test_loss' : avg_test_loss,
                    'avg_test_top1_acc' : avg_test_top1_acc,
                    'std_train_loss' : std_train_loss,
                    'std_train_top1_acc' : std_train_top1_acc,
                    'std_test_loss' : std_test_loss,
                    'std_test_top1_acc' : std_test_top1_acc,
                    'avg_train_time' : avg_train_time,
                    'avg_inference_time' : avg_inference_time,
                    'std_train_time' : std_train_time,
                    'std_inference_time' : std_inference_time,
                }
                
                RESULTS_PATH = f'./results/{dataset}/{architecture}/{activation}/merged_results.json'
                with open(RESULTS_PATH, "w", encoding="utf-8") as outfile:
                    json.dump(results, outfile)
                    
                df.loc[counter, 'Dataset'] = dataset
                df.loc[counter, 'Architecture'] = architecture
                df.loc[counter, 'Activation'] = activation
                df.loc[counter, 'Best Test Top-1 Acc'] = best_test_top1_acc
                df.loc[counter, 'Avg Train Loss'] = avg_train_loss
                df.loc[counter, 'Avg Train Top-1 Acc'] = avg_train_top1_acc
                df.loc[counter, 'Avg Test Loss'] = avg_test_loss
                df.loc[counter, 'Avg Test Top-1 Acc'] = avg_test_top1_acc
                df.loc[counter, 'Std Train Loss'] = std_train_loss
                df.loc[counter, 'Std Train Top-1 Acc'] = std_train_top1_acc
                df.loc[counter, 'Std Test Loss'] = std_test_loss
                df.loc[counter, 'Std Test Top-1 Acc'] = std_test_top1_acc
                df.loc[counter, 'Avg Training Time (Mins)'] = avg_train_time
                df.loc[counter, 'Avg Inference Time (Mins)'] = avg_inference_time
                df.loc[counter, 'Std Training Time (Mins)'] = std_train_time
                df.loc[counter, 'Std Inference Time (Mins)'] = std_inference_time
                
            all_files = [file for file in all_files if file not in result_files]
            counter += 1

df.to_excel(excel_path, index=False)
