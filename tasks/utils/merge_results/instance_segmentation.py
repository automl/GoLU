"""
This file helps to merge all the results for different architectures trained.
"""

import json
import numpy as np
import pandas as pd
from mlxtend.file_io import find_files


# Keeping the Architectures and Activations for which merged results are to be generated ----------
datasets_types = ['coco']
architectures = ['maskrcnn_resnet50_fpn']
activations = [
    'LeakyReLU', 'ReLU', 'GELU', 'ELU', 'Swish', 'Mish', 'GoLU',
]

excel_path = './results/ins_seg.xlsx'

# Generating Merged Results in the json format ----------------------------------------------------
all_files = find_files(substring=".json", path="./results", recursive=True)
all_files = [file for file in all_files if 'merged' not in file]
all_files = [file for file in all_files if 'opt' not in file]
all_files = [file.replace('\\', '/') for file in all_files]
print(f'File path looks like this - {all_files[0]}')
print(f'Total number of files - {len(all_files)}')

# Dataframe to store the results
columns = [
    'Dataset', 'Model', 'Activation', 'Best Test Box MAP', 'Best Test Mask MAP', 'Avg Train Loss',
    'Avg Test Box MAP', 'Avg Test Mask MAP', 'Std Train Loss', 'Std Test Box MAP', 'Std Test Mask MAP',
    'Avg Training Time (Mins)', 'Avg Inference Time (Mins)', 'Std Training Time (Mins)', 'Std Inference Time (Mins)'
]
df = pd.DataFrame(columns=columns)
counter = 0

for dataset in datasets_types:
    for architecture in architectures:
        for activation in activations:
            
            print(f'Generating Merged JSON for - {dataset} | {architecture} | {activation}')
            
            # Defining some variables that accumulate the results
            TRAIN_LOSS = []
            TEST_BOX_MAP = []
            TEST_MASK_MAP = []
            TRAIN_TIME = []
            INFERENCE_TIME = []
            
            best_test_box_map = 0
            best_test_mask_map = 0
            avg_train_loss = 0
            avg_test_box_map = 0
            avg_test_mask_map = 0
            std_train_loss = 0
            std_test_box_map = 0
            std_test_mask_map = 0
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
                    TEST_BOX_MAP.append(data['test_box_map'])
                    TEST_MASK_MAP.append(data['test_mask_map'])
                    TRAIN_TIME.append(data['train_time'])
                    INFERENCE_TIME.append(data['inference_time'])
                
                best_test_box_map = np.max(TEST_BOX_MAP)
                best_test_mask_map = np.max(TEST_MASK_MAP)
                avg_train_loss = np.mean(TRAIN_LOSS)
                avg_test_box_map = np.mean(TEST_BOX_MAP)
                avg_test_mask_map = np.mean(TEST_MASK_MAP)
                std_train_loss = np.std(TRAIN_LOSS)
                std_test_box_map = np.std(TEST_BOX_MAP)
                std_test_mask_map = np.std(TEST_MASK_MAP)
                avg_train_time = np.mean(TRAIN_TIME)
                avg_inference_time = np.mean(INFERENCE_TIME)
                std_train_time = np.std(TRAIN_TIME)
                std_inference_time = np.std(INFERENCE_TIME)

                results = {
                    'best_test_box_map' : best_test_box_map,
                    'best_test_mask_map' : best_test_mask_map,
                    'avg_train_loss' : avg_train_loss,
                    'avg_test_box_map' : avg_test_box_map,
                    'avg_test_mask_map' : avg_test_mask_map,
                    'std_train_loss' : std_train_loss,
                    'std_test_box_map' : std_test_box_map,
                    'std_test_mask_map' : std_test_mask_map,
                    'avg_train_time' : avg_train_time,
                    'avg_inference_time' : avg_inference_time,
                    'std_train_time' : std_train_time,
                    'std_inference_time' : std_inference_time,
                }
                
                RESULTS_PATH = f'./results/{dataset}/{architecture}/{activation}/merged_results.json'
                with open(RESULTS_PATH, "w", encoding="utf-8") as outfile:
                    json.dump(results, outfile)
                    
                df.loc[counter, 'Dataset'] = dataset
                df.loc[counter, 'Model'] = architecture
                df.loc[counter, 'Activation'] = activation
                df.loc[counter, 'Best Test Box MAP'] = best_test_box_map
                df.loc[counter, 'Best Test Mask MAP'] = best_test_mask_map
                df.loc[counter, 'Avg Train Loss'] = avg_train_loss
                df.loc[counter, 'Avg Test Box MAP'] = avg_test_box_map
                df.loc[counter, 'Avg Test Mask MAP'] = avg_test_mask_map
                df.loc[counter, 'Std Train Loss'] = std_train_loss
                df.loc[counter, 'Std Test Box MAP'] = std_test_box_map
                df.loc[counter, 'Std Test Mask MAP'] = std_test_mask_map
                df.loc[counter, 'Avg Training Time (Mins)'] = avg_train_time
                df.loc[counter, 'Avg Inference Time (Mins)'] = avg_inference_time
                df.loc[counter, 'Std Training Time (Mins)'] = std_train_time
                df.loc[counter, 'Std Inference Time (Mins)'] = std_inference_time
                
            all_files = [file for file in all_files if file not in result_files]
            counter += 1

df.to_excel(excel_path, index=False)
