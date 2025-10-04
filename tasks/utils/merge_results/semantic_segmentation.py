"""
This file helps to merge all the results for different architectures trained.
"""

import json
import numpy as np
import pandas as pd
from mlxtend.file_io import find_files


# Keeping the Architectures and Activations for which merged results are to be generated ----------
datasets_types = ['coco']
architectures = ['deeplabv3_resnet50_0.02', 'deeplabv3_resnet50_0.01']
activations = [
    'LeakyReLU', 'ReLU', 'GELU', 'ELU', 'Swish', 'Mish', 'GoLU',
]

excel_path = './results/sem_seg.xlsx'

# Generating Merged Results in the json format ----------------------------------------------------
all_files = find_files(substring=".json", path="./results", recursive=True)
all_files = [file for file in all_files if 'merged' not in file]
all_files = [file for file in all_files if 'opt' not in file]
all_files = [file.replace('\\', '/') for file in all_files]
print(f'File path looks like this - {all_files[0]}')
print(f'Total number of files - {len(all_files)}')

# Dataframe to store the results
columns = [
    'Dataset', 'Model', 'Activation', 'Best Test mIoU', 'Best Test Pixel Accuracy', 'Avg Train Loss', 'Avg Train mIoU',
    'Avg Train Pixel Accuracy', 'Avg Test Loss', 'Avg Test mIoU', 'Avg Test Pixel Accuracy', 'Std Train Loss',
    'Std Train mIoU', 'Std Train Pixel Accuracy', 'Std Test Loss', 'Std Test mIoU', 'Std Test Pixel Accuracy',
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
            TRAIN_MIOU = []
            TRAIN_PA = []
            TEST_LOSS = []
            TEST_MIOU = []
            TEST_PA = []
            TRAIN_TIME = []
            INFERENCE_TIME = []
            
            best_test_miou = 0
            best_test_pixel_accuracy = 0
            avg_train_loss = 0
            avg_train_miou = 0
            avg_train_pixel_accuracy = 0
            avg_test_loss = 0
            avg_test_miou = 0
            avg_test_pixel_accuracy = 0
            std_train_loss = 0
            std_train_miou = 0
            std_train_pixel_accuracy = 0
            std_test_loss = 0
            std_test_miou = 0
            std_test_pixel_accuracy = 0
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
                    TRAIN_MIOU.append(data['train_miou'])
                    TRAIN_PA.append(data['train_pixel_accuracy'])
                    TEST_LOSS.append(data['test_loss'])
                    TEST_MIOU.append(data['test_miou'])
                    TEST_PA.append(data['test_pixel_accuracy'])
                    TRAIN_TIME.append(data['train_time'])
                    INFERENCE_TIME.append(data['inference_time'])
                
                best_test_miou = np.max(TEST_MIOU)
                best_test_pixel_accuracy = np.max(TEST_PA)
                avg_train_loss = np.mean(TRAIN_LOSS)
                avg_train_miou = np.mean(TRAIN_MIOU)
                avg_train_pixel_accuracy = np.mean(TRAIN_PA)
                avg_test_loss = np.mean(TEST_LOSS)
                avg_test_miou = np.mean(TEST_MIOU)
                avg_test_pixel_accuracy = np.mean(TEST_PA)
                std_train_loss = np.std(TRAIN_LOSS)
                std_train_miou = np.std(TRAIN_MIOU)
                std_train_pixel_accuracy = np.std(TRAIN_PA)
                std_test_loss = np.std(TEST_LOSS)
                std_test_miou = np.std(TEST_MIOU)
                std_test_pixel_accuracy = np.std(TEST_PA)
                avg_train_time = np.mean(TRAIN_TIME)
                avg_inference_time = np.mean(INFERENCE_TIME)
                std_train_time = np.std(TRAIN_TIME)
                std_inference_time = np.std(INFERENCE_TIME)

                results = {
                    'best_test_miou' : best_test_miou,
                    'best_test_pixel_accuracy' : best_test_pixel_accuracy,
                    'avg_train_loss' : avg_train_loss,
                    'avg_train_miou' : avg_train_miou,
                    'avg_train_pixel_accuracy' : avg_train_pixel_accuracy,
                    'avg_test_loss' : avg_test_loss,
                    'avg_test_miou' : avg_test_miou,
                    'avg_test_pixel_accuracy' : avg_test_pixel_accuracy,
                    'std_train_loss' : std_train_loss,
                    'std_train_miou' : std_train_miou,
                    'std_train_pixel_accuracy' : std_train_pixel_accuracy,
                    'std_test_loss' : std_test_loss,
                    'std_test_miou' : std_test_miou,
                    'std_test_pixel_accuracy' : std_test_pixel_accuracy,
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
                df.loc[counter, 'Best Test mIoU'] = best_test_miou
                df.loc[counter, 'Best Test Pixel Accuracy'] = best_test_pixel_accuracy
                df.loc[counter, 'Avg Train Loss'] = avg_train_loss
                df.loc[counter, 'Avg Train mIoU'] = avg_train_miou
                df.loc[counter, 'Avg Train Pixel Accuracy'] = avg_train_pixel_accuracy
                df.loc[counter, 'Avg Test Loss'] = avg_test_loss
                df.loc[counter, 'Avg Test mIoU'] = avg_test_miou
                df.loc[counter, 'Avg Test Pixel Accuracy'] = avg_test_pixel_accuracy
                df.loc[counter, 'Std Train Loss'] = std_train_loss
                df.loc[counter, 'Std Train mIoU'] = std_train_miou
                df.loc[counter, 'Std Train Pixel Accuracy'] = std_train_pixel_accuracy
                df.loc[counter, 'Std Test Loss'] = std_test_loss
                df.loc[counter, 'Std Test mIoU'] = std_test_miou
                df.loc[counter, 'Std Test Pixel Accuracy'] = std_test_pixel_accuracy
                df.loc[counter, 'Avg Training Time (Mins)'] = avg_train_time
                df.loc[counter, 'Avg Inference Time (Mins)'] = avg_inference_time
                df.loc[counter, 'Std Training Time (Mins)'] = std_train_time
                df.loc[counter, 'Std Inference Time (Mins)'] = std_inference_time
                
            all_files = [file for file in all_files if file not in result_files]
            counter += 1

df.to_excel(excel_path, index=False)
