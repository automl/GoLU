import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --------------------------------------------------------------------------------------
# 1. Define the directory structure and hyperparameters
# --------------------------------------------------------------------------------------

base_path = ""

# The subfolder for the classification set, might vary in your setup.
folder = "language_modeling"

root_path = f"./results/ablation_studies/learning_rate/{folder}"
os.makedirs(root_path, exist_ok=True)

# We have four architectures and associated learning rates
architectures_lrs = {
    "baby_gpt":  [0.1, 0.01, 0.001, 0.0001],
}
archtecture_display_names = {
    'baby_gpt': 'BabyGPT',
}

default_lrs = {
    'baby_gpt': 0.001,
}

best_lrs = {
    'baby_gpt': 0.001,
}

# Activations (renaming "GoLUCUDA" -> "GoLU" for plotting)
activations = ["ReLU", "LeakyReLU", "ELU", "GELU", "Swish", "Mish", "GoLUCUDA"]
activation_display_names = {
    "ReLU": "ReLU",
    "LeakyReLU": "LeakyReLU",
    "ELU": "ELU",
    "GELU": "GELU",
    "Swish": "Swish",
    "Mish": "Mish",
    "GoLUCUDA": "GoLU"
}

# Seeds
seeds = [1, 2, 3]

# Dataset folder (as an example, e.g. "imagenet_1k")
dataset_name = "tiny_stories"
dataset_display_name = {
    'tiny_stories': 'Tiny Stories'
}

metric = 'test_loss'
plot_title_string = 'Test Loss'
ablation_name = 'Learning Rate'

# --------------------------------------------------------------------------------------
# 2. Parse JSON and collect accuracy metrics
# --------------------------------------------------------------------------------------

# data_store will hold: data_store[architecture][activation][lr] = list of accuracies from seeds
data_store = {}

for arch, lrs in architectures_lrs.items():
    data_store[arch] = {}
    for activation in activations:
        data_store[arch][activation] = {}
        for lr in lrs:
            data_store[arch][activation][lr] = []  # list of top-1 accuracies across seeds
            
            for seed in seeds:
                # Example path construction:
                json_path = os.path.join(
                    base_path,
                    folder,
                    str(lr),
                    dataset_name,
                    arch,
                    activation,
                    f"results_{seed}.json"
                )
                
                if not os.path.isfile(json_path):
                    print(f"WARNING: File not found: {json_path}")
                    continue
                
                with open(json_path, "r") as f:
                    results = json.load(f)
                
                # Extract test_top1_acc from the JSON
                # (Adjust the key to match exactly how it's stored in your JSON)
                test_top1_acc = results.get(metric, None)
                
                if test_top1_acc is not None:
                    data_store[arch][activation][lr].append(test_top1_acc)
                else:
                    print(f"WARNING: 'test_top1_acc' not found in {json_path}")


# --------------------------------------------------------------------------------------
# 3. Compute mean and standard error across seeds
# --------------------------------------------------------------------------------------

# We will store final results as dataframes for easy plotting with seaborn
architecture_dfs = {}

for arch, act_dict in data_store.items():
    
    # Construct a table with:
    #   rows = activations
    #   columns = learning rates
    # and cell values = "mean ± std_error"
    # We might also keep them separate if you prefer to show separate heatmaps or text overlays.
    
    # For seaborn heatmaps, we typically store only the "mean" in the dataframe.
    # We can store standard errors separately if we want to show them as text in the cells.
    
    df_mean = pd.DataFrame(index=[activation_display_names[a] for a in activations],
                           columns=architectures_lrs[arch], 
                           dtype=float)
    df_stderr = pd.DataFrame(index=[activation_display_names[a] for a in activations],
                             columns=architectures_lrs[arch], 
                             dtype=float)
    
    for activation in activations:
        row_name = activation_display_names[activation]
        for lr in architectures_lrs[arch]:
            acc_list = act_dict[activation][lr]
            if len(acc_list) == 0:
                mean_acc = np.nan
                stderr_acc = np.nan
            else:
                mean_acc = np.mean(acc_list)
                # Standard error of the mean (SEM) = std / sqrt(n)
                stderr_acc = np.std(acc_list, ddof=1) / np.sqrt(len(acc_list))
            
            df_mean.loc[row_name, lr] = mean_acc
            df_stderr.loc[row_name, lr] = stderr_acc
    
    architecture_dfs[arch] = (df_mean, df_stderr)

# --------------------------------------------------------------------------------------
# 4. Plotting heatmaps
# --------------------------------------------------------------------------------------

for arch, (df_mean, df_stderr) in architecture_dfs.items():
    
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        df_mean,
        annot=True,          # show the mean values in each cell
        fmt=".2f",           # format floats to 2 decimal places
        cmap="coolwarm",
        cbar=True
    )
    
    # Optional: If you also want to incorporate the standard error in the annotation,
    # you could manually construct annotation strings with both mean ± se.
    # For example:
    
    annotation_df = df_mean.copy()
    for row in annotation_df.index:
        for col in annotation_df.columns:
            m = df_mean.loc[row, col]
            s = df_stderr.loc[row, col]
            if pd.isna(m) or pd.isna(s):
                annotation_df.loc[row, col] = ""
            else:
                annotation_df.loc[row, col] = f"{m:.2f}\n±{s:.3f}"
    
    plt.figure(figsize=(10,6))
    ax = sns.heatmap(
        df_mean,  # used as numerical values
        annot=annotation_df,  # used for text
        fmt="",  # no additional formatting
        cmap="coolwarm",
        cbar=True
    )
    
    # Rectangle for default lr
    default_lr = default_lrs[arch]
    col_idx = df_mean.columns.get_loc(default_lr)
    ax.add_patch(Rectangle(
        (col_idx, 0),       # (x, y) in data coordinates
        1,                  # width
        df_mean.shape[0],   # height
        fill=False,         # don't fill, just the border
        edgecolor='#000000',    # color of the border
        linewidth=4
    ))
    
    # Rectangle for best lr
    best_lr = best_lrs[arch]
    col_idx = df_mean.columns.get_loc(best_lr)
    ax.add_patch(Rectangle(
        (col_idx, 0),       # (x, y) in data coordinates
        1,                  # width
        df_mean.shape[0],   # height
        fill=False,         # don't fill, just the border
        edgecolor='#20c997',    # color of the border
        linewidth=4
    ))
    
    ax.set_title(f"{plot_title_string} - {archtecture_display_names[arch]} - {dataset_display_name[dataset_name]}")
    ax.set_xlabel(ablation_name)
    # ax.set_ylabel("Activation")
    
    plt.tight_layout()
    plt.savefig(f'./results/ablation_studies/learning_rate/{folder}/{arch}.png')
