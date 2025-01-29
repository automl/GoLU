import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description='Plot Ablation Heatmaps from CSV.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset name (e.g. mnist, cifar10, etc.)')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load the CSV file
    df = pd.read_csv(f'./results/{args.dataset}_ablation_results.csv')

    # Filter only for the 'blocks_vs_lr' ablation (example)
    df = df[df['ablation'] == 'blocks_vs_lr']

    # We want to average out seeds for each combination of activation, num_blocks, learning_rate, etc.
    group_cols = ['activation', 'ablation', 'num_blocks', 'learning_rate', 'epochs', 'batch_size']
    agg_df = df.groupby(group_cols, as_index=False).mean()

    # Replace "GoLUCUDA" with "GoLU"
    agg_df['activation'] = agg_df['activation'].replace({'GoLUCUDA': 'GoLU'})

    # Fix the order of activations if desired
    # (Here we explicitly set it to a chosen order, else we could sort)
    activations = ['ReLU', 'GELU', 'Swish', 'Mish', 'GoLU']

    # Pivot tables for each activation (for test_loss and test_accuracy)
    test_loss_matrices = {}
    test_accuracy_matrices = {}

    for activation in activations:
        sub_df = agg_df[agg_df['activation'] == activation]

        # Pivot table: index = num_blocks, columns = learning_rate
        loss_pivot = sub_df.pivot(index='num_blocks', columns='learning_rate', values='test_loss')
        acc_pivot = sub_df.pivot(index='num_blocks', columns='learning_rate', values='test_accuracy')

        # Optionally sort rows/columns to keep consistent order
        loss_pivot = loss_pivot.sort_index().sort_index(axis=1)
        acc_pivot = acc_pivot.sort_index().sort_index(axis=1)

        test_loss_matrices[activation] = loss_pivot
        test_accuracy_matrices[activation] = acc_pivot

    # Find global min/max for test_loss and test_accuracy for consistent color scales
    all_loss_values = np.concatenate([mat.values.flatten() for mat in test_loss_matrices.values()])
    all_acc_values  = np.concatenate([mat.values.flatten() for mat in test_accuracy_matrices.values()])

    loss_min, loss_max = all_loss_values.min(), all_loss_values.max()
    acc_min,  acc_max  = all_acc_values.min(),  all_acc_values.max()

    # Metrics to plot
    metrics       = ['test_loss', 'test_accuracy']
    metric_labels = {'test_loss': 'Test Loss', 'test_accuracy': 'Test Accuracy'}

    # Prepare subplots:
    #   Row 1 -> Test Loss (across all activations)
    #   Row 2 -> Test Accuracy (across all activations)
    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(activations),
        figsize=(5 * len(activations), 5 * len(metrics)),
        gridspec_kw={'hspace': 0.3, 'wspace': 0.25}
    )

    # If there is only 1 activation, ensure axes is 2D
    if len(activations) == 1:
        axes = np.array([axes]).T  # shape => (2,1)

    # Colormap
    cmap = 'coolwarm'

    # Grab row/column labels from one pivot table for consistent tick labels
    sample_activation = activations[0]
    nb_values = test_loss_matrices[sample_activation].index.tolist()   # num_blocks
    lr_values = test_loss_matrices[sample_activation].columns.tolist() # learning_rate

    # Loop over metrics (rows)
    for i, metric in enumerate(metrics):
        # Decide global min/max and pivot dictionary
        if metric == 'test_loss':
            vmin, vmax = loss_min, loss_max
            data_dict  = test_loss_matrices
        else:
            vmin, vmax = acc_min, acc_max
            data_dict  = test_accuracy_matrices

        # Plot each activation in columns
        for j, activation in enumerate(activations):
            ax = axes[i, j] if len(metrics) > 1 else axes[j]

            # Extract the pivot table
            pivot = data_dict[activation]

            # Show the colorbar only on the rightmost column
            cbar = (j == len(activations) - 1)

            # Label the colorbar only if we're showing it
            cbar_label = metric_labels[metric] if cbar else None

            # Plot the heatmap
            sns.heatmap(
                pivot,
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                annot=True,
                fmt='.4f',
                annot_kws={"size": 10, "weight": "bold"},
                cbar=False,
                # cbar_kws={
                #     "shrink": 0.8,
                #     "label": cbar_label
                # }
            )

            # Titles & labels
            if i == 0:  # top row
                ax.set_title(activation, fontsize=20)
            if j == 0:  # leftmost column
                ax.set_ylabel("Num Blocks", fontsize=15)
            else:
                ax.set_ylabel('')
            ax.set_xlabel('Learning Rate', fontsize=15)

            # Tick labels
            ax.set_xticklabels(lr_values, rotation=45, ha='right')
            ax.set_yticklabels(nb_values, rotation=0)

    # Reserve space on the right so that colorbars do not squeeze the last column
    # 'rect' is [left, bottom, right, top] all in 0..1 range
    # plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.savefig(f'./results/{args.dataset}_lr_heatmaps.png')
    plt.show()

if __name__ == '__main__':
    main()
