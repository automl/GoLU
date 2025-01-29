import argparse
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

parser = argparse.ArgumentParser(description='Ablation Study on MNIST/CIFAR-10 with various activations.')
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                    help='Dataset to use: mnist or cifar10')
args = parser.parse_args()

# Load the data
df = pd.read_csv(f'./results/{args.dataset}_ablation_results.csv')

# Average over seeds
group_cols = ['activation', 'ablation', 'num_blocks', 'learning_rate', 'epochs', 'batch_size']
mean_df = df.groupby(group_cols, as_index=False).mean()

# Rename activation 'GoLUCUDA' to 'GoLU'
mean_df['activation'] = mean_df['activation'].replace({'GoLUCUDA': 'GoLU'})

# Define the ablations and the associated x/y parameters
ablations_params = {
    'blocks_vs_epochs': ('num_blocks', 'epochs'),
    'blocks_vs_batch_size': ('num_blocks', 'batch_size')
}
ablations_labels = {
    'epochs': 'Epochs',
    'batch_size': 'Batch Size'
}

# Define the metrics we want to plot
metrics = [
    'test_loss',
    'test_accuracy'
]
metric_labels = {
    'test_loss': 'Test Loss',
    'test_accuracy': 'Test Accuracy'
}

# Get the list of activations (columns order)
activations = ['ReLU', 'GELU', 'Swish', 'Mish', 'GoLU']

for ablation, (x_param, y_param) in ablations_params.items():
    # Filter dataframe for this ablation
    ablation_df = mean_df[mean_df['ablation'] == ablation]

    # Get unique x and y values sorted
    x_values = sorted(ablation_df[x_param].unique())
    y_values = sorted(ablation_df[y_param].unique())

    # Compute global min/max for each metric to ensure uniform color scale per row
    metric_global_min = {}
    metric_global_max = {}
    for metric in metrics:
        metric_global_min[metric] = ablation_df[metric].min()
        metric_global_max[metric] = ablation_df[metric].max()

    # Create subplots: rows = #metrics, cols = #activations
    fig = make_subplots(
        rows=len(metrics), 
        cols=len(activations),
        horizontal_spacing=0.05, 
        vertical_spacing=0.10,
        shared_xaxes=True,
        shared_yaxes=True
    )

    # Plot each metric in rows, each activation in columns
    for i, metric in enumerate(metrics, start=1):
        for j, activation in enumerate(activations, start=1):

            subset = ablation_df[ablation_df['activation'] == activation]
            pivot = subset.pivot(index=y_param, columns=x_param, values=metric)
            # Ensure rows of pivot are sorted by y_values and columns by x_values
            pivot = pivot.reindex(index=y_values, columns=x_values)
            z_values = pivot.values

            # Decide whether to show scale (only for rightmost column in each row)
            if j == len(activations):
                showscale = True
                # Position colorbar differently for row 1 vs row 2
                # (You can tweak y and len to suit your needs)
                if i == 1:   # top row
                    cb_y = 0.75
                else:        # second row (if you have more rows, adjust accordingly)
                    cb_y = 0.25

                colorbar_dict = dict(
                    title=metric_labels[metric],
                    titleside='right',
                    x=1.03,      # shift colorbar slightly to the right
                    y=cb_y,
                    len=0.45     # length of the colorbar
                )
            else:
                showscale = False
                colorbar_dict = {}

            trace = go.Contour(
                x=x_values,
                y=y_values,
                z=z_values,
                contours=dict(coloring='heatmap'),
                colorscale='viridis',
                showscale=showscale,
                colorbar=colorbar_dict,
                zmin=metric_global_min[metric],
                zmax=metric_global_max[metric]
            )
            fig.add_trace(trace, row=i, col=j)

            # Label the y-axis on the leftmost column
            if j == 1:
                # Combine metric label and y_param if you wish
                fig.update_yaxes(
                    title_text=f"{ablations_labels[y_param]}",
                    row=i, 
                    col=j,
                )

            # Label the x-axis on the bottom row
            if i == len(metrics):
                fig.update_xaxes(title_text=x_param.replace('_', ' ').title(), row=i, col=j)

            # Label each column with the activation function name at the top
            if i == 1:  # top row
                fig.add_annotation(
                    x=0.5,
                    y=1.15,
                    xref='x domain',
                    yref='y domain',
                    showarrow=False,
                    text=activation,
                    font=dict(size=30),
                    row=i, 
                    col=j
                )

    # Update layout
    fig.update_layout(
        font=dict(size=25),
        title_text=ablation.replace('_', ' ').title(),
        title_font=dict(size=30),
        width=420 * len(activations),
        height=400 * len(metrics),
        margin=dict(l=150, r=100, t=100, b=100)
    )

    # Save the figure
    fig.write_image(f'./results/{args.dataset}_{ablation}_plots.png')
