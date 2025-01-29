# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 10 * 4 # for 4 GPUs else 5 * 8 for 8 GPUs

# eval stuff
eval_interval = 1000
eval_iters = 50 # For 4 GPUs else 25 for 8 GPUs - accounts to 200 iterations across N GPUs
log_interval = 100

# weight decay
weight_decay = 1e-1

# dataset splits
dataset_splits = ['train', 'test']
