# train on tiny stories with baby-gpt

eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 50 # For 4 GPUs else 25 for 8 GPUs
log_interval = 10 # don't print too too often

always_save_checkpoint = True

gradient_accumulation_steps = 10 * 4 # for 4 GPUs else 5 * 8 for 8 GPUs
batch_size = 16
block_size = 256

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

dataset_splits = ['train', 'validation']
