"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)

The following code has been taken from https://github.com/karpathy/nanoGPT/blob/master/train.py
"""

import datetime
import argparse
import logging
import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp
from torch.utils.tensorboard import SummaryWriter

from tasks.gpt.model import GPTConfig, GPT
from golu.activation_utils import replace_activation_by_torch_module
from tasks.utils.utils import load_json, save_results, create_directory, str_to_bool
from tasks.utils.plots import save_gpt_plots


cmdline_parser = argparse.ArgumentParser('Train GPT Models')

# Get general HPs --------------------------------------------------------------------------------------------------
cmdline_parser.add_argument('-mn', '--model_name',
                            default='gpt2s',
                            choices=['gpt2s', 'baby_gpt'],
                            help='Name of the model',
                            type=str)
cmdline_parser.add_argument('-d', '--dataset',
                            default='open_web_text',
                            choices=['open_web_text', 'tiny_stories'],
                            help='Name of the dataset',
                            type=str)
cmdline_parser.add_argument('-dp', '--dataset_path',
                            default='./data',
                            help='Where to store the dataset',
                            type=str)
cmdline_parser.add_argument('-s', '--seed',
                            default=0,
                            help='Seed to fix randomization',
                            type=int)
cmdline_parser.add_argument('-a', '--activation',
                            default='',
                            help='Activation function to train with',
                            type=str)
cmdline_parser.add_argument('-rp', '--results_path',
                            default='./results',
                            help='Path to store all the results',
                            type=str)
cmdline_parser.add_argument('-initfrom', '--init_from',
                            default='scratch',
                            choices=['scratch', 'resume'],
                            help='Flag to load model from scratch or resume training',
                            type=str)
cmdline_parser.add_argument('-esf', '--early_stop_flag',
                            default='False',
                            help='Flag to early stop',
                            type=str_to_bool)
cmdline_parser.add_argument('-esi', '--early_stop_iters',
                            default=0,
                            help='Iterations after which early stop is done',
                            type=int)
cmdline_parser.add_argument('-mi', '--max_iters',
                            default=600000,
                            help='Iterations to run for',
                            type=int)
cmdline_parser.add_argument('-lri', '--lr_decay_iters',
                            default=600000,
                            help='Iterations to run scheduler for',
                            type=int)

logging.basicConfig(level=logging.INFO)
args, unknowns = cmdline_parser.parse_known_args()

if unknowns:
    logging.warning('Found unknown arguments!')
    logging.warning(str(unknowns))
    logging.warning('These will be ignored')


# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
model_name = args.model_name
seed = int(args.seed)
activation = args.activation
init_from = args.init_from # 'scratch' or 'resume' or 'gpt2*'
out_dir = f'{args.results_path}/{args.dataset}/{model_name}/{activation}'
eval_interval = 1000
log_interval = 100
eval_iters = 25  # For 8 GPUs in one node it is 25 else 50 for 4 GPUs and so on
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
# data
dataset = args.dataset
dataset_path = f'{args.dataset_path}/{dataset}'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
dataset_splits = ['train', 'test']
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = int(args.max_iters)
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = int(args.lr_decay_iters) # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(f'./tasks/gpt/config/train_{model_name}_{dataset}.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
early_stop = bool(args.early_stop_flag)
early_stop_iters = max_iters if int(args.early_stop_iters) == 0 else int(args.early_stop_iters)

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend, timeout=datetime.timedelta(seconds=3600))
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = seed * ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = seed
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process:
    logging.info(f'Training Dataset - {dataset} | Model - {model_name} | Activation - {activation} | Seed - {seed}')
    logging.info(f'Training starts at - {datetime.datetime.now()} | Device - {device} | Training Dtype - {dtype}')
    logging.info(f'The config is - \n{config}')
    logging.info(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    logging.info(f'All checkpoints will be stored at - {out_dir}')
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
if master_process:
    logging.info(f'Training is done with {dtype}')
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
# data_dir = os.path.join(dataset_path, dataset)
if dataset == 'open_web_text':
    dataset_path = f'{dataset_path}/datasets'
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    data = np.memmap(os.path.join(dataset_path, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

logging_path= f'{out_dir}/{seed}'
checkpoint_path = f'{out_dir}/checkpoint_{seed}.pth'
opt_trace_path = f'{out_dir}/opt_trace_{seed}.json'
final_results_path = f'{out_dir}/results_{seed}.json'
plots_path = f'{out_dir}/plot_{seed}.png'
create_directory(logging_path)

if master_process:
    tensorboard_writer = SummaryWriter(log_dir=logging_path)

opt_trace = {
    'iteration': [],
    'train_loss': [],
    'train_acc': [],
    'test_loss': [],
    'test_acc': [],
    'train_time': 0,
    'inference_time': 0
}
if init_from == 'resume' and os.path.exists(opt_trace_path):
    opt_trace = load_json(path=opt_trace_path)
    if master_process:
        logging.info(f'Loaded optimization trace from path - {opt_trace_path}')

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(dataset_path, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    if master_process:
        logging.info(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=None, dropout=dropout
) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    if master_process:
        logging.info("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None and master_process:
        logging.info("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    replace_activation_by_torch_module(model, nn.GELU, activation)
    
elif init_from == 'resume':
    if master_process:
        logging.info(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    replace_activation_by_torch_module(model, nn.GELU, activation)
    
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif init_from.startswith('gpt2'):
    if master_process:
        logging.info(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

if master_process:
    logging.info(f'The model looks like - \n{model}')

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    if master_process:
        logging.info("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    acc = {}
    model.eval()
    for split in dataset_splits:
        losses = torch.tensor(0.0).to(device)
        accuracies = torch.tensor(0.0).to(device)
        num_datapoints = torch.tensor(0.0).to(device)
        for _ in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            if scaler is not None:
                scaler.scale(loss)
            preds = logits.argmax(dim=-1).view(-1)
            Y = Y.view(-1)
            num_datapoints += Y.size(0)
            losses += loss * Y.size(0)
            accuracies += torch.sum(preds == Y)
            torch.cuda.empty_cache()
        all_reduce(losses, op=ReduceOp.SUM)
        all_reduce(accuracies, op=ReduceOp.SUM)
        all_reduce(num_datapoints, op=ReduceOp.SUM)
        out[split] = (losses / num_datapoints).item()
        acc[split] = (accuracies / num_datapoints).item() * 100
    model.train()
    return out, acc

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# training loop
X, Y = get_batch(dataset_splits[0]) # fetch the very first training batch
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    if iter_num == 0 and master_process:
        logging.info(f'Training Starts at {datetime.datetime.now()}')
    
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if (iter_num + 1) % eval_interval == 0:
        inference_start_time = time.time()
        losses, accs = estimate_loss()
        opt_trace['inference_time'] += (time.time() - inference_start_time) / 60  # Converts from seconds to minutes
        opt_trace['iteration'].append(iter_num + 1)
        opt_trace['train_loss'].append(losses[dataset_splits[0]])
        opt_trace['test_loss'].append(losses[dataset_splits[1]])
        opt_trace['train_acc'].append(accs[dataset_splits[0]])
        opt_trace['test_acc'].append(accs[dataset_splits[1]])
        inferencing_time = opt_trace['inference_time']
        
        if master_process:
            logging.info(f"{datetime.datetime.now()} | Iteration [{iter_num + 1}/{max_iters}]: train loss - {losses[dataset_splits[0]]:.4f}, train acc - {accs[dataset_splits[0]]:.4f}, test loss - {losses[dataset_splits[1]]:.4f}, test acc - {accs[dataset_splits[1]]:.4f}, inference time - {inferencing_time:.4f} mins")
        
        # Tensorboard logging
        if master_process:
            for stat_name, value in opt_trace.items():
                if stat_name in ['train_loss', 'test_loss', 'train_acc', 'test_acc']:
                    tensorboard_writer.add_scalar(stat_name, value[-1], iter_num + 1)
        
        if losses[dataset_splits[1]] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses[dataset_splits[1]]
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num + 1,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            if master_process:
                torch.save(checkpoint, checkpoint_path)
                save_results(opt_trace, opt_trace_path)
                logging.info(f"saving checkpoint to {checkpoint_path}")
    
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    train_time_start = time.time()
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        torch.cuda.empty_cache()
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch(dataset_splits[0])
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        del logits  # Free up memory
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()

    # timing and logging
    dt = (time.time() - train_time_start) / 60
    opt_trace['train_time'] += dt  # Convert from seconds to minutes
    if (iter_num + 1) % log_interval == 0 and master_process:
        
        if master_process:
            # Tensorboard logging --------------------------------------------------------------------------------------
            weight_stats = {}
            grad_stats = {}
            with torch.no_grad():
                for name, param in model.named_parameters():
                    
                    # Get the weight stats -----------------------------------------------------------------------------
                    weight_stats[f'param/{name}/min'] = param.min().item()
                    weight_stats[f'param/{name}/max'] = param.max().item()
                    weight_stats[f'param/{name}/mean'] = param.mean().item()
                    weight_stats[f'param/{name}/std'] = param.std().item()
                    weight_stats[f'param/{name}/l2n'] = param.norm(2).item()
                    
                    # Get the grad stats -------------------------------------------------------------------------------
                    if param.grad is not None:
                        grad_stats[f'grad/{name}/min'] = param.grad.min().item()
                        grad_stats[f'grad/{name}/max'] = param.grad.max().item()
                        grad_stats[f'grad/{name}/mean'] = param.grad.mean().item()
                        grad_stats[f'grad/{name}/std'] = param.grad.std().item()
                        grad_stats[f'grad/{name}/l2n'] = param.grad.norm(2).item()
            
            for stat_name, value in weight_stats.items():
                tensorboard_writer.add_scalar(stat_name, value, iter_num + 1)
            
            for stat_name, value in grad_stats.items():
                tensorboard_writer.add_scalar(stat_name, value, iter_num + 1)
        
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt * 60)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        train_time = opt_trace['train_time']
        logging.info(f"{datetime.datetime.now()} | Iteration {iter_num + 1}: loss - {lossf:.4f}, train time - {train_time:.4f} mins, mfu - {running_mfu*100:.4f}%")
    
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
    
    if early_stop and iter_num == early_stop_iters:
        break

results = dict()
if master_process:
    results['train_loss'] = opt_trace['train_loss'][-1]
    results['train_acc'] = opt_trace['train_acc'][-1]
    results['test_loss'] = opt_trace['test_loss'][-1]
    results['test_acc'] = opt_trace['test_acc'][-1]
    results['train_time'] = opt_trace['train_time']
    results['inference_time'] = opt_trace['inference_time']
    save_results(results_dict=opt_trace, path=opt_trace_path)
    save_results(results_dict=results, path=final_results_path)
    save_gpt_plots(model_name=model_name, activation=activation, seed=seed, opt_trace=opt_trace, path=plots_path)
    logging.info('Training Completed!')

if ddp:
    destroy_process_group()
