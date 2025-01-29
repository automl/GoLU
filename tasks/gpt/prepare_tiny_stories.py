# Taken from Karpathy's NanoGPT repository (https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py)
# and modified to work with TinyStories

import argparse
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets


# When downloading data pass "path_to_dataset/tiny_stories"
# When training the model pass only "path_to_dataset" - tiny_stories is added in train.py file from the dataset name


if __name__ == '__main__':

    cmdline_parser = argparse.ArgumentParser('Prepare Tiny Stories')
    cmdline_parser.add_argument('-cd', '--cache_dir',
                                default='',
                                help='The directory where the data .arrow files are stored',
                                type=str)
    cmdline_parser.add_argument('-nw', '--num_workers',
                                default=1,
                                help='Number of workers used to process data',
                                type=int)
    args, unknowns = cmdline_parser.parse_known_args()
    
    # number of workers in .map() call
    # good number to use is ~order number of cpu cores // 2
    num_proc = int(args.num_workers)

    # number of workers in load_dataset() call
    # best number might be different from num_proc above as it also depends on NW speed.
    # it is better than 1 usually though
    num_proc_load_dataset = num_proc
    
    tinystories_datapath = args.cache_dir

    if not os.path.exists(tinystories_datapath):
        os.makedirs(tinystories_datapath)
    
    dataset = load_dataset("roneneldan/TinyStories", num_proc=num_proc_load_dataset, cache_dir=tinystories_datapath)

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    enc = tiktoken.get_encoding("gpt2")
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(tinystories_datapath, f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
