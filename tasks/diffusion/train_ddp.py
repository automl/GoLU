"""
The following code has been taken from - https://github.com/KimRass/DDPM/blob/main/train_ddp.py
"""

# References:
    # https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1
    # https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb#scrollTo=e3eb5811-c10b-4dae-a58d-9583c42e7f57
    # https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/modules.py

import os
import datetime
import logging
import torch
from torch import nn
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
import argparse
from pathlib import Path
import math
from time import time
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler

from tasks.diffusion.utils import (
    get_grad_scaler,
    modify_state_dict,
    image_to_grid,
    save_image,
)
from tasks.diffusion.data import get_train_and_val_dls_ddp
from tasks.diffusion.unet import UNet
from tasks.diffusion.ddpm import DDPM
from tasks.utils.utils import set_seed, setup_ddp, create_directory, load_json, \
    save_results, num_parameters, DEVICES_GPU, DDP_BACKENDS
from tasks.utils.plots import save_diffusion_plots

from golu.activation_utils import replace_activation_by_torch_module


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", default="./results", type=str)
    parser.add_argument("--n_epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--n_warmup_steps", default=1000, type=int)
    parser.add_argument("--img_size", default=64, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--activation", default="ReLU", type=str)
    parser.add_argument("--model_name", default="ddpm", type=str)
    parser.add_argument("--dataset", default="celeba", type=str)
    parser.add_argument("--data_dir", default="", type=str)
    parser.add_argument("--device_type", type=str, default="cuda", choices=DEVICES_GPU)
    parser.add_argument("--backend", type=str, default="nccl", choices=DDP_BACKENDS)
    args = parser.parse_args()
    return args


class Trainer(object):
    def __init__(self, args, train_dl, val_dl, device):
        self.args = args
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device
        self.rank = args.rank
        self.master_process = args.master_process

        self.save_dir = args.save_dir
        self.save_sample_dir = args.save_sample_dir
        self.checkpoint_path = args.checkpoint_path
        self.opt_trace_path = args.opt_trace_path
        self.results_path = args.results_path
        self.plots_path = args.plots_path
        
        self.opt_trace = {
            'epoch': [],
            'train_loss': [],
            'test_loss': [],
            'train_time': 0,
            'inference_time': 0
        }
        if os.path.exists(self.opt_trace_path):
            self.opt_trace = load_json(self.opt_trace_path)
        
        self.result = dict()

    def train_for_one_epoch(self, epoch, model, optim, scaler):
        self.train_dl.sampler.set_epoch(epoch)

        train_loss = 0
        if self.master_process:
            pbar = tqdm(self.train_dl, leave=False)
        else:
            pbar = self.train_dl
        for step_idx, ori_image in enumerate(pbar): # "$x_{0} \sim q(x_{0})$"
            if self.master_process:
                pbar.set_description("Training...")

            ori_image = ori_image.to(self.device)
            loss = model.module.get_loss(ori_image)
            train_loss += (loss.item() / len(self.train_dl))

            optim.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

            self.scheduler.step(epoch * len(self.train_dl) + step_idx)
        return train_loss

    @torch.inference_mode()
    def validate(self, model):
        val_loss = 0
        if self.master_process:
            pbar = tqdm(self.val_dl, leave=False)
        else:
            pbar = self.val_dl
        for ori_image in pbar:
            if self.master_process:
                pbar.set_description("Validating...")

            ori_image = ori_image.to(self.device)
            loss = model.module.get_loss(ori_image.detach())
            val_loss += (loss.item() / len(self.val_dl))
        return val_loss

    @staticmethod
    def save_model_params(model, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(modify_state_dict(model.module.state_dict()), str(save_path))
        logging.info(f"Saved model params as '{str(save_path)}'.")

    def save_ckpt(self, epoch, model, optim, min_val_loss, scaler):
        if self.master_process:
            ckpt = {
                "epoch": epoch + 1,
                "model": modify_state_dict(model.state_dict()),
                "optimizer": optim.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "min_val_loss": min_val_loss,
            }
            if scaler is not None:
                ckpt["scaler"] = scaler.state_dict()
            torch.save(ckpt, self.checkpoint_path)

    @torch.inference_mode()
    def test_sampling(self, epoch, model, batch_size):
        if self.master_process:
            gen_image = model.module.sample(batch_size=batch_size)
            gen_grid = image_to_grid(gen_image, n_cols=int(batch_size ** 0.5))
            sample_path = f'{self.save_sample_dir}/sample_epoch_{epoch+1}.jpg'
            save_image(gen_grid, save_path=sample_path)

    def train(self, n_epochs, model, model_without_ddp, optim, scaler, n_warmup_steps):
        
        self.scheduler = CosineLRScheduler(
            optimizer=optim,
            t_initial=n_epochs * len(self.train_dl),
            warmup_t=n_warmup_steps,
            warmup_lr_init=optim.param_groups[0]["lr"] * 0.1,
            warmup_prefix=True,
            t_in_epochs=False,
        )

        init_epoch = 0
        min_val_loss = math.inf
        
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path)
            model_without_ddp.load_state_dict(checkpoint["model"])
            optim.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            if scaler is not None:
                scaler.load_state_dict(checkpoint['scaler'])
            init_epoch = checkpoint['epoch']
            min_val_loss = checkpoint['min_val_loss']
        
        for epoch in range(init_epoch, n_epochs):
            
            train_start_time = time()
            train_loss = self.train_for_one_epoch(
                epoch=epoch, model=model, optim=optim, scaler=scaler,
            )
            train_time = (time() - train_start_time) / 60
            
            inference_start_time = time()
            val_loss = self.validate(model)
            inference_time = (time() - inference_start_time) / 60
            
            self.opt_trace['epoch'].append(epoch + 1)
            self.opt_trace['train_loss'].append(train_loss)
            self.opt_trace['test_loss'].append(val_loss)
            self.opt_trace['train_time'] += train_time
            self.opt_trace['inference_time'] += inference_time
            
            if self.master_process:
                self.save_ckpt(
                    epoch=epoch,
                    model=model_without_ddp,
                    optim=optim,
                    min_val_loss=min_val_loss,
                    scaler=scaler,
                )
                save_results(self.opt_trace, self.opt_trace_path)

            self.test_sampling(epoch=epoch, model=model, batch_size=16)

            if self.master_process:
                logging.info(f'Epoch [{epoch + 1}/{n_epochs}] | Train Loss - {train_loss:.4f} | Test Loss - {val_loss:.4f} | Train Time - {self.opt_trace["train_time"]:.4f} | Inference Time - {self.opt_trace["inference_time"]:.4f}')
        
        # Evaluate FID -------------------------------------------------------------------------------------------------
        if self.master_process:
            self.result['train_loss'] = self.opt_trace['train_loss'][-1]
            self.result['test_loss'] = self.opt_trace['test_loss'][-1]
            self.result['train_time'] = self.opt_trace['train_time']
            self.result['inference_time'] = self.opt_trace['inference_time']
            save_results(self.result, self.results_path)
            save_diffusion_plots(
                self.args.model_name, self.args.activation, self.args.seed, self.opt_trace, self.plots_path
            )


class DistDataParallel(object):
    def __init__(self, args):
        self.args = args

    def setup(self):
        self.args.rank, self.args.world_size, self.args.device, self.args.master_process = setup_ddp(
            backend=self.args.backend,
            device_type=self.args.device_type,
            seed=int(self.args.seed),
            port=23456,
            timeout=3600
        )
        
    def cleanup(self):
        dist.destroy_process_group()

    def main_worker(self):
        self.setup()
        if self.args.master_process:
            logging.info(f'Training Dataset - {self.args.dataset} | Model - {self.args.model_name} | Activation - {self.args.activation} | Seed - {self.args.seed}')
            logging.info(f'Training starts at - {datetime.datetime.now()} | Device - {self.args.device}')
            logging.info(f'The config is - \n{self.args}')
        
        DEVICE = self.args.device
        set_seed(self.args.seed, self.args.device_type)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # Get batch size per GPU
        batch_size_per_gpu = int(self.args.batch_size / self.args.world_size)
        if self.args.master_process:
            logging.info(f'Provided batch size - {self.args.batch_size} | World Size - {self.args.world_size} | Batch Size per GPU - {batch_size_per_gpu}')
        
        self.args.save_dir = f'{self.args.save_dir}/{self.args.dataset}/{self.args.model_name}/{self.args.activation}'
        self.args.save_sample_dir = f'{self.args.save_dir}/{self.args.seed}'
        self.args.checkpoint_path = f'{self.args.save_dir}/checkpoint_{self.args.seed}.pth'
        self.args.opt_trace_path = f'{self.args.save_dir}/opt_trace_{self.args.seed}.json'
        self.args.results_path = f'{self.args.save_dir}/results_{self.args.seed}.json'
        self.args.plots_path = f'{self.args.save_dir}/plot_{self.args.seed}.png'
        create_directory(self.args.save_sample_dir)
        
        train_dl, val_dl = get_train_and_val_dls_ddp(
            data_dir=self.args.data_dir,
            img_size=self.args.img_size,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers
        )
        trainer = Trainer(
            args=self.args,
            train_dl=train_dl,
            val_dl=val_dl,
            device=DEVICE,
        )

        net = UNet()
        net = replace_activation_by_torch_module(net, nn.SiLU, self.args.activation)
        model = DDPM(model=net, img_size=self.args.img_size, device=DEVICE)
        model.to(DEVICE)
        if self.args.master_process:
            num_parameters(model)
        model_without_ddp = model
        model = DDP(model, device_ids=[self.args.rank])
        model_without_ddp = model.module
        optim = AdamW(model.parameters(), lr=self.args.lr)
        scaler = get_grad_scaler(device=DEVICE)

        trainer.train(
            n_epochs=self.args.n_epochs,
            model=model,
            model_without_ddp=model_without_ddp,
            optim=optim,
            scaler=scaler,
            n_warmup_steps=self.args.n_warmup_steps,
        )

        self.cleanup()


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO)
    ddp = DistDataParallel(args)
    ddp.main_worker()


if __name__ == "__main__":
    main()