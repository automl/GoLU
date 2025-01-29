"""
The following code has been taken from - https://github.com/KimRass/DDPM/blob/main/eval.py
"""

# References:
    # https://wandb.ai/wandb_fc/korean/reports/-Frechet-Inception-distance-FID-GANs---Vmlldzo0MzQ3Mzc
    # https://m.blog.naver.com/chrhdhkd/222013835684
    # https://notou10.github.io/deep%20learning/2021/05/31/FID.html

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import scipy
from tqdm import tqdm
import math
import argparse

from tasks.diffusion.inceptionv3 import InceptionV3
from tasks.diffusion.data import CelebADS, ImageGridDataset


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--real_data_dir", type=str, required=True)
    parser.add_argument("--gen_data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_eval_imgs", type=int, required=True)

    parser.add_argument("--n_cpus", type=int, required=False, default=0)
    parser.add_argument("--padding", type=int, required=False, default=1)
    parser.add_argument("--n_cells", type=int, required=False, default=100)

    args = parser.parse_args()
    return args


def get_matrix_sqrt(x):
    conv_mean = scipy.linalg.sqrtm(x)
    if np.iscomplexobj(conv_mean):
        conv_mean = conv_mean.real
    return conv_mean


def get_mean_and_cov(embed):
    mu = embed.mean(axis=0)
    sigma = np.cov(embed, rowvar=False)
    return mu, sigma


def get_frechet_distance(mu1, mu2, sigma1, sigma2):
    cov_mean = get_matrix_sqrt(sigma1 @ sigma2)
    fd = ((mu1 - mu2) ** 2).sum() + np.trace(sigma1 + sigma2 - 2 * cov_mean)
    return fd.item()


def get_fid(embed1, embed2):
    mu1, sigma1 = get_mean_and_cov(embed1)
    mu2, sigma2 = get_mean_and_cov(embed2)
    fd = get_frechet_distance(mu1=mu1, mu2=mu2, sigma1=sigma1, sigma2=sigma2)
    return fd


def get_inception_score(prob, eps=1e-16):
    p_yx = prob # $p(y|x)$
    p_y = p_yx.mean(axis=0, keepdims=True) # $p(y)$
    kld = p_yx * np.log((p_yx + eps) / (p_y + eps)) # $p(y|x)\log(P(y|x) / P(y))$
    sum_kld = kld.sum(axis=1)
    avg_kld = sum_kld.mean()
    inception_score = np.exp(avg_kld)
    return inception_score


def get_dls(real_data_dir, gen_data_dir, batch_size, img_size, n_cpus, n_cells, padding):
    real_ds = CelebADS(data_dir=real_data_dir, img_size=img_size)
    real_dl = DataLoader(
        real_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpus,
        pin_memory=False,
        drop_last=False,
    )
    gen_ds = ImageGridDataset(
        data_dir=gen_data_dir,
        img_size=img_size,
        n_cells=n_cells,
        padding=padding,
    )
    gen_dl = DataLoader(
        gen_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpus,
        pin_memory=False,
        drop_last=False,
    )
    return real_dl, gen_dl


class Evaluator(object):
    def __init__(self, ddpm, n_eval_imgs, batch_size, real_dl, gen_dl, mode, device):

        self.ddpm = ddpm
        self.n_eval_imgs = n_eval_imgs
        self.batch_size = batch_size
        self.real_dl = real_dl
        self.gen_dl = gen_dl
        self.mode = mode
        self.device = device

        self.ddpm.eval()

        self.model1 = InceptionV3(output_blocks=[3]).to(device)
        if mode in ["is", "both"]:
            self.model2 = InceptionV3(output_blocks=[3, 4]).to(device)
        else:
            self.model2 = self.model1
        self.model1.eval()
        self.model2.eval()

        self.process_real_dl()

    @torch.no_grad()
    def process_real_dl(self):
        embeds = list()
        gen_di = iter(self.gen_dl)
        for _ in tqdm(range(math.ceil(self.n_eval_imgs / self.batch_size))):
            x0 = next(gen_di)
            x0 = x0.to(self.device)

            out = self.model1(x0.detach())
            embed = out[0]
            embeds.append(embed.squeeze().detach().cpu().numpy())
        self.real_embed = np.concatenate(embeds)[: self.n_eval_imgs]

    @torch.no_grad()
    def process_gen_dl(self):
        embeds = list()
        probs = list()
        real_di = iter(self.real_dl)
        for _ in tqdm(range(math.ceil(self.n_eval_imgs / self.batch_size))):
            x0 = next(real_di)
            x0 = x0.to(self.device)

            out = self.model2(x0.detach())
            embed = out[0]
            embeds.append(embed.squeeze().detach().cpu().numpy())

            if self.mode in ["is", "both"]:
                logit = out[1]
                prob = F.softmax(logit, dim=1)
                probs.append(prob.detach().cpu().numpy())
        gen_embed = np.concatenate(embeds)[: self.n_eval_imgs]
        if self.mode in ["is", "both"]:
            gen_prob = np.concatenate(probs)[: self.n_eval_imgs]
        return gen_embed, gen_prob if self.mode in ["is", "both"] else gen_embed

    def evaluate(self):
        gen_embed, gen_prob = self.process_gen_dl()
        fid = get_fid(self.real_embed, gen_embed)
        print(f"[ FID: {fid:.2f} ]")
        if self.mode in ["is", "both"]:
            inception_score = get_inception_score(gen_prob)
            print(f"[ IS: {inception_score:.2f} ]")

        if self.mode in ["is", "both"]:
            return fid, inception_score
        else:
            return fid
