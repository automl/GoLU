"""
The following code has been taken from - https://github.com/KimRass/DDPM/blob/main/ddpm.py
"""

# References:
    # https://nn.labml.ai/diffusion/ddpm/index.html
    # https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/08_diffusion/01_ddm/ddm.ipynb
    # https://huggingface.co/blog/annotated-diffusion

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import imageio
from tqdm import tqdm

from tasks.diffusion.data import CelebADS
from tasks.diffusion.utils import image_to_grid


class DDPM(nn.Module):
    def get_linear_beta_schdule(self):
        # "We set the forward process variances to constants increasing linearly."
        # return torch.linspace(init_beta, fin_beta, n_diffusion_steps) # "$\beta_{t}$"
        self.beta = torch.linspace(
            self.init_beta,
            self.fin_beta,
            self.n_diffusion_steps,
            device=self.device,
        ) # "The forward process variances $\beta_{t}$"

    # "We set T = 1000 without a sweep."
    # "We chose a linear schedule from $\beta_{1} = 10^{-4}$ to  $\beta_{T} = 0:02$."
    def __init__(
        self,
        model,
        img_size,
        device,
        image_channels=3,
        n_diffusion_steps=1000,
        init_beta=0.0001,
        fin_beta=0.02,
    ):
        super().__init__()

        self.img_size = img_size
        self.device = device
        self.image_channels = image_channels
        self.n_diffusion_steps = n_diffusion_steps
        self.init_beta = init_beta
        self.fin_beta = fin_beta

        self.model = model.to(device)

        self.get_linear_beta_schdule()
        self.alpha = 1 - self.beta # "$\alpha_{t} = 1 - \beta_{t}$"
        # "$\bar{\alpha_{t}} = \prod^{t}_{s=1}{\alpha_{s}}$"
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    @staticmethod
    def index(x, diffusion_step):
        return x[diffusion_step][:, None, None, None]

    def sample_noise(self, batch_size):
        return torch.randn(
            size=(batch_size, self.image_channels, self.img_size, self.img_size),
            device=self.device,
        )

    def sample_diffusion_step(self, batch_size):
        return torch.randint(
            0, self.n_diffusion_steps, size=(batch_size,), device=self.device,
        )

    def batchify_diffusion_steps(self, diffusion_step_idx, batch_size):
        return torch.full(
            size=(batch_size,),
            fill_value=diffusion_step_idx,
            dtype=torch.long,
            device=self.device,
        )

    def perform_diffusion_process(self, ori_image, diffusion_step, rand_noise=None):
        # "$\bar{\alpha_{t}}$"
        alpha_bar_t = self.index(self.alpha_bar, diffusion_step=diffusion_step)
        mean = (alpha_bar_t ** 0.5) * ori_image # $\sqrt{\bar{\alpha_{t}}}x_{0}$
        var = 1 - alpha_bar_t # $(1 - \bar{\alpha_{t}})\mathbf{I}$
        if rand_noise is None:
            rand_noise = self.sample_noise(batch_size=ori_image.size(0))
        noisy_image = mean + (var ** 0.5) * rand_noise
        return noisy_image

    def forward(self, noisy_image, diffusion_step):
        # "where $\epsilon_{\theta}$ is a function approximator intended to predict $\epsilon$ from $x_{t}$."
        return self.model(noisy_image=noisy_image, diffusion_step=diffusion_step)

    def get_loss(self, ori_image):
        # "Algorithm 1-3: $t \sim Uniform(\{1, \ldots, T\})$"
        rand_diffusion_step = self.sample_diffusion_step(batch_size=ori_image.size(0))
        rand_noise = self.sample_noise(batch_size=ori_image.size(0))
        # "Algorithm 1-4: $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$"
        noisy_image = self.perform_diffusion_process(
            ori_image=ori_image,
            diffusion_step=rand_diffusion_step,
            rand_noise=rand_noise,
        )
        
        with torch.amp.autocast(enabled=True, dtype=torch.float16, device_type='cuda'):
            pred_noise = self(noisy_image=noisy_image, diffusion_step=rand_diffusion_step)
            return F.mse_loss(pred_noise, rand_noise, reduction="mean")

    @torch.inference_mode()
    def take_denoising_step(self, noisy_image, diffusion_step_idx):
        diffusion_step = self.batchify_diffusion_steps(
            diffusion_step_idx=diffusion_step_idx, batch_size=noisy_image.size(0),
        )
        alpha_t = self.index(self.alpha, diffusion_step=diffusion_step)
        beta_t = self.index(self.beta, diffusion_step=diffusion_step)
        alpha_bar_t = self.index(self.alpha_bar, diffusion_step=diffusion_step)
        pred_noise = self(noisy_image=noisy_image.detach(), diffusion_step=diffusion_step)
        # # "Algorithm 2-4:
        # $x_{t - 1} = \frac{1}{\sqrt{\alpha_{t}}}
        # \Big(x_{t} - \frac{\beta_{t}}{\sqrt{1 - \bar{\alpha_{t}}}}z_{\theta}(x_{t}, t)\Big)
        # + \sigma_{t}z"$
        model_mean = (1 / (alpha_t ** 0.5)) * (
            noisy_image - ((beta_t / ((1 - alpha_bar_t) ** 0.5)) * pred_noise)
        )
        # "At the end of sampling, we display $\mu_{\theta}(x_{1}, 1)$ noiselessly."
        model_var = beta_t # "$\sigma_{t}$"

        if diffusion_step_idx > 0:
            rand_noise = self.sample_noise(batch_size=noisy_image.size(0)) # "$z$"
        else:
            rand_noise = torch.zeros(
                size=(noisy_image.size(0), self.image_channels, self.img_size, self.img_size),
                device=self.device,
            )
        return model_mean + (model_var ** 0.5) * rand_noise

    @staticmethod
    def _get_frame(x):
        grid = image_to_grid(x, n_cols=int(x.size(0) ** 0.5))
        frame = np.array(grid)
        return frame

    def perform_denoising_process(self, noisy_image, start_diffusion_step_idx, n_frames=None):
        if n_frames is not None:
            frames = list()

        x = noisy_image
        pbar = tqdm(range(start_diffusion_step_idx, -1, -1), leave=False)
        for diffusion_step_idx in pbar:
            pbar.set_description("Denoising...")

            x = self.take_denoising_step(x, diffusion_step_idx=diffusion_step_idx)

            if n_frames is not None and (
                diffusion_step_idx % (self.n_diffusion_steps // n_frames) == 0
            ):
                frames.append(self._get_frame(x))
        return frames if n_frames is not None else x

    def sample(self, batch_size):
        rand_noise = self.sample_noise(batch_size=batch_size) # "$x_{T}$"
        return self.perform_denoising_process(
            noisy_image=rand_noise,
            start_diffusion_step_idx=self.n_diffusion_steps - 1,
            n_frames=None,
        )

    def vis_denoising_process(self, batch_size, save_path, n_frames=100):
        rand_noise = self.sample_noise(batch_size=batch_size)
        frames = self.perform_denoising_process(
            noisy_image=rand_noise,
            start_diffusion_step_idx=self.n_diffusion_steps - 1,
            n_frames=n_frames,
        )
        imageio.mimsave(save_path, frames)

    def _get_ori_images(self, data_dir, image_idx1, image_idx2):
        test_ds = CelebADS(
            data_dir=data_dir, split="test", img_size=self.img_size, hflip=False,
        )
        ori_image1 = test_ds[image_idx1][None, ...].to(self.device)
        ori_image2 = test_ds[image_idx2][None, ...].to(self.device)
        return ori_image1, ori_image2

    def _get_linearly_interpolated_image(self, x, y, n_points):
        weight = torch.linspace(
            start=0, end=1, steps=n_points, device=self.device,
        )[:, None, None, None]
        return (1 - weight) * x + weight * y

    def interpolate(self, data_dir, image_idx1, image_idx2, interpolate_at=500, n_points=10):
        ori_image1, ori_image2 = self._get_ori_images(
            data_dir=data_dir, image_idx1=image_idx1, image_idx2=image_idx2,
        )

        diffusion_step = self.batchify_diffusion_steps(interpolate_at, batch_size=1)
        noisy_image1 = self.perform_diffusion_process(
            ori_image=ori_image1, diffusion_step=diffusion_step,
        )
        noisy_image2 = self.perform_diffusion_process(
            ori_image=ori_image2, diffusion_step=diffusion_step,
        )

        x = self._get_linearly_interpolated_image(noisy_image1, noisy_image2, n_points=n_points)
        denoised_image = self.perform_denoising_process(
            noisy_image=x,
            start_diffusion_step_idx=interpolate_at,
            n_frames=None,
        )
        return torch.cat([ori_image1, denoised_image, ori_image2], dim=0)

    def coarse_to_fine_interpolate(self, data_dir, image_idx1, image_idx2, n_rows=9, n_points=10):
        rows = list()
        pbar = tqdm(
            range(
                self.n_diffusion_steps - 1,
                -1,
                - self.n_diffusion_steps // (n_rows - 1),
            ),
            leave=False,
        )
        for interpolate_at in pbar:
            pbar.set_description("Coarse to fine interpolating...")

            row = self.interpolate(
                data_dir=data_dir,
                image_idx1=image_idx1,
                image_idx2=image_idx2,
                interpolate_at=interpolate_at,
                n_points=n_points,
            )
            rows.append(row)
        return torch.cat(rows, dim=0)
