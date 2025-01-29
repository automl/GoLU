"""
The following code has been taken from - https://github.com/KimRass/DDPM/blob/main/data.py
"""

# Source: https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ
# References:
    # https://github.com/KimRass/DCGAN/blob/main/celeba.py

from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as T
from torchvision.datasets import CelebA
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
import numpy as np
from pathlib import Path

from tasks.utils.utils import set_worker_sharing_strategy


# The CelebA dataset has 202599 images of size 178x218 pixels
# Train-Set Size - 162770 | Val-Set Size - 19867 | Test-Set Size - 19962

class CelebADS(Dataset):
    def __init__(self, data_dir, split, img_size, hflip):
        self.ds = CelebA(root=data_dir, split=split, download=True)

        transforms = [
            A.HorizontalFlip(p=0.5),
            A.SmallestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
            A.CenterCrop(height=img_size, width=img_size),
            # "We assume that image data consists of integers in $\{0, 1, \ldots, 255\}$ scaled linearly
            # to $[-1, 1]$. This ensures that the neural network reverse process operates
            # on consistently scaled inputs starting from the standard normal prior $p(x_{T})$."
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
        if not hflip:
            transforms = transforms[1:]
        self.transform = A.Compose(transforms)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, _ = self.ds[idx]
        return self.transform(image=np.array(image))["image"]


def get_train_and_val_dls(data_dir, img_size, batch_size, num_workers):
    train_ds = CelebADS(data_dir=data_dir, split="train", img_size=img_size, hflip=True)
    val_ds = CelebADS(data_dir=data_dir, split="valid", img_size=img_size, hflip=False)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        num_workers=num_workers,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        num_workers=num_workers,
    )
    return train_dl, val_dl


def get_test_dl(data_dir, img_size, batch_size, num_workers):
    test_ds = CelebADS(data_dir=data_dir, split="test", img_size=img_size, hflip=False)
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=True,
        persistent_workers=False,
        num_workers=num_workers,
    )


def get_train_and_val_dls_ddp(
    data_dir, img_size, batch_size, num_workers
):
    train_ds = CelebADS(data_dir=data_dir, split="train", img_size=img_size, hflip=True)
    val_ds = CelebADS(data_dir=data_dir, split="valid", img_size=img_size, hflip=False)
    train_sampler = DistributedSampler(
        train_ds, shuffle=True,
    )
    val_sampler = DistributedSampler(
        val_ds, shuffle=False,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        num_workers=num_workers,
        worker_init_fn=set_worker_sharing_strategy
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        num_workers=num_workers,
        worker_init_fn=set_worker_sharing_strategy
    )
    return train_dl, val_dl


def get_test_dl(data_dir, img_size, batch_size, num_workers):
    test_ds = CelebADS(data_dir=data_dir, split="test", img_size=img_size, hflip=False)
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=True,
        persistent_workers=False,
        num_workers=num_workers,
        worker_init_fn=set_worker_sharing_strategy
    )


class ImageGridDataset(Dataset):
    def __init__(self, data_dir, img_size, n_cells=100, padding=1):
        super().__init__()

        self.img_paths = sorted(list(Path(data_dir).glob("**/*.jpg")))
        self.img_size = img_size
        self.padding = padding
        self.n_cells = n_cells

        self.transformer = T.Compose(
            [T.ToTensor(), T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))],
        )

    def __len__(self):
        return len(self.img_paths) * self.n_cells

    def _idx_to_dimension(self, idx):
        return self.padding * (idx + 1) + self.img_size * idx

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx // self.n_cells]).convert("RGB")
        image = self.transformer(image)
        row_idx = (idx % self.n_cells) // int((self.n_cells ** 0.5))
        col_idx = (idx % self.n_cells) % int((self.n_cells ** 0.5))
        return image[
            :,
            self._idx_to_dimension(row_idx): self._idx_to_dimension(row_idx) + self.img_size,
            self._idx_to_dimension(col_idx): self._idx_to_dimension(col_idx) + self.img_size,
        ]
