import os
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.datasets as datasets
import pytorch_lightning as pl
import torchvision.transforms as T


def is_valid_file(path):
    valid_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"]
    _, ext = os.path.splitext(path)

    if ext.lower not in valid_extensions:
        return False

    try:
        with Image.open(path) as img:
            img.verify()
            img.load()
        return True
    except Exception as e:
        print(f"Invalid file: {path}, error - {e}")
        return False

# Custom PyTorch Lightning datamodule
class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        input_shape,
        batch,
        test_split,
        val_split,
        transform=None,
    ):
        self.data_dir = data_dir
        self.batch = batch
        self.test_split = test_split
        self.val_split = val_split

        # If no transform is provided then apply the default transform
        if transform is None:
            self.transform = T.Compose(
                [T.Resize((input_shape, input_shape)), T.ToTensor()]
            )
        else:
            self.transform = transform

    def setup(self, stage=None):
        self.full_dataset = datasets.ImageFolder(
            root=self.data_dir, transform=self.transform, is_valid_file=is_valid_file
        )

        self.train_ds, self.val_ds, self.test_ds = random_split(
            self.full_dataset, [0.7, 0.2, 0.1]
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch)

    def val_dataloader(self):
        return DataLoader(self.train_ds, shuffle=False, batch_size=self.batch)

    def test_dataloader(self):
        return DataLoader(self.train_ds, shuffle=False, batch_size=self.batch)
