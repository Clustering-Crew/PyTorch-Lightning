import os
import argparse
from glob import glob
import numpy as np
from PIL import Image
import torch
import pytorch_lightning as pl
from config import get_config
from dataset import LitDataModule
from model import CustomModel
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import RichProgressBar, EarlyStopping


def dataset_sanity_check(data_dir):
    valid_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.gif"]

    paths = []

    for ext in valid_extensions:
        pattern = os.path.join(data_dir, "**", ext)
        paths.extend(glob(pattern, recursive=True))

    for path in paths:
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            print(f"Error - {e}")
            os.remove(path)


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    opt = get_config()

    # Define and create the folder system to save the training and validation losses and curves.
    save_dir = os.path.join(opt.save_dir, opt.name)
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)

    # Do a dataset sanity check
    print("Starting Dataset Sanity Check...")
    dataset_sanity_check(opt.data_dir)
    print("Ending Dataset Sanity Check...")

    # Load the Pytorch lightning datamodule
    datamodule = LitDataModule(
        data_dir=opt.data_dir,
        input_shape=opt.imgsz,
        batch=opt.batch,
        test_split=opt.test_split,
        val_split=opt.val_split,
    )
    datamodule.setup()
    # Load the dataloaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    # Define the callbacks
    callback = [
        EarlyStopping(monitor="val_loss", patience=10),
        RichProgressBar(leave=True),
    ]

    # Define the csv logger
    logger = CSVLogger(
        save_dir=os.path.join(save_dir, "logs"),
        name=f"{opt.name}_lightning_logs",
    )

    # Define the model
    model = CustomModel(
        num_classes=opt.num_classes,
        average=opt.average,
        epochs=opt.epochs,
        lr=opt.lr,
        device=device
    )

    # Define the trainer
    trainer = pl.Trainer(
        accelerator="gpu", max_epochs=opt.epochs, callbacks=callback, logger=logger
    )

    if opt.tune:
        trainer.tune()

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model=model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
