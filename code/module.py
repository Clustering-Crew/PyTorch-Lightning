import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import random_split
import pytorch_lightning as pl
from torch.


# Skeleton of Lightning module based models
# Constructor
# Forward method
# Training step
# Validation step
# Testing step
# Get optimizer

class NN(pl.LightningModule):
    # Constructor
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        # Custom model
        self.fc1 = nn.Linear(input_shape, 256)
        self.fc2 = nn.Linear(256, 64)
        self.ouptut = nn.Linear(64, num_classes)

    # Forward method
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.ouptut(x)
        return x
    
    # Training step 
    def training_step(self, batch, batch_idx):
        '''
            Helps integrate with Trainer
        '''
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = F.cross_entropy(scores, y)
        return loss
    
    # Validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = F.cross_entropy(scores, y)
        return loss
    
    # Configure optimizer and scheduler
    def configure_optimizer(self):
        return optim.Adam(self.parameters(), lr=0.001)


def main():
    
    entire_dataset = datasets.MNIST(
        root="dataset/",
        train=True,
        transform=T.ToTensor(),
        download=True,
    )

    train_ds, val_ds = random_split(entire_dataset, [50000, 10000])

    test_ds = datasets.MNIST(
        root="dataset/",
        train=False,
        transform=T.ToTensor(),
        download=True,
    )

    model = NN((256, 256, 3), 2).to(device)
    
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=0,
        min_epochs=1,
        max_epochs=3,
        precision=16,
    )
    trainer.fit(
        model,
        train_dataloader,
        val_dataloader
    )

if __name__ == "__main__":
    main()

