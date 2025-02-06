import os
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import pytorch_lightning as pl
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score


class CustomModel(pl.LightningModule):
    def __init__(self, num_classes, average, epochs, lr, device):
        super().__init__()
        self.lr = lr
        self.epochs = epochs
        self.num_classes = num_classes
        self.d = device
        self.average = average  # Averaging technique
        self.task_type = "multiclass" if num_classes > 2 else "binary"
        self.accuracy = Accuracy(self.task_type, num_classes=self.num_classes)
        self.precision = Precision(
            self.task_type, num_classes=self.num_classes, average=self.average
        )
        self.recall = Recall(
            self.task_type, num_classes=self.num_classes, average=self.average
        )

        self.loss_fn = nn.CrossEntropyLoss()

        # Load the Resnet50 model from model zoo
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Gradient calc not needed for frozen weights
        for param in self.model.parameters():
            param.requires_grad = False

        # Get the shape of the features out from the feature extractor (Flattened)
        self.num_features = self.model.fc.in_features

        # Fully Connected or classification head
        self.model.fc = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes),
        )
        
        # for name, param in self.model.named_parameters():
        #     print(name, param.requires_grad)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.d).type(torch.float32), y.to(self.d).type(torch.float32)
        logits = self(x)
        logits = torch.argmax(logits, dim=1).type(torch.float32)
        train_loss = self.loss_fn(logits, y)
        train_loss.requires_grad = True
        self.log(
            name="train_loss",
            value=train_loss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.d).type(torch.float32), y.to(self.d).type(torch.float32)

        logits = self(x)
        logits = torch.argmax(logits, dim=1).type(torch.float32)
        val_loss = self.loss_fn(logits, y)
        val_loss.requires_grad = True
        val_acc = self.accuracy(logits, y)
        val_precision = self.precision(logits, y)
        val_recall = self.recall(logits, y)

        self.log_dict(
            {
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
            },
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        # self.log(
        #     name="val_loss",
        #     value=val_loss,
        #     prog_bar=True,
        #     logger=True,
        #     on_step=False,
        #     on_epoch=True,
        # )

        return val_loss, val_acc, val_precision, val_recall

    def test_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.d).type(torch.float32), y.to(self.d).type(torch.float32)
        logits = self(x)
        logits = torch.argmax(logits, dim=1).type(torch.float32)
        test_acc = self.accuracy(logits, y)
        test_precision = self.precision(logits, y)
        test_recall = self.recall(logits, y)

        self.log_dict(
            {
                "test_accuracy": test_acc,
                "test_precisio": test_precision,
                "test_recall": test_recall,
            },
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=[0.5 * self.epochs, 0.75 * self.epochs],
                gamma=0.1,
            ),
        }
