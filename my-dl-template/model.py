import json

import lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR


class myModel(pl.LightningModule):
    """
    A base class to implement a Pytorch Lightning Model.
    Hyper-parameters are specified in a separate config.json file.
    """

    def __init__(
        self,
        config_path,
    ):
        super().__init__()

        # Load configuration file
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # The actual model
        self.model = nn.Identity()

    def forward(self, idx, targets=None) -> dict:
        """
        Implements the forward pass.
        If multiple components are required (e.g. encoder-decoder), combine them here.
        """

        # Required output
        logits, loss = self.model(idx, targets)

        return logits, loss

    def _common_step(self, batch, batch_idx):
        """
        Handles one step of the model, from reading in the batch to returning the loss.
        """
        # Read in the batch
        X, y = batch

        # Forward pass
        logits, loss = self.forward(X, y)

        return loss

    def training_step(self, batch, batch_idx):
        # Get the loss
        loss = self._common_step(batch, batch_idx)

        # Get model state
        current_lr = self.optimizers().param_groups[0]["lr"]

        # Logging
        self.log_dict({
            "train/loss": loss,
            "learning_rate": current_lr,
        }, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Get the loss
        loss = self._common_step(batch, batch_idx)

        # Logging
        self.log("valid/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer and the learning rate scheduling policy.
        """

        # Optimizer
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=0.01,  #  Reset to max_lr / div_factor by the OneCyleLR
        )

        # The 1cycle policy (warm-up + annealing)
        # use default params
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config["max_lr"],
            total_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,  # Check after each step
            },
        }

    @torch.no_grad()
    def generate(self):
        pass
