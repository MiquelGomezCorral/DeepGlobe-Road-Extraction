"""Road segmentation model.

Road Segmentation Model Lightning wrapper for Segmentation Models PyTorch.
"""
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from src.config import Configuration
from torch.optim import lr_scheduler


class RoadSegmentationModel(pl.LightningModule):
    """Road Segmentation Model Lightning wrapper for Segmentation Models PyTorch."""

    def __init__(self, CONFIG: Configuration, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch=CONFIG.architecture,
            encoder_name=CONFIG.encoder_name,
            in_channels=CONFIG.in_channels,
            classes=CONFIG.out_classes,
            **kwargs,
        )
        self.lr = CONFIG.learning_rate
        self.t_max = CONFIG.max_steps
        self.eta_min = CONFIG.learning_rate / 1000

        # Select loss function
        if CONFIG.loss_function == "DiceLoss":
            self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        elif CONFIG.loss_function == "BCEWithLogitsLoss":
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
        elif CONFIG.loss_function == "BCEDice":
            self.loss_fn = (
                lambda pred, target: (
                    torch.nn.BCEWithLogitsLoss()(pred, target)
                    + smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)(pred, target)
                )
                / 2
            )
        else:
            raise ValueError(f"Unknown loss function: {CONFIG.loss_function}")

    def forward(self, x):
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Do Training step."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        """Do Validation step."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.t_max, eta_min=self.eta_min
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
