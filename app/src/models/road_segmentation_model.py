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
            CONFIG.architecture,
            encoder_name=CONFIG.encoder_name,
            in_channels=CONFIG.in_channels,
            classes=CONFIG.out_classes,
            **kwargs,
        )
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.lr = CONFIG.learning_rate
        self.t_max = CONFIG.max_steps
        self.eta_min = CONFIG.learning_rate / 1000

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
