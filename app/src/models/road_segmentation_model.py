"""Road segmentation model.

Road Segmentation Model Lightning wrapper for Segmentation Models PyTorch.
"""
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from src.config import ModelConfiguration
from torch.optim import lr_scheduler


class RoadSegmentationModel(pl.LightningModule):
    """Road Segmentation Model Lightning wrapper for Segmentation Models PyTorch."""

    def __init__(self, M_CONFIG: ModelConfiguration, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            M_CONFIG.architecture,
            encoder_name=M_CONFIG.encoder_name,
            in_channels=M_CONFIG.in_channels,
            classes=M_CONFIG.out_classes,
            **kwargs,
        )
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.lr = M_CONFIG.learning_rate
        self.t_max = M_CONFIG.max_steps
        self.eta_min = M_CONFIG.learning_rate / 1000

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
