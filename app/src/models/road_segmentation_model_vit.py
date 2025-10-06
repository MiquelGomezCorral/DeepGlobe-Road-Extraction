"""Road segmentation VIT model.

Road Segmentation Model Lightning wrapper for a VIT Segmentation Model.
"""
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from src.config import Configuration
from torch.optim import lr_scheduler
from transformers import SegformerForSemanticSegmentation


class RoadSegmentationModelVIT(pl.LightningModule):
    """Road Segmentation Model Lightning wrapper for Segmentation Models PyTorch."""

    def __init__(self, CONFIG: Configuration, **kwargs):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",  # Pretrained VIT model for segmentation
            num_labels=CONFIG.out_classes,
            ignore_mismatched_sizes=True,
            **kwargs,
        )
        self.lr = CONFIG.learning_rate
        self.t_max = CONFIG.max_steps
        self.eta_min = CONFIG.learning_rate / 1000
        self.out_classes = CONFIG.out_classes

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
        """Do Forward pass."""
        outputs = self.model(pixel_values=x)
        return outputs.logits  # [B, num_labels, H, W]

    def training_step(self, batch, batch_idx):
        """Do Training step."""
        x, y = batch
        logits = self(x)

        # Resize logits if needed
        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)

        # Binary vs multiclass loss
        if self.out_classes == 1:
            if y.ndim == 3:
                y = y.unsqueeze(1)
            y = y.to(dtype=logits.dtype, device=logits.device)
            if y.max() > 1.5:
                y = (y / 255.0).clamp(0.0, 1.0)
        else:
            if y.ndim == 4:
                y = y.squeeze(1)
            y = y.long().to(device=logits.device)

        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Do Validation step."""
        x, y = batch
        logits = self(x)

        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)

        if self.out_classes == 1:
            if y.ndim == 3:
                y = y.unsqueeze(1)
            y = y.to(dtype=logits.dtype, device=logits.device)
            if y.max() > 1.5:
                y = (y / 255.0).clamp(0.0, 1.0)
        else:
            if y.ndim == 4:
                y = y.squeeze(1)
            y = y.long().to(device=logits.device)

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
