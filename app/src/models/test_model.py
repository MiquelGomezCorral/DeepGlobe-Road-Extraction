"""Test model.

Functions to test the model.
"""


import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import auc, precision_recall_curve
from src.utils import get_device
from torch.utils.data import DataLoader


def test_model(model: pl.LightningModule, dataloader: DataLoader):
    """Aply test for a model with the dataloader.

    Args:
        model (pl.LightningModule): Trained model to be evaluated
        dataloader (DataLoader): Dataloader for testing

    Returns:
        all_metrics (dict): mean_metrics -> Mean metrics over all batches
    """
    model.eval()
    all_metrics = []
    for batch in dataloader:
        x, y = batch
        with torch.no_grad():
            logits = model(x.to(get_device()))
        metrics = compute_segmentation_metrics(logits.cpu(), y.cpu(), threshold=0.5)
        all_metrics.append(metrics)

    # ================== Aggregate all metrics ==================
    mean_metrics = {}
    numeric_keys = [
        "IoU_per_image",
        "Dice_per_image",
        "Precision_per_image",
        "Recall_per_image",
        "PixelAcc_per_image",
    ]

    for k in numeric_keys:
        vals = [m[k] for m in all_metrics if k in m]
        if vals:
            mean_metrics[k.replace("_per_image", "_mean")] = float(np.mean(np.concatenate(vals)))

    # Add PR_AUC mean if exists
    pr_aucs = [m["PR_AUC"] for m in all_metrics if m.get("PR_AUC") is not None]
    if pr_aucs:
        mean_metrics["PR_AUC_mean"] = float(np.mean(pr_aucs))

    return mean_metrics


def compute_segmentation_metrics(logits, targets, threshold=0.5, reduce="mean"):
    """Compute segmentation metrics.

    Args:
        logits (torch.Tensor), shape [B,1,H,W] or [B,H,W] (raw logits)
        targets (torch.Tensor), shape [B,1,H,W] or [B,H,W] (binary 0/1 or 0/255)

    Returns:
        dict with IoU, Dice, Precision, Recall, PixelAcc, PR_AUC
    """
    EPS = 1e-7

    # normalize shapes: [B,H,W]
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)
    # Ensure logits and targets are tensors on CPU
    probs = torch.sigmoid(logits).detach().cpu()
    targets = targets.detach().cpu().float()

    # If spatial shapes differ (e.g., model upsamples/downsamples), resize probs to match targets
    # probs: [B, H1, W1], targets: [B, H2, W2]
    if probs.ndim == 3 and targets.ndim == 3 and probs.shape[1:] != targets.shape[1:]:
        import torch.nn.functional as F

        # add channel dim for interpolate: [B,1,H,W]
        probs = probs.unsqueeze(1)
        probs = F.interpolate(probs, size=targets.shape[1:], mode="bilinear", align_corners=False)
        probs = probs.squeeze(1)
    # if masks in 0/255
    if targets.max() > 1.5:
        targets = (targets / 255.0).clamp(0, 1)

    B = probs.shape[0]

    ious = []
    dices = []
    precisions = []
    recalls = []
    pixel_accs = []
    # For PR AUC compute per-batch flattened PR curve then AUC of precision-recall
    # Make sure shapes match before flattening
    if probs.numel() == 0 or targets.numel() == 0 or probs.numel() != targets.numel():
        pr_auc = None
    else:
        flat_probs = probs.view(-1).numpy()
        flat_targets = targets.view(-1).numpy()
        # avoid degenerate PR AUC when no positives
        pr_auc = None
        if flat_targets.sum() > 0:
            prec, rec, _ = precision_recall_curve(flat_targets, flat_probs)
            pr_auc = auc(rec, prec)

    for i in range(B):
        p = (probs[i] > threshold).float()
        t = targets[i].float()

        tp = (p * t).sum().item()
        fp = (p * (1 - t)).sum().item()
        fn = ((1 - p) * t).sum().item()
        tn = ((1 - p) * (1 - t)).sum().item()

        iou = tp / (tp + fp + fn + EPS)
        dice = (2 * tp) / (2 * tp + fp + fn + EPS)
        prec = tp / (tp + fp + EPS)
        rec = tp / (tp + fn + EPS)
        acc = (tp + tn) / (tp + fp + fn + tn + EPS)

        ious.append(iou)
        dices.append(dice)
        precisions.append(prec)
        recalls.append(rec)
        pixel_accs.append(acc)

    out = {
        "IoU_per_image": np.array(ious),
        "Dice_per_image": np.array(dices),
        "Precision_per_image": np.array(precisions),
        "Recall_per_image": np.array(recalls),
        "PixelAcc_per_image": np.array(pixel_accs),
        # aggregated
        "IoU_mean": float(np.mean(ious)) if reduce == "mean" else None,
        "Dice_mean": float(np.mean(dices)) if reduce == "mean" else None,
        "Precision_mean": float(np.mean(precisions)) if reduce == "mean" else None,
        "Recall_mean": float(np.mean(recalls)) if reduce == "mean" else None,
        "PixelAcc_mean": float(np.mean(pixel_accs)) if reduce == "mean" else None,
        "PR_AUC": float(pr_auc) if pr_auc is not None else None,
    }
    return out
