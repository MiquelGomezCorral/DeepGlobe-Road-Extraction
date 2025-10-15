"""Refactor for training model script.

Functions used to train models.
"""
import math
import os

import matplotlib.pyplot as plt
import torch
from maikol_utils.print_utils import print_log
from PIL import Image
from src.config import Configuration
from src.data import AUG_PIPELINES
from src.models import RoadSegmentationDataset, RoadSegmentationModel
from src.utils import get_device, to_device
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_data_loaders(CONFIG: Configuration):
    """Get data loaders from CONFIG for training, validation, and test datasets.

    Args:
        CONFIG (Configuration): Configuration

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]:
            train_dataloader, valid_dataloader, test_dataloader
    """
    train_dataset = RoadSegmentationDataset(
        CONFIG.train_folder, CONFIG, AUG_PIPELINES[CONFIG.augmentation_set]
    )
    valid_dataset = RoadSegmentationDataset(CONFIG.val_folder, CONFIG)
    test_dataset = RoadSegmentationDataset(CONFIG.test_folder, CONFIG)

    print_log(f" - Train samples: {len(train_dataset):8_}")
    print_log(f" - Val samples:   {len(valid_dataset):8_}")
    print_log(f" - Test samples:  {len(test_dataset):8_}")

    n_cpu = max(os.cpu_count() // 2, 1)
    train_dataloader = DataLoader(
        train_dataset, batch_size=CONFIG.batch_size, shuffle=True, num_workers=n_cpu
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=CONFIG.batch_size, shuffle=False, num_workers=n_cpu
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=CONFIG.batch_size, shuffle=False, num_workers=n_cpu
    )

    return train_dataloader, valid_dataloader, test_dataloader


def visualize_model_predictions(
    CONFIG: Configuration,
    model: RoadSegmentationModel,
    test_dataloader: RoadSegmentationDataset,
    idx: int = 0,
    max_samples: int = 20,
    cols_per_row: int = 5,
    show: bool = False,
):
    """Visualize model predictions on test dataset.

    Args:
        model (RoadSegmentationModel): The trained model for inference.
        test_dataloader (RoadSegmentationDataset): DataLoader for the test dataset.
        max_samples (int, optional): Maximum number of samples to visualize. Defaults to 20.
        cols_per_row (int, optional): Number of columns per row in the visualization. Defaults to 5.
    """
    print_log(" - Visualizing model predictions...")
    # Collect predictions
    all_images, all_masks, all_preds = [], [], []
    for batch in tqdm(test_dataloader, desc="Processing batches"):
        images, masks = batch
        images, masks = to_device(images), to_device(masks)
        model = model.to(get_device()).eval()
        with torch.inference_mode():
            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).float()

        all_images.append(images.cpu())
        all_masks.append(masks.cpu())
        all_preds.append(preds.cpu())
        if sum(img.shape[0] for img in all_images) >= max_samples:
            break

    # Flatten and limit
    all_images = torch.cat(all_images)[:max_samples]
    all_masks = torch.cat(all_masks)[:max_samples]
    all_preds = torch.cat(all_preds)[:max_samples]

    n_samples = all_images.shape[0]
    n_rows = math.ceil(n_samples / cols_per_row)

    fig, axes = plt.subplots(n_rows * 3, cols_per_row, figsize=(3 * cols_per_row, 3 * n_rows * 3))

    for i in range(n_samples):
        img_np = all_images[i].permute(1, 2, 0).numpy()
        mask_np = all_masks[i].squeeze(0).numpy()
        pred_np = all_preds[i].squeeze(0).numpy()

        row_block = (i // cols_per_row) * 3
        col = i % cols_per_row

        axes[row_block, col].imshow(img_np)
        axes[row_block, col].axis("off")
        axes[row_block, col].set_title(f"Image {i}")

        axes[row_block + 1, col].imshow(mask_np, cmap="gray")
        axes[row_block + 1, col].axis("off")

        axes[row_block + 2, col].imshow(pred_np, cmap="gray")
        axes[row_block + 2, col].axis("off")

    plt.tight_layout()
    save_path = os.path.join(
        CONFIG.log_folder, f"model_predictions_{model.__class__.__name__}{idx}.png"
    )
    plt.savefig(save_path)
    if show:
        plt.show()

    plt.close()
    return save_path


def combine_prediction_plots(image_paths, output_name="combined.png", show=False):
    """Combine multiple prediction plots vertically into a single image."""
    images = [Image.open(p) for p in image_paths if os.path.exists(p)]
    if not images:
        raise ValueError("No valid images found to combine")

    widths, heights = zip(*(img.size for img in images))
    total_height = sum(heights)
    max_width = max(widths)

    combined = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))

    y_offset = 0
    for img in images:
        combined.paste(img, (0, y_offset))
        y_offset += img.height

    os.makedirs(os.path.dirname(output_name) or ".", exist_ok=True)
    combined.save(output_name)

    if show:
        plt.figure(figsize=(len(image_paths) * 20, len(image_paths) * 10))
        plt.imshow(combined)
        plt.axis("off")
        plt.title("Combined Predictions")
        plt.show()

    return output_name
