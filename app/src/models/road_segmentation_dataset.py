"""Road Segmentation Dataset.

Dataset class for the road segmentation task.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from maikol_utils.file_utils import list_dir_files
from src.config import ModelConfiguration
from src.data import PipeType, SampleImage, apply_pipeline
from src.utils import split_seed
from torch.utils.data import Dataset


class RoadSegmentationDataset(Dataset):
    """Dataset class for road segmentation task with optional data augmentation.

    This class loads image-mask pairs from a specified directory, applies optional
    data augmentation pipelines, and prepares the data for training a segmentation model.
    """

    def __init__(
        self, root_dir: str, M_CONFIG: ModelConfiguration, pipelines: list[PipeType] = None
    ):
        self.augmentation_chance = M_CONFIG.augmentation_chance
        self.pipelines = pipelines

        original_files, n = list_dir_files(
            root_dir,
            nat_sorting=True,
            absolute_path=True,
            recursive=True,
        )
        path_images_X = [img for img in original_files if "_sat" in img][: M_CONFIG.max_samples]
        path_images_Y = [img for img in original_files if "_mask" in img][: M_CONFIG.max_samples]

        self.sample_points: list[SampleImage] = [
            SampleImage(path_img_x, path_img_y)
            for path_img_x, path_img_y in zip(path_images_X, path_images_Y)
        ]
        self.N = len(self.sample_points)
        self.seeds = split_seed(seed=M_CONFIG.seed, n=self.N)
        print(f"Dataset initialized with {self.N} samples from {root_dir}")

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.N

    def __getitem__(self, idx):
        """Retrieve the image and mask for the given index, applying augmentation if specified."""
        # If augmentation is to be applied, select a random pipeline
        # Else just load the original image and mask
        if self.pipelines is not None and np.random.rand() < self.augmentation_chance:
            seed = self.seeds[idx]
            np.random.seed(seed)
            pipe = self.pipelines[np.random.randint(0, len(self.pipelines))]

            x, y = apply_pipeline(self.sample_points[idx], pipe, seed)
        else:
            x, y = self.sample_points[idx].get_images(keep_in_memory=False)

        # Convert to tensors and normalize to [0, 1]
        # Make sure the channels are aligned with the models library (C, H, W)
        x = torch.tensor(np.array(x)).permute(2, 0, 1).float() / 255.0
        y = torch.tensor(np.array(y)).unsqueeze(0).float() / 255.0

        return x, y

    def plot_sample(self, idx: int):
        """Plot the image and mask for the given index."""
        x, y = self[idx]  # uses __getitem__

        # Convert tensors back to numpy for plotting
        img_np = x.permute(1, 2, 0).numpy()
        mask_np = y.squeeze(0).numpy()  # remove channel dim

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_np)
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(mask_np, cmap="gray")
        axes[1].set_title("Mask")
        axes[1].axis("off")

        plt.show()
