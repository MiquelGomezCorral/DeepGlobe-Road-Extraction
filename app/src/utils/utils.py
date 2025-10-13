"""Project utility functions.

This module contains various utility functions that are used throughout the project.
"""
import os
import random
import re

import numpy as np
import torch
from maikol_utils.file_utils import list_dir_files
from maikol_utils.print_utils import print_error, print_log
from PIL import Image
from src.config import Configuration


class PathParser:
    """Class to parse model path and extract configuration."""

    pattern = re.compile(r"ARC_(.+?)-EN_(.+?)-BS_\d+-EP_\d+-LR_[\d.]+-AUG_(.+?)-LSS_(.+?)/")

    def __init__(self, path):
        self.path = path

        match = self.pattern.search(path)
        if match:
            arc, enc, aug, lss = match.groups()
            self.arc = arc
            self.enc = enc
            self.aug = aug
            self.lss = lss
            self.name = f"ARC={arc}, ENC={enc}, AUG={aug}, LSS={lss}"

    def get_config(self):
        """Get configuration from parsed path."""
        return Configuration(
            architecture=self.arc,
            encoder_name=self.enc,
            augmentation_set=self.aug,
            loss_function=self.lss,
            create_folders=False,
        )

    def __repr__(self):
        """Get String representation of the PathParser object."""
        return f"{self.name}"


def set_seed(seed: int) -> None:
    """Set the seed for random number generators in various libraries.

    This function sets the seed for the built-in random module, NumPy, and PyTorch
    to ensure reproducibility of results across different runs.

    Args:
        seed (int): The seed value to be set for the random number generators.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print_log(f" - Random seed set to: {seed}")


def split_seed(seed, n):
    """Split a single seed into `n` unique deterministic values.

    Args:
        seed (int): The base seed value used to generate deterministic random numbers.
        n (int): The number of unique seed values to generate.

    Returns:
        list[int]: A list of `n` unique deterministic values generated from the base seed.

    """
    rng = random.Random(seed)  # Create a random generator with the given seed
    return [rng.randint(0, 2**32 - 1) for _ in range(n)]


def to_device(x, device=None):
    """Move a tensor or model to the specified device.

    Args:
        x : torch.Tensor or torch.nn.Module
            The tensor or model to be moved.
        device : torch.device or None, optional
            The target device to move the tensor or model to. If None, it defaults to:
            - CUDA device if available.
            - CPU otherwise.

    Returns:
        torch.Tensor or torch.nn.Module
            The input tensor or model moved to the specified or default device.
    """
    if device is not None:
        return x.to(device)
    elif torch.cuda.is_available():
        return x.cuda()
    else:
        return x.cpu()


def get_device():
    """Get the available device (GPU or CPU).

    Returns:
        torch.device: The available device (GPU or CPU).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def load_image(image_path: str, to_bw: bool = False) -> Image.Image | None:
    """Load an image from a specified file path.

    Args:
        image_path (str): Path to the image file.
        to_bw (bool): Whether to convert the image to black and white.

    Returns:
        Image.Image | None: Loaded image or None if loading fails.
    """
    try:
        # Open the file inside a context manager so the file descriptor is
        # closed immediately. Return a copy/converted image that does not
        # keep the underlying file open.
        with Image.open(image_path) as _img:
            if to_bw:
                # Convert to single-channel and return a copy so the underlying
                # file descriptor is closed when exiting the context manager.
                img = _img.convert("L").copy()
            else:
                # Ensure color images are in RGB mode so downstream code that
                # expects 3 channels (C, H, W) works reliably. copy() detaches
                # the returned Image from the underlying file so the file
                # descriptor is closed when exiting the context manager.
                img = _img.convert("RGB").copy()
        return img
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def ensure_monochannel(folder_path: str) -> None:
    """Ensure all images in a folder are converted to single-channel black-and-white format.

    This function processes all files in the specified folder, converting each image
    to monochannel format, and overwrites the original files.

    Args:
        folder_path (str): Path to the folder containing images to be converted.

    Returns:
        None: The images are processed and saved in-place.

    """
    for file_path in list_dir_files(folder_path):
        if not os.path.isfile(file_path):
            continue  # Check if it's a file

        try:
            image = Image.open(file_path)
            monochannel_image = image.convert("L")  # Convert to single-channel
            monochannel_image.save(file_path)  # Overwrite with the same name
            print_log(f"Processed and saved: {file_path}")
        except Exception as e:
            print_error(f"Failed to process {file_path}: {e}")


def get_data_paths_from_config(CONFIG: Configuration) -> tuple[str, str]:
    """Get data paths from the configuration current partition.

    Args:
        CONFIG (Configuration): The configuration object containing data paths.

    Returns:
        tuple[str, str]: img_folder, gt_folder: A tuple with img and gt paths.
    """
    if CONFIG.partition not in ["train", "val", "test"]:
        raise ValueError(
            f"Invalid partition: {CONFIG.partition}. Must be 'train', 'val', or 'test'."
        )

    if CONFIG.partition == "train":
        img_folder = CONFIG.train_img_folder
        gt_folder = CONFIG.train_gt_folder
    elif CONFIG.partition == "val":
        img_folder = CONFIG.val_img_folder
        gt_folder = CONFIG.val_gt_folder
    else:  # CONFIG.partition == "test"
        img_folder = CONFIG.test_img_folder
        gt_folder = CONFIG.test_gt_folder

    return img_folder, gt_folder
