"""Project utility functions.

This module contains various utility functions that are used throughout the project.
"""
import os
import random

import torch
from maikol_utils.file_utils import list_dir_files
from maikol_utils.print_utils import print_error, print_log
from PIL import Image


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


def load_image(image_path: str, to_bw: bool = False) -> Image.Image | None:
    """Load an image from a specified file path.

    Args:
        image_path (str): Path to the image file.
        to_bw (bool): Whether to convert the image to black and white.

    Returns:
        Image.Image | None: Loaded image or None if loading fails.
    """
    try:
        img = Image.open(image_path) if to_bw else Image.open(image_path).convert("L")
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
