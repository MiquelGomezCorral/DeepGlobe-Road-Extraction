"""Train model script.

Scrip to train a model and save it.
"""
# import segmentation_models_pytorch as smp
# import torch
from maikol_utils.print_utils import print_separator  # , print_log
from src.config.config import Configuration, ModelConfiguration


def train_model(CONFIG: Configuration, M_CONFIG: ModelConfiguration):
    """Train a model and save it to disk.

    Args:
        CONFIG (Configuration): Configuration.
    """
    print_separator("TRAINING MODEL", sep_type="START")

    print_separator("DONE!", sep_type="START")
