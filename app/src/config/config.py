"""Configuration file.

Configuration of project variables that we want to have available
everywhere and considered configuration.
"""
import dataclasses
import os
from argparse import Namespace
from dataclasses import dataclass


@dataclass
class Configuration:
    """Configuration class for the project.

    This class contains all the configuration variables for the project.
    """

    seed: int = 42
    augmented: bool = False
    max_samples: int = None
    # ========================= PATHS ==========================
    MODELS_FOLDER: str = "../models"
    LOGS_FOLDER: str = "../logs"
    TEMP_FOLDER: str = "../temp"
    DATA_FOLDER: str = "../data"
    DATA_BASIC_FOLDER: str = os.path.join(DATA_FOLDER, "basic")
    DATA_AUG_FOLDER: str = os.path.join(DATA_FOLDER, "augmented")

    metadata_path: str = os.path.join(DATA_FOLDER, "metadata.csv")
    original_data_path: str = os.path.join(DATA_FOLDER, "raw", "original")

    train_folder: str = os.path.join(DATA_BASIC_FOLDER, "train")
    train_img_folder: str = os.path.join(DATA_BASIC_FOLDER, "train", "images")
    train_gt_folder: str = os.path.join(DATA_BASIC_FOLDER, "train", "groundtruth")

    val_folder: str = os.path.join(DATA_BASIC_FOLDER, "validation")
    val_img_folder: str = os.path.join(DATA_BASIC_FOLDER, "validation", "images")
    val_gt_folder: str = os.path.join(DATA_BASIC_FOLDER, "validation", "groundtruth")

    test_folder: str = os.path.join(DATA_BASIC_FOLDER, "test")
    test_img_folder: str = os.path.join(DATA_BASIC_FOLDER, "test", "images")
    test_gt_folder: str = os.path.join(DATA_BASIC_FOLDER, "test", "groundtruth")

    logs_path: str = os.path.join(LOGS_FOLDER, "log.log")

    def __post_init__(self):
        """Post-initialization."""
        if not self.augmented:
            return

        # ========================= PARTITION OF DATA ==========================
        self.train_folder = os.path.join(self.DATA_AUG_FOLDER, "train")
        self.train_img_folder = os.path.join(self.DATA_AUG_FOLDER, "train", "images")
        self.train_gt_folder = os.path.join(self.DATA_AUG_FOLDER, "train", "groundtruth")

        self.val_folder = os.path.join(self.DATA_AUG_FOLDER, "validation")
        self.val_img_folder = os.path.join(self.DATA_AUG_FOLDER, "validation", "images")
        self.val_gt_folder = os.path.join(self.DATA_AUG_FOLDER, "validation", "groundtruth")

        self.test_folder = os.path.join(self.DATA_AUG_FOLDER, "test")
        self.test_img_folder = os.path.join(self.DATA_AUG_FOLDER, "test", "images")
        self.test_gt_folder = os.path.join(self.DATA_AUG_FOLDER, "test", "groundtruth")


def args_to_config(args: Namespace):
    """From the args namespace, create a Configuration.

    It will change all the fields that have ben added to the args.
    If a field is not added in the args will be ignored.
    Fields in the args that are not in the Config this will be ignored.

    Args:
        args (Namespace): Parsed arguments.

    Returns:
        Configuration: Configuration with args values.
    """
    fields = {f.name for f in dataclasses.fields(Configuration)}
    filtered = {k: v for k, v in vars(args).items() if k in fields}
    return Configuration(**filtered)
