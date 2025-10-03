"""Configuration file.

Configuration of project variables that we want to have available
everywhere and considered configuration.
"""
import dataclasses
import os
from argparse import Namespace
from dataclasses import dataclass
from typing import Literal

from maikol_utils.file_utils import make_dirs


@dataclass
class Configuration:
    """Configuration class for the project.

    This class contains all the configuration variables for the project.
    """

    seed: int = 42
    max_samples: int = None
    augmented: bool = False
    copy_original: bool = True
    augmented_samples_per_image: int = 3

    val_split: float = 0.15
    test_split: float = 0.15
    partition: Literal["train", "val", "test"] = "train"
    # ========================= PATHS ==========================
    MODELS_FOLDER: str = "../models"
    LOGS_FOLDER: str = "../logs"
    TEMP_FOLDER: str = "../temp"
    DATA_FOLDER: str = "../data"
    DATA_BASIC_FOLDER: str = os.path.join(DATA_FOLDER, "basic")
    DATA_AUG_FOLDER: str = os.path.join(DATA_FOLDER, "augmented")
    DATA_IMG_FOLDER_NAME: str = "images"
    DATA_GT_FOLDER_NAME: str = "groundtruth"

    metadata_path: str = os.path.join(DATA_FOLDER, "metadata.csv")
    original_data_path: str = os.path.join(DATA_FOLDER, "raw", "original")

    train_folder: str = os.path.join(DATA_BASIC_FOLDER, "train")
    train_img_folder: str = os.path.join(DATA_BASIC_FOLDER, "train", DATA_IMG_FOLDER_NAME)
    train_gt_folder: str = os.path.join(DATA_BASIC_FOLDER, "train", DATA_GT_FOLDER_NAME)

    val_folder: str = os.path.join(DATA_BASIC_FOLDER, "validation")
    val_img_folder: str = os.path.join(DATA_BASIC_FOLDER, "validation", DATA_IMG_FOLDER_NAME)
    val_gt_folder: str = os.path.join(DATA_BASIC_FOLDER, "validation", DATA_GT_FOLDER_NAME)

    test_folder: str = os.path.join(DATA_BASIC_FOLDER, "test")
    test_img_folder: str = os.path.join(DATA_BASIC_FOLDER, "test", DATA_IMG_FOLDER_NAME)
    test_gt_folder: str = os.path.join(DATA_BASIC_FOLDER, "test", DATA_GT_FOLDER_NAME)

    logs_path: str = os.path.join(LOGS_FOLDER, "log.log")

    def __post_init__(self):
        """Post-initialization."""
        if self.augmented:
            # ========================= PARTITION OF DATA ==========================
            self.train_folder = os.path.join(self.DATA_AUG_FOLDER, "train")
            self.train_img_folder = os.path.join(
                self.DATA_AUG_FOLDER, "train", self.DATA_IMG_FOLDER_NAME
            )
            self.train_gt_folder = os.path.join(
                self.DATA_AUG_FOLDER, "train", self.DATA_GT_FOLDER_NAME
            )

            self.val_folder = os.path.join(self.DATA_AUG_FOLDER, "validation")
            self.val_img_folder = os.path.join(
                self.DATA_AUG_FOLDER, "validation", self.DATA_IMG_FOLDER_NAME
            )
            self.val_gt_folder = os.path.join(
                self.DATA_AUG_FOLDER, "validation", self.DATA_GT_FOLDER_NAME
            )

            self.test_folder = os.path.join(self.DATA_AUG_FOLDER, "test")
            self.test_img_folder = os.path.join(
                self.DATA_AUG_FOLDER, "test", self.DATA_IMG_FOLDER_NAME
            )
            self.test_gt_folder = os.path.join(
                self.DATA_AUG_FOLDER, "test", self.DATA_GT_FOLDER_NAME
            )
            # =====================================================================
        # END IF

        make_dirs(
            [
                self.train_img_folder,
                self.train_gt_folder,
                self.val_img_folder,
                self.val_gt_folder,
                self.test_img_folder,
                self.test_gt_folder,
            ]
        )


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
    filtered = {k: v for k, v in vars(args).items() if k in fields and v is not None}
    return Configuration(**filtered)
