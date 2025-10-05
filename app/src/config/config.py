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
from maikol_utils.print_utils import print_log, print_separator


@dataclass
class Configuration:
    """Configuration class for the project.

    This class contains all the configuration variables for the project.
    """

    # ========================= MODEL PARAMETERS ==========================
    seed: int = 42
    max_samples: int = None
    val_split: float = 0.15
    test_split: float = 0.15

    epochs: int = 10
    batch_size: int = 4
    max_steps: int = 1000
    learning_rate: float = 0.0001

    in_channels: int = 3
    out_classes: int = 1

    architecture: Literal["Unet", "FPN", "PSPNet"] = "Unet"
    encoder_name: Literal["resnet18", "resnet34"] = "resnet34"
    loss_function: Literal["DiceLoss", "BCEDice"] = "DiceLoss"
    augset: Literal["none", "simple", "double", "all"] = "none"

    augmentation_chance: float = 0.75
    # ========================= PATHS ==========================
    MODELS_FOLDER: str = "../models"
    LOGS_FOLDER: str = "../logs"
    TEMP_FOLDER: str = "../temp"
    DATA_FOLDER: str = "../data"
    DATA_BASIC_FOLDER: str = os.path.join(DATA_FOLDER, "basic")
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

    model_name: str = "basic_model"
    model_folder: str = os.path.join(MODELS_FOLDER, model_name)
    model_path: str = os.path.join(model_folder, f"{model_name}.ckpt")

    log_folder: str = os.path.join(LOGS_FOLDER, model_name)
    log_file: str = os.path.join(log_folder, f"{model_name}.log")

    def __post_init__(self):
        """Post-initialization."""
        if self.max_samples is not None:
            self.max_steps = self.epochs * (self.max_samples // self.batch_size)

        # ==================== MODEL NAME ====================
        self.model_name = (
            f"log-ARC{self.architecture}"
            f"-EN{self.encoder_name}"
            f"-BS{self.batch_size}"
            f"-EP{self.epochs}"
            f"-LR{self.learning_rate}"
            f"-AUG{self.augset}"
        )
        if self.max_samples is not None:
            self.model_name += f"-MS{self.max_samples}"

        # ==================== MODEL FOLDER ====================
        self.model_folder = os.path.join(self.MODELS_FOLDER, self.model_name)
        self.model_path = os.path.join(self.model_folder, f"{self.model_name}.ckpt")

        # ==================== LOGS ====================
        self.log_folder = os.path.join(self.LOGS_FOLDER, self.model_name)
        self.log_file = os.path.join(self.log_folder, f"{self.model_name}.log")

        make_dirs(
            [
                self.train_img_folder,
                self.train_gt_folder,
                self.val_img_folder,
                self.val_gt_folder,
                self.test_img_folder,
                self.test_gt_folder,
                self.model_folder,
                self.log_folder,
            ]
        )

    def print_config(self):
        """Print the configuration."""
        print_separator("CONFIGURATION", sep_type="LONG")
        print_log(f" - Model name:          {self.model_name}")
        print_log(f" - Model folder:        {self.model_folder}")
        print_log(f" - Log folder:          {self.log_folder}")
        print_log(f" - Seed:                {self.seed}")
        print_log(f" - Max samples:         {self.max_samples}")
        print_log(f" - Val split:           {self.val_split}")
        print_log(f" - Test split:          {self.test_split}")
        print_separator(" ", sep_type="SHORT")
        print_log(f" - Epochs:              {self.epochs}")
        print_log(f" - Batch size:          {self.batch_size}")
        print_log(f" - Max steps:           {self.max_steps}")
        print_log(f" - Learning rate:       {self.learning_rate}")
        print_log(f" - In channels:         {self.in_channels}")
        print_log(f" - Out classes:         {self.out_classes}")
        print_separator(" ", sep_type="SHORT")
        print_log(f" - Architecture:        {self.architecture}")
        print_log(f" - Encoder name:        {self.encoder_name}")
        print_log(f" - Loss function:       {self.loss_function}")
        print_log(f" - Augmentation set:    {self.augset}")
        print_log(f" - Augmentation chance: {self.augmentation_chance}")


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
