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
    """Configuration class for the project."""

    seed: int = 42

    # ========================= PATHS ==========================
    DATA_FOLDER: str = "../data"
    MODELS_FOLDER: str = "../models"
    LOGS_FOLDER: str = "../logs"

    metadata_path: str = os.path.join(DATA_FOLDER, "metadata.csv")
    train_folder: str = os.path.join(DATA_FOLDER, "train")
    val_folder: str = os.path.join(DATA_FOLDER, "validation")
    test_folder: str = os.path.join(DATA_FOLDER, "test")

    logs_path: str = os.path.join(LOGS_FOLDER, "log.log")

    def __post_init__(self):
        """Post-initialization."""
        ...


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
