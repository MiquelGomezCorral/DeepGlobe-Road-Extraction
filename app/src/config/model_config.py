"""ModelConfiguration file.

Configuration of the model parameters and so on.
"""
import dataclasses
from argparse import Namespace
from dataclasses import dataclass


@dataclass
class ModelConfiguration:
    """Configuration class for the project.

    This class contains all the configuration variables for the project.
    """

    seed: int = 42
    max_samples: int = None
    epochs: int = 10
    batch_size: int = 4
    max_steps: int = 1000
    learning_rate: float = 0.0001

    in_channels: int = 3
    out_classes: int = 1

    architecture: str = "Unet"
    encoder_name: str = "resnet34"

    augmentation_chance: float = 0.75

    def __post_init__(self):
        """Post-initialization."""
        if self.max_samples is not None:
            self.max_steps = self.epochs * (self.max_samples // self.batch_size)


def args_to_model_config(args: Namespace):
    """From the args namespace, create a ModelConfiguration.

    It will change all the fields that have ben added to the args.
    If a field is not added in the args will be ignored.
    Fields in the args that are not in the Config this will be ignored.

    Args:
        args (Namespace): Parsed arguments.

    Returns:
        ModelConfiguration: ModelConfiguration with args values.
    """
    fields = {f.name for f in dataclasses.fields(ModelConfiguration)}
    filtered = {k: v for k, v in vars(args).items() if k in fields and v is not None}
    return ModelConfiguration(**filtered)
