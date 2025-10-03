"""Data.

Functions to manage, clean and process data.
"""
from ..data.pipelines import PipeType  # noqa: F401
from . import pipelines  # noqa: F401
from .image_augmentation import apply_pipeline  # noqa: F401
from .image_transformation import (  # noqa: F401
    Transformation,
    add_noise_to_image,
    add_random_circles,
    invert_colors,
    mirror_image,
    random_subimage,
    rotate_image,
    set_brightness,
    shift_color_towards_random,
    shuffle_image,
    split_image_into_grid,
)
from .sample_image import SampleImage  # noqa: F401
