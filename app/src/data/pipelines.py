"""Pipelines definitions for image transformations and augmentations."""
from itertools import combinations

from .image_transformation import Transformation

PipeType = list[list[Transformation]]

# Single augmentations (already defined)
single_pipelines = [
    [Transformation.ROTATE],
    [Transformation.MIRROR],
    [Transformation.SUB],
    [Transformation.SHUFFLE],
    [Transformation.CIRCLES],
    [Transformation.INVERT],
    [Transformation.SHIFT_COLOR],
    [Transformation.NOISE],
]

# Pairs of augmentations
pair_pipelines = [list(c) for c in combinations([t[0] for t in single_pipelines], 2)]

#  Full pipeline (all augmentations)
full_pipeline = [t[0] for t in single_pipelines]


# Combine into dict for easy reference
augmentation_experiments = {
    "single": single_pipelines,
    "pairs": pair_pipelines,
    "full": [full_pipeline],
}
