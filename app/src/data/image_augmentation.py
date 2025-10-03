"""Image augmentation and transformation utilities."""

from PIL import Image
from src.config import Configuration
from src.utils import get_data_paths_from_config, split_seed

from .image_transformation import VALID_GT_TRANSFORMATIONS, Transformation
from .pipelines import PipeType
from .sample_image import SampleImage


def apply_transformation(
    img: Image.Image, gt: Image.Image, transformations: PipeType, seed: int
) -> tuple[Image.Image, Image.Image]:
    """Apply a series of transformations to an image and its ground truth.

    The function modifies the provided image and ground truth based on the list
    of transformations, using a consistent random seed for reproducibility.

    Args:
        image_in (PIL.Image.Image): The input image to be transformed.
        truth_in (PIL.Image.Image): The ground truth image to be transformed.
        transformations (list): List of transformations to apply.
        seed (int): Seed for random operations to ensure reproducibility.

    Returns:
        tuple: A tuple containing the transformed image and ground truth.

    """
    img_copy = img.copy()
    gt_copy = gt.copy()

    for transformation in transformations:
        transform_func = transformation.value[1]
        img_copy = transform_func(img_copy, seed)

        # Not all the transformations should be applied to the ground truth
        if transformation in VALID_GT_TRANSFORMATIONS:
            gt_copy = transform_func(gt_copy, seed)

    return img_copy, gt_copy


def apply_pipeline(sample: SampleImage, pipelines: PipeType, CONFIG: Configuration):
    """Apply transformation pipelines to a sample image and its ground truth.

    This function processes a sample point by applying a set of transformation
    pipelines and saves the results to specified directories.

    Args:
        sample (SampleImage): The sample containing paths to an image and its ground truth.
        pipelines (list): List of transformation pipelines to apply.
        base_index (int): Base index for naming the output files.
        out_directory (str): Path to the output directory.
        settings (Settings): Configuration settings including seeds and other parameters.

    Returns:
        None: The transformed images and ground truths are saved to the specified directory.

    """
    img, gt = sample.get_images()
    img_name, gt_name = sample.get_names()
    img_folder, gt_folder = get_data_paths_from_config(CONFIG)

    seeds = split_seed(CONFIG.seed, len(pipelines))
    for i, (pipeline, seed) in enumerate(zip(pipelines, seeds)):
        img_aug, gt_aug = apply_transformation(img, gt, pipeline, seed)
        gt_aug = gt_aug.convert("L")

        new_name_x = f"{img_name}_{i}_{Transformation.pipe_to_name(pipeline)}.png"
        new_name_y = f"{gt_name}_{i}_{Transformation.pipe_to_name(pipeline)}.png"

        aug_sample = SampleImage(
            new_name_x,
            new_name_y,
        )
        aug_sample.set_images(img_aug, gt_aug)
        aug_sample.save_images(img_folder, gt_folder)
