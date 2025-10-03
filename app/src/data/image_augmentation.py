"""Image augmentation and transformation utilities."""

import os
from typing import Literal

from PIL import Image
from src.data import SampleImage, Transformation, split_image_into_grid
from src.utils import load_image, split_seed

from .pipelines import PipeType


def create_base_images(sample_directory, output_directory):
    """Create base images and ground truth data by splitting input images into grids.

    This function generates directories for base images and ground truth, splits images
    into grids of different sizes (2x2, 3x3, 4x4), and saves them with unique names.

    Args:
        sample_directory (str): Input path directory containing 'images' and 'groundtruth' folders.
        output_directory (str): Output Path directory where results will be saved.

    Returns:
        None: The base images and ground truths are saved to the specified output directory.

    """
    # PART ONE: CREATE DIRECTORIES
    custom_dir = os.path.join(output_directory, "base")
    os.makedirs(custom_dir, exist_ok=True)

    images_dir = os.path.join(custom_dir, "images")
    groundtruth_dir = os.path.join(custom_dir, "groundtruth")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(groundtruth_dir, exist_ok=True)

    # PART TWO: CREATE BASE
    image_files = sorted(os.listdir(os.path.join(sample_directory, "images")))
    truth_files = sorted(os.listdir(os.path.join(sample_directory, "groundtruth")))

    index_counter = 1

    for image_file, truth_file in zip(image_files, truth_files):
        original_image = Image.open(os.path.join(sample_directory, "images", image_file))
        original_truth = Image.open(os.path.join(sample_directory, "groundtruth", truth_file))

        # Split images into grids
        two_split_image = split_image_into_grid(original_image, 2)
        two_split_truth = split_image_into_grid(original_truth, 2)

        three_split_image = split_image_into_grid(original_image, 3)
        three_split_truth = split_image_into_grid(original_truth, 3)

        four_split_image = split_image_into_grid(original_image, 4)
        four_split_truth = split_image_into_grid(original_truth, 4)

        # Combine all splits into one list
        images = [original_image] + two_split_image + three_split_image + four_split_image
        truths = [original_truth] + two_split_truth + three_split_truth + four_split_truth

        # Save split images and truths with unique names
        for img, truth in zip(images, truths):
            img.save(os.path.join(images_dir, f"satImage_{index_counter}.png"))
            truth.save(os.path.join(groundtruth_dir, f"satImage_{index_counter}.png"))
            index_counter += 1


def apply_transformation(image_in, truth_in, transformations, seed):
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
    image_copy = image_in.copy()
    truth_copy = truth_in.copy()

    for transformation in transformations:
        transform_func = transformation.value[1]
        # Apply transformation to the image_copy
        image_copy = transform_func(image_copy, seed)

        # Apply transformation to truth_copy only if it is one of the specified transformations
        if transformation in [
            Transformation.ROTATE,
            Transformation.MIRROR,
            Transformation.SUB,
            Transformation.SHUFFLE,
            Transformation.CIRCLES,
        ]:
            truth_copy = transform_func(truth_copy, seed)

    return image_copy, truth_copy


def apply_pipeline(sample: SampleImage, pipelines: PipeType, seed: int):
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
    image, truth = sample.get()
    image_in = load_image(image)
    truth_in = load_image(truth)

    seeds = split_seed(seed)
    for i, (pipeline, seed) in enumerate(zip(pipelines, seeds)):
        image_out, truth_out = apply_transformation(image_in, truth_in, pipeline, seed)
        truth_out = truth_out.convert("L")

        image_path = f"{i}_{Transformation.pipe_to_name(pipeline)}.png"
        truth_path = f"{i}_{Transformation.pipe_to_name(pipeline)}.png"

        image_out.save(image_path)
        truth_out.save(truth_path)


def create_training_data(
    sample_directory,
    output_directory,
    output_directory_name,
    pipelines,
    settings,
    n_divisions: Literal[0, 1, 2, 3] = 1,
):
    """Create training data by splitting images into grids and applying transformations.

    The function generates a dataset by splitting input images into grids of varying sizes
    and applying specified transformation pipelines.

    Args:
        sample_directory (str): Input Path directory containing 'images' and 'groundtruth' folders.
        output_directory (str): Output Path directory where results will be saved.
        output_directory_name (str): Folder Name within the output directory to store results.
        pipelines (list): List of transformation pipelines to apply.
        settings (Settings): Configuration settings including seeds and parameters.
        n_divisions (Literal[0, 1, 2, 3], optional): Number of divisions for splitting the images.

    Returns:
        None: The training data is saved to the specified output directory.

    Example:
        create_training_data("input_dir", "output_dir", "training_data", pipelines, settings)
    """
    # PART ONE: CREATE DIRECTORIES
    # custom_dir = os.path.join(output_directory, output_directory_name)
    # os.makedirs(custom_dir, exist_ok=True)

    # images_dir = os.path.join(custom_dir, "images")
    # groundtruth_dir = os.path.join(custom_dir, "groundtruth")
    # base_image_dir = os.path.join(custom_dir, "base_image")
    # base_truth_dir = os.path.join(custom_dir, "base_truth")

    # os.makedirs(images_dir, exist_ok=True)
    # os.makedirs(groundtruth_dir, exist_ok=True)
    # os.makedirs(base_image_dir, exist_ok=True)
    # os.makedirs(base_truth_dir, exist_ok=True)

    # # PART TWO: CREATE BASE
    # image_files = sorted(os.listdir(os.path.join(sample_directory, "images")))
    # truth_files = sorted(os.listdir(os.path.join(sample_directory, "groundtruth")))
    # assert len(image_files) == len(truth_files), "Mismatch between images and groundtruth files."

    # index_counter = 1

    # for image_file, truth_file in zip(image_files, truth_files):
    #     original_image = Image.open(os.path.join(sample_directory, "images", image_file))
    #     original_truth = Image.open(
    #         os.path.join(sample_directory, "groundtruth", truth_file)
    #     ).convert("L")

    #     # Split images into grids
    #     if n_divisions == 0:
    #         images = [original_image]
    #         truths = [original_truth]
    #     else:  # n_divisions >= 1
    #         two_split_image = split_image_into_grid(original_image, 2)
    #         two_split_truth = split_image_into_grid(original_truth, 2)
    #         three_split_image = []
    #         three_split_truth = []
    #         four_split_image = []
    #         four_split_truth = []
    #         if n_divisions >= 2:
    #             three_split_image = split_image_into_grid(original_image, 3)
    #             three_split_truth = split_image_into_grid(original_truth, 3)
    #         if n_divisions >= 3:
    #             four_split_image = split_image_into_grid(original_image, 4)
    #             four_split_truth = split_image_into_grid(original_truth, 4)

    #         # Combine all splits into one list
    #         images = two_split_image + three_split_image + four_split_image
    #         truths = two_split_truth + three_split_truth + four_split_truth

    #     # Save split images and truths with unique names
    #     for img, truth in zip(images, truths):
    #         img.save(os.path.join(base_image_dir, f"satImage_{index_counter}.png"))
    #         truth.convert("L").save(os.path.join(base_truth_dir, f"satImage_{index_counter}.png"))
    #         index_counter += 1

    # # PART THREE: APPLY PIPELINE TO BASE FOLDERS
    # # Load all base images and truths into SampleImage objects
    # base_images = sorted(os.listdir(base_image_dir))
    # base_truths = sorted(os.listdir(base_truth_dir))
    # assert len(base_images) == len(base_truths), "Mismatch between base images and truths."

    # sample_points = [
    #     SampleImage(os.path.join(base_image_dir, img), os.path.join(base_truth_dir, truth))
    #     for img, truth in zip(base_images, base_truths)
    # ]

    # for i, sample in enumerate(sample_points):
    #     original_image = Image.open(sample.get_image())
    #     original_truth = Image.open(sample.get_groundtruth()).convert(
    #         "L"
    #     )  # Black and white monochannel

    #     # Save the original image and truth to the custom folder
    #     original_image.save(os.path.join(images_dir, f"satImage_{i + 1}_0_0.png"))
    #     original_truth.save(os.path.join(groundtruth_dir, f"satImage_{i + 1}_0_0.png"))

    #     # Apply pipeline transformations
    #     apply_pipeline(sample, pipelines, i + 1, custom_dir, settings)

    # # PART FOUR: DELETE BASE IMAGE AND TRUTH FOLDERS
    # shutil.rmtree(base_image_dir)
    # shutil.rmtree(base_truth_dir)
