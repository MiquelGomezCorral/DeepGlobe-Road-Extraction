"""Split data functions.

Refactored functions from generate_data script.
"""
from dataclasses import replace

from maikol_utils.file_utils import list_dir_files
from maikol_utils.print_utils import print_log, print_separator
from sklearn.model_selection import train_test_split
from src.config import Configuration
from src.data import SampleImage, apply_pipeline, pipelines
from src.utils import set_seed
from tqdm import tqdm


def split_data(CONFIG: Configuration):
    """Split the dataset into train, validation and test sets.

    Args:
        CONFIG (Configuration): Configuration object containing dataset paths and parameters.

    Returns:
        tuple: train, val, test. SampleImage objects for train, validation, and test sets.
    """
    # =================================================================
    #                           Get image pairs
    # =================================================================
    original_files, n = list_dir_files(
        CONFIG.original_data_path,
        nat_sorting=True,  # Number to be properly sorted
        absolute_path=True,
        # x2 because we have image and groundtruth.
        max_files=2 * CONFIG.max_samples if CONFIG.max_samples is not None else None,
    )
    path_imges_X = [img for img in original_files if "_sat" in img]
    path_imges_Y = [img for img in original_files if "_mask" in img]

    sample_points = [
        SampleImage(path_img_x, path_img_y)
        for path_img_x, path_img_y in zip(path_imges_X, path_imges_Y)
    ]

    # =================================================================
    #                           SPLIT THE DATA
    # =================================================================
    train, temp = train_test_split(
        sample_points, test_size=CONFIG.val_split + CONFIG.test_split, random_state=CONFIG.seed
    )
    val, test = train_test_split(
        temp,
        test_size=CONFIG.test_split / (CONFIG.val_split + CONFIG.test_split),
        random_state=CONFIG.seed,
    )

    print_separator("DATA_RESUME", sep_type="SHORT")
    print_log(f" - Total samples:      {len(sample_points):6_}")
    print_log(f" - Train samples:      {len(train):6_}")
    print_log(f" - Validation samples: {len(val):6_}")
    print_log(f" - Test samples:       {len(test):6_}")

    return train, val, test


def save_splited_data(
    CONFIG: Configuration, train: list[SampleImage], val: list[SampleImage], test: list[SampleImage]
):
    """Save the split dataset to disk.

    Args:
        CONFIG (Configuration): Configuration object containing dataset paths and parameters.
        train (list[SampleImage]): List of training samples.
        val (list[SampleImage]): List of validation samples.
        test (list[SampleImage]): List of test samples.
    """
    print_separator("SAVING IMAGES IN DISK")
    print_log(f" - Train images folder:      {CONFIG.train_img_folder}")
    print_log(f" - Train groundtruth folder: {CONFIG.train_gt_folder}")
    print_log(f" - Val images folder:        {CONFIG.val_img_folder}")
    print_log(f" - Val groundtruth folder:   {CONFIG.val_gt_folder}")
    print_log(f" - Test images folder:       {CONFIG.test_img_folder}")
    print_log(f" - Test groundtruth folder:  {CONFIG.test_gt_folder}")

    print_separator("SAVING TRAIN", sep_type="SHORT")
    for train_sample in tqdm(train):
        train_sample.save_images(
            CONFIG.train_img_folder,
            CONFIG.train_gt_folder,
        )

    print_separator("SAVING VALIDATION", sep_type="SHORT")
    for val_sample in tqdm(val):
        val_sample.save_images(
            CONFIG.val_img_folder,
            CONFIG.val_gt_folder,
        )

    print_separator("SAVING TEST", sep_type="SHORT")
    for test_sample in tqdm(test):
        test_sample.save_images(
            CONFIG.test_img_folder,
            CONFIG.test_gt_folder,
        )


def augment_data(
    CONFIG: Configuration, train: list[SampleImage], val: list[SampleImage], test: list[SampleImage]
):
    """Generate and save augmented data.

    Args:
        CONFIG (Configuration): Configuration object containing dataset paths and parameters.
        train (list[SampleImage]): List of training samples.
        val (list[SampleImage]): List of validation samples.
        test (list[SampleImage]): List of test samples.
    """
    CONFIG_AUG = replace(CONFIG, augmented=True)

    set_seed(CONFIG_AUG.seed)
    selected_pipelines = pipelines.simple_pipeline

    print_log(f" - Train images folder:      {CONFIG_AUG.train_img_folder}")
    print_log(f" - Train groundtruth folder: {CONFIG_AUG.train_gt_folder}")
    print_log(f" - Val images folder:        {CONFIG_AUG.val_img_folder}")
    print_log(f" - Val groundtruth folder:   {CONFIG_AUG.val_gt_folder}")
    print_log(f" - Test images folder:       {CONFIG_AUG.test_img_folder}")
    print_log(f" - Test groundtruth folder:  {CONFIG_AUG.test_gt_folder}")

    print_separator("GENERATING TRAIN", sep_type="SHORT")
    CONFIG_AUG.partition = "train"
    for train_sample in tqdm(train):
        apply_pipeline(train_sample, selected_pipelines, CONFIG_AUG)

    print_separator("GENERATING VALIDATION", sep_type="SHORT")
    CONFIG_AUG.partition = "val"
    for val_sample in tqdm(val):
        apply_pipeline(val_sample, selected_pipelines, CONFIG_AUG)

    print_separator("GENERATING TEST", sep_type="SHORT")
    CONFIG_AUG.partition = "test"
    for test_sample in tqdm(test):
        apply_pipeline(test_sample, selected_pipelines, CONFIG_AUG)
