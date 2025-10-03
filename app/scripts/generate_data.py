"""Generate data script.

Script that generates a copy and an augmented version from a dataset.
"""
from maikol_utils.print_utils import print_separator
from src.config import Configuration
from src.script_refactor import augment_data, save_splited_data, split_data


def generate_dataset(CONFIG: Configuration):
    """Generate a dataset by splitting, saving, and augmenting the data.

    Args:
        CONFIG (Configuration): Configuration.
    """
    print_separator("GENERATING DATASET", sep_type="START")
    # =================================================================
    #                           Configuration
    # =================================================================
    print_separator("CONFIGURATION", sep_type="LONG")
    CONFIG = Configuration()

    # =================================================================
    #                           SPLIT DATA
    # =================================================================
    print_separator("SPLIT DATA", sep_type="LONG")
    train, val, test = split_data(CONFIG)

    # =================================================================
    #                           SAVE DATA
    # =================================================================
    print_separator("PREPEARING BASIC DATA", sep_type="LONG")
    save_splited_data(CONFIG, train, val, test)

    # =================================================================
    #                           AUGMENT DATA
    # =================================================================
    print_separator("PREPEARING AUGMENTED DATA", sep_type="LONG")
    augment_data(CONFIG, train, val, test)

    print_separator("GENERATING DATASET", sep_type="DONE")
