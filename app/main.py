"""Main file for scripts with arguments and call other functions."""

import argparse

import dotenv
from scripts import generate_dataset, train_model
from src.config import (
    Configuration,
    ModelConfiguration,
    args_to_config,
    args_to_model_config,
)


def cmd_generate_dataset(args: argparse.Namespace):
    """Call read_extract_from_config_list with the given args."""
    CONFIG: Configuration = args_to_config(args)
    generate_dataset(CONFIG)


def cmd_train_model(args: argparse.Namespace):
    """Call read_extract_from_config_list with the given args."""
    CONFIG: Configuration = args_to_config(args)
    M_CONFIG: ModelConfiguration = args_to_model_config(args)
    train_model(CONFIG, M_CONFIG)


def cmd_test_model(args: argparse.Namespace):
    """Call read_extract_from_config_list with the given args."""
    # CONFIG: Configuration = args_to_config(args)
    ...


# ======================================================================================
#                                       ARGUMENTS
# ======================================================================================
if __name__ == "__main__":
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(prog="app", description="Main Application CLI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    subparsers = parser.add_subparsers(dest="function", required=True)

    # ======================================================================================
    #                                       read_extract
    # ======================================================================================
    p_generate = subparsers.add_parser("generate-dataset", help="Generate dataset.")
    p_generate.add_argument(
        "-o", "--original_data_path", type=str, default=None, help="Path to the original data"
    )
    p_generate.add_argument(
        "-m", "--max_samples", type=int, default=None, help="Max samples to load"
    )
    p_generate.set_defaults(func=cmd_generate_dataset)

    # ======================================================================================
    #                                       train
    # ======================================================================================
    p_train = subparsers.add_parser("train-model", help="Train model.")
    p_train.add_argument("-m", "--max_samples", type=int, default=None, help="Max samples to load")
    p_train.add_argument("-e", "--epochs", type=int, default=None, help="Number of epochs")
    p_train.add_argument("-b", "--batch_size", type=int, default=None, help="Batch size")
    p_train.add_argument("-s", "--max_steps", type=int, default=None, help="Max steps")
    p_train.add_argument("-lr", "--learning_rate", type=float, default=None, help="Learning rate")
    p_train.add_argument(
        "-aug", "--augmentation_chance", type=float, default=None, help="Augmentation chance"
    )

    p_train.add_argument(
        "-arc", "--architecture", type=str, default=None, help="Model architecture"
    )
    p_train.add_argument("-enc", "--encoder_name", type=str, default=None, help="Encoder name")
    p_train.set_defaults(func=cmd_train_model)

    # ======================================================================================
    #                                       test
    # ======================================================================================
    p_test = subparsers.add_parser("test-model", help="Test model.")
    # p_test.add_argument(
    #     "-m", "--max_samples", type=int, default=None, help="Max samples to load"
    # )
    p_test.set_defaults(func=cmd_test_model)

    # ======================================================================================
    #                                       CALL
    # ======================================================================================
    args = parser.parse_args()
    args.func(args)
