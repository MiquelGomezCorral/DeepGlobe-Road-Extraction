"""Main file for scripts with arguments and call other functions."""

import argparse

import dotenv
from scripts import generate_data
from src.config import Configuration, args_to_config


def cmd_generate_dataset(args: argparse.Namespace):
    """Call read_extract_from_config_list with the given args."""
    CONFIG: Configuration = args_to_config(args)
    generate_data.generate_dataset(CONFIG)


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
        "-m", "--max_samples", type=int, default=None, help="Max samples to load"
    )
    p_generate.add_argument(
        "-o", "--original_data_path", type=str, default=None, help="Path to the original data"
    )
    p_generate.set_defaults(func=cmd_generate_dataset)

    # ======================================================================================
    #                                       trains
    # ======================================================================================

    # ======================================================================================
    #                                       CALL
    # ======================================================================================
    args = parser.parse_args()
    args.func(args)
