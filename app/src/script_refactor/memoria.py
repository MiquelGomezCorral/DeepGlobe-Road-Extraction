"""Paper refactor.

Functions refactored from the papaer notebook.
"""
import re


def get_minutes(path: str) -> int:
    """From a path, load the log file and extract the training time in minutes.

    Args:
        path (str): Path to the log file.

    Returns:
        int: Training time in minutes.
    """
    with open(path) as f:
        content = f.read()

    match = re.search(r"Training time.*?: (.*)", content)
    if match:
        training_time = match.group(1).strip()

        match = re.search(r"(\d+)\s*hrs?", training_time)
        print
        hours = 0
        if match:
            hours = int(match.group(1))
            # print(f"Training time in hrs: {hours}")

        match = re.search(r"(\d+)\s*mins?", training_time)
        mins = 0
        if match:
            mins = int(match.group(1))
            # print(f"Training time in minutes: {mins}")

    return hours * 60 + mins
