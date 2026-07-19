import os
import pandas as pd


def create_directory(path):
    """
    Create a directory if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def load_csv(file_path):
    """
    Load a CSV file into a DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")

    return pd.read_csv(file_path)


def save_csv(df, file_path):
    """
    Save a DataFrame as a CSV file.
    """
    directory = os.path.dirname(file_path)

    if directory:
        create_directory(directory)

    df.to_csv(file_path, index=False)


def print_heading(title):
    """
    Print a formatted heading.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_success(message):
    """
    Print a success message.
    """
    print(f"✓ {message}")


def print_error(message):
    """
    Print(f"Error: {message}")
    """
    print(f"✗ {message}")


if __name__ == "__main__":

    print_heading("Utility Module")

    print_success("Utility functions loaded successfully.")