import os
import pandas as pd

DATA_PATH = "data/raw/Global_Mental_Health_Crisis_Index_2026.csv"

def load_dataset():
    """
    Load the healthcare dataset.
    """

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"\nDataset not found!\n"
            f"Please place the CSV file at:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    return df


def display_dataset_info(df):
    """
    Display basic dataset information.
    """

    print("\n" + "=" * 60)
    print("Healthcare System Stress Intelligence")
    print("=" * 60)
    print(f"\nRows               : {df.shape[0]}")
    print(f"Columns            : {df.shape[1]}")

    print("\nColumn Names")
    print("-" * 60)

    for column in df.columns:
        print(column)

    print("\nDataset Preview")
    print("-" * 60)
    print(df.head())

    print("\nDataset Information")
    print("-" * 60)
    print(df.info())

    print("\nMissing Values")
    print("-" * 60)
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print("-" * 60)
    print(df.duplicated().sum())

    print("\nSummary Statistics")
    print("-" * 60)
    print(df.describe())


def main():

    try:
        df = load_dataset()
        print("\nDataset Loaded Successfully!")
        display_dataset_info(df)
    except Exception as e:
        print(f"\nError : {e}")


if __name__ == "__main__":
    main()