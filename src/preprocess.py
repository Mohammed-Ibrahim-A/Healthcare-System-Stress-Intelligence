import os
import pandas as pd

RAW_DATA = "data/raw/Global_Mental_Health_Crisis_Index_2026.csv"
OUTPUT_DATA = "data/processed/cleaned_mental_health.csv"

def load_data():
    """Load the raw dataset."""
    return pd.read_csv(RAW_DATA)

def clean_data(df):
    """Perform data cleaning."""

    print("\nRemoving duplicate records...")
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Removed {before - after} duplicate rows.")

    print("\nHandling missing values...")

    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

def encode_features(df):
    """Encode categorical variables."""

    print("\nEncoding categorical features...")

    # Yes / No columns
    yes_no_columns = [
        "mh_policy_exists",
        "mh_law_exists"
    ]

    for col in yes_no_columns:
        df[col] = df[col].map({
            "Yes": 1,
            "No": 0
        })

    # One-hot encoding
    categorical_columns = [
        "region",
        "income_group",
        "social_media_mental_health_risk"
    ]
    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True
    )

    return df

def save_dataset(df):
    """Save processed dataset."""

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(
        OUTPUT_DATA,
        index=False
    )
    print(f"\nProcessed dataset saved to:\n{OUTPUT_DATA}")

def main():

    print("=" * 60)
    print("Healthcare System Stress Intelligence")
    print("Data Preprocessing")
    print("=" * 60)
    df = load_data()
    print(f"\nOriginal Shape : {df.shape}")
    df = clean_data(df)
    df = encode_features(df)
    print(f"\nProcessed Shape : {df.shape}")
    print("\nRemaining Missing Values")
    print(df.isnull().sum().sum())
    save_dataset(df)
    print("\nPreprocessing Completed Successfully!")


if __name__ == "__main__":
    main()