import pandas as pd
import os

INPUT_DATA = "data/processed/cleaned_mental_health.csv"
OUTPUT_DATA = "data/processed/final_dataset.csv"

def create_healthcare_features(df):
    """
    Create new healthcare-related features.
    """

    print("\nCreating engineered features...")

    # Overall Mental Health Burden
    df["mental_health_burden"] = (
        df["depression_pct"] +
        df["anxiety_pct"] +
        df["treatment_gap_pct"]
    ) / 3

    # Healthcare Capacity
    df["healthcare_capacity"] = (
        df["psychiatrists_per100k"] *
        df["mh_budget_pct_health"]
    )

    # Digital Exposure
    df["digital_exposure"] = (
        df["social_media_hours_daily"] *
        df["internet_penetration_pct"]
    ) / 100

    # Spending Efficiency
    df["spending_efficiency"] = (
        df["mh_spend_usd_per_capita"] /
        (df["treatment_gap_pct"] + 1)
    )

    return df


def create_stress_class(df):
    """
    Create Healthcare Stress Classification.
    """

    print("Creating Healthcare Stress Classes...")
    def classify(score):
        if score < 40:
            return "Low Stress"
        elif score < 70:
            return "Medium Stress"
        else:
            return "High Stress"

    df["stress_class"] = df["mh_crisis_index"].apply(classify)
    return df

def remove_unused_columns(df):
    """
    Remove columns not needed for ML.
    """

    print("Removing unnecessary columns...")
    columns_to_remove = [
        "country",
        "iso3",
        "data_source",
        "data_year"
    ]

    df = df.drop(columns=columns_to_remove)

    return df

def save_dataset(df):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(
        OUTPUT_DATA,
        index=False
    )

def main():
    print("=" * 60)
    print("Healthcare System Stress Intelligence")
    print("Feature Engineering")
    print("=" * 60)
    df = pd.read_csv(INPUT_DATA)
    print(f"\nInput Shape : {df.shape}")
    df = create_healthcare_features(df)
    df = create_stress_class(df)
    df = remove_unused_columns(df)
    print(f"\nFinal Shape : {df.shape}")
    save_dataset(df)
    print("\nFeature Engineering Completed Successfully!")
    print(f"Dataset saved to:\n{OUTPUT_DATA}")


if __name__ == "__main__":
    main()