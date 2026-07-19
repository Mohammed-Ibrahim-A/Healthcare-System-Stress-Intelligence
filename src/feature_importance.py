import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/final_dataset.csv"
MODEL_PATH = "models/regression_model.pkl"

IMAGE_PATH = "images/feature_importance.png"
REPORT_PATH = "reports/feature_importance.csv"


def load_data():
    return pd.read_csv(DATA_PATH)


def load_model():
    return joblib.load(MODEL_PATH)


def calculate_feature_importance(model, X):

    importance = model.feature_importances_

    feature_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    ).reset_index(drop=True)

    feature_df.index += 1
    feature_df.index.name = "Rank"

    return feature_df


def save_report(feature_df):

    os.makedirs("reports", exist_ok=True)

    feature_df.to_csv(
        REPORT_PATH
    )

    print("\nFeature Importance Report Saved Successfully!")
    print(feature_df.head(10))

    print(f"\nSaved to:\n{REPORT_PATH}")


def plot_feature_importance(feature_df):

    os.makedirs("images", exist_ok=True)

    plt.figure(figsize=(12,7))

    plt.barh(
        feature_df["Feature"][:10],
        feature_df["Importance"][:10]
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.title("Top 10 Healthcare Stress Features")

    plt.tight_layout()

    plt.savefig(IMAGE_PATH)

    plt.close()

    print("\nFeature Importance Chart Saved Successfully!")
    print(IMAGE_PATH)


def main():

    print("=" * 60)
    print("Healthcare Feature Importance")
    print("=" * 60)

    df = load_data()

    X = df.drop(columns=[
        "mh_crisis_index",
        "stress_class"
    ])

    model = load_model()

    feature_df = calculate_feature_importance(
        model,
        X
    )

    save_report(feature_df)

    plot_feature_importance(feature_df)

    print("\nFeature Importance Analysis Completed Successfully!")


if __name__ == "__main__":
    main()