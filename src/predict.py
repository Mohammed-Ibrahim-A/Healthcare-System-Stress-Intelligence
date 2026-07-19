import os
import joblib
import pandas as pd

REGRESSION_MODEL = "models/regression_model.pkl"
CLASSIFICATION_MODEL = "models/classification_model.pkl"

DATA_PATH = "data/processed/final_dataset.csv"
RAW_DATA_PATH = "data/raw/Global_Mental_Health_Crisis_Index_2026.csv"

REPORT_PATH = "reports/healthcare_predictions.csv"


def load_models():

    regression_model = joblib.load(REGRESSION_MODEL)
    classification_model = joblib.load(CLASSIFICATION_MODEL)

    return regression_model, classification_model


def load_data():

    processed = pd.read_csv(DATA_PATH)
    raw = pd.read_csv(RAW_DATA_PATH)

    return processed, raw


def predict_all(regression_model, classification_model, processed, raw):

    X = processed.drop(columns=[
        "mh_crisis_index",
        "stress_class"
    ])

    score_predictions = regression_model.predict(X)

    class_predictions = classification_model.predict(X)

    output = pd.DataFrame({
        "Country": raw["country"],
        "Predicted_Healthcare_Stress_Score": score_predictions.round(2),
        "Predicted_Stress_Category": class_predictions
    })

    os.makedirs("reports", exist_ok=True)

    output.to_csv(
        REPORT_PATH,
        index=False
    )

    print("=" * 75)
    print("Healthcare Stress Predictions")
    print("=" * 75)

    for _, row in output.iterrows():

        print(
            f"{row['Country']:<20} : "
            f"{row['Predicted_Healthcare_Stress_Score']:>6.2f} : "
            f"{row['Predicted_Stress_Category']}"
        )

    print("\nPrediction report saved to:")
    print(REPORT_PATH)


def main():

    regression_model, classification_model = load_models()

    processed, raw = load_data()

    predict_all(
        regression_model,
        classification_model,
        processed,
        raw
    )


if __name__ == "__main__":
    main()