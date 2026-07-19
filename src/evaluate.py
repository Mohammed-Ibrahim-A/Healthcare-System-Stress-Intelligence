import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)

DATA_PATH = "data/processed/final_dataset.csv"

REGRESSION_MODEL = "models/regression_model.pkl"
CLASSIFICATION_MODEL = "models/classification_model.pkl"


def load_data():
    return pd.read_csv(DATA_PATH)


def evaluate_regression(df):

    print("\n" + "=" * 60)
    print("Regression Model Evaluation")
    print("=" * 60)

    X = df.drop(columns=["mh_crisis_index", "stress_class"])
    y = df["mh_crisis_index"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = joblib.load(REGRESSION_MODEL)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")


def evaluate_classification(df):

    print("\n" + "=" * 60)
    print("Classification Model Evaluation")
    print("=" * 60)

    X = df.drop(columns=["mh_crisis_index", "stress_class"])
    y = df["stress_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = joblib.load(CLASSIFICATION_MODEL)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy : {accuracy:.4f}")

    print("\nClassification Report\n")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix\n")
    print(confusion_matrix(y_test, predictions))


def main():

    print("=" * 60)
    print("Healthcare System Stress Intelligence")
    print("Model Evaluation")
    print("=" * 60)

    if not os.path.exists(REGRESSION_MODEL):
        print("Regression model not found.")
        return

    if not os.path.exists(CLASSIFICATION_MODEL):
        print("Classification model not found.")
        return

    df = load_data()

    evaluate_regression(df)

    evaluate_classification(df)

    print("\nEvaluation Completed Successfully!")


if __name__ == "__main__":
    main()