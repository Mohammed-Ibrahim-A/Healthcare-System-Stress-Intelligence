import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

DATA_PATH = "data/processed/final_dataset.csv"
RAW_DATA_PATH = "data/raw/Global_Mental_Health_Crisis_Index_2026.csv"
MODEL_PATH = "models/classification_model.pkl"
REPORT_PATH = "reports/healthcare_stress_classification.csv"


def load_dataset():
    return pd.read_csv(DATA_PATH)


def prepare_data(df):

    X = df.drop(columns=[
        "mh_crisis_index",
        "stress_class"
    ])

    y = df["stress_class"]

    return X, y


def split_data(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


def train_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def evaluate(model, X_test, y_test):

    prediction = model.predict(X_test)

    print("\nClassification Results")
    print("=" * 50)

    print(f"\nAccuracy : {accuracy_score(y_test, prediction):.4f}")

    print("\nClassification Report")
    print(classification_report(y_test, prediction))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, prediction))


def save_model(model):

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved at:\n{MODEL_PATH}")


def generate_classification_report(model, X):

    raw = pd.read_csv(RAW_DATA_PATH)

    predictions = model.predict(X)

    output = pd.DataFrame({
        "Country": raw["country"],
        "Predicted_Stress": predictions
    })

    os.makedirs("reports", exist_ok=True)

    output.to_csv(
        REPORT_PATH,
        index=False
    )

    print("\nHealthcare Stress Classification Saved Successfully!")
    print(output.head())

    print(f"\nSaved to:\n{REPORT_PATH}")


def main():

    print("=" * 60)
    print("Healthcare Stress Classification")
    print("=" * 60)

    df = load_dataset()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    evaluate(model, X_test, y_test)

    save_model(model)

    generate_classification_report(model, X)

    print("\nClassification Model Completed Successfully!")


if __name__ == "__main__":
    main()