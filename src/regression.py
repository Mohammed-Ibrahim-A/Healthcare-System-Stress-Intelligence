import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

DATA_PATH = "data/processed/final_dataset.csv"
RAW_DATA_PATH = "data/raw/Global_Mental_Health_Crisis_Index_2026.csv"
MODEL_PATH = "models/regression_model.pkl"
REPORT_PATH = "reports/healthcare_stress_scores.csv"


def load_dataset():
    return pd.read_csv(DATA_PATH)


def prepare_data(df):

    X = df.drop(columns=[
        "mh_crisis_index",
        "stress_class"
    ])

    y = df["mh_crisis_index"]

    return X, y


def split_data(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )


def train_model(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def evaluate(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\nRegression Evaluation")
    print("-" * 40)
    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")


def save_model(model):

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to:\n{MODEL_PATH}")


def generate_healthcare_scores(model, X):

    raw = pd.read_csv(RAW_DATA_PATH)

    predicted_scores = model.predict(X)

    output = pd.DataFrame({
        "Country": raw["country"],
        "Healthcare_Stress_Score": predicted_scores.round(2)
    })

    os.makedirs("reports", exist_ok=True)

    output.to_csv(
        REPORT_PATH,
        index=False
    )

    print("\nHealthcare Stress Scores Saved Successfully!")
    print(output.head())

    print(f"\nSaved to:\n{REPORT_PATH}")


def main():

    print("=" * 60)
    print("Healthcare Stress Score Prediction")
    print("=" * 60)

    df = load_dataset()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    evaluate(model, X_test, y_test)

    save_model(model)

    generate_healthcare_scores(model, X)

    print("\nRegression Model Training Completed Successfully!")


if __name__ == "__main__":
    main()