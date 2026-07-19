import os
import pandas as pd

DATA_PATH = "data/processed/final_dataset.csv"
REPORT_PATH = "reports/healthcare_trend_analysis.txt"


def load_data():
    return pd.read_csv(DATA_PATH)


def generate_trend_analysis(df):

    os.makedirs("reports", exist_ok=True)

    average = df["mh_crisis_index"].mean()
    highest = df["mh_crisis_index"].max()
    lowest = df["mh_crisis_index"].min()

    report = []

    report.append("=" * 70)
    report.append("Healthcare Trend Analysis")
    report.append("=" * 70)

    report.append("")
    report.append("Dataset Summary")
    report.append("-" * 40)
    report.append(f"Number of Countries : {len(df)}")
    report.append(f"Average Stress Score: {average:.2f}")
    report.append(f"Highest Stress Score: {highest:.2f}")
    report.append(f"Lowest Stress Score : {lowest:.2f}")

    report.append("")
    report.append("Trend Analysis")
    report.append("-" * 40)

    report.append(
        "This dataset contains healthcare indicators for a single year (2026)."
    )

    report.append(
        "Because historical data is unavailable, long-term healthcare trends "
        "cannot be calculated."
    )

    report.append("")

    report.append("The following analyses require multi-year data:")
    report.append("• Countries with increasing healthcare pressure")
    report.append("• Countries showing recovery")
    report.append("• Seasonal healthcare event patterns")
    report.append("• Long-term healthcare reporting trends")

    report.append("")
    report.append("Future Recommendation")
    report.append("-" * 40)

    report.append(
        "Collect healthcare data across multiple years "
        "(for example, 2022–2026) to enable trend analysis."
    )

    report.append(
        "A time-series dataset would allow forecasting, seasonal analysis, "
        "and monitoring of healthcare system changes over time."
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(report))

    print("\n".join(report))

    print("\nHealthcare Trend Analysis saved to:")
    print(REPORT_PATH)


def main():

    print("=" * 60)
    print("Healthcare Trend Analysis")
    print("=" * 60)

    df = load_data()

    generate_trend_analysis(df)

    print("\nTrend Analysis Completed Successfully!")


if __name__ == "__main__":
    main()