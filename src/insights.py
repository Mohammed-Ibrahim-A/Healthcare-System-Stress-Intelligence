import os
import pandas as pd

DATA_PATH = "data/processed/final_dataset.csv"
REPORT_PATH = "reports/business_policy_insights.txt"


def load_data():
    return pd.read_csv(DATA_PATH)


def generate_insights(df):

    os.makedirs("reports", exist_ok=True)

    average_stress = df["mh_crisis_index"].mean()
    highest_stress = df["mh_crisis_index"].max()
    lowest_stress = df["mh_crisis_index"].min()

    stress_distribution = df["stress_class"].value_counts()

    top_high_stress = df.sort_values(
        by="mh_crisis_index",
        ascending=False
    ).head(10)

    numeric_df = df.select_dtypes(include="number")

    correlations = numeric_df.corr()["mh_crisis_index"] \
        .drop("mh_crisis_index") \
        .abs() \
        .sort_values(ascending=False)

    report = []

    report.append("=" * 70)
    report.append("Healthcare Business & Policy Insights")
    report.append("=" * 70)

    report.append("")
    report.append(f"Average Healthcare Stress Score : {average_stress:.2f}")
    report.append(f"Highest Healthcare Stress Score : {highest_stress:.2f}")
    report.append(f"Lowest Healthcare Stress Score  : {lowest_stress:.2f}")

    report.append("")
    report.append("Stress Category Distribution")
    report.append("-" * 40)

    for category, count in stress_distribution.items():
        report.append(f"{category:<15}: {count}")

    report.append("")
    report.append("Top 10 Highest Stress Countries")
    report.append("-" * 40)

    if "country" in df.columns:
        for _, row in top_high_stress.iterrows():
            report.append(
                f"{row['country']:<25} {row['mh_crisis_index']:.2f}"
            )
    else:
        report.append(
            "Country names are not available in the processed dataset."
        )

    report.append("")
    report.append("Top Factors Associated with Healthcare Stress")
    report.append("-" * 40)

    for feature, value in correlations.head(10).items():
        report.append(f"{feature:<35} {value:.3f}")

    report.append("")
    report.append("Business & Policy Recommendations")
    report.append("-" * 40)

    report.append("• Increase investment in mental healthcare services.")
    report.append("• Reduce treatment gaps through awareness programmes.")
    report.append("• Expand psychiatrist and psychologist availability.")
    report.append("• Improve youth mental health programmes.")
    report.append("• Strengthen digital mental healthcare platforms.")
    report.append("• Prioritize countries with consistently high stress scores.")
    report.append("• Allocate healthcare resources based on identified risk factors.")
    report.append("• Continuously monitor healthcare indicators for early intervention.")

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(report))

    print("\n".join(report))

    print("\nBusiness & Policy Insights saved to:")
    print(REPORT_PATH)


def main():

    print("=" * 60)
    print("Generating Business & Policy Insights")
    print("=" * 60)

    df = load_data()

    generate_insights(df)

    print("\nBusiness & Policy Insights Completed Successfully!")


if __name__ == "__main__":
    main()