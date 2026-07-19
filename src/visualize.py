import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/final_dataset.csv"
RAW_DATA = "data/raw/Global_Mental_Health_Crisis_Index_2026.csv"

os.makedirs("images", exist_ok=True)


def load_data():
    processed = pd.read_csv(DATA_PATH)
    raw = pd.read_csv(RAW_DATA)
    return processed, raw


# =====================================================
# 1. Stress Distribution
# =====================================================

def stress_distribution(df):

    plt.figure(figsize=(8,5))

    plt.hist(
        df["mh_crisis_index"],
        bins=10
    )

    plt.title("Healthcare Stress Score Distribution")

    plt.xlabel("Healthcare Stress Score")

    plt.ylabel("Number of Countries")

    plt.tight_layout()

    plt.savefig("images/stress_distribution.png")

    plt.close()


# =====================================================
# 2. Top 10 Countries
# =====================================================

def top10_countries(raw):

    top = raw.sort_values(
        by="mh_crisis_index",
        ascending=False
    ).head(10)

    plt.figure(figsize=(12,6))

    plt.bar(
        top["country"],
        top["mh_crisis_index"]
    )

    plt.xticks(rotation=45)

    plt.ylabel("Healthcare Stress Score")

    plt.title("Top 10 Countries by Healthcare Stress")

    plt.tight_layout()

    plt.savefig("images/top10_countries.png")

    plt.close()


# =====================================================
# 3. Region Comparison
# =====================================================

def region_comparison(raw):

    region = raw.groupby(
        "region"
    )["mh_crisis_index"].mean().sort_values()

    plt.figure(figsize=(10,6))

    plt.bar(
        region.index,
        region.values
    )

    plt.xticks(rotation=30)

    plt.ylabel("Average Stress Score")

    plt.title("Average Healthcare Stress by Region")

    plt.tight_layout()

    plt.savefig("images/region_comparison.png")

    plt.close()


# =====================================================
# 4. Income Group Comparison
# =====================================================

def income_group(raw):

    income = raw.groupby(
        "income_group"
    )["mh_crisis_index"].mean()

    plt.figure(figsize=(8,5))

    plt.bar(
        income.index,
        income.values
    )

    plt.xticks(rotation=30)

    plt.ylabel("Average Stress Score")

    plt.title("Healthcare Stress by Income Group")

    plt.tight_layout()

    plt.savefig("images/income_group_comparison.png")

    plt.close()


# =====================================================
# 5. Stress Category Pie Chart
# =====================================================

def stress_category(df):

    counts = df["stress_class"].value_counts()

    plt.figure(figsize=(6,6))

    plt.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title("Healthcare Stress Categories")

    plt.tight_layout()

    plt.savefig("images/stress_category_pie.png")

    plt.close()

    # =====================================================
# 6. Correlation Heatmap
# =====================================================

def correlation_heatmap(df):

    corr = df.select_dtypes(include="number").corr()

    plt.figure(figsize=(12,10))

    plt.imshow(corr)

    plt.colorbar()

    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=90
    )

    plt.yticks(
        range(len(corr.columns)),
        corr.columns
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.savefig("images/correlation_heatmap.png")

    plt.close()


# =====================================================
# 7. Depression vs Anxiety
# =====================================================

def depression_vs_anxiety(raw):

    plt.figure(figsize=(8,6))

    plt.scatter(
        raw["depression_pct"],
        raw["anxiety_pct"]
    )

    plt.xlabel("Depression (%)")

    plt.ylabel("Anxiety (%)")

    plt.title("Depression vs Anxiety")

    plt.tight_layout()

    plt.savefig("images/depression_vs_anxiety.png")

    plt.close()


# =====================================================
# 8. Treatment Gap Distribution
# =====================================================

def treatment_gap(raw):

    plt.figure(figsize=(8,5))

    plt.hist(
        raw["treatment_gap_pct"],
        bins=10
    )

    plt.xlabel("Treatment Gap (%)")

    plt.ylabel("Countries")

    plt.title("Treatment Gap Distribution")

    plt.tight_layout()

    plt.savefig("images/treatment_gap_distribution.png")

    plt.close()


# =====================================================
# 9. Mental Health Budget Distribution
# =====================================================

def budget_distribution(raw):

    plt.figure(figsize=(8,5))

    plt.hist(
        raw["mh_budget_pct_health"],
        bins=10
    )

    plt.xlabel("Mental Health Budget (%)")

    plt.ylabel("Countries")

    plt.title("Mental Health Budget Distribution")

    plt.tight_layout()

    plt.savefig("images/budget_distribution.png")

    plt.close()


# =====================================================
# 10. Youth Crisis Score Distribution
# =====================================================

def youth_score(raw):

    plt.figure(figsize=(8,5))

    plt.hist(
        raw["youth_mh_crisis_score"],
        bins=10
    )

    plt.xlabel("Youth Mental Health Crisis Score")

    plt.ylabel("Countries")

    plt.title("Youth Mental Health Crisis Score Distribution")

    plt.tight_layout()

    plt.savefig("images/youth_score_distribution.png")

    plt.close()


# =====================================================
# Main
# =====================================================

def main():

    print("=" * 60)
    print("Healthcare Data Visualization")
    print("=" * 60)

    processed, raw = load_data()

    stress_distribution(processed)
    print("✓ Stress Distribution")

    top10_countries(raw)
    print("✓ Top 10 Countries")

    region_comparison(raw)
    print("✓ Region Comparison")

    income_group(raw)
    print("✓ Income Group Comparison")

    stress_category(processed)
    print("✓ Stress Category Pie Chart")

    correlation_heatmap(processed)
    print("✓ Correlation Heatmap")

    depression_vs_anxiety(raw)
    print("✓ Depression vs Anxiety")

    treatment_gap(raw)
    print("✓ Treatment Gap Distribution")

    budget_distribution(raw)
    print("✓ Mental Health Budget Distribution")

    youth_score(raw)
    print("✓ Youth Crisis Score Distribution")

    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()