import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/processed/final_dataset.csv"
RAW_DATA_PATH = "data/raw/Global_Mental_Health_Crisis_Index_2026.csv"

OUTPUT_PATH = "data/processed/clustered_dataset.csv"
REPORT_PATH = "reports/healthcare_region_clusters.csv"
IMAGE_PATH = "images/cluster_visualization.png"


def load_data():
    return pd.read_csv(DATA_PATH)


def prepare_features(df):

    X = df.drop(columns=[
        "mh_crisis_index",
        "stress_class"
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


def perform_clustering(X_scaled):

    print("\nTraining K-Means Model...")

    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(X_scaled)

    return model, clusters


def save_cluster_reports(df, clusters):

    raw = pd.read_csv(RAW_DATA_PATH)

    df["Cluster"] = clusters

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    report = pd.DataFrame({
        "Country": raw["country"],
        "Cluster": clusters
    })

    os.makedirs("reports", exist_ok=True)
    report.to_csv(
        REPORT_PATH,
        index=False
    )

    print("\nHealthcare Region Cluster Report Saved Successfully!")
    print(report.head())

    print(f"\nSaved to:\n{REPORT_PATH}")


def visualize_clusters(X_scaled, clusters):

    os.makedirs("images", exist_ok=True)

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8,6))

    plt.scatter(
        reduced[:,0],
        reduced[:,1],
        c=clusters
    )

    plt.title("Healthcare Region Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.savefig(IMAGE_PATH)
    plt.close()

    print("\nVisualization saved to:")
    print(IMAGE_PATH)


def main():

    print("=" * 60)
    print("Healthcare Region Clustering")
    print("=" * 60)

    df = load_data()

    X_scaled = prepare_features(df)

    model, clusters = perform_clustering(X_scaled)

    save_cluster_reports(df, clusters)

    print("\nCluster Summary")
    print(pd.Series(clusters).value_counts().sort_index())

    visualize_clusters(X_scaled, clusters)

    print("\nClustering Completed Successfully!")


if __name__ == "__main__":
    main()