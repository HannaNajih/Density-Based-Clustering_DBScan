import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure output directories exist
os.makedirs('../outputs/plots', exist_ok=True)

def load_cluster_labels():
    """Load cluster labels from all methods."""
    try:
        dbscan = pd.read_csv('../outputs/clusters/dbscan_labels.csv').values.flatten()
        agnes = pd.read_csv('../outputs/clusters/agnes_labels.csv').values.flatten()
        kmeans = pd.read_csv('../outputs/clusters/kmeans_labels.csv').values.flatten()
        return dbscan, agnes, kmeans
    except Exception as e:
        print(f"❌ Error loading cluster labels: {e}")
        raise

def calculate_metrics(X_scaled, dbscan, agnes, kmeans):
    """Compute silhouette scores and cluster counts."""
    metrics = []
    
    for name, labels in zip(
        ['DBSCAN', 'AGNES', 'K-Means'],
        [dbscan, agnes, kmeans]
    ):
        n_clusters = len(np.unique(labels[labels != -1]))  # Exclude noise (-1)
        noise = np.sum(labels == -1)
        
        if n_clusters > 1:  # Silhouette score requires >=2 clusters
            score = silhouette_score(X_scaled, labels)
        else:
            score = np.nan
        
        metrics.append({
            'Method': name,
            'Clusters': n_clusters,
            'Noise Points': noise,
            'Silhouette Score': score
        })
    
    return pd.DataFrame(metrics)

def plot_results(metrics_df):
    """Visualize comparison metrics."""
    plt.figure(figsize=(10, 5))
    
    # Silhouette scores bar plot
    plt.subplot(1, 2, 1)
    sns.barplot(data=metrics_df, x='Method', y='Silhouette Score', palette='PuRd')
    plt.title('Silhouette Score Comparison')
    plt.ylim(0, 1)
    
    # Cluster counts bar plot
    plt.subplot(1, 2, 2)
    sns.barplot(data=metrics_df, x='Method', y='Clusters', palette='magma')
    plt.title('Number of Clusters')
    
    plt.tight_layout()
    plt.savefig('../outputs/plots/comparison_results.png')
    plt.close()

if __name__ == "__main__":
    # Load data and labels
    X_scaled = pd.read_csv('../outputs/scaled_data.csv').values
    dbscan, agnes, kmeans = load_cluster_labels()
    
    # Compare methods
    metrics_df = calculate_metrics(X_scaled, dbscan, agnes, kmeans)
    print("\n⭐ Clustering Performance Comparison:")
    print(metrics_df.to_markdown(index=False))
    
    # Save and plot results
    metrics_df.to_csv('../outputs/clustering_metrics.csv', index=False)
    plot_results(metrics_df)
    print("\n📊 Visualizations saved to '../outputs/plots/comparison_results.png'")