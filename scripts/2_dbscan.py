# Import libraries
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_excel('data/BusinessDS.xlsx')
    return df

def preprocess(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numeric_cols])
    return X_scaled

def find_optimal_eps(X_scaled, k=4):
    nn = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    distances = np.sort(distances[:, -1])
    plt.plot(distances)
    plt.savefig('outputs/plots/k_distance_graph.png')
    plt.close()

def run_dbscan(X_scaled, eps=0.3, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)
    pd.Series(clusters).to_csv('outputs/clusters/dbscan_labels.csv', index=False)
    return clusters

if __name__ == "__main__":
    df = load_data()
    X_scaled = preprocess(df)
    find_optimal_eps(X_scaled)
    clusters = run_dbscan(X_scaled)
    print(f"Clusters: {len(set(clusters))}, Noise: {list(clusters).count(-1)}")