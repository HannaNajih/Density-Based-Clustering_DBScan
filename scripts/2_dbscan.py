import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load data
df = pd.read_csv('data/processed/cleaned_data.csv')
X = df.select_dtypes(include=['float64'])

# Find optimal eps (Elbow Method)
nn = NearestNeighbors(n_neighbors=2).fit(X)
distances, _ = nn.kneighbors(X)
distances = np.sort(distances[:, 1], axis=0)

plt.plot(distances)
plt.title('K-Distance Graph for Eps Selection')
plt.savefig('results/plots/k_distance.png')
plt.close()

# Run DBSCAN (adjust eps/min_samples)
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['cluster'] = dbscan.fit_predict(X)

# Save results
df.to_csv('results/clusters/dbscan_results.csv', index=False)

# Visualize
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df['cluster'])
plt.title('DBSCAN Clustering')
plt.savefig('results/plots/dbscan_clusters.png')