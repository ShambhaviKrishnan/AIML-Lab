import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

# Generate dataset
X, _ = make_blobs(n_samples=300, centers=5, random_state=42)

# -----------------------------
# Elbow Method to find K
# -----------------------------
wcss = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(6,4))
plt.plot(K_range, wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.grid()
plt.show()

# -----------------------------
# Silhouette Score to choose best K
# -----------------------------
best_k = 2
best_score = -1

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)

    print(f"K = {k}, Silhouette Score = {score:.4f}")

    if score > best_score:
        best_score = score
        best_k = k

print("\nOptimal K (based on Silhouette Score):", best_k)

# -----------------------------
# Final K-Means with optimal K
# -----------------------------
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(6,4))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200, c='red', marker='X', label='Centroids')

plt.title(f"K-Means Clustering (K = {best_k})")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# Final Silhouette Score
# -----------------------------
final_score = silhouette_score(X, labels)
print("Final Silhouette Score:", round(final_score, 4))
