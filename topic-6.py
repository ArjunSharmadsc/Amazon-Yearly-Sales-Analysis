# ─── TOPIC 6: K-Means Clustering ───

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA  # only for 2D visualisation

# ── 1. Load & Prepare Data ────────────────────────────────────────────
df = pd.read_excel(r"C:\Users\arjun\OneDrive\Desktop\amazon_sales_dataset.xlsx")

# Select numerical features for clustering
cluster_features = ['price', 'discount_percent', 'quantity_sold',
                    'rating', 'review_count', 'total_revenue']

X = df[cluster_features].copy()

# Scale features (K-Means is distance-based, scaling is essential)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 2. Elbow Method – Find Optimal K ─────────────────────────────────
wcss = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o', color='steelblue', linewidth=2)
plt.title("Elbow Method – Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.xticks(K_range)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("elbow_plot.png", dpi=150)
plt.show()

# ── 3. Silhouette Score for K = 2 to 10 ──────────────────────────────
sil_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), sil_scores, marker='s', color='tomato', linewidth=2)
plt.title("Silhouette Score vs Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(range(2, 11))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("silhouette_plot.png", dpi=150)
plt.show()

print("Silhouette Scores:")
for k, s in zip(range(2, 11), sil_scores):
    print(f"  K={k}: {round(s, 4)}")

# ── 4. Final K-Means Model (use K=3 or best from elbow) ──────────────
OPTIMAL_K = 3

kmeans = KMeans(n_clusters=OPTIMAL_K, init='k-means++',
                random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print(f"\nCluster Distribution (K={OPTIMAL_K}):")
print(df['Cluster'].value_counts().sort_index())

# ── 5. Cluster Profile – Mean values per cluster ──────────────────────
cluster_profile = df.groupby('Cluster')[cluster_features].mean().round(2)
print("\nCluster Profiles (Mean values):\n", cluster_profile)

# Heatmap of cluster profiles
plt.figure(figsize=(10, 4))
sns.heatmap(cluster_profile.T, annot=True, fmt='.1f', cmap='YlOrRd',
            linewidths=0.5)
plt.title("Cluster Profiles – Mean Feature Values")
plt.xlabel("Cluster")
plt.tight_layout()
plt.savefig("cluster_heatmap.png", dpi=150)
plt.show()

# ── 6. 2D Visualisation using PCA ────────────────────────────────────
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(9, 6))
colors = ['#E63946', '#457B9D', '#2A9D8F']
for i in range(OPTIMAL_K):
    mask = df['Cluster'] == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                label=f'Cluster {i}', alpha=0.4, s=10, color=colors[i])

# Plot centroids
centroids_pca = pca_2d.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
            c='black', marker='X', s=200, zorder=5, label='Centroids')

plt.title(f"K-Means Clusters (K={OPTIMAL_K}) – PCA 2D View")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("kmeans_scatter.png", dpi=150)
plt.show()

# ── 7. Cluster vs Product Category ───────────────────────────────────
ct = pd.crosstab(df['Cluster'], df['product_category'])
ct.plot(kind='bar', figsize=(9, 5), colormap='Set2', edgecolor='black')
plt.title("Cluster Distribution by Product Category")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.legend(title="Category", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig("cluster_category.png", dpi=150)
plt.show()