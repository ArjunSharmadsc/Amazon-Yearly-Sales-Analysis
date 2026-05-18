# ─── TOPIC 8: Principal Component Analysis (PCA) ───

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# ── 1. Load & Prepare Data ────────────────────────────────────────────
df = pd.read_excel(r"C:\Users\arjun\OneDrive\Desktop\amazon_sales_dataset.xlsx")

# Encode categorical columns
le = LabelEncoder()
df['cat_encoded'] = le.fit_transform(df['product_category'])
df['region_encoded'] = le.fit_transform(df['customer_region'])
df['pay_encoded'] = le.fit_transform(df['payment_method'])

features = [
    'price', 'discount_percent', 'quantity_sold', 'rating',
    'review_count', 'discounted_price', 'total_revenue',
    'cat_encoded', 'region_encoded', 'pay_encoded'
]

X = df[features].copy()
print("Original shape:", X.shape)

# ── 2. Standardise Data ───────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 3. Fit Full PCA ───────────────────────────────────────────────────
pca_full = PCA()
pca_full.fit(X_scaled)

explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print("\nExplained Variance per Component:")
for i, (ev, cv) in enumerate(zip(explained_var, cumulative_var)):
    print(f"PC{i+1}: {ev*100:.2f}% | Cumulative: {cv*100:.2f}%")

# ── 4. Scree Plot ─────────────────────────────────────────────────────
n_components = len(explained_var)
x_ticks = range(1, n_components + 1)

fig, ax1 = plt.subplots(figsize=(9, 5))

ax1.bar(x_ticks, explained_var * 100, color='steelblue',
        alpha=0.7, label='Individual Variance')
ax1.set_xlabel("Principal Component")
ax1.set_ylabel("Explained Variance (%)", color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.set_xticks(x_ticks)

ax2 = ax1.twinx()
ax2.plot(x_ticks, cumulative_var * 100, color='tomato',
         marker='o', linewidth=2, label='Cumulative Variance')
ax2.axhline(95, color='green', linestyle='--', linewidth=1,
            label='95% Threshold')
ax2.set_ylabel("Cumulative Variance (%)", color='tomato')
ax2.tick_params(axis='y', labelcolor='tomato')
ax2.set_ylim(0, 105)

fig.suptitle("Scree Plot – PCA Explained Variance")
fig.legend(loc='lower right', bbox_to_anchor=(0.9, 0.15))
plt.tight_layout()
plt.savefig("pca_scree.png", dpi=150)
plt.show()

# ── 5. Choose Number of Components for 95% Variance ──────────────────
n_95 = np.argmax(cumulative_var >= 0.95) + 1
print(f"\nComponents needed to explain 95% variance: {n_95}")

# ── 6. Apply PCA with Selected Components ─────────────────────────────
pca = PCA(n_components=n_95)
X_pca = pca.fit_transform(X_scaled)

print(f"Reduced shape: {X_pca.shape}")
print(f"Total variance retained: {pca.explained_variance_ratio_.sum()*100:.2f}%")

# ── 7. Component Loading Matrix Heatmap ───────────────────────────────
loadings = pd.DataFrame(
    pca.components_.T,
    index=features,
    columns=[f'PC{i+1}' for i in range(n_95)]
)

plt.figure(figsize=(max(8, n_95 * 1.2), 6))
sns.heatmap(loadings, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5)
plt.title("PCA Component Loadings")
plt.tight_layout()
plt.savefig("pca_loadings.png", dpi=150)
plt.show()

# ── 8. PCA 2D Scatter Plot ────────────────────────────────────────────
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
categories = df['product_category'].unique()
palette = sns.color_palette("Set2", len(categories))

for cat, col in zip(categories, palette):
    mask = df['product_category'] == cat
    plt.scatter(
        X_2d[mask, 0],
        X_2d[mask, 1],
        label=cat,
        alpha=0.3,
        s=10,
        color=col
    )

plt.title("PCA 2D Projection – Coloured by Product Category")
plt.xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.legend(markerscale=3, title="Category")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("pca_2d_scatter.png", dpi=150)
plt.show()

# ── 9. Biplot ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

# Safe sampling: works even if data has less than 1000 rows
sample_size = min(1000, len(X_2d))
sample_idx = np.random.choice(len(X_2d), size=sample_size, replace=False)

ax.scatter(
    X_2d[sample_idx, 0],
    X_2d[sample_idx, 1],
    alpha=0.2,
    s=8,
    color='steelblue'
)

# Feature vectors
scale = 3
for i, feat in enumerate(features):
    ax.annotate(
        '',
        xy=(pca_2d.components_[0, i] * scale,
            pca_2d.components_[1, i] * scale),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color='red', lw=1.5)
    )

    ax.text(
        pca_2d.components_[0, i] * scale * 1.15,
        pca_2d.components_[1, i] * scale * 1.15,
        feat,
        fontsize=8,
        color='darkred'
    )

ax.axhline(0, color='grey', linewidth=0.5)
ax.axvline(0, color='grey', linewidth=0.5)
ax.set_title("PCA Biplot – Data Points + Feature Directions")
ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
plt.tight_layout()
plt.savefig("pca_biplot.png", dpi=150)
plt.show()

# ── 10. Dimensionality Reduction Summary ──────────────────────────────
print("\n─── Dimensionality Reduction Summary ───")
print(f"Original dimensions : {X.shape[1]}")
print(f"Reduced dimensions  : {n_95}")
print(f"Reduction           : {round((1 - n_95 / X.shape[1]) * 100, 1)}%")
print(f"Variance retained   : {pca.explained_variance_ratio_.sum()*100:.2f}%")