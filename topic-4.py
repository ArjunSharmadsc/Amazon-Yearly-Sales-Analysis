import os
os.chdir(r"C:\Users\arjun\OneDrive\Desktop\data mining project")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold, RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

sns.set_theme(style="whitegrid")

# -----------------------------------------------
# STEP 1: Load and Prepare Dataset
# -----------------------------------------------
df = pd.read_excel(r"C:\Users\arjun\OneDrive\Desktop\amazon_sales_dataset.xlsx")
print("Dataset loaded. Shape:", df.shape)

# Encode categorical columns
le = LabelEncoder()
df['product_category_enc'] = le.fit_transform(df['product_category'])
df['customer_region_enc']  = le.fit_transform(df['customer_region'])
df['payment_method_enc']   = le.fit_transform(df['payment_method'])

# Define features and target
features = ['price', 'discount_percent', 'quantity_sold', 'rating',
            'review_count', 'discounted_price',
            'product_category_enc', 'customer_region_enc', 'payment_method_enc']

X = df[features]
y = df['total_revenue']

print("\nFeatures used:", features)
print("Target: total_revenue")

# -----------------------------------------------
# STEP 2: Filter Method 1 — Variance Threshold
# -----------------------------------------------
print("\n--- Variance Threshold ---")
selector = VarianceThreshold(threshold=0.1)
selector.fit(X)

variances = pd.Series(selector.variances_, index=features).sort_values(ascending=False)
print(variances.round(3))

# Features retained after threshold
retained = X.columns[selector.get_support()].tolist()
removed  = X.columns[~selector.get_support()].tolist()
print(f"\nRetained Features ({len(retained)}): {retained}")
print(f"Removed Features  ({len(removed)}): {removed}")

# Plot variances
plt.figure(figsize=(10, 5))
variances.plot(kind='bar', color='steelblue', edgecolor='white')
plt.title('Feature Variances', fontsize=14, fontweight='bold')
plt.ylabel('Variance')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('variance_threshold.png', dpi=150)
plt.show()
print("Saved: variance_threshold.png")

# -----------------------------------------------
# STEP 3: Filter Method 2 — Correlation with Target
# -----------------------------------------------
print("\n--- Correlation with Target (total_revenue) ---")
corr_with_target = X.copy()
corr_with_target['total_revenue'] = y
corr_values = corr_with_target.corr()['total_revenue'].drop('total_revenue').sort_values(ascending=False)
print(corr_values.round(3))

plt.figure(figsize=(10, 5))
corr_values.plot(kind='bar', color='coral', edgecolor='white')
plt.title('Feature Correlation with Total Revenue', fontsize=14, fontweight='bold')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=30)
plt.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('correlation_target.png', dpi=150)
plt.show()
print("Saved: correlation_target.png")

# -----------------------------------------------
# STEP 4: Filter Method 3 — SelectKBest (f_regression)
# -----------------------------------------------
print("\n--- SelectKBest with f_regression (Top 5 Features) ---")
skb = SelectKBest(score_func=f_regression, k=5)
skb.fit(X, y)

scores = pd.Series(skb.scores_, index=features).sort_values(ascending=False)
print(scores.round(2))

best_features = X.columns[skb.get_support()].tolist()
print(f"\nTop 5 Selected Features: {best_features}")

plt.figure(figsize=(10, 5))
scores.plot(kind='bar', color='mediumseagreen', edgecolor='white')
plt.title('SelectKBest — F-Regression Scores', fontsize=14, fontweight='bold')
plt.ylabel('F-Score')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('selectkbest_scores.png', dpi=150)
plt.show()
print("Saved: selectkbest_scores.png")

# -----------------------------------------------
# STEP 5: Wrapper Method — RFE with Linear Regression
# -----------------------------------------------
print("\n--- RFE with Linear Regression (Top 5 Features) ---")
lr = LinearRegression()
rfe = RFE(estimator=lr, n_features_to_select=5)
rfe.fit(X, y)

rfe_df = pd.DataFrame({
    'Feature': features,
    'Selected': rfe.support_,
    'Ranking': rfe.ranking_
}).sort_values('Ranking')

print(rfe_df.to_string(index=False))

rfe_selected = rfe_df[rfe_df['Selected'] == True]['Feature'].tolist()
print(f"\nRFE Selected Features: {rfe_selected}")

plt.figure(figsize=(10, 5))
plt.barh(rfe_df['Feature'], rfe_df['Ranking'], color='mediumpurple')
plt.xlabel('Ranking (1 = Best)')
plt.title('RFE Feature Rankings', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('rfe_rankings.png', dpi=150)
plt.show()
print("Saved: rfe_rankings.png")

# -----------------------------------------------
# STEP 6: Embedded Method — Random Forest Importance
# -----------------------------------------------
print("\n--- Random Forest Feature Importance ---")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(importance_df.to_string(index=False))

plt.figure(figsize=(10, 5))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='darkorange')
plt.xlabel('Importance Score')
plt.title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('rf_importance.png', dpi=150)
plt.show()
print("Saved: rf_importance.png")

# -----------------------------------------------
# STEP 7: Final Summary — Feature Selection Results
# -----------------------------------------------
print("\n--- Feature Selection Summary ---")
summary = pd.DataFrame({
    'Feature': features,
    'Variance': variances[features].values.round(2),
    'Corr_with_Target': corr_values[features].values.round(3),
    'F_Score': scores[features].values.round(2),
    'RFE_Rank': rfe_df.set_index('Feature').loc[features, 'Ranking'].values,
    'RF_Importance': importance_df.set_index('Feature').loc[features, 'Importance'].values.round(4)
})
print(summary.to_string(index=False))

print("\nFinal Recommended Features (based on RF Importance):")
top_features = importance_df.head(5)['Feature'].tolist()
print(top_features)