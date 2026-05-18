# ─── TOPIC 5: Classification – Decision Tree & Logistic Regression ───

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# ── 1. Load Data ──────────────────────────────────────────────────────
df = pd.read_excel(r"C:\Users\arjun\OneDrive\Desktop\amazon_sales_dataset.xlsx")
print("Shape:", df.shape)
print(df.head())

# ── 2. Create Binary Target Variable ─────────────────────────────────
# High Revenue = 1 if total_revenue >= median, else 0
median_rev = df['total_revenue'].median()
df['high_revenue'] = (df['total_revenue'] >= median_rev).astype(int)
print("\nTarget Distribution:\n", df['high_revenue'].value_counts())

# ── 3. Feature Selection ──────────────────────────────────────────────
# Drop non-predictive columns and the target source column
features = ['price', 'discount_percent', 'quantity_sold',
            'rating', 'review_count', 'product_category',
            'customer_region', 'payment_method']

X = df[features].copy()
y = df['high_revenue']

# ── 4. Encode Categorical Columns ────────────────────────────────────
le = LabelEncoder()
for col in ['product_category', 'customer_region', 'payment_method']:
    X[col] = le.fit_transform(X[col])

print("\nFeature matrix sample:\n", X.head())

# ── 5. Train-Test Split ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

# ── 6. Feature Scaling (needed for Logistic Regression) ───────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ═══════════════════════════════════════════════════════
#  MODEL 1: DECISION TREE
# ═══════════════════════════════════════════════════════

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

print("\n─── Decision Tree Results ───")
print("Accuracy:", round(accuracy_score(y_test, y_pred_dt) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))

# Confusion Matrix – Decision Tree
cm_dt = confusion_matrix(y_test, y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt,
                                  display_labels=["Low Revenue", "High Revenue"])
disp_dt.plot(cmap='Blues')
plt.title("Decision Tree – Confusion Matrix")
plt.tight_layout()
plt.savefig("dt_confusion_matrix.png", dpi=150)
plt.show()

# Visualise the Decision Tree (top 3 levels)
plt.figure(figsize=(20, 8))
plot_tree(dt_model, feature_names=features,
          class_names=["Low", "High"],
          filled=True, max_depth=3, fontsize=10)
plt.title("Decision Tree Visualization (depth=3)")
plt.tight_layout()
plt.savefig("decision_tree_plot.png", dpi=150)
plt.show()

# Feature Importance – Decision Tree
importances = pd.Series(dt_model.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=importances.values, y=importances.index, palette='viridis')
plt.title("Decision Tree – Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("dt_feature_importance.png", dpi=150)
plt.show()

# ═══════════════════════════════════════════════════════
#  MODEL 2: LOGISTIC REGRESSION
# ═══════════════════════════════════════════════════════

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)

print("\n─── Logistic Regression Results ───")
print("Accuracy:", round(accuracy_score(y_test, y_pred_lr) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))

# Confusion Matrix – Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr,
                                  display_labels=["Low Revenue", "High Revenue"])
disp_lr.plot(cmap='Oranges')
plt.title("Logistic Regression – Confusion Matrix")
plt.tight_layout()
plt.savefig("lr_confusion_matrix.png", dpi=150)
plt.show()

# Coefficients – Logistic Regression
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': lr_model.coef_[0]
}).sort_values('Coefficient', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='coolwarm')
plt.title("Logistic Regression – Feature Coefficients")
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig("lr_coefficients.png", dpi=150)
plt.show()

# ═══════════════════════════════════════════════════════
#  COMPARISON TABLE
# ═══════════════════════════════════════════════════════

dt_acc  = accuracy_score(y_test, y_pred_dt)
lr_acc  = accuracy_score(y_test, y_pred_lr)

comparison = pd.DataFrame({
    'Model':    ['Decision Tree', 'Logistic Regression'],
    'Accuracy': [round(dt_acc*100, 2), round(lr_acc*100, 2)]
})
print("\n─── Model Accuracy Comparison ───")
print(comparison.to_string(index=False))

plt.figure(figsize=(6, 4))
sns.barplot(x='Model', y='Accuracy', data=comparison, palette='Set2')
plt.ylim(0, 100)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
for i, v in enumerate(comparison['Accuracy']):
    plt.text(i, v + 0.5, f"{v}%", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()