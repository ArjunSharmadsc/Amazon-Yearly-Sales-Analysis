# ─── TOPIC 9: Model Evaluation & Comparison ───

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc, RocCurveDisplay)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ── 1. Load & Prepare Data ────────────────────────────────────────────
df = pd.read_excel(r"C:\Users\arjun\OneDrive\Desktop\amazon_sales_dataset.xlsx")

# Binary target: High Revenue = 1 if total_revenue >= median
df['high_revenue'] = (df['total_revenue'] >= df['total_revenue'].median()).astype(int)

# Features
le = LabelEncoder()
df['cat_enc']    = le.fit_transform(df['product_category'])
df['region_enc'] = le.fit_transform(df['customer_region'])
df['pay_enc']    = le.fit_transform(df['payment_method'])

features = ['price', 'discount_percent', 'quantity_sold',
            'rating', 'review_count', 'cat_enc', 'region_enc', 'pay_enc']

X = df[features]
y = df['high_revenue']

print("Dataset shape:", X.shape)
print("Target distribution:\n", y.value_counts())

# ── 2. Train-Test Split ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Scale (required for LR, KNN)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 3. Define All Models ──────────────────────────────────────────────
models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"       : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN"                 : KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes"         : GaussianNB()
}

# Models that need scaled data
scaled_models = {"Logistic Regression", "KNN"}

# ── 4. Train, Predict & Collect Metrics ──────────────────────────────
results = []

for name, model in models.items():
    X_tr = X_train_sc if name in scaled_models else X_train
    X_te = X_test_sc  if name in scaled_models else X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        'Model'     : name,
        'Accuracy'  : round(acc  * 100, 2),
        'Precision' : round(prec * 100, 2),
        'Recall'    : round(rec  * 100, 2),
        'F1-Score'  : round(f1   * 100, 2)
    })

    print(f"\n{'─'*40}")
    print(f"Model: {name}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall   : {rec*100:.2f}%")
    print(f"  F1-Score : {f1*100:.2f}%")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Low Rev','High Rev'])}")

results_df = pd.DataFrame(results).set_index('Model')
print("\n─── Final Comparison Table ───")
print(results_df.to_string())

# ── 5. Grouped Bar Chart – All Metrics ───────────────────────────────
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(results_df))
width = 0.2
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

fig, ax = plt.subplots(figsize=(13, 6))
for i, (metric, color) in enumerate(zip(metrics, colors)):
    bars = ax.bar(x + i * width, results_df[metric],
                  width, label=metric, color=color, alpha=0.85, edgecolor='white')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{bar.get_height():.1f}",
                ha='center', va='bottom', fontsize=7.5)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(results_df.index, fontsize=10)
ax.set_ylim(0, 115)
ax.set_ylabel("Score (%)")
ax.set_title("Model Evaluation – Accuracy, Precision, Recall, F1-Score Comparison")
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("model_comparison_grouped.png", dpi=150)
plt.show()

# ── 6. Heatmap of All Metrics ─────────────────────────────────────────
plt.figure(figsize=(8, 5))
sns.heatmap(results_df, annot=True, fmt='.2f', cmap='YlGn',
            linewidths=0.5, linecolor='grey',
            vmin=50, vmax=100)
plt.title("Model Performance Heatmap (all metrics in %)")
plt.tight_layout()
plt.savefig("model_heatmap.png", dpi=150)
plt.show()

# ── 7. Confusion Matrices – All 5 Models (grid) ───────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for idx, (name, model) in enumerate(models.items()):
    X_te = X_test_sc if name in scaled_models else X_test
    y_pred = model.predict(X_te)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Low Rev', 'High Rev'])
    disp.plot(ax=axes[idx], cmap='Blues', colorbar=False)
    axes[idx].set_title(name, fontsize=11, fontweight='bold')

axes[-1].set_visible(False)   # hide 6th empty subplot
fig.suptitle("Confusion Matrices – All Models", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("all_confusion_matrices.png", dpi=150)
plt.show()

# ── 8. ROC Curves – All Models ────────────────────────────────────────
plt.figure(figsize=(9, 7))
roc_colors = ['#4C72B0','#DD8452','#55A868','#C44E52','#9B59B6']

for (name, model), color in zip(models.items(), roc_colors):
    X_te = X_test_sc if name in scaled_models else X_test
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_te)[:, 1]
    else:
        y_prob = model.decision_function(X_te)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, linewidth=2,
             label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Guess (AUC = 0.500)')
plt.title("ROC Curves – All Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.legend(loc='lower right', fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.show()

# ── 9. Cross-Validation (5-Fold) ─────────────────────────────────────
print("\n─── 5-Fold Cross-Validation (F1-Score) ───")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}
for name, model in models.items():
    X_cv = X_train_sc if name in scaled_models else X_train
    scores = cross_val_score(model, X_cv, y_train,
                             cv=cv, scoring='f1', n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:<22}: mean={scores.mean()*100:.2f}%  std=±{scores.std()*100:.2f}%")

# CV Box Plot
cv_df = pd.DataFrame(cv_results) * 100
plt.figure(figsize=(9, 5))
cv_df.boxplot(column=list(models.keys()), patch_artist=True,
              boxprops=dict(facecolor='lightblue'),
              medianprops=dict(color='red', linewidth=2))
plt.title("5-Fold Cross-Validation – F1-Score Distribution")
plt.ylabel("F1-Score (%)")
plt.xticks(rotation=15)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("cv_boxplot.png", dpi=150)
plt.show()

# ── 10. Radar Chart – Best Model Visualisation ───────────────────────
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # close the loop

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
radar_colors = ['#4C72B0','#DD8452','#55A868','#C44E52','#9B59B6']

for (_, row), color in zip(results_df.iterrows(), radar_colors):
    values = row[labels].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, color=color, label=row.name)
    ax.fill(angles, values, alpha=0.07, color=color)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_ylim(50, 100)
ax.set_title("Radar Chart – Model Comparison", fontsize=13,
             fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9)
plt.tight_layout()
plt.savefig("radar_chart.png", dpi=150)
plt.show()

# ── 11. Final Summary Table ───────────────────────────────────────────
print("\n" + "═"*55)
print("        FINAL MODEL EVALUATION SUMMARY")
print("═"*55)
print(results_df.to_string())
print("═"*55)
best = results_df['F1-Score'].idxmax()
print(f"\n★  Best Model (by F1-Score): {best}")
print(f"   F1-Score : {results_df.loc[best,'F1-Score']}%")
print(f"   Accuracy : {results_df.loc[best,'Accuracy']}%")