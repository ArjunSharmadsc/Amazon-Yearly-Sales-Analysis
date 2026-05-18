# ─── TOPIC 7: Association Rule Mining ───

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Install if needed: pip install mlxtend
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ── 1. Load Data ──────────────────────────────────────────────────────
df = pd.read_excel("amazon_sales_dataset.xlsx")

# ── 2. Build Transactions ─────────────────────────────────────────────
# Each order = one transaction with: category + payment + region + revenue_tier

df['revenue_tier'] = pd.cut(df['total_revenue'],
                             bins=[0, 300, 800, 9999],
                             labels=['Low_Rev', 'Mid_Rev', 'High_Rev'])

df['discount_tier'] = pd.cut(df['discount_percent'],
                              bins=[-1, 10, 25, 100],
                              labels=['Low_Disc', 'Mid_Disc', 'High_Disc'])

# Combine items per transaction
transactions = df.apply(lambda row: [
    str(row['product_category']),
    str(row['payment_method']),
    str(row['customer_region']),
    str(row['revenue_tier']),
    str(row['discount_tier'])
], axis=1).tolist()

print(f"Total transactions: {len(transactions)}")
print("Sample transaction:", transactions[0])

# ── 3. Encode Transactions ────────────────────────────────────────────
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
te_df = pd.DataFrame(te_array, columns=te.columns_)

print(f"\nTransaction matrix shape: {te_df.shape}")
print(te_df.head(3))

# ── 4A. Apriori Algorithm ─────────────────────────────────────────────
print("\n─── Running Apriori ───")
frequent_itemsets_apriori = apriori(te_df, min_support=0.05,
                                     use_colnames=True, max_len=3)
frequent_itemsets_apriori['length'] = frequent_itemsets_apriori['itemsets'].apply(len)

print(f"Frequent Itemsets found (Apriori): {len(frequent_itemsets_apriori)}")
print(frequent_itemsets_apriori.sort_values('support', ascending=False).head(10))

# ── 4B. FP-Growth Algorithm ───────────────────────────────────────────
print("\n─── Running FP-Growth ───")
frequent_itemsets_fp = fpgrowth(te_df, min_support=0.05,
                                 use_colnames=True, max_len=3)
frequent_itemsets_fp['length'] = frequent_itemsets_fp['itemsets'].apply(len)

print(f"Frequent Itemsets found (FP-Growth): {len(frequent_itemsets_fp)}")
print(frequent_itemsets_fp.sort_values('support', ascending=False).head(10))

# ── 5. Generate Association Rules ─────────────────────────────────────
rules = association_rules(frequent_itemsets_fp,
                           metric="lift",
                           min_threshold=1.0,
                           num_itemsets=len(frequent_itemsets_fp))

rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)

print(f"\nTotal Rules Generated: {len(rules)}")
print("\nTop 10 Rules by Lift:")
print(rules[['antecedents', 'consequents',
             'support', 'confidence', 'lift']].head(10).to_string())

# Filter strong rules
strong_rules = rules[(rules['confidence'] >= 0.5) & (rules['lift'] > 1.2)]
print(f"\nStrong Rules (confidence≥0.5, lift>1.2): {len(strong_rules)}")
print(strong_rules[['antecedents', 'consequents',
                     'support', 'confidence', 'lift']].head(10).to_string())

# ── 6. Visualisations ─────────────────────────────────────────────────

# Support vs Confidence scatter coloured by Lift
plt.figure(figsize=(9, 6))
scatter = plt.scatter(rules['support'], rules['confidence'],
                      c=rules['lift'], cmap='RdYlGn', alpha=0.7, s=60)
plt.colorbar(scatter, label='Lift')
plt.title("Association Rules – Support vs Confidence (colour = Lift)")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("association_scatter.png", dpi=150)
plt.show()

# Top 15 rules by lift – bar chart
top15 = rules.head(15).copy()
top15['rule'] = top15.apply(
    lambda r: f"{set(r['antecedents'])} → {set(r['consequents'])}", axis=1)

plt.figure(figsize=(10, 7))
sns.barplot(x='lift', y='rule', data=top15, palette='viridis')
plt.title("Top 15 Association Rules by Lift")
plt.xlabel("Lift")
plt.ylabel("Rule")
plt.tight_layout()
plt.savefig("top_rules_lift.png", dpi=150)
plt.show()

# Frequent itemsets by length
plt.figure(figsize=(6, 4))
freq_lengths = frequent_itemsets_fp['length'].value_counts().sort_index()
sns.barplot(x=freq_lengths.index, y=freq_lengths.values, palette='Set1')
plt.title("Frequent Itemsets by Length")
plt.xlabel("Itemset Length")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("itemset_length.png", dpi=150)
plt.show()

# Support distribution
plt.figure(figsize=(7, 4))
plt.hist(frequent_itemsets_fp['support'], bins=30, color='steelblue', edgecolor='white')
plt.title("Distribution of Support Values")
plt.xlabel("Support")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("support_distribution.png", dpi=150)
plt.show()