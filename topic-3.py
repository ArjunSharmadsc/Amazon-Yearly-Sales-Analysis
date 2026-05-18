# Topic 3 - Exploratory Data Analysis (EDA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# -----------------------------------------------
# STEP 1: Load Dataset
# -----------------------------------------------
df = pd.read_excel(r"C:\Users\arjun\OneDrive\Desktop\amazon_sales_dataset.xlsx")
print("Dataset loaded. Shape:", df.shape)

# -----------------------------------------------
# STEP 2: Descriptive Statistics
# -----------------------------------------------
print("\n--- Basic Statistical Summary ---")
print(df.describe())

print("\n--- Mean of Numerical Columns ---")
print(df[['price', 'discount_percent', 'quantity_sold',
          'rating', 'review_count', 'total_revenue']].mean().round(2))

print("\n--- Median of Numerical Columns ---")
print(df[['price', 'discount_percent', 'quantity_sold',
          'rating', 'review_count', 'total_revenue']].median())

print("\n--- Mode of Categorical Columns ---")
print("Product Category:", df['product_category'].mode()[0])
print("Customer Region:", df['customer_region'].mode()[0])
print("Payment Method:", df['payment_method'].mode()[0])

print("\n--- Standard Deviation ---")
print(df[['price', 'total_revenue', 'rating']].std().round(2))

print("\n--- Skewness ---")
print(df[['price', 'total_revenue', 'rating', 'review_count']].skew().round(3))

print("\n--- Kurtosis ---")
print(df[['price', 'total_revenue', 'rating', 'review_count']].kurtosis().round(3))

print("\n--- Value Counts: Product Category ---")
print(df['product_category'].value_counts())

print("\n--- Value Counts: Customer Region ---")
print(df['customer_region'].value_counts())

print("\n--- Value Counts: Payment Method ---")
print(df['payment_method'].value_counts())

# -----------------------------------------------
# STEP 3: Histogram — Distribution of Price
# -----------------------------------------------
plt.figure(figsize=(10, 5))
plt.hist(df['price'], bins=30, color='steelblue', edgecolor='white')
plt.title('Distribution of Product Price', fontsize=14, fontweight='bold')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('hist_price.png', dpi=150)
plt.show()
print("Saved: hist_price.png")

# -----------------------------------------------
# STEP 4: Histogram — Distribution of Total Revenue
# -----------------------------------------------
plt.figure(figsize=(10, 5))
plt.hist(df['total_revenue'], bins=30, color='coral', edgecolor='white')
plt.title('Distribution of Total Revenue', fontsize=14, fontweight='bold')
plt.xlabel('Total Revenue')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('hist_revenue.png', dpi=150)
plt.show()
print("Saved: hist_revenue.png")

# -----------------------------------------------
# STEP 5: Histogram — Distribution of Rating
# -----------------------------------------------
plt.figure(figsize=(10, 5))
plt.hist(df['rating'], bins=20, color='mediumseagreen', edgecolor='white')
plt.title('Distribution of Customer Ratings', fontsize=14, fontweight='bold')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('hist_rating.png', dpi=150)
plt.show()
print("Saved: hist_rating.png")

# -----------------------------------------------
# STEP 6: Bar Chart — Sales by Product Category
# -----------------------------------------------
category_revenue = df.groupby('product_category')['total_revenue'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
category_revenue.plot(kind='bar', color='steelblue', edgecolor='white')
plt.title('Total Revenue by Product Category', fontsize=14, fontweight='bold')
plt.xlabel('Product Category')
plt.ylabel('Total Revenue')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('bar_category_revenue.png', dpi=150)
plt.show()
print("Saved: bar_category_revenue.png")

# -----------------------------------------------
# STEP 7: Bar Chart — Orders by Customer Region
# -----------------------------------------------
region_counts = df['customer_region'].value_counts()

plt.figure(figsize=(8, 5))
region_counts.plot(kind='bar', color='mediumpurple', edgecolor='white')
plt.title('Number of Orders by Customer Region', fontsize=14, fontweight='bold')
plt.xlabel('Customer Region')
plt.ylabel('Number of Orders')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('bar_region_orders.png', dpi=150)
plt.show()
print("Saved: bar_region_orders.png")

# -----------------------------------------------
# STEP 8: Bar Chart — Payment Method Distribution
# -----------------------------------------------
payment_counts = df['payment_method'].value_counts()

plt.figure(figsize=(9, 5))
payment_counts.plot(kind='bar', color='darkorange', edgecolor='white')
plt.title('Orders by Payment Method', fontsize=14, fontweight='bold')
plt.xlabel('Payment Method')
plt.ylabel('Number of Orders')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('bar_payment.png', dpi=150)
plt.show()
print("Saved: bar_payment.png")

# -----------------------------------------------
# STEP 9: Pie Chart — Product Category Share
# -----------------------------------------------
category_counts = df['product_category'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%',
        startangle=140, colors=sns.color_palette("pastel"))
plt.title('Product Category Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('pie_category.png', dpi=150)
plt.show()
print("Saved: pie_category.png")

# -----------------------------------------------
# STEP 10: Box Plot — Price by Product Category
# -----------------------------------------------
plt.figure(figsize=(11, 6))
sns.boxplot(data=df, x='product_category', y='price', palette='Set2')
plt.title('Price Distribution by Product Category', fontsize=14, fontweight='bold')
plt.xlabel('Product Category')
plt.ylabel('Price')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('box_price_category.png', dpi=150)
plt.show()
print("Saved: box_price_category.png")

# -----------------------------------------------
# STEP 11: Box Plot — Revenue by Region
# -----------------------------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='customer_region', y='total_revenue', palette='Set3')
plt.title('Revenue Distribution by Customer Region', fontsize=14, fontweight='bold')
plt.xlabel('Customer Region')
plt.ylabel('Total Revenue')
plt.tight_layout()
plt.savefig('box_revenue_region.png', dpi=150)
plt.show()
print("Saved: box_revenue_region.png")

# -----------------------------------------------
# STEP 12: Correlation Heatmap
# -----------------------------------------------
numerical_cols = ['price', 'discount_percent', 'quantity_sold',
                  'rating', 'review_count', 'discounted_price', 'total_revenue']

corr_matrix = df[numerical_cols].corr()

print("\n--- Correlation Matrix ---")
print(corr_matrix.round(2))

plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
            linewidths=0.5, square=True)
plt.title('Correlation Heatmap of Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('heatmap_correlation.png', dpi=150)
plt.show()
print("Saved: heatmap_correlation.png")

# -----------------------------------------------
# STEP 13: Scatter Plot — Price vs Total Revenue
# -----------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(df['price'], df['total_revenue'], alpha=0.3, color='steelblue', s=10)
plt.title('Price vs Total Revenue', fontsize=14, fontweight='bold')
plt.xlabel('Price')
plt.ylabel('Total Revenue')
plt.tight_layout()
plt.savefig('scatter_price_revenue.png', dpi=150)
plt.show()
print("Saved: scatter_price_revenue.png")

# -----------------------------------------------
# STEP 14: Scatter Plot — Rating vs Review Count
# -----------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(df['rating'], df['review_count'], alpha=0.3, color='coral', s=10)
plt.title('Rating vs Review Count', fontsize=14, fontweight='bold')
plt.xlabel('Rating')
plt.ylabel('Review Count')
plt.tight_layout()
plt.savefig('scatter_rating_reviews.png', dpi=150)
plt.show()
print("Saved: scatter_rating_reviews.png")

print("\n--- EDA Complete. All plots saved. ---")