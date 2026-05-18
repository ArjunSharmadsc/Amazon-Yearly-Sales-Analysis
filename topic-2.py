# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Step 2: Load the dataset
df = pd.read_excel(r"C:\Users\arjun\OneDrive\Desktop\amazon_sales_dataset.xlsx")
import random
random.seed(42)

# Randomly insert NaN into 200 rows for 'rating' and 150 rows for 'customer_region'
missing_indices_rating = random.sample(range(len(df)), 200)
missing_indices_region = random.sample(range(len(df)), 150)

df.loc[missing_indices_rating, 'rating'] = np.nan
df.loc[missing_indices_region, 'customer_region'] = np.nan

print("\nMissing values after simulation:")
print(df.isnull().sum())

# -----------------------------------------------
# STEP 3: Introduce Duplicate Records (Simulation)
# -----------------------------------------------
duplicate_rows = df.iloc[0:50]  # Take first 50 rows
df = pd.concat([df, duplicate_rows], ignore_index=True)

print(f"\nShape after adding duplicates: {df.shape}")
print(f"Total duplicate rows found: {df.duplicated().sum()}")

# -----------------------------------------------
# STEP 4: Handle Missing Values
# -----------------------------------------------

# For numerical column 'rating' -> fill with mean
mean_rating = df['rating'].mean()
df['rating'].fillna(round(mean_rating, 2), inplace=True)
print(f"\nFilled missing 'rating' values with mean: {round(mean_rating, 2)}")

# For categorical column 'customer_region' -> fill with mode
mode_region = df['customer_region'].mode()[0]
df['customer_region'].fillna(mode_region, inplace=True)
print(f"Filled missing 'customer_region' values with mode: {mode_region}")

# Verify no missing values remain
print("\nMissing values after handling:")
print(df.isnull().sum())

# -----------------------------------------------
# STEP 5: Remove Duplicate Records
# -----------------------------------------------
df = df.drop_duplicates()
print(f"\nShape after removing duplicates: {df.shape}")
print("Duplicate rows remaining:", df.duplicated().sum())

# -----------------------------------------------
# STEP 6: Encoding Categorical Data
# -----------------------------------------------

# --- Label Encoding ---
# Suitable for: product_category, customer_region, payment_method

le = LabelEncoder()

df['product_category_encoded'] = le.fit_transform(df['product_category'])
print("\nLabel Encoding - product_category:")
print(dict(zip(le.classes_, le.transform(le.classes_))))

df['customer_region_encoded'] = le.fit_transform(df['customer_region'])
print("\nLabel Encoding - customer_region:")
print(dict(zip(le.classes_, le.transform(le.classes_))))

df['payment_method_encoded'] = le.fit_transform(df['payment_method'])
print("\nLabel Encoding - payment_method:")
print(dict(zip(le.classes_, le.transform(le.classes_))))

# --- One-Hot Encoding ---
# Suitable for nominal categories where no order exists
df_ohe = pd.get_dummies(df[['product_category', 'customer_region', 'payment_method']], 
                         prefix=['cat', 'region', 'pay'])
print("\nOne-Hot Encoded Columns:")
print(df_ohe.columns.tolist())
print("\nSample One-Hot Encoded Data (first 3 rows):")
print(df_ohe.head(3))

# -----------------------------------------------
# STEP 7: Final Preprocessed Dataset Summary
# -----------------------------------------------
print("\n--- Final Preprocessed Dataset ---")
print("Shape:", df.shape)
print("Missing Values:", df.isnull().sum().sum())
print("Duplicates:", df.duplicated().sum())
print("\nSample of preprocessed data:")
print(df[['order_id', 'product_category', 'product_category_encoded',
          'customer_region', 'customer_region_encoded',
          'payment_method', 'payment_method_encoded',
          'rating']].head(10))