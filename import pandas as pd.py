import pandas as pd
import numpy as np

df = pd.read_excel(r"C:\Users\arjun\OneDrive\Desktop\amazon_sales_dataset.xlsx")
print(df)
# Step 3: Confirm successful loading
print("Dataset loaded successfully!")
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")

# Step 4: View first 5 records
print("\nFirst 5 Records:")
print(df.head())

# Step 5: View column names and data types
print("\nColumn Names and Data Types:")
print(df.dtypes)

# Step 6: Basic dataset information
print("\nDataset Info:")
print(df.info())

# Step 7: Check for missing values
print("\nMissing Values per Column:")
print(df.isnull().sum())

# Step 8: Basic statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Step 9: Check unique values in categorical columns
print("\nUnique Product Categories:", df['product_category'].unique())
print("Unique Customer Regions:", df['customer_region'].unique())
print("Unique Payment Methods:", df['payment_method'].unique())

# Step 10: Check date range
print(f"\nDate Range: {df['order_date'].min()} to {df['order_date'].max()}")