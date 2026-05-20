# Amazon Sales Data — Data Warehousing & Mining Project

# Overview
This project performs end-to-end data warehousing and mining operations on a real-world Amazon Sales Dataset sourced from Kaggle. 
It covers the complete data analytics lifecycle — from raw data loading and preprocessing to exploratory analysis and feature engineering — using Python and its core data science libraries.

# Dataset
| Property | Details |
|---|---|
| Source | [Kaggle — Amazon Sales Dataset](https://www.kaggle.com/) |
| Records | 50,000 rows × 13 columns |
| Date Range | January 2022 — December 2023 |
| Features | Order details, product categories, pricing, discounts, customer regions, payment methods, ratings, revenue |

# Project Structure
├── amazon_sales_dataset.csv         # Raw dataset
│
├── topic1_data_loading.py           # Data collection and loading
├── topic2_preprocessing.py          # Missing values, duplicates, encoding
├── topic3_eda.py                    # Descriptive statistics and visualizations
├── topic4_feature_selection.py      # Feature selection techniques
│
└── README.md                        # Project documentation
## 📚 Topics Covered

| # | Topic | Key Techniques |
|---|---|---|
| 1 | Data Collection & Loading | `pd.read_csv`, `df.info()`, `df.describe()` |
| 2 | Data Preprocessing | Mean/mode imputation, `drop_duplicates()`, `LabelEncoder`, `get_dummies()` |
| 3 | Exploratory Data Analysis | Histograms, bar charts, box plots, heatmaps, scatter plots |
| 4 | Feature Selection | Variance Threshold, SelectKBest, RFE, Random Forest Importance |

##  Key Findings
- `quantity_sold` and `price` are the strongest predictors of total revenue
- All 6 product categories and 4 customer regions are uniformly distributed across the dataset
- Strong positive correlation (~0.99) exists between `price` and `discounted_price`
- No missing values or duplicates exist in the original dataset
- Random Forest identified `quantity_sold`, `price`, and `discounted_price` as the top 3 most important features
