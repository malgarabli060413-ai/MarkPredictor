"""
Preprocess Exam Score Prediction Dataset
----------------------------------------
This script performs data preprocessing including:
    1. Handling missing values
    2. Encoding categorical variables
    3. Detecting and flagging outliers
    4. Normalizing appropriate numerical features

Author: [Your Name]
Date: 2025-10-27
"""

# =============================
# 1. Import Required Libraries
# =============================
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# =============================
# 2. Load Dataset
# =============================
# Replace 'data.csv' with your actual dataset filename
df = pd.read_csv('data.csv')

print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())

# =============================
# 3. Handle Missing Values
# =============================
print("\n🔹 Checking missing values:\n", df.isnull().sum())

# Strategy:
# - Numeric columns: fill with median (less sensitive to outliers)
# - Categorical columns: fill with mode (most frequent value)

for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\n✅ Missing values handled successfully!")

# =============================
# 4. Encode Non-Numerical (Categorical) Columns
# =============================

# Identify categorical columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical columns:", cat_cols)

# Encode with LabelEncoder (for simplicity)
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("✅ Categorical columns encoded successfully!")

# =============================
# 5. Detect and Flag Outliers
# =============================

# Function to detect outliers using IQR
def detect_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return ((data[col] < lower) | (data[col] > upper))

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
outlier_flags = pd.DataFrame(index=df.index)

for col in numeric_cols:
    outlier_flags[col + '_outlier'] = detect_outliers_iqr(df, col)

# Add an overall "has_outlier" flag
df['has_outlier'] = outlier_flags.any(axis=1).astype(int)

print("✅ Outliers flagged successfully!")
print(f"Total rows with potential outliers: {df['has_outlier'].sum()}")

# =============================
# 6. Normalize Numeric Values
# =============================

# Choose features to normalize (exclude target 'exam_score')
features_to_normalize = [col for col in numeric_cols if col != 'exam_score']

scaler = StandardScaler()  # You can use MinMaxScaler() as an alternative
df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

print("✅ Numeric features normalized successfully!")

# =============================
# 7. Save Processed Dataset
# =============================

df.to_csv('processed_exam_data.csv', index=False)
print("\n💾 Processed dataset saved as 'processed_exam_data.csv'")

# =============================
# 8. Optional: Quick Visualization
# =============================

# Show distribution of target variable
sns.histplot(df['exam_score'], kde=True)
plt.title('Exam Score Distribution (After Preprocessing)')
plt.show()

print("\n🎉 Preprocessing completed successfully!")
