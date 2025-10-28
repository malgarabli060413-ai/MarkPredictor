"""
Preprocess Exam Score Prediction Dataset (Fixed)
----------------------------------------
This script performs data preprocessing including:
    1. Handling missing values
    2. Encoding categorical variables
    3. Detecting and flagging outliers
    4. Normalizing appropriate numeric features (excluding categorical columns)

Author: [Your Name]
Date: 2025-10-27
"""

# =============================
# 1. Import Required Libraries
# =============================
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# =============================
# 2. Load Dataset
# =============================
df = pd.read_csv('student_habits_performance.csv')  # Update filename if needed

print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())

# =============================
# 3. Handle Missing Values
# =============================
print("\n🔹 Checking missing values:\n", df.isnull().sum())

for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("✅ Missing values handled successfully!")

# =============================
# 4. Encode Categorical Columns
# =============================
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical columns:", cat_cols)

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

categorical_encoded_cols = cat_cols.copy()
print("✅ Categorical columns encoded successfully!")

# =============================
# 5. Detect and Flag Outliers
# =============================
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

df['has_outlier'] = outlier_flags.any(axis=1).astype(int)

print("✅ Outliers flagged successfully!")
print(f"Total rows with potential outliers: {df['has_outlier'].sum()}")

# =============================
# 6. Normalize Numeric Values
# =============================
# Only normalize true numeric features (exclude categorical columns, target, and has_outlier)
true_numeric_cols = [
    'age', 'study_hours_per_day', 'social_media_hours',
    'netflix_hours', 'attendance_percentage', 'sleep_hours',
    'exercise_frequency', 'mental_health_rating'
]

scaler = StandardScaler()
df[true_numeric_cols] = scaler.fit_transform(df[true_numeric_cols])

print("✅ Numeric features normalized successfully!")
print("Columns normalized:", true_numeric_cols)

# =============================
# 7. Save Processed Dataset
# =============================
df.to_csv('processed_exam_data.csv', index=False)
print("\n💾 Processed dataset saved as 'processed_exam_data.csv'")

# =============================
# 8. Optional: Quick Visualization
# =============================
sns.histplot(df['exam_score'], kde=True)
plt.title('Exam Score Distribution (After Preprocessing)')
plt.show()

print("\n🎉 Preprocessing completed successfully!")
