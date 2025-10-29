"""
Train Machine Learning Model for Exam Score Prediction
=====================================================
This script trains a model to predict exam scores based on student habits and performance metrics.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load preprocessed dataset
print("📊 Loading preprocessed data...")
df = pd.read_csv('../data/processed_exam_data.csv')
print(f"Dataset shape: {df.shape}")

# Prepare features and target
# Drop student_id (now just an index) and exam_score (target)
X = df.drop(['student_id', 'exam_score'], axis=1)
y = df['exam_score']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Features: {list(X.columns)}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Train models
print("\n🤖 Training models...")

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Evaluate models
print("\n📈 Model Performance:")
print("-" * 50)

# Linear Regression metrics
lr_mse = mean_squared_error(y_test, lr_pred)
lr_rmse = np.sqrt(lr_mse)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

print("Linear Regression:")
print(f"  - R² Score: {lr_r2:.4f}")
print(f"  - RMSE: {lr_rmse:.4f}")
print(f"  - MAE: {lr_mae:.4f}")

# Random Forest metrics
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest:")
print(f"  - R² Score: {rf_r2:.4f}")
print(f"  - RMSE: {rf_rmse:.4f}")
print(f"  - MAE: {rf_mae:.4f}")

# Choose best model
best_model = rf_model if rf_r2 > lr_r2 else lr_model
best_model_name = "Random Forest" if rf_r2 > lr_r2 else "Linear Regression"
print(f"\n✅ Best model: {best_model_name}")

# Feature importance (for Random Forest)
if best_model_name == "Random Forest":
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🎯 Top 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))

# Save the best model
print("\n💾 Saving model and scaler...")
joblib.dump(best_model, 'exam_score_model.pkl')
print("Model saved as 'exam_score_model.pkl'")

# Create and save a scaler for the original features
# Note: We need the original feature names from the preprocessing step
original_features = ['age', 'gender', 'study_hours_per_day', 'social_media_hours', 
                    'netflix_hours', 'part_time_job', 'attendance_percentage', 
                    'sleep_hours', 'diet_quality', 'exercise_frequency', 
                    'parental_education_level', 'internet_quality', 
                    'mental_health_rating', 'extracurricular_participation', 
                    'has_outlier']

# Save feature names for prediction
joblib.dump(original_features, 'model_features.pkl')
print("Feature names saved as 'model_features.pkl'")

print("\n✅ Training completed successfully!")