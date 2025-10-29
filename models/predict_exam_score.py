"""
Predict Exam Scores Using Saved Model
====================================
This script demonstrates how to make predictions using the saved model.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load saved model and features
print("📊 Loading saved model...")
model = joblib.load('exam_score_model.pkl')
feature_names = joblib.load('model_features.pkl')
print("Model loaded successfully!")

# Load the original preprocessing script's mappings
# We need to match the preprocessing done during training
print("\n🔄 Loading preprocessing parameters...")

# Sample student data (before preprocessing)
sample_student = {
    'age': 20,
    'gender': 'Female',
    'study_hours_per_day': 4.5,
    'social_media_hours': 2.0,
    'netflix_hours': 1.5,
    'part_time_job': 'No',
    'attendance_percentage': 85.0,
    'sleep_hours': 7.0,
    'diet_quality': 'Good',
    'exercise_frequency': 3,
    'parental_education_level': 'Bachelor',
    'internet_quality': 'Good',
    'mental_health_rating': 7,
    'extracurricular_participation': 'Yes'
}

print("\n👤 Sample Student Data:")
for key, value in sample_student.items():
    print(f"  - {key}: {value}")

# Preprocessing function
def preprocess_student_data(student_data):
    """Preprocess a single student's data to match training preprocessing"""
    
    # Create DataFrame
    df = pd.DataFrame([student_data])
    
    # Encode categorical variables
    # Gender encoding
    gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
    df['gender'] = df['gender'].map(gender_map)
    
    # Part-time job encoding
    job_map = {'No': 0, 'Yes': 1}
    df['part_time_job'] = df['part_time_job'].map(job_map)
    
    # Diet quality encoding
    diet_map = {'Poor': 0, 'Fair': 1, 'Good': 2}
    df['diet_quality'] = df['diet_quality'].map(diet_map)
    
    # Parental education encoding
    edu_map = {'None': 0, 'High School': 1, 'Bachelor': 2, 'Master': 3}
    df['parental_education_level'] = df['parental_education_level'].map(edu_map)
    
    # Internet quality encoding
    internet_map = {'Poor': 0, 'Average': 1, 'Good': 2}
    df['internet_quality'] = df['internet_quality'].map(internet_map)
    
    # Extracurricular encoding
    extra_map = {'No': 0, 'Yes': 1}
    df['extracurricular_participation'] = df['extracurricular_participation'].map(extra_map)
    
    # Normalize numeric features
    numeric_features = ['age', 'study_hours_per_day', 'social_media_hours',
                       'netflix_hours', 'attendance_percentage', 'sleep_hours',
                       'exercise_frequency', 'mental_health_rating']
    
    # Apply standard scaling (using approximate mean/std from training)
    # Note: In production, you'd load the actual scaler used during training
    scaler = StandardScaler()
    
    # Approximate scaling based on typical ranges
    scaling_params = {
        'age': {'mean': 20.5, 'std': 2.0},
        'study_hours_per_day': {'mean': 3.5, 'std': 1.5},
        'social_media_hours': {'mean': 2.5, 'std': 1.2},
        'netflix_hours': {'mean': 1.8, 'std': 1.0},
        'attendance_percentage': {'mean': 85.0, 'std': 10.0},
        'sleep_hours': {'mean': 6.5, 'std': 1.5},
        'exercise_frequency': {'mean': 3.0, 'std': 2.0},
        'mental_health_rating': {'mean': 5.5, 'std': 2.5}
    }
    
    for feature in numeric_features:
        mean = scaling_params[feature]['mean']
        std = scaling_params[feature]['std']
        df[feature] = (df[feature] - mean) / std
    
    # Add outlier flag (0 for this example - no outliers assumed)
    df['has_outlier'] = 0
    
    return df[feature_names]

# Preprocess the sample student data
print("\n🔧 Preprocessing student data...")
processed_data = preprocess_student_data(sample_student)

# Make prediction
print("\n🎯 Making prediction...")
predicted_score = model.predict(processed_data)[0]

print(f"\n📊 Predicted Exam Score: {predicted_score:.2f}")

# Interpret the score
if predicted_score >= 90:
    performance = "Excellent"
elif predicted_score >= 80:
    performance = "Very Good"
elif predicted_score >= 70:
    performance = "Good"
elif predicted_score >= 60:
    performance = "Average"
else:
    performance = "Needs Improvement"

print(f"📈 Performance Level: {performance}")

# Batch prediction example
print("\n\n📚 Batch Prediction Example:")
print("-" * 50)

# Multiple students
batch_students = [
    {
        'age': 19,
        'gender': 'Male',
        'study_hours_per_day': 6.0,
        'social_media_hours': 1.0,
        'netflix_hours': 0.5,
        'part_time_job': 'No',
        'attendance_percentage': 95.0,
        'sleep_hours': 8.0,
        'diet_quality': 'Good',
        'exercise_frequency': 4,
        'parental_education_level': 'Master',
        'internet_quality': 'Good',
        'mental_health_rating': 8,
        'extracurricular_participation': 'Yes'
    },
    {
        'age': 22,
        'gender': 'Female',
        'study_hours_per_day': 2.0,
        'social_media_hours': 4.0,
        'netflix_hours': 3.0,
        'part_time_job': 'Yes',
        'attendance_percentage': 70.0,
        'sleep_hours': 5.0,
        'diet_quality': 'Poor',
        'exercise_frequency': 1,
        'parental_education_level': 'High School',
        'internet_quality': 'Poor',
        'mental_health_rating': 3,
        'extracurricular_participation': 'No'
    }
]

for i, student in enumerate(batch_students, 1):
    processed = preprocess_student_data(student)
    score = model.predict(processed)[0]
    print(f"\nStudent {i}:")
    print(f"  - Study Hours: {student['study_hours_per_day']}")
    print(f"  - Attendance: {student['attendance_percentage']}%")
    print(f"  - Predicted Score: {score:.2f}")