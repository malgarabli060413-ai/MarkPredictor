"""
Flask Web Application for Exam Score Prediction
==============================================
This app serves the MarkPredictor web interface and handles predictions
using the trained machine learning model.
"""

from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Initialize Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Enable CORS for Live Server and network access
CORS(app, origins=['*'])  # Allow all origins for classroom use

# Load the trained model and feature names
print("Loading model...")
model = joblib.load('models/exam_score_model.pkl')
feature_names = joblib.load('models/model_features.pkl')
print("Model loaded successfully!")

# Simple health check endpoint
@app.route('/')
def home():
    """API health check"""
    return jsonify({
        'status': 'API is running',
        'message': 'Use Live Server to access the frontend'
    })

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests from the frontend
    Expects JSON data with student information
    Returns predicted exam score
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        print("\n📥 RECEIVED REQUEST:")
        print(f"Raw data: {data}")
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract and validate required fields from the form
        # Note: The HTML form has limited fields, so we'll need to set defaults for missing ones
        
        # Map form fields to model features - now all fields come from the form
        student_data = {
            'age': int(data.get('age', 20)),
            'gender': data.get('gender', 'Female'),
            'study_hours_per_day': float(data.get('studyHours', 3.5)),
            'social_media_hours': float(data.get('socialMediaHours', 2.0)),
            'netflix_hours': float(data.get('netflixHours', 1.5)),
            'part_time_job': data.get('partTimeJob', 'No'),
            'attendance_percentage': float(data.get('attendance', 85.0)),
            'sleep_hours': float(data.get('sleepHours', 7.0)),
            'diet_quality': data.get('dietQuality', 'Fair'),
            'exercise_frequency': int(data.get('exerciseFrequency', 3)),
            'parental_education_level': data.get('parentEdu', 'Bachelor'),
            'internet_quality': data.get('internetQuality', 'Average'),
            'mental_health_rating': int(data.get('mentalHealthRating', 7)),
            'extracurricular_participation': data.get('extracurricular', 'No')
        }
        
        # Preprocess the data
        processed_data = preprocess_student_data(student_data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        
        # Print prediction info to console
        print("\n" + "="*50)
        print("🎯 NEW PREDICTION MADE")
        print("="*50)
        print(f"Input Data:")
        print(f"  - Age: {data.get('age')}")
        print(f"  - Gender: {data.get('gender')}")
        print(f"  - Study Hours/Day: {data.get('studyHours')}")
        print(f"  - Social Media Hours: {data.get('socialMediaHours')}")
        print(f"  - Netflix Hours: {data.get('netflixHours')}")
        print(f"  - Part-time Job: {data.get('partTimeJob')}")
        print(f"  - Attendance: {data.get('attendance')}%")
        print(f"  - Sleep Hours: {data.get('sleepHours')}")
        print(f"  - Diet Quality: {data.get('dietQuality')}")
        print(f"  - Exercise Frequency: {data.get('exerciseFrequency')}")
        print(f"  - Parent Education: {data.get('parentEdu')}")
        print(f"  - Internet Quality: {data.get('internetQuality')}")
        print(f"  - Mental Health Rating: {data.get('mentalHealthRating')}")
        print(f"  - Extracurricular: {data.get('extracurricular')}")
        print(f"\n📊 PREDICTED EXAM SCORE: {prediction:.2f}%")
        
        # Determine performance level
        if prediction >= 90:
            performance = "Excellent"
        elif prediction >= 80:
            performance = "Very Good"
        elif prediction >= 70:
            performance = "Good"
        elif prediction >= 60:
            performance = "Average"
        else:
            performance = "Needs Improvement"
        
        print(f"🎓 Performance Level: {performance}")
        print("="*50 + "\n")
        
        # Return prediction result
        return jsonify({
            'success': True,
            'predicted_score': round(prediction, 2),
            'performance_level': performance,
            'message': f'Based on your inputs, your predicted exam score is {round(prediction, 2)}%'
        })
        
    except Exception as e:
        # Handle any errors
        import traceback
        print("\n❌ ERROR OCCURRED:")
        print(f"Error message: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred while making the prediction',
            'details': traceback.format_exc()
        }), 500

def preprocess_student_data(student_data):
    """
    Preprocess student data to match the format expected by the model
    """
    # Create DataFrame
    df = pd.DataFrame([student_data])
    
    # No special conversions needed - form now collects data in the right format
    
    # Encode categorical variables
    # Gender encoding
    gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
    df['gender'] = df['gender'].map(gender_map).fillna(0)
    
    # Part-time job encoding
    job_map = {'No': 0, 'Yes': 1}
    df['part_time_job'] = df['part_time_job'].map(job_map).fillna(0)
    
    # Diet quality encoding
    diet_map = {'Poor': 0, 'Fair': 1, 'Good': 2}
    df['diet_quality'] = df['diet_quality'].map(diet_map).fillna(1)
    
    # Parental education encoding
    edu_map = {
        'None': 0, 
        'High-School Diploma': 1, 
        'High School': 1,
        "Bachelor's Degree": 2,
        'Bachelor': 2,
        'Masters': 3,
        'Master': 3,
        'Doctorate': 3
    }
    df['parental_education_level'] = df['parental_education_level'].map(edu_map).fillna(1)
    
    # Internet quality encoding
    internet_map = {'Poor': 0, 'Average': 1, 'Good': 2}
    df['internet_quality'] = df['internet_quality'].map(internet_map).fillna(1)
    
    # Extracurricular encoding
    extra_map = {'No': 0, 'Yes': 1}
    df['extracurricular_participation'] = df['extracurricular_participation'].map(extra_map).fillna(0)
    
    # Normalize numeric features (using approximate parameters from training)
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
    
    numeric_features = ['age', 'study_hours_per_day', 'social_media_hours',
                       'netflix_hours', 'attendance_percentage', 'sleep_hours',
                       'exercise_frequency', 'mental_health_rating']
    
    for feature in numeric_features:
        mean = scaling_params[feature]['mean']
        std = scaling_params[feature]['std']
        df[feature] = (df[feature] - mean) / std
    
    # Add outlier flag (0 - no outliers assumed for user input)
    df['has_outlier'] = 0
    
    # Ensure all features are present in correct order
    return df[feature_names]

# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Page not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

# Run the app
if __name__ == '__main__':
    # Run in debug mode for development
    app.run(debug=True, host='0.0.0.0', port=5001)
    print("Server running on http://localhost:5001")