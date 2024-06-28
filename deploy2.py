from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import joblib

# Thresholds for categorizing risk tolerance
threshold_low = 33.3
threshold_high = 66.6

# Function to categorize risk
def categorize(value):
    if value <= threshold_low:
        return 'Low'
    elif value > threshold_low and value <= threshold_high:
        return 'Medium'
    else:
        return 'High'

app = Flask(__name__)

# Load your trained model
model = joblib.load("risk_tolerance.joblib")
# Feature names based on your training data
feature_names = [
    'age', 'education_level', 'married_state', 'no_of_kids', 'life_statge',
    'occupational_category', 'income', 'risk', 'eager'
]

@app.route('/')
def home():
    return #render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form
        features = [float(request.form[feature]) for feature in feature_names]
        # Convert features to a DataFrame with proper feature names
        features_df = pd.DataFrame([features], columns=feature_names)
        # Make prediction
        prediction = model.predict(features_df)
        # Render the prediction result on the web page
        return render_template("index.html", prediction_text=f'Your Risk Tolerance is {categorize(prediction[0])}')
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/predict', methods=['GET'])
def api_predict():
    try:
        # Extract features from JSON request
        data = request.json
        features = [float(data[feature]) for feature in feature_names]
        # Convert features to a DataFrame with proper feature names
        features_df = pd.DataFrame([features], columns=feature_names)
        # Make prediction
        prediction = model.predict(features_df)
        # Return prediction as JSON
        return jsonify({'Your risk tolerance is ': categorize(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

