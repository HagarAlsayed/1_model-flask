from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib

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

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form
        features = [float(x) for x in request.form.values()]
        # Convert features to a NumPy array and reshape for the model
        features_array = np.array(features).reshape(1, -1)
        # Make prediction
        prediction = model.predict(features_array)
        # Render the prediction result on the web page
        return render_template("index.html", prediction_text=f'Your Risk Tolerance is {categorize(prediction[0])}')
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

'''
from flask import Flask, request, jsonify
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
    return "Welcome to Risk Tolerance Prediction API"

@app.route('/api/predict', methods=['GET'])
def api_predict():
    try:
        # Extract features from query parameters
        age = float(request.args.get('age'))
        education_level = float(request.args.get('education_level'))
        married_state = float(request.args.get('married_state'))
        no_of_kids = float(request.args.get('no_of_kids'))
        life_stage = float(request.args.get('life_stage'))
        occupational_category = float(request.args.get('occupational_category'))
        income = float(request.args.get('income'))
        risk = float(request.args.get('risk'))
        eager = float(request.args.get('eager'))
        
        # Create a DataFrame with the extracted features
        features_df = pd.DataFrame([[age, education_level, married_state, no_of_kids, life_stage,
                                     occupational_category, income, risk, eager]],
                                   columns=feature_names)
        
        # Make prediction
        prediction = model.predict(features_df)
        
        # Categorize the prediction
        risk_tolerance = categorize(prediction[0])
        
        # Return prediction as JSON
        return jsonify({'your risk tolerance is ': risk_tolerance})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)'''