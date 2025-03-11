
from flask import Flask, request, jsonify
import pickle
import numpy as np

from joblib import load

# Load the model
model = load('wine_classifier_pca.pkl')

# Initialize Flask app
app = Flask(__name__)

# Expected feature keys
expected_features = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalch", "oldpeak", "slope"
]

# Home route
@app.route('/')
def home():
    return "Welcome to the Wine Classifier API! Use the /predict endpoint for predictions."

# Prediction route (expects 10 features as input)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Check if all expected features are present in the request
        missing_features = [feature for feature in expected_features if feature not in data]
        
        if missing_features:
            return jsonify({'error': f'Missing features: {", ".join(missing_features)}'}), 400
        
        # Extract the features
        features = np.array([[data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'],
                               data['fbs'], data['restecg'], data['thalch'], data['oldpeak'], data['slope']]])
        
        # Make the prediction
        prediction = model.predict(features)
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': str(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)