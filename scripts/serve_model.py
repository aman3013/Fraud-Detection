import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import logging
from logging.handlers import RotatingFileHandler

# Load model and scaler
model = joblib.load('../models/random_forest_model.pkl')
scaler = joblib.load('../models/scaler.pkl')

# Initialize Flask app
app = Flask(__name__)

# Configure logging
handler = RotatingFileHandler('api.log', maxBytes=10000, backupCount=3)
logging.basicConfig(level=logging.INFO)
app.logger.addHandler(handler)

@app.route('/')
def welcome(): 
    return "Welcome to the Fraud Detection API. Use the /predict endpoint to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract JSON data
        data = request.get_json()
        
        # Check for required fields
        if 'signup_time' not in data or 'purchase_time' not in data:
            return jsonify({'error': 'Both signup_time and purchase_time are required to calculate time_to_purchase.'}), 400
        
        # Calculate time_to_purchase in hours
        signup_time = pd.to_datetime(data['signup_time'])
        purchase_time = pd.to_datetime(data['purchase_time'])
        time_to_purchase = (purchase_time - signup_time).total_seconds() / 3600  # Convert to hours
        
        # Prepare the feature array in the correct order
        features = np.array([[
            data['purchase_value'],   # purchase_value
            data['age'],              # age
            time_to_purchase,        # time_to_purchase
            0,                       # Placeholder for device_id (or encode if you have the encoder)
            0,                       # Placeholder for source (or encode if you have the encoder)
            0,                       # Placeholder for browser (or encode if you have the encoder)
            0,                       # Placeholder for sex (or encode if you have the encoder)
            0,                       # Placeholder for user_id (or encode if you have the encoder)
            0                        # Placeholder for ip_address (or encode if you have the encoder)
        ]])

        # Scale numerical features (ensure correct column slicing)
        features[:, [0, 1, 2]] = scaler.transform(features[:, [0, 1, 2]])

        # Make prediction
        prediction = model.predict(features)
        
        # Log the prediction
        app.logger.info(f'Prediction: {prediction[0]} for data: {data}')
        
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        app.logger.error(f'Error: {str(e)}')
        return jsonify({'error': str(e)}), 500

# Define health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
  app.run(debug=True)
