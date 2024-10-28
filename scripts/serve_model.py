# import pandas as pd
# import numpy as np
# import pickle
# from flask import Flask, request, jsonify
# from sklearn.preprocessing import OneHotEncoder

# app = Flask(__name__)

# # Load the trained model
# model = pickle.load(open('../models/radom_forest_model.pkl', 'rb'))

# # Initialize OneHotEncoder for categorical variables
# encoder = OneHotEncoder(drop='first')

# @app.route('/')
# def welcome():
#     return "Welcome to the Fraud Detection API. Use the /predict endpoint to get predictions."

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()

#     # Ensure data is in the correct format (list of dicts)
#     if not isinstance(data, list):
#         return jsonify({'error': 'Input data must be a list of records.'}), 400

#     # Create a DataFrame from the incoming JSON data
#     df = pd.DataFrame(data)

#     # Convert timestamp columns to numerical values (Unix timestamp)
#     if 'signup_time' in df.columns:
#         df['signup_time'] = pd.to_datetime(df['signup_time']).astype(np.int64) // 10**9  # Convert to seconds
#     if 'purchase_time' in df.columns:
#         df['purchase_time'] = pd.to_datetime(df['purchase_time']).astype(np.int64) // 10**9  # Convert to seconds

#     # Select the categorical columns to encode
#     categorical_cols = ['device_id', 'source', 'browser', 'sex']

#     # Check if there are any categorical columns
#     if not all(col in df.columns for col in categorical_cols):
#         return jsonify({'error': 'Some categorical columns are missing.'}), 400

#     # Fit the encoder on the categorical columns (you may want to fit this on your training data separately)
#     df_encoded = encoder.transform(df[categorical_cols])
#     df_encoded = pd.DataFrame(df_encoded, columns=encoder.get_feature_names_out(categorical_cols))

#     # Drop original categorical columns and concatenate the encoded columns
#     df = df.drop(columns=categorical_cols)
#     df = pd.concat([df.reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)

#     # Make predictions
#     predictions = model.predict(df)

#     # Return the predictions as JSON
#     return jsonify({'predictions': predictions.tolist()})


# if __name__ == '__main__':
#     app.run(debug=True)

# import numpy as np
# import pickle
# from flask import Flask, request, jsonify
# from datetime import datetime
# from sklearn.preprocessing import StandardScaler

# app = Flask(__name__)

# # Load the trained model and scaler
# model = pickle.load(open('../models/random_forest_model.pkl', 'rb'))
# scaler = pickle.load(open('../models/scaler.pkl', 'rb'))

# @app.route('/')
# def welcome():
#     return "Welcome to the Fraud Detection API. Use the /predict endpoint to get predictions."

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()

#     # Ensure data is a list of dictionaries
#     if not isinstance(data, list) or not all(isinstance(record, dict) for record in data):
#         return jsonify({'error': 'Input data must be a list of dictionaries.'}), 400

#     # Process each record
#     processed_data = []
#     for record in data:
#         try:
#             # Convert timestamp columns to Unix time (seconds)
#             signup_time = int(datetime.strptime(record['signup_time'], '%Y-%m-%d %H:%M:%S').timestamp())
#             purchase_time = int(datetime.strptime(record['purchase_time'], '%Y-%m-%d %H:%M:%S').timestamp())

#             # Calculate time to purchase in seconds
#             time_to_purchase = purchase_time - signup_time

#             # Extract numerical features
#             features = [
#                 signup_time,
#                 purchase_time,
#                 time_to_purchase,
#                 record['purchase_value'],
#                 record['age'],
#                 record['ip_address']  # Assuming this is numerical; otherwise, handle accordingly
#             ]

#             processed_data.append(features)

#         except KeyError as e:
#             return jsonify({'error': f"Missing key in data: {str(e)}"}), 400
#         except Exception as e:
#             return jsonify({'error': f"Error processing record: {str(e)}"}), 400

#     # Convert the processed data to a numpy array for scaling
#     processed_data = np.array(processed_data)

#     # Scale the numerical features
#     scaled_data = scaler.transform(processed_data)

#     # Ensure the data is compatible with the model's input features
#     if model.n_features_in_ != scaled_data.shape[1]:
#         return jsonify({'error': 'Input data does not match the model training format.'}), 400

#     # Make predictions
#     predictions = model.predict(scaled_data)

#     # Return predictions as JSON
#     return jsonify({'predictions': predictions.tolist()})

# if __name__ == '__main__':
#     app.run(debug=True)

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
