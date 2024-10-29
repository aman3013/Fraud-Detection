from flask import Flask, jsonify
import pandas as pd

# Create a Flask server
app = Flask(__name__)

# Load the data
data = pd.read_csv('../data/fraud_data.csv')  # Update the path if necessary
data['purchase_time'] = pd.to_datetime(data['purchase_time'])
data['signup_time'] = pd.to_datetime(data['signup_time'])

# Define an API endpoint for summary statistics
@app.route('/api/stats', methods=['GET'])
def get_stats():
    # Calculate statistics
    total_transactions = len(data)
    total_frauds = len(data[data['class'] == 1])
    fraud_percentage = (total_frauds / total_transactions) * 100 if total_transactions > 0 else 0

    # Return results as JSON
    return jsonify({
        "total_transactions": total_transactions,
        "total_frauds": total_frauds,
        "fraud_percentage": fraud_percentage
    })

if __name__ == '__main__':
    app.run(debug=True)
