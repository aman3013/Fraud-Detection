import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(filepath):
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(data):
    """Preprocess the dataset by handling dates, encoding, and scaling."""
    # Handle date columns
    data['signup_time'] = pd.to_datetime(data['signup_time'])
    data['purchase_time'] = pd.to_datetime(data['purchase_time'])
    data['time_to_purchase'] = (data['purchase_time'] - data['signup_time']).dt.total_seconds()
    data.drop(['signup_time', 'purchase_time'], axis=1, inplace=True)
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    categorical_columns = ['device_id', 'source', 'browser', 'sex']
    for col in categorical_columns:
        data[col] = label_encoder.fit_transform(data[col])
    
    # Split features and target
    X = data.drop('class', axis=1)
    y = data['class']
    
    return X, y

def scale_data(X_train, X_test, numerical_columns):
    """Scale the training and test data using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])
    
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train, y_train):
    """Train a Random Forest Classifier."""
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print performance metrics."""
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def save_model_and_scaler(model, scaler, model_path='models/random_forest_model.pkl', scaler_path='models/scaler.pkl'):
    """Save the trained model and scaler to specified paths."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

def main():
    # Define file paths and columns
    filepath = 'fraud_data.csv'
    numerical_columns = ['purchase_value', 'age', 'time_to_purchase']
    
    # Load and preprocess data
    data = load_data(filepath)
    X, y = preprocess_data(data)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale data
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test, numerical_columns)
    
    # Train the model
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test_scaled, y_test)
    
    # Save the model and scaler
    save_model_and_scaler(model, scaler)

if __name__ == "__main__":
    main()
