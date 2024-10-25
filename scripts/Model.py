import os
import sys
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import parallel_backend
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# Define the function to prepare, preprocess, and train models on a dataset
def process_dataset(df, target_column, dataset_name, is_fraud_data=False):
    print(f"Processing {dataset_name} dataset")
    
    # Data preparation
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Identify numerical and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Data transformation pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Splitting the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)
    
    # Optional downsampling for quicker execution
    if X_train.shape[0] > 10000:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=0.1, stratify=y_train, random_state=42)
        print(f"Downsampled {dataset_name} to 10% of the original size for faster training")

    # Determine if it's a classification or regression task
    unique_values = np.unique(y_train)
    is_classification = len(unique_values) <= 10

    if is_classification:
        # Convert to categorical if needed
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        
        # Define models with parameters adjusted for quick training
        models = {
            'Logistic Regression': LogisticRegression(max_iter=500, n_jobs=-1),
            'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5),
            'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=10, min_samples_leaf=5, max_features='sqrt', n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, validation_fraction=0.2, n_iter_no_change=10),
            'MLP': MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, early_stopping=True, n_iter_no_change=5)
        }
    else:
        print("Regression task detected. Skipping classification models.")
        return

    # MLflow experiment for model training and logging
    with mlflow.start_run(run_name=f"{dataset_name}_experiment"):
        for model_name, model in models.items():
            print(f"Training {model_name}")
            
            with mlflow.start_run(run_name=model_name, nested=True):
                with parallel_backend('threading', n_jobs=-1):
                    model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                # Evaluate and log metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred)
                }
                
                # Log parameters, metrics, and the model
                mlflow.log_params(model.get_params())
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                mlflow.sklearn.log_model(model, model_name)
                print(f"{model_name} metrics: {metrics}")

        # Log the preprocessor
        mlflow.sklearn.log_model(preprocessor, "preprocessor")

# Sample datasets
datasets = {
    'credit_card': ('../data/creditcard.csv', 'Class'),
    'fraud_data': ('../data/fraud_data_preprocessed.csv', 'class')
}

# Process each dataset
for dataset_name, (file_path, target_column) in datasets.items():
    df = pd.read_csv(file_path)
    process_dataset(df, target_column, dataset_name)
