import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Data Loading
def load_data(creditcard_path, fraud_path):
    creditcard_data = pd.read_csv(creditcard_path)
    fraud_data = pd.read_csv(fraud_path)
    return creditcard_data, fraud_data

# Data Preparation
def prepare_data(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    return accuracy

# SHAP Explainability
def shap_explainability(model, X_test, target_class=1):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print("\n=== SHAP Summary Plot ===")
    shap.summary_plot(shap_values, X_test)
    
    # Force Plot for a Single Instance
    instance = 0
    plt.title("SHAP Force Plot for Instance 0")
    shap.force_plot(explainer.expected_value[target_class], shap_values[target_class][instance], X_test.iloc[instance, :], matplotlib=True)
    plt.show()
    
    # Dependence Plot for a Feature
    feature = 'V14'  # Change as needed
    print(f"\n=== SHAP Dependence Plot for Feature '{feature}' ===")
    shap.dependence_plot(feature, shap_values[target_class], X_test)
    plt.show()
    return shap_values

# LIME Explainability
def lime_explainability(model, X_train, X_test, instance_idx=0):
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                            feature_names=X_train.columns,
                                                            class_names=['Not Fraud', 'Fraud'],
                                                            mode='classification')
    instance = X_test.iloc[instance_idx].values.reshape(1, -1)
    lime_exp = lime_explainer.explain_instance(instance[0], model.predict_proba, num_features=5)
    print("\n=== LIME Explanation for Instance ===")
    lime_exp.show_in_notebook(show_table=True)
    lime_exp.as_pyplot_figure()
    plt.show()

# Main Function to Execute the Workflow
def main(creditcard_path, fraud_path):
    # Load Data
    creditcard_data, fraud_data = load_data(creditcard_path, fraud_path)
    
    # Prepare Data
    X_train_credit, X_test_credit, y_train_credit, y_test_credit = prepare_data(creditcard_data, 'Class')
    
    # Train Model
    model_credit = train_model(X_train_credit, y_train_credit)
    
    # Evaluate Model
    evaluate_model(model_credit, X_test_credit, y_test_credit)
    
    # SHAP Explainability
    shap_values_credit = shap_explainability(model_credit, X_test_credit)
    
    # LIME Explainability
    lime_explainability(model_credit, X_train_credit, X_test_credit)

# Run the main function with your file paths
if __name__ == "__main__":
    creditcard_path = '../data/creditcard.csv'
    fraud_path = '../data/fraud_data_preprocessed.csv'
    main(creditcard_path, fraud_path)
