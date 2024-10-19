import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import ipaddress
from datetime import datetime, timedelta

# Suppress warnings
import warnings 
warnings.filterwarnings('ignore')

def load_datasets():
    """Load the datasets from CSV files."""
    fraud_data = pd.read_csv('../data/Fraud_Data.csv')
    ip_country_data = pd.read_csv('../data/IpAddress_to_Country.csv')
    credit_card_data = pd.read_csv('../data/creditcard.csv')
    return fraud_data, ip_country_data, credit_card_data

def check_missing_values(datasets):
    """Check for missing values in the datasets."""
    for name, df in datasets.items():
        print(f"Missing values in {name}: ", df.isnull().sum())

def remove_duplicates(datasets):
    """Remove duplicate entries from the datasets."""
    return {name: df.drop_duplicates() for name, df in datasets.items()}

def clean_data(df):
    """Clean and preprocess the data."""
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    return df

def perform_eda(fraud_data, credit_card_data):
    """Perform exploratory data analysis."""
    print(fraud_data.describe())
    print(credit_card_data.describe())

    # Age distribution
    sns.histplot(fraud_data['age'], bins=30, kde=True)
    plt.show()

    # Fraud distribution
    sns.countplot(x='class', data=fraud_data)
    plt.show()

    # Bivariate analysis
    plt.figure(figsize=(10, 6))
    sns.countplot(x='class', data=fraud_data)
    plt.title('Fraudulent Transactions')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.countplot(x='Class', data=credit_card_data)
    plt.title('Fraudulent Transactions')
    plt.show()

    # Purchase Value by Fraud Class
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='class', y='purchase_value', data=fraud_data)
    plt.title('Purchase Value by Fraud Class')
    plt.xlabel('Fraud Class')
    plt.ylabel('Purchase Value')
    plt.show()

def handle_outliers(df, column):
    """Handle outliers in the specified column."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])

    return df

def convert_ip_to_int(df):
    """Convert IP addresses to integer format."""
    df['ip_address'] = pd.to_numeric(df['ip_address'], errors='coerce').fillna(0).astype(int)
    return df

def find_country(ip, ip_country_data):
    """Find country based on IP address."""
    ip_int = int(ip)
    country_row = ip_country_data[
        (ip_country_data['lower_bound_ip_address'] <= ip_int) & 
        (ip_country_data['upper_bound_ip_address'] >= ip_int)
    ]
    if not country_row.empty:
        return country_row.iloc[0]['country']
    return "Unknown"

def merge_data(fraud_data, ip_country_data):
    """Merge fraud data with country information."""
    merged_data = fraud_data.copy()
    merged_data['country'] = merged_data['ip_address'].apply(lambda ip: find_country(ip, ip_country_data))
    return merged_data

def engineer_features(df):
    """Engineer new features for fraud detection."""
    # Transaction Frequency
    df['user_transaction_count'] = df.groupby('user_id')['user_id'].transform('count')

    # Transaction Velocity
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
    df['transaction_velocity'] = df['user_transaction_count'] / df['time_since_signup']

    # Time-Based Features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Time since last transaction
    df = df.sort_values(['user_id', 'purchase_time'])
    df['time_since_last_transaction'] = df.groupby('user_id')['purchase_time'].diff().dt.total_seconds() / 3600

    # Purchase value related features
    df['avg_purchase_value'] = df.groupby('user_id')['purchase_value'].transform('mean')
    df['purchase_value_difference'] = df['purchase_value'] - df['avg_purchase_value']

    return df

def create_time_buckets(df):
    """Create time bucket features."""
    first_transaction_time = df['purchase_time'].min()
    df['time_since_first_transaction'] = (df['purchase_time'] - first_transaction_time).dt.total_seconds() / 3600
    df['hour_bucket'] = df['purchase_time'].dt.floor('H')
    df['day_bucket'] = df['purchase_time'].dt.floor('D')
    df['week_bucket'] = df['purchase_time'].dt.to_period('W').apply(lambda r: r.start_time)
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    df['country'] = df['country'].fillna('Unknown')
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    return df

def save_processed_data(df, output_path):
    """Save the processed data to a CSV file."""
    df.to_csv(output_path, index=False)
    print(f"Processed data saved successfully to {output_path}")

def main():
    # Load datasets
    fraud_data, ip_country_data, credit_card_data = load_datasets()

    # Check for missing values
    check_missing_values({'Fraud_Data': fraud_data, 'IpAddress_to_Country': ip_country_data, 'creditcard': credit_card_data})

    # Remove duplicates
    fraud_data = remove_duplicates({'fraud_data': fraud_data})['fraud_data']
    ip_country_data = remove_duplicates({'ip_country_data': ip_country_data})['ip_country_data']
    credit_card_data = remove_duplicates({'credit_card_data': credit_card_data})['credit_card_data']

    # Clean and preprocess data
    fraud_data = clean_data(fraud_data)

    # Perform EDA
    perform_eda(fraud_data, credit_card_data)

    # Handle outliers
    fraud_data = handle_outliers(fraud_data, 'purchase_value')

    # Convert IP to int
    fraud_data = convert_ip_to_int(fraud_data)
    ip_country_data['lower_bound_ip_address'] = pd.to_numeric(ip_country_data['lower_bound_ip_address'], errors='coerce').fillna(0).astype(int)
    ip_country_data['upper_bound_ip_address'] = pd.to_numeric(ip_country_data['upper_bound_ip_address'], errors='coerce').fillna(0).astype(int)

    # Merge data
    merged_data = merge_data(fraud_data, ip_country_data)

    # Engineer features
    merged_data = engineer_features(merged_data)

    # Create time buckets
    merged_data = create_time_buckets(merged_data)

    # Handle missing values
    merged_data = handle_missing_values(merged_data)

    # Save processed data
    save_processed_data(merged_data, '../data/processed_fraud_data.csv')

    print("Data processing and feature engineering completed successfully!")

    # Display summary of engineered features
    print("\nSummary of engineered features:")
    print(merged_data[['user_transaction_count', 'transaction_velocity', 'hour_of_day', 'day_of_week', 
                       'is_weekend', 'time_since_last_transaction', 'avg_purchase_value', 
                       'purchase_value_difference', 'time_since_first_transaction']].describe())

    # Check for any remaining missing values
    missing_values = merged_data.isnull().sum()
    if missing_values.sum() > 0:
        print("\nRemaining missing values:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values in the processed dataset.")

if __name__ == "__main__":
    main()