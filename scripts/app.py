import os
import sys
from flask import Flask, jsonify
from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import numpy as np
from scipy import stats

# Create a Flask server to serve the Dash application
server = Flask(__name__)

# Load your data from a CSV file into a DataFrame
data = pd.read_csv('../data/fraud_data.csv')
# Convert the purchase_time and signup_time columns to datetime objects for easier filtering and analysis
data['purchase_time'] = pd.to_datetime(data['purchase_time'])
data['signup_time'] = pd.to_datetime(data['signup_time'])

# Create the Dash application, linking it to the Flask server
app = Dash(__name__, server=server, url_base_pathname='/dashboard/')

# Define custom colors for styling the dashboard
colors = {
    'background': '#f2f2f2',
    'text': '#333333',
    'primary': '#007bff',
    'secondary': '#6c757d'
}

# Define the layout for the Dash app
app.layout = html.Div([
    # Main title of the dashboard
    html.H1("Fraud Detection Dashboard", style={'textAlign': 'center', 'color': colors['primary']}),

    # Overview section providing context about the project
    html.Div([
        html.H2("Overview", style={'color': colors['secondary']}),
        html.P([
            "This dashboard is part of a fraud detection project at Adey Innovations Inc., ",
            "a leading company in the financial technology sector. Our goal is to improve ",
            "the detection of fraud cases for e-commerce transactions and bank credit transactions."
        ]),
        
        html.P([
            "The project involves analyzing transaction data, engineering relevant features, ",
            "building and training machine learning models, and deploying these models for ",
            "real-time fraud detection. This dashboard provides insights into the transaction ",
            "data and fraud patterns to support the development and monitoring of our fraud ",
            "detection system."
        ]),
    ], style={'backgroundColor': colors['background'], 'padding': '20px', 'margin': '20px 0', 'borderRadius': '10px'}),

    # Date picker for filtering the data based on purchase dates
    dcc.DatePickerRange(
        id='date-picker',
        start_date=data['purchase_time'].min(),  # Minimum purchase date
        end_date=data['purchase_time'].max(),    # Maximum purchase date
        display_format='YYYY-MM-DD',              # Date display format
        style={'margin': '10px'}                  # Margin for styling
    ),

    # Div for displaying summary statistics
    html.Div(id='summary-stats', style={'margin': '20px', 'padding': '20px', 'backgroundColor': colors['background']}),

    # Various graphs for visualizations, each accompanied by descriptive text
    dcc.Graph(id='time-series-fraud', config={'displayModeBar': False}),
    html.P("This time series chart tracks daily transactions and fraud cases, helping to monitor trends in fraud activity over time."),

    dcc.Graph(id='browser-fraud-graph', config={'displayModeBar': False}),
    html.P("The bar chart shows transactions and fraud counts by browser. It highlights if certain browsers are linked with higher fraud risk."),

    dcc.Graph(id='device-fraud-graph', config={'displayModeBar': False}),
    html.P("This bar chart presents transactions and fraud cases by device, helping identify devices with higher fraud cases."),

    dcc.Graph(id='fraud-type-pie-chart', config={'displayModeBar': False}),
    html.P("The pie chart gives a quick view of the percentage of fraud versus non-fraud cases."),

    dcc.Graph(id='purchase-value-distribution', config={'displayModeBar': False}),
    html.P("The histogram shows the distribution of purchase values, providing insight into typical purchase ranges for fraud and non-fraud cases."),

    dcc.Graph(id='age-distribution', config={'displayModeBar': False}),
    html.P("This histogram shows the distribution of customer ages, highlighting if fraud is more common in certain age groups."),

    dcc.Graph(id='correlation-heatmap', config={'displayModeBar': False}),
    html.P("The heatmap shows correlations among variables, useful for identifying features that might be relevant for fraud detection.")
])

# Define callback function to update the dashboard based on date selection
@app.callback(
    Output('summary-stats', 'children'),
    Output('time-series-fraud', 'figure'),
    Output('browser-fraud-graph', 'figure'),
    Output('device-fraud-graph', 'figure'),
    Output('fraud-type-pie-chart', 'figure'),
    Output('purchase-value-distribution', 'figure'),
    Output('age-distribution', 'figure'),
    Output('correlation-heatmap', 'figure'),
    Input('date-picker', 'start_date'),  # Input for start date
    Input('date-picker', 'end_date'),    # Input for end date
)
def update_dashboard(start_date, end_date):
    # Filter data based on the selected date range
    filtered_data = data[(data['purchase_time'] >= start_date) & (data['purchase_time'] <= end_date)]
    
    # Summary Statistics Calculation
    total_transactions = len(filtered_data)  # Total number of transactions in the filtered data
    total_frauds = len(filtered_data[filtered_data['class'] == 1])  # Total number of fraud cases
    fraud_percentage = (total_frauds / total_transactions) * 100 if total_transactions > 0 else 0  # Fraud percentage calculation
    avg_purchase_value = filtered_data['purchase_value'].mean()  # Average purchase value calculation
    
    # Create a summary statistics section to be displayed on the dashboard
    summary_stats = html.Div([
        html.H3("Summary Statistics"),
        html.P(f"Total Transactions: {total_transactions:,}"),
        html.P(f"Total Fraud Cases: {total_frauds:,}"),
        html.P(f"Fraud Percentage: {fraud_percentage:.2f}%"),
        html.P(f"Average Purchase Value: ${avg_purchase_value:.2f}")
    ])
    
    # Time Series Analysis: Daily transactions and fraud cases
    time_series_data = filtered_data.groupby(filtered_data['purchase_time'].dt.date).agg({
        'class': ['count', lambda x: (x == 1).sum()]  # Count total transactions and fraud cases
    }).reset_index()
    time_series_data.columns = ['date', 'total_transactions', 'fraud_count']  # Rename columns for clarity
    # Create a line plot for the time series
    time_series_fig = px.line(time_series_data, x='date', y=['total_transactions', 'fraud_count'],
                              title='Transactions and Fraud Cases Over Time')
    
    # Browser Fraud Analysis
    browser_fraud_counts = filtered_data.groupby('browser').agg({
        'class': ['count', lambda x: (x == 1).sum()]  # Count total transactions and fraud cases by browser
    }).reset_index()
    browser_fraud_counts.columns = ['browser', 'total_transactions', 'fraud_count']  # Rename columns for clarity
    # Create a bar plot for browser fraud analysis
    browser_fraud_fig = px.bar(browser_fraud_counts, x='browser', y=['total_transactions', 'fraud_count'],
                               title='Transactions and Fraud Cases by Browser',
                               labels={'value': 'Count', 'variable': 'Type'},
                               barmode='group')
    
    # Device Fraud Analysis
    device_fraud_counts = filtered_data.groupby('device_id').agg({
        'class': ['count', lambda x: (x == 1).sum()]  # Count total transactions and fraud cases by device
    }).reset_index()
    device_fraud_counts.columns = ['device_id', 'total_transactions', 'fraud_count']  # Rename columns for clarity
    # Create a bar plot for device fraud analysis
    device_fraud_fig = px.bar(device_fraud_counts, x='device_id', y=['total_transactions', 'fraud_count'],
                              title='Transactions and Fraud Cases by Device',
                              labels={'value': 'Count', 'variable': 'Type'},
                              barmode='group')
    
    # Fraud Type Distribution Pie Chart
    fraud_type_fig = px.pie(filtered_data, names='class', title='Fraud Cases Distribution')
    
    # Purchase Value Distribution Histogram
    purchase_value_fig = px.histogram(filtered_data, x='purchase_value', color='class', 
                                      title='Purchase Value Distribution',
                                      labels={'purchase_value': 'Purchase Value', 'class': 'Fraud Status'},
                                      marginal='box',  # Adding box plot to marginal for better insight
                                      hover_data=filtered_data.columns)  # Show all columns in hover data

    # Age Distribution Histogram
    age_fig = px.histogram(filtered_data, x='age', color='class', 
                           title='Age Distribution',
                           labels={'age': 'Age', 'class': 'Fraud Status'},
                           marginal='box',  # Adding box plot to marginal for better insight
                           hover_data=filtered_data.columns)

    # Correlation Heatmap
    numeric_columns = filtered_data.select_dtypes(include=['int64', 'float64']).columns  # Select only numeric columns
    correlation_matrix = filtered_data[numeric_columns].corr()  # Compute the correlation matrix
    correlation_fig = px.imshow(correlation_matrix,
                                x=correlation_matrix.columns,
                                y=correlation_matrix.columns,
                                color_continuous_scale='RdBu_r',
                                title='Correlation Heatmap')

    return (summary_stats, time_series_fig, browser_fraud_fig, device_fraud_fig, fraud_type_fig,
            purchase_value_fig, age_fig, correlation_fig)

# Run the Dash application
if __name__ == '__main__':
    app.run_server(debug=True)
