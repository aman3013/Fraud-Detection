# Fraud Detection System for E-commerce and Banking Transactions

## Overview

This project, developed by Adey Innovations Inc., aims to improve the detection of fraud cases in e-commerce transactions and bank credit transactions using advanced machine learning models. It addresses unique challenges posed by both types of transaction data and leverages geolocation and transaction pattern analysis to enhance fraud detection accuracy.

The solution focuses on providing enhanced security, real-time monitoring, and efficient fraud detection to reduce financial losses and increase customer trust. 

## Project Goals

1. **Data Preprocessing and Cleaning:** Prepare transaction data for analysis.
2. **Feature Engineering:** Identify features to help spot fraud patterns.
3. **Model Building:** Train and test multiple machine learning models.
4. **Model Explainability:** Utilize SHAP and LIME to interpret models.
5. **Deployment and API Development:** Deploy models with Flask, Dockerize, and create APIs.
6. **Visualization Dashboard:** Build an interactive dashboard for fraud detection insights.

## Data Sources

- **Fraud_Data.csv:** Contains e-commerce transaction data.
- **IpAddress_to_Country.csv:** Maps IP address ranges to countries.
- **creditcard.csv:** Contains bank transaction data curated for fraud detection.

## Dataset Details

### Fraud_Data.csv
- **user_id:** Unique identifier for the user.
- **signup_time:** Timestamp when the user signed up.
- **purchase_time:** Timestamp of purchase.
- **purchase_value:** Value of the transaction.
- **device_id:** Unique device identifier.
- **source:** Source of site access (SEO, Ads).
- **browser:** Browser used for the transaction.
- **sex:** Gender of the user (M/F).
- **age:** Age of the user.
- **ip_address:** IP address for the transaction.
- **class:** Target variable (1 for fraud, 0 for non-fraud).

### IpAddress_to_Country.csv
- **lower_bound_ip_address:** Lower bound of IP range.
- **upper_bound_ip_address:** Upper bound of IP range.
- **country:** Country corresponding to the IP range.

### creditcard.csv
- **Time:** Seconds between this and the first transaction.
- **V1 to V28:** PCA-transformed features.
- **Amount:** Transaction amount.
- **Class:** Target variable (1 for fraud, 0 for non-fraud).

---

## Project Tasks

### Task 1 - Data Analysis and Preprocessing

- **Missing Values:** Impute or drop as necessary.
- **Data Cleaning:** Remove duplicates, ensure correct data types.
- **Exploratory Data Analysis (EDA):** Conduct univariate and bivariate analysis.
- **Geolocation Analysis:** Map IP addresses to countries.
- **Feature Engineering:** Create transaction frequency, time-based features, and normalize/scale data.
- **Encoding Categorical Variables.**

### Task 2 - Model Building and Training

- **Data Preparation:** Separate features and target (`Class` or `class`), split data.
- **Model Selection:** Train and evaluate multiple models including:
  - Logistic Regression, Decision Tree, Random Forest
  - Gradient Boosting, MLP, CNN, RNN, LSTM
- **MLOps Steps:** Version and track experiments with MLflow.

### Task 3 - Model Explainability

- **SHAP:** Visualize feature contributions using:
  - Summary, Force, and Dependence Plots.
- **LIME:** Use LIME to understand feature importance on individual predictions.

### Task 4 - Model Deployment and API Development

- **Flask API:** Set up and serve models with Flask.
- **Dockerization:** Containerize the Flask app for scalable deployment.
- **Logging:** Integrate logging for request and error tracking.

### Task 5 - Build a Dashboard with Flask and Dash

- **Data Visualization:** Develop an interactive dashboard with Dash.
  - Display transaction summaries, fraud trends, and geolocation of fraud.
  - Visualize fraud across devices, browsers, and countries.

---

## Technologies Used

- **Languages/Tools:** Python, Flask, Docker, SQL, Git, GitHub CI/CD.
- **Libraries:** Pandas, Matplotlib, Scikit-learn, SHAP, LIME, MLflow.
- **Database:** MySQL for database management.
- **APIs:** Flask API for model serving, REST API for dashboard data.
- **Containerization:** Docker for deploying the Flask app.
- **Visualization:** Dash for an interactive dashboard.


## Deployment Steps

### 1. Set Up Flask API

```bash
# In project root
$ mkdir flask-app
$ cd flask-app

### Dockerfile

FROM python:3.8-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "serve_model.py"]

### Build and run the Docker container

$ docker build -t fraud-detection-model .
$ docker run -p 5000:5000 fraud-detection-model

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
