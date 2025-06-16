# Coffee Sales Analysis
# Author: Haripriya Bhallam
# Date: 2025
# Objective: Explore customer behavior and forecast coffee sales using a data science.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# Load data
DATA_PATH = '../data/coffee_sales.csv'
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Please place the coffee_sales.csv file in the 'data' folder.")

df = pd.read_csv(DATA_PATH)

# Preprocessing
df['date'] = pd.to_datetime(df['date'])
df['datetime'] = pd.to_datetime(df['datetime'])
df['card'] = df['card'].fillna("CASH_USER")
df['month'] = df['date'].dt.to_period('M').astype(str)
df['weekday'] = df['date'].dt.day_name()
df['hour'] = df['datetime'].dt.hour

# Prepare data for ML
features = ['coffee_name', 'cash_type', 'hour', 'weekday']
df_ml = df[features + ['money']].copy()
df_ml = pd.get_dummies(df_ml, columns=['coffee_name', 'cash_type', 'weekday'], drop_first=True)

X = df_ml.drop(columns=['money'])
y = df_ml['money']

# coffee_sales_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings

warnings.filterwarnings("ignore")

# Load Data
DATA_PATH = '../data/coffee_sales.csv'
if not os.path.exists(DATA_PATH):
    print("Please place the coffee_sales.csv file in the 'data' folder.")
else:
    df = pd.read_csv(DATA_PATH)

    # Initial Inspection
    print("\n--- Data Head ---")
    print(df.head())

    print("\n--- Info ---")
    print(df.info())

    print("\n--- Null Values ---")
    print(df.isnull().sum())

    print("\n--- Duplicate Rows ---")
    print(df.duplicated().sum())

    # Convert dates
    df['date'] = pd.to_datetime(df['date'])
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Fill missing values in 'card'
    df['card'] = df['card'].fillna("CASH_USER")

    # Add new time features
    df['month'] = df['date'].dt.to_period('M')
    df['weekday'] = df['date'].dt.day_name()
    df['hour'] = df['datetime'].dt.hour

    # EDA: Sales by Product
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, y='coffee_name', order=df['coffee_name'].value_counts().index)
    plt.title("Coffee Sales by Product")
    plt.xlabel("Count")
    plt.ylabel("Coffee Type")
    plt.tight_layout()
    plt.savefig("../reports/sales_by_product.png")
    plt.close()

    # EDA: Hourly Trends
    hourly_trend = df.groupby('hour').size()
    hourly_trend.plot(kind='bar', figsize=(10, 4), title="Sales by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Sales")
    plt.tight_layout()
    plt.savefig("../reports/hourly_sales.png")
    plt.close()

    # Monthly Trends
    monthly = df.groupby(df['month']).size()
    monthly.plot(kind='line', marker='o', title="Monthly Sales Trends", figsize=(10, 4))
    plt.xlabel("Month")
    plt.ylabel("Sales Count")
    plt.tight_layout()
    plt.savefig("../reports/monthly_trends.png")
    plt.close()

    print("\n--- Analysis Complete. Visuals saved in reports/ ---")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

-- coffee_sales_queries.sql

-- 1. Total sales by day
SELECT date, COUNT(*) AS total_sales
FROM coffee_sales
GROUP BY date
ORDER BY date;

-- 2. Top 5 best-selling coffee types
SELECT coffee_name, COUNT(*) AS sales_count
FROM coffee_sales
GROUP BY coffee_name
ORDER BY sales_count DESC
LIMIT 5;

-- 3. Sales distribution by hour
SELECT EXTRACT(HOUR FROM datetime) AS sale_hour, COUNT(*) AS total_sales
FROM coffee_sales
GROUP BY sale_hour
ORDER BY sale_hour;

-- 4. Most active customers (non-cash)
SELECT card, COUNT(*) AS purchases
FROM coffee_sales
WHERE card IS NOT NULL AND card != 'CASH_USER'
GROUP BY card
ORDER BY purchases DESC
LIMIT 10;

-- 5. Sales by weekday
SELECT TO_CHAR(date, 'Day') AS weekday, COUNT(*) AS total_sales
FROM coffee_sales
GROUP BY weekday
ORDER BY total_sales DESC;

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("ML Model Results")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.4f}")

# Save coefficients
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"])
coeff_df.to_csv("../reports/ml_model_coefficients.csv", index=True)

print("Model coefficients saved to /reports/ml_model_coefficients.csv")



    

    


   
