# ==========================================
# Retail Sales Prediction Project
# ==========================================

# Step 1: Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load Dataset

df = pd.read_csv("train.csv")

print(df.head())
print(df.info())

# Step 3: Data Cleaning

print(df.isnull().sum())

# Convert date into datetime

df["date"] = pd.to_datetime(df["date"])

# Feature Engineering

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek

# Step 4: Convert categorical variables

df = pd.get_dummies(df, columns=["family"], drop_first=True)

# Step 5: Define Features

X = df.drop(columns=["sales", "date"])
y = df["sales"]

# Step 6: Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Step 7: Train Model

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Step 8: Prediction

predictions = model.predict(X_test)

# Step 9: Evaluation

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("-------------------")
print("MAE :", mae)
print("RMSE:", rmse)
print("R² Score:", r2)

# Step 10: Feature Importance

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10,6))
importance.head(15).plot(kind="bar")
plt.title("Top 15 Important Features")
plt.ylabel("Importance")
plt.show()

# Step 11: Actual vs Predicted

plt.figure(figsize=(8,6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# Step 12: Sales Trend

monthly_sales = df.groupby("month")["sales"].mean()

plt.figure(figsize=(8,5))
monthly_sales.plot(marker='o')
plt.title("Average Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Average Sales")
plt.grid(True)
plt.show()
