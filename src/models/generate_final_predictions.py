"""
Final 2024 Food Inflation Forecast Generator

Creates monthly forecasts:
January 2024 - December 2024
"""

import pandas as pd
import numpy as np
import os

from sklearn.ensemble import GradientBoostingRegressor


print("=" * 80)
print("FINAL FOOD INFLATION FORECAST")
print("=" * 80)


# --------------------------------------------------
# Load Data
# --------------------------------------------------

INPUT_FILE = (
    "data/processed/"
    "food_inflation_features.csv"
)


df = pd.read_csv(INPUT_FILE)


df["Date"] = pd.to_datetime(
    df["Date"]
)


df = df.sort_values(
    "Date"
).reset_index(drop=True)


TARGET = "FAO_CP_23012"


# --------------------------------------------------
# Prepare Training Data
# --------------------------------------------------

X = df.drop(
    columns=[
        "Date",
        "FAO_CP_23012",
        "FAO_CP_23013",
        "FAO_CP_23014"
    ]
)


y = df[TARGET]


X = X.fillna(
    X.median()
)


# --------------------------------------------------
# Train Final Model
# --------------------------------------------------

print("\nTraining Gradient Boosting...")


model = GradientBoostingRegressor(
    random_state=42
)


model.fit(
    X,
    y
)


# --------------------------------------------------
# Create Future Features
# --------------------------------------------------

last_features = X.iloc[-1].copy()


future_dates = pd.date_range(
    start="2024-01-01",
    periods=12,
    freq="MS"
)


future_predictions = []


for date in future_dates:

    future_row = last_features.copy()


    # Update time features if they exist

    if "Year" in future_row.index:
        future_row["Year"] = date.year

    if "Month" in future_row.index:
        future_row["Month"] = date.month

    if "Quarter" in future_row.index:
        future_row["Quarter"] = date.quarter


    prediction = model.predict(
        pd.DataFrame(
            [future_row]
        )
    )[0]


    future_predictions.append(
        prediction
    )


# --------------------------------------------------
# Save
# --------------------------------------------------

forecast = pd.DataFrame(
    {
        "Date": future_dates,
        "FAO_CP_23012_Prediction": future_predictions
    }
)


os.makedirs(
    "submissions",
    exist_ok=True
)


OUTPUT = (
    "submissions/"
    "final_predictions_2024.csv"
)


forecast.to_csv(
    OUTPUT,
    index=False
)


print("\nForecast:")
print(forecast)


print("\nSaved:")
print(OUTPUT)


print("=" * 80)
print("FINAL FORECAST COMPLETE")
print("=" * 80)