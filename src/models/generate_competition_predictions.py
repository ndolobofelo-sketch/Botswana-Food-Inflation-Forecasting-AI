"""
Competition Food Inflation Forecast Generator

Creates:
January 2024 - December 2024 predictions
"""

import pandas as pd
import numpy as np
import os

from sklearn.ensemble import GradientBoostingRegressor


print("=" * 80)
print("COMPETITION FORECAST GENERATION")
print("=" * 80)


# Load data

df = pd.read_csv(
    "data/processed/food_inflation_features.csv"
)


df["Date"] = pd.to_datetime(
    df["Date"]
)


df = df.sort_values(
    "Date"
).reset_index(drop=True)


TARGET = "FAO_CP_23012"


# Prepare training data

feature_columns = [
    col for col in df.columns
    if col not in [
        "Date",
        "FAO_CP_23012",
        "FAO_CP_23013",
        "FAO_CP_23014"
    ]
]


X = df[feature_columns]

y = df[TARGET]


X = X.fillna(
    X.median()
)


# Train model

print("\nTraining Gradient Boosting model...")


model = GradientBoostingRegressor(
    random_state=42
)


model.fit(
    X,
    y
)


# Start from last known month

last_row = df.iloc[-1].copy()


predictions = []


future_dates = pd.date_range(
    "2024-01-01",
    "2024-12-01",
    freq="MS"
)


previous_prediction = last_row[TARGET]


for date in future_dates:

    future = last_row.copy()


    future["Year"] = date.year
    future["Month"] = date.month
    future["Quarter"] = date.quarter


    # update lag features

    if "FAO_CP_23012_lag1" in future:
        future["FAO_CP_23012_lag1"] = previous_prediction


    if "FAO_CP_23012_lag3" in future:
        future["FAO_CP_23012_lag3"] = previous_prediction


    if "FAO_CP_23012_lag6" in future:
        future["FAO_CP_23012_lag6"] = previous_prediction


    prediction = model.predict(
        pd.DataFrame(
            [future[feature_columns]]
        )
    )[0]


    predictions.append(
        prediction
    )


    previous_prediction = prediction



forecast = pd.DataFrame(
    {
        "Date": future_dates,
        "FAO_CP_23012_Prediction": predictions
    }
)


os.makedirs(
    "submissions",
    exist_ok=True
)


forecast.to_csv(
    "submissions/competition_predictions_2024.csv",
    index=False
)


print("\nForecast:")
print(forecast)


print("\nSaved:")
print(
    "submissions/competition_predictions_2024.csv"
)


print("=" * 80)
print("DONE")
print("=" * 80)