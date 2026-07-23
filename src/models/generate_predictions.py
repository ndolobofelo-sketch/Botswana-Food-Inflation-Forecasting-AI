"""
Generate 2024 Food Inflation Forecasts

Purpose:
--------
Train final model using historical data
and generate monthly predictions for 2024.
"""

import pandas as pd
import os


from sklearn.ensemble import GradientBoostingRegressor


print("=" * 80)
print("GENERATING 2024 FOOD PRICE FORECASTS")
print("=" * 80)


# ==================================================
# Load Feature Dataset
# ==================================================

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


# ==================================================
# Target
# ==================================================

TARGET = "FAO_CP_23012"


# ==================================================
# Prepare Training Data
# ==================================================

X = df.drop(
    columns=[
        "Date",
        "FAO_CP_23012",
        "FAO_CP_23013",
        "FAO_CP_23014"
    ]
)


y = df[TARGET]


# Fill missing values

X = X.fillna(
    X.median()
)


print("\nTraining data:")
print(X.shape)


# ==================================================
# Train Final Model
# ==================================================

print("\nTraining Gradient Boosting model...")


model = GradientBoostingRegressor(
    random_state=42
)


model.fit(
    X,
    y
)


# ==================================================
# Create Future Dates
# ==================================================

future_dates = pd.date_range(
    start="2024-01-01",
    end="2024-12-01",
    freq="MS"
)


# ==================================================
# Forecast
# ==================================================

print("\nGenerating predictions...")


# Use last available feature row
# as future feature approximation

future_features = pd.DataFrame(
    [
        X.iloc[-1].values
    ] * len(future_dates),
    columns=X.columns
)


predictions = model.predict(
    future_features
)


# ==================================================
# Save Predictions
# ==================================================

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


OUTPUT = (
    "submissions/"
    "predictions_2024.csv"
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
print("FORECAST COMPLETE")
print("=" * 80)