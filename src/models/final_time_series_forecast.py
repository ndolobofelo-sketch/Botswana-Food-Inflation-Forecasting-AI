"""
Final Food Inflation Forecast Model

Model:
Linear Regression

Forecast:
January 2024 - December 2024
"""


import pandas as pd
import numpy as np
import os


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression



print("=" * 80)
print("FINAL FOOD INFLATION FORECAST")
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



# Features

features = [

    "FAO_CP_23012_lag1",
    "FAO_CP_23012_lag3",
    "FAO_CP_23012_lag6",
    "FAO_CP_23012_lag12",

    "FAO_CP_23012_rolling3",
    "FAO_CP_23012_rolling6",

    "Brent_USD_per_barrel",
    "policy_rate",

    "Year",
    "Month",
    "Quarter"

]



X = df[features]

y = df["FAO_CP_23012"]



X = X.fillna(
    X.median()
)



# Train model

print("\nTraining Linear Regression...")


model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LinearRegression()
        )
    ]
)



model.fit(
    X,
    y
)



# Recursive forecast

history = list(
    y.values
)



future_dates = pd.date_range(
    start="2024-01-01",
    periods=12,
    freq="MS"
)


predictions = []



for date in future_dates:


    row = pd.DataFrame(
        {

        "FAO_CP_23012_lag1":
            [history[-1]],


        "FAO_CP_23012_lag3":
            [history[-3]],


        "FAO_CP_23012_lag6":
            [history[-6]],


        "FAO_CP_23012_lag12":
            [history[-12]],


        "FAO_CP_23012_rolling3":
            [np.mean(history[-3:])],


        "FAO_CP_23012_rolling6":
            [np.mean(history[-6:])],


        "Brent_USD_per_barrel":
            [df["Brent_USD_per_barrel"].iloc[-1]],


        "policy_rate":
            [df["policy_rate"].iloc[-1]],


        "Year":
            [date.year],


        "Month":
            [date.month],


        "Quarter":
            [date.quarter]

        }
    )



    prediction = model.predict(row)[0]


    predictions.append(
        prediction
    )


    history.append(
        prediction
    )



forecast = pd.DataFrame(
    {

    "Date":future_dates,

    "FAO_CP_23012_Prediction":
        predictions

    }
)



os.makedirs(
    "submissions",
    exist_ok=True
)



forecast.to_csv(
    "submissions/final_predictions.csv",
    index=False
)



print("\nForecast:")
print(forecast)



print("\nSaved:")
print(
"submissions/final_predictions.csv"
)


print("=" * 80)
print("COMPLETE")
print("=" * 80)