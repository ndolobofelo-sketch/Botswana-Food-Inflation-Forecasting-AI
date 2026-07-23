"""
Improved Food Inflation Forecasting Pipeline

Removes target leakage and evaluates
true forecasting performance.
"""

import pandas as pd
import numpy as np
import os


from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


INPUT_FILE = (
    "data/processed/"
    "food_inflation_features.csv"
)


OUTPUT_FILE = (
    "reports/"
    "forecasting_model_results.csv"
)


print("=" * 80)
print("FOOD INFLATION FORECASTING MODELS")
print("=" * 80)


# Load data

df = pd.read_csv(INPUT_FILE)


df["Date"] = pd.to_datetime(df["Date"])


df = df.sort_values(
    "Date"
).reset_index(drop=True)


TARGET = "FAO_CP_23012"


# Remove direct food indicators

remove_columns = [
    "Date",
    "FAO_CP_23012",
    "FAO_CP_23013",
    "FAO_CP_23014",
]


X = df.drop(
    columns=remove_columns
)


y = df[TARGET]


# Fill missing values

X = X.fillna(
    X.median()
)


# Time split

split = int(
    len(df) * 0.8
)


X_train = X.iloc[:split]

X_test = X.iloc[split:]


y_train = y.iloc[:split]

y_test = y.iloc[split:]


print("\nTraining size:", len(X_train))

print("Testing size:", len(X_test))


models = {

    "Random Forest":

    RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ),


    "Gradient Boosting":

    GradientBoostingRegressor(
        random_state=42
    )

}


results = []


for name, model in models.items():

    print("\nTraining:", name)


    model.fit(
        X_train,
        y_train
    )


    pred = model.predict(
        X_test
    )


    mae = mean_absolute_error(
        y_test,
        pred
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred
        )
    )


    mape = (
        np.mean(
            np.abs(
                (y_test - pred)
                /
                y_test
            )
        )
        * 100
    )


    results.append(
        {
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        }
    )


results_df = pd.DataFrame(results)


print("\nRESULTS")
print(results_df)


os.makedirs(
    "reports",
    exist_ok=True
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved:")
print(OUTPUT_FILE)


print("=" * 80)
print("COMPLETE")
print("=" * 80)