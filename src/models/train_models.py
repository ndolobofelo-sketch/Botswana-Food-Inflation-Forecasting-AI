"""
Machine Learning Model Training Pipeline

Purpose:
--------
Train and compare multiple models
for food price forecasting.

Leakage control:
----------------
Only information available before
the prediction period is used.
"""


import pandas as pd
import numpy as np
import os


from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# Paths
# ============================================================

INPUT_FILE = (
    "data/processed/"
    "food_inflation_features.csv"
)


OUTPUT_FILE = (
    "reports/"
    "model_results.csv"
)


print("=" * 80)
print("FOOD INFLATION MODEL TRAINING")
print("=" * 80)


# ============================================================
# Load Data
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


print("\nDataset shape:")
print(df.shape)


# Convert date

df["Date"] = pd.to_datetime(
    df["Date"]
)


df = df.sort_values(
    "Date"
).reset_index(drop=True)



# ============================================================
# Target
# ============================================================

TARGET = "FAO_CP_23012"


print("\nTarget variable:")
print(TARGET)



df = df.dropna(
    subset=[TARGET]
)



# ============================================================
# Feature Preparation
# Leakage Removed
# ============================================================

print("\nRemoving leakage features...")


remove_columns = [

    "Date",

    # Main target
    "FAO_CP_23012",

    # Same month FAO information
    "FAO_CP_23013",
    "FAO_CP_23014",


    # Botswana
    "BWA_FAO_CP_23012",
    "BWA_FAO_CP_23013",
    "BWA_FAO_CP_23014",


    # Kenya
    "KEN_FAO_CP_23012",
    "KEN_FAO_CP_23013",
    "KEN_FAO_CP_23014",


    # Namibia
    "NAM_FAO_CP_23012",
    "NAM_FAO_CP_23013",
    "NAM_FAO_CP_23014",


    # South Africa
    "ZAF_FAO_CP_23012",
    "ZAF_FAO_CP_23013",
    "ZAF_FAO_CP_23014",


    # Zimbabwe
    "ZWE_FAO_CP_23012",
    "ZWE_FAO_CP_23013",
    "ZWE_FAO_CP_23014"
]


X = df.drop(
    columns=remove_columns
)


y = df[TARGET]



# Fill missing values

X = X.fillna(
    X.median()
)



print("\nFeatures:")
print(X.shape)



# ============================================================
# Time Split
# ============================================================

split = int(
    len(df) * 0.8
)


X_train = X.iloc[:split]

X_test = X.iloc[split:]


y_train = y.iloc[:split]

y_test = y.iloc[split:]



print("\nTraining samples:")
print(len(X_train))


print("Testing samples:")
print(len(X_test))



# ============================================================
# Models
# ============================================================


models = {


    "Linear Regression":

    Pipeline(
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
    ),



    "Random Forest":

    RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ),



    "Gradient Boosting":

    GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        random_state=42
    )

}



# ============================================================
# Training
# ============================================================


results = []


for name, model in models.items():


    print("\nTraining:")
    print(name)


    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )


    mae = mean_absolute_error(
        y_test,
        predictions
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )


    mape = (
        np.mean(
            np.abs(
                (y_test - predictions)
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



# ============================================================
# Results
# ============================================================


results_df = pd.DataFrame(
    results
)


print("\nMODEL RESULTS")

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
print("MODEL TRAINING COMPLETE")
print("=" * 80)