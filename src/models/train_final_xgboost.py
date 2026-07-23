import pandas as pd
import numpy as np
import os

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


print("=" * 80)
print("FINAL XGBOOST FOOD INFLATION MODEL")
print("=" * 80)


# ==================================================
# LOAD DATA
# ==================================================

DATA_PATH = "data/processed/final_feature_dataset.csv"


if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(DATA_PATH)


df = pd.read_csv(DATA_PATH)


print("\nDataset loaded")
print(df.head())

print("\nOriginal shape:")
print(df.shape)



# ==================================================
# DATE SORTING
# ==================================================

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)



# ==================================================
# TARGET
# ==================================================

TARGET = "Food_Price_Index"



if TARGET not in df.columns:
    raise ValueError(
        f"Target {TARGET} not found"
    )



# ==================================================
# MISSING VALUE HANDLING
# ==================================================

print("\nMissing values before cleaning:")

print(
    df.isnull().sum().sum()
)


# We only work with numerical columns
numeric_columns = df.select_dtypes(
    include=["number"]
).columns


# Forward fill time series
df[numeric_columns] = (
    df[numeric_columns]
    .ffill()
)


# Back fill remaining first rows
df[numeric_columns] = (
    df[numeric_columns]
    .bfill()
)


# Final safety replacement
for col in numeric_columns:

    if df[col].isnull().sum() > 0:

        df[col] = df[col].fillna(
            df[col].median()
        )


print("\nMissing values after cleaning:")

print(
    df.isnull().sum().sum()
)


print("\nFinal shape:")
print(df.shape)



# ==================================================
# FEATURES AND TARGET
# ==================================================

X = df.drop(
    columns=[
        "Date",
        TARGET
    ]
)


y = df[TARGET]



print("\nNumber of features:")
print(X.shape[1])



# ==================================================
# TIME SERIES TRAIN TEST SPLIT
# ==================================================

split = int(
    len(df) * 0.8
)


X_train = X.iloc[:split]

X_test = X.iloc[split:]


y_train = y.iloc[:split]

y_test = y.iloc[split:]



print("\nTraining:")
print(X_train.shape)


print("Testing:")
print(X_test.shape)



# Safety check

if len(X_train) == 0 or len(X_test) == 0:

    raise ValueError(
        "Training or testing dataset is empty"
    )



# ==================================================
# XGBOOST MODEL
# ==================================================

model = XGBRegressor(

    n_estimators=300,

    learning_rate=0.03,

    max_depth=4,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42

)



print("\nTraining model...")


model.fit(
    X_train,
    y_train
)



# ==================================================
# PREDICTION
# ==================================================

predictions = model.predict(
    X_test
)



# ==================================================
# EVALUATION
# ==================================================

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


r2 = r2_score(
    y_test,
    predictions
)



print("\nRESULTS")
print("----------------")

print("MAE :", mae)

print("RMSE:", rmse)

print("R2  :", r2)



# ==================================================
# SAVE MODEL
# ==================================================

os.makedirs(
    "models",
    exist_ok=True
)


model.save_model(
    "models/final_xgboost_food_inflation.json"
)



# ==================================================
# SAVE PREDICTIONS
# ==================================================

os.makedirs(
    "reports",
    exist_ok=True
)


prediction_results = pd.DataFrame({

    "Actual":

    y_test.values,


    "Prediction":

    predictions

})


prediction_results.to_csv(

    "reports/final_xgboost_predictions.csv",

    index=False

)



# ==================================================
# FEATURE IMPORTANCE
# ==================================================

feature_importance = pd.DataFrame({

    "Feature":

    X.columns,


    "Importance":

    model.feature_importances_

})


feature_importance = feature_importance.sort_values(

    by="Importance",

    ascending=False

)


feature_importance.to_csv(

    "reports/final_xgboost_feature_importance.csv",

    index=False

)



print("\nSaved files:")

print(
"models/final_xgboost_food_inflation.json"
)

print(
"reports/final_xgboost_predictions.csv"
)

print(
"reports/final_xgboost_feature_importance.csv"
)


print("=" * 80)

print("FINAL XGBOOST COMPLETE")

print("=" * 80)