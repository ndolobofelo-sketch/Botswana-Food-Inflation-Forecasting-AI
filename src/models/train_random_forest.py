import pandas as pd
import numpy as np
import os
import joblib


from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


print("=" * 80)
print("RANDOM FOREST FOOD INFLATION FORECASTING MODEL")
print("=" * 80)



# ==================================================
# Load Data
# ==================================================

DATA_PATH = "data/processed/hcp_bwa_wide.csv"


df = pd.read_csv(DATA_PATH)


df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")



print("\nDataset loaded")

print(df.head())

print("\nShape:")
print(df.shape)



# ==================================================
# Prepare Data
# ==================================================

features = df.select_dtypes(
    include=["float64","int64"]
).columns



data = df[features].copy()



# Fix missing values

data = data.interpolate()

data = data.bfill()

data = data.ffill()



# ==================================================
# Create Lag Features
# ==================================================

window = 12


for i in range(1, window + 1):

    data[f"lag_{i}"] = data[
        features[0]
    ].shift(i)



data = data.dropna()



# Target

y = data[features[0]]



X = data.drop(
    columns=[features[0]]
)



print("\nFeature shape:")
print(X.shape)



# ==================================================
# Train Test Split
# ==================================================

split = int(
    len(X)*0.8
)


X_train = X.iloc[:split]

X_test = X.iloc[split:]


y_train = y.iloc[:split]

y_test = y.iloc[split:]



print("\nTraining:")
print(X_train.shape)


print("Testing:")
print(X_test.shape)



# ==================================================
# Train Random Forest
# ==================================================

model = RandomForestRegressor(

    n_estimators=300,

    max_depth=8,

    random_state=42,

    min_samples_split=5

)



print("\nTraining model...")


model.fit(
    X_train,
    y_train
)



# ==================================================
# Prediction
# ==================================================

predictions = model.predict(
    X_test
)



# ==================================================
# Evaluation
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
# Save Model
# ==================================================

os.makedirs(
    "models",
    exist_ok=True
)



joblib.dump(

    model,

    "models/random_forest_food_inflation.pkl"

)



# Save Predictions

os.makedirs(
    "reports",
    exist_ok=True
)


results = pd.DataFrame({

    "Actual": y_test,

    "Predicted": predictions

})


results.to_csv(

    "reports/random_forest_predictions.csv",

    index=False

)



# Feature importance

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_

})


importance.sort_values(

    by="Importance",

    ascending=False

).to_csv(

    "reports/random_forest_feature_importance.csv",

    index=False

)



print("\nSaved files:")

print("models/random_forest_food_inflation.pkl")

print("reports/random_forest_predictions.csv")

print("reports/random_forest_feature_importance.csv")


print("\nRANDOM FOREST TRAINING COMPLETE")