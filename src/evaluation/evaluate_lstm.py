import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("=" * 80)
print("LSTM MODEL EVALUATION")
print("=" * 80)


# Files we need

DATA_PATH = "data/processed/hcp_bwa_wide.csv"

MODEL_PATH = "models/lstm_food_inflation.keras"

SCALER_PATH = "models/lstm_scaler.pkl"



# Load dataset

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")



# Select numerical columns

features = df.select_dtypes(
    include=["float64", "int64"]
).columns


data = df[features].copy()



# Fix missing values

data = data.interpolate()

data = data.bfill()

data = data.ffill()



# Load scaler

scaler = joblib.load(
    SCALER_PATH
)



scaled_data = scaler.transform(
    data
)



# Create sequences

window = 12

X = []
y = []


for i in range(len(scaled_data)-window):

    X.append(
        scaled_data[i:i+window]
    )

    y.append(
        scaled_data[i+window, 0]
    )



X = np.array(X)

y = np.array(y)



# Use testing data

split = int(len(X)*0.8)


X_test = X[split:]

y_test = y[split:]



# Load trained model

model = load_model(
    MODEL_PATH
)



# Make predictions

predictions = model.predict(
    X_test
)



# Calculate accuracy

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



# Save predictions

results = pd.DataFrame({

    "Actual": y_test,

    "Predicted": predictions.flatten()

})


results.to_csv(
    "reports/lstm_predictions.csv",
    index=False
)



print("\nSaved:")
print("reports/lstm_predictions.csv")