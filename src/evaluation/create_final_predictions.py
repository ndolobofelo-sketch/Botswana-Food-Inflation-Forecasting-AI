import pandas as pd
import numpy as np
import os
import joblib

from tensorflow.keras.models import load_model



print("=" * 80)
print("CREATING FINAL FOOD INFLATION PREDICTIONS")
print("=" * 80)



# ==================================================
# Load dataset
# ==================================================

DATA_PATH = "data/processed/hcp_bwa_wide.csv"


df = pd.read_csv(DATA_PATH)


df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")



# ==================================================
# Prepare features
# ==================================================

features = df.select_dtypes(
    include=["float64","int64"]
).columns



data = df[features].copy()



# Fill missing values

data = data.interpolate()

data = data.bfill()

data = data.ffill()



# ==================================================
# Load scaler and model
# ==================================================

print("\nLoading LSTM model...")


model = load_model(
    "models/lstm_food_inflation.keras"
)



scaler = joblib.load(
    "models/lstm_scaler.pkl"
)



print("Model loaded successfully")



# ==================================================
# Scale data
# ==================================================

scaled_data = scaler.transform(
    data
)



# ==================================================
# Create last sequence
# ==================================================

WINDOW = 12


last_sequence = scaled_data[-WINDOW:]


X_future = np.array(
    [last_sequence]
)



# ==================================================
# Forecast next 12 months
# ==================================================

future_predictions = []


current_sequence = last_sequence.copy()



for i in range(12):


    prediction = model.predict(
        current_sequence.reshape(
            1,
            WINDOW,
            len(features)
        ),
        verbose=0
    )


    future_predictions.append(
        prediction[0][0]
    )


    # Update sequence

    new_row = current_sequence[-1].copy()

    new_row[0] = prediction[0][0]


    current_sequence = np.vstack(
        [
            current_sequence[1:],
            new_row
        ]
    )



# ==================================================
# Convert back to original scale
# ==================================================

future_array = np.zeros(
    (
        12,
        len(features)
    )
)



future_array[:,0] = future_predictions



future_original = scaler.inverse_transform(
    future_array
)



final_values = future_original[:,0]



# ==================================================
# Create submission file
# ==================================================

future_dates = pd.date_range(

    start="2024-01-01",

    periods=12,

    freq="MS"

)



submission = pd.DataFrame({

    "Date": future_dates,

    "Food_Price_Forecast": final_values

})



os.makedirs(

    "submissions",

    exist_ok=True

)



submission.to_csv(

    "submissions/final_predictions.csv",

    index=False

)



print("\nFINAL PREDICTIONS")

print(submission)



print("\nSaved:")
print(
    "submissions/final_predictions.csv"
)


print("=" * 80)
print("FINAL FORECAST COMPLETE")
print("=" * 80)