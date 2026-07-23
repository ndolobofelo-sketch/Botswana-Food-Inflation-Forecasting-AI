import pandas as pd
import numpy as np
import os
import random
import joblib


from sklearn.preprocessing import MinMaxScaler


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import EarlyStopping


# ===================================================
# Reproducibility
# ===================================================

SEED = 42

np.random.seed(SEED)
random.seed(SEED)


print("=" * 80)
print("LSTM FOOD INFLATION FORECASTING MODEL")
print("=" * 80)



# ===================================================
# Paths
# ===================================================

DATA_PATH = "data/processed/hcp_bwa_wide.csv"

MODEL_PATH = "models/lstm_food_inflation.keras"

SCALER_PATH = "models/lstm_scaler.pkl"

HISTORY_PATH = "models/lstm_training_history.csv"



# ===================================================
# Load Dataset
# ===================================================

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )


df = pd.read_csv(DATA_PATH)



print("\nDataset loaded successfully")

print(df.head())

print("\nDataset shape:")
print(df.shape)



# ===================================================
# Date Processing
# ===================================================

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")



# ===================================================
# Select Features
# ===================================================

numeric_features = df.select_dtypes(
    include=["float64","int64"]
).columns



print("\nFeatures used:")

print(list(numeric_features))



data = df[numeric_features].copy()



# ===================================================
# Missing Value Handling
# ===================================================

print("\nMissing values before cleaning:")

print(data.isnull().sum())



# interpolate missing values

data = data.interpolate(
    method="linear"
)


# backward fill remaining

data = data.bfill()


# forward fill remaining

data = data.ffill()



print("\nMissing values after cleaning:")

print(data.isnull().sum())



# ===================================================
# Scaling
# ===================================================


scaler = MinMaxScaler()


scaled_data = scaler.fit_transform(
    data
)



# Save scaler

os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(
    scaler,
    SCALER_PATH
)


print("\nScaler saved:")
print(SCALER_PATH)



# ===================================================
# Sequence Creation
# ===================================================


def create_sequences(
        data,
        window=12
):

    X = []
    y = []


    for i in range(
        len(data)-window
    ):

        X.append(
            data[i:i+window]
        )


        # Predict first feature

        y.append(
            data[i+window,0]
        )


    return np.array(X), np.array(y)




WINDOW = 12



X,y = create_sequences(
    scaled_data,
    WINDOW
)



print("\nSequence shapes")

print("X:", X.shape)

print("y:", y.shape)



# ===================================================
# Train Test Split
# ===================================================


split = int(
    len(X)*0.8
)


X_train = X[:split]

X_test = X[split:]


y_train = y[:split]

y_test = y[split:]



print("\nTraining data:")
print(X_train.shape)


print("\nTesting data:")
print(X_test.shape)




# ===================================================
# Build LSTM Network
# ===================================================


model = Sequential()



model.add(
    Input(
        shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    )
)



model.add(
    LSTM(
        64,
        return_sequences=True
    )
)



model.add(
    Dropout(0.2)
)



model.add(
    LSTM(
        32
    )
)



model.add(
    Dropout(0.2)
)



model.add(
    Dense(1)
)



model.compile(

    optimizer="adam",

    loss="mse",

    metrics=[
        "mae"
    ]

)



print("\nMODEL SUMMARY")

model.summary()



# ===================================================
# Training
# ===================================================


early_stop = EarlyStopping(

    monitor="val_loss",

    patience=15,

    restore_best_weights=True

)



history = model.fit(

    X_train,

    y_train,

    validation_data=(

        X_test,

        y_test

    ),

    epochs=100,

    batch_size=8,

    callbacks=[
        early_stop
    ],

    verbose=1

)



# ===================================================
# Save Training History
# ===================================================


history_df = pd.DataFrame(
    history.history
)


history_df.to_csv(
    HISTORY_PATH,
    index=False
)


print("\nTraining history saved:")
print(HISTORY_PATH)



# ===================================================
# Save Model
# ===================================================


model.save(
    MODEL_PATH
)



print("\nMODEL SAVED SUCCESSFULLY")

print(MODEL_PATH)



print("="*80)
print("LSTM TRAINING COMPLETE")
print("="*80)
