"""
Feature Engineering Pipeline

Creates machine learning features
for food price forecasting.
"""

import pandas as pd
import os


INPUT_FILE = "data/processed/master_food_inflation_dataset.csv"

OUTPUT_FILE = "data/processed/food_inflation_features.csv"


print("=" * 80)
print("CREATING MACHINE LEARNING FEATURES")
print("=" * 80)


# Load dataset

df = pd.read_csv(INPUT_FILE)


print("\nOriginal shape:")
print(df.shape)


# Convert date

df["Date"] = pd.to_datetime(df["Date"])


# Sort by time

df = df.sort_values("Date").reset_index(drop=True)


# ==================================================
# Time Features
# ==================================================

print("\nCreating time features...")


df["Year"] = df["Date"].dt.year

df["Month"] = df["Date"].dt.month

df["Quarter"] = df["Date"].dt.quarter


# ==================================================
# Target Feature Lags
# ==================================================

print("Creating lag features...")


target_columns = [
    "FAO_CP_23012",
    "FAO_CP_23013",
    "FAO_CP_23014"
]


for col in target_columns:

    df[f"{col}_lag1"] = df[col].shift(1)

    df[f"{col}_lag3"] = df[col].shift(3)

    df[f"{col}_lag6"] = df[col].shift(6)

    df[f"{col}_lag12"] = df[col].shift(12)


# ==================================================
# Rolling Features
# ==================================================

print("Creating rolling features...")


for col in target_columns:

    df[f"{col}_rolling3"] = (
        df[col]
        .rolling(3)
        .mean()
    )

    df[f"{col}_rolling6"] = (
        df[col]
        .rolling(6)
        .mean()
    )


# ==================================================
# Economic Change Features
# ==================================================

print("Creating economic changes...")


df["Brent_change"] = (
    df["Brent_USD_per_barrel"]
    .pct_change()
)


df["Policy_change"] = (
    df["policy_rate"]
    .diff()
)


# ==================================================
# Missing values check
# ==================================================

print("\nMissing values before saving:")

print(
    df.isnull()
    .sum()
    .sort_values(
        ascending=False
    )
    .head(10)
)


# Save

os.makedirs(
    "data/processed",
    exist_ok=True
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nFinal shape:")
print(df.shape)


print("\nSaved:")
print(OUTPUT_FILE)


print("=" * 80)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 80)