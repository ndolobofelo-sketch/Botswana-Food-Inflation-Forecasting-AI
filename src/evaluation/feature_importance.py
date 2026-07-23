"""
Feature Importance Analysis

Purpose:
--------
Identify which variables contribute most
to food price predictions.

Model:
------
Gradient Boosting Regressor
"""

import pandas as pd
import os

from sklearn.ensemble import GradientBoostingRegressor


# ==================================================
# START
# ==================================================

print("=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)


# ==================================================
# Load Dataset
# ==================================================

INPUT_FILE = (
    "data/processed/"
    "food_inflation_features.csv"
)


df = pd.read_csv(INPUT_FILE)


print("\nDataset loaded:")
print(df.shape)


# Convert Date

df["Date"] = pd.to_datetime(
    df["Date"]
)


# Sort by time

df = df.sort_values(
    "Date"
).reset_index(drop=True)


# ==================================================
# Define Target
# ==================================================

TARGET = "FAO_CP_23012"


print("\nTarget:")
print(TARGET)


# ==================================================
# Prepare Features
# ==================================================

print("\nPreparing features...")


X = df.drop(
    columns=[
        "Date",
        "FAO_CP_23012",
        "FAO_CP_23013",
        "FAO_CP_23014"
    ]
)


y = df[TARGET]


# Handle missing values

X = X.fillna(
    X.median()
)


print("Feature shape:")
print(X.shape)


# ==================================================
# Train Gradient Boosting Model
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
# Calculate Feature Importance
# ==================================================

print("\nCalculating feature importance...")


importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }
)


importance = importance.sort_values(
    by="Importance",
    ascending=False
)


# ==================================================
# Display Results
# ==================================================

print("\nTop 15 Important Features:")

print(
    importance.head(15)
)


# ==================================================
# Save Results
# ==================================================

os.makedirs(
    "reports",
    exist_ok=True
)


OUTPUT_FILE = (
    "reports/"
    "feature_importance.csv"
)


importance.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved successfully:")
print(OUTPUT_FILE)


print("=" * 80)
print("FEATURE IMPORTANCE COMPLETE")
print("=" * 80)