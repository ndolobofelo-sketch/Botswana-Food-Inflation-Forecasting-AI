"""
Botswana Policy Rate Dataset Preprocessing

Purpose:
--------
Prepare Botswana interest rate data
for food inflation forecasting.

Input:
------
data/raw/03_botswana_policy_rate.csv

Output:
-------
data/processed/policy_rate_monthly.csv
"""

import pandas as pd
import os


INPUT_FILE = "data/raw/03_botswana_policy_rate.csv"
OUTPUT_FILE = "data/processed/policy_rate_monthly.csv"


print("=" * 80)
print("PROCESSING BOTSWANA POLICY RATE DATASET")
print("=" * 80)


# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(INPUT_FILE)


policy = pd.read_csv(INPUT_FILE)


print("\nRows loaded:", len(policy))

print("\nColumns:")
print(policy.columns.tolist())


print("\nFirst rows:")
print(policy.head())


# ------------------------------------------------------------
# Convert Date
# ------------------------------------------------------------

print("\nConverting Date column...")


policy["Date"] = pd.to_datetime(policy["Date"])


policy["Date"] = (
    policy["Date"]
    .dt.to_period("M")
    .dt.to_timestamp()
)


# ------------------------------------------------------------
# Duplicate Check
# ------------------------------------------------------------

duplicates = policy.duplicated().sum()

print(
    "\nDuplicate rows:",
    duplicates
)


if duplicates > 0:
    policy = policy.drop_duplicates()


# ------------------------------------------------------------
# Missing Values
# ------------------------------------------------------------

print("\nMissing values:")
print(policy.isnull().sum())


# ------------------------------------------------------------
# Sort
# ------------------------------------------------------------

policy = policy.sort_values("Date")


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

os.makedirs(
    "data/processed",
    exist_ok=True
)


policy.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nFinal shape:")
print(policy.shape)


print("\nSaved successfully:")
print(OUTPUT_FILE)


print("=" * 80)
print("POLICY RATE PROCESSING COMPLETE")
print("=" * 80)