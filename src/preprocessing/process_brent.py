"""
Brent Crude Monthly Dataset Preprocessing

Purpose:
--------
Clean and prepare Brent crude oil prices
for food inflation forecasting.

Input:
------
data/raw/02_brent_crude_monthly.csv

Output:
-------
data/processed/brent_monthly.csv
"""

import pandas as pd
import os


INPUT_FILE = "data/raw/02_brent_crude_monthly.csv"
OUTPUT_FILE = "data/processed/brent_monthly.csv"


print("=" * 80)
print("PROCESSING BRENT CRUDE DATASET")
print("=" * 80)


# ------------------------------------------------------------
# Load Data
# ------------------------------------------------------------

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(INPUT_FILE)


brent = pd.read_csv(INPUT_FILE)


print("\nRows loaded:", len(brent))

print("\nColumns:")
print(brent.columns.tolist())


# ------------------------------------------------------------
# Inspect Data
# ------------------------------------------------------------

print("\nFirst rows:")
print(brent.head())


# ------------------------------------------------------------
# Standardize Date
# ------------------------------------------------------------

print("\nConverting Date column...")


brent["Date"] = pd.to_datetime(brent["Date"])


# Convert to monthly period start

brent["Date"] = (
    brent["Date"]
    .dt.to_period("M")
    .dt.to_timestamp()
)


# ------------------------------------------------------------
# Duplicate Check
# ------------------------------------------------------------

duplicates = brent.duplicated().sum()

print(
    "\nDuplicate rows:",
    duplicates
)


if duplicates > 0:
    brent = brent.drop_duplicates()


# ------------------------------------------------------------
# Missing Values
# ------------------------------------------------------------

print("\nMissing values:")
print(brent.isnull().sum())


# ------------------------------------------------------------
# Sort
# ------------------------------------------------------------

brent = brent.sort_values("Date")


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

os.makedirs(
    "data/processed",
    exist_ok=True
)


brent.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nFinal shape:")
print(brent.shape)


print("\nSaved successfully:")
print(OUTPUT_FILE)


print("=" * 80)
print("BRENT PROCESSING COMPLETE")
print("=" * 80)