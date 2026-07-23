"""
Master Dataset Creation Pipeline

Purpose:
--------
Merge all processed datasets into one
machine learning forecasting dataset.

Datasets:
---------
1. Baltic Dry Index
2. Brent Crude Oil
3. Botswana Policy Rate
4. FAO Botswana Prices
5. Human Capital Project

Output:
-------
data/processed/master_food_inflation_dataset.csv
"""

import pandas as pd
import os


print("=" * 90)
print("CREATING MASTER FOOD INFLATION DATASET")
print("=" * 90)


# ============================================================
# File Paths
# ============================================================

DATA_PATH = "data/processed"


FILES = {
    "bdi": "bdi_monthly_features.csv",
    "brent": "brent_monthly.csv",
    "policy": "policy_rate_monthly.csv",
    "fao": "fao_wide.csv",
    "hcp": "hcp_wide.csv"
}


# ============================================================
# Load datasets
# ============================================================

datasets = {}


for name, file in FILES.items():

    path = os.path.join(DATA_PATH, file)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])

    datasets[name] = df

    print(
        f"{name.upper()} loaded:",
        df.shape
    )


# ============================================================
# Check Date Ranges
# ============================================================

print("\nDATE RANGES")

for name, df in datasets.items():

    print(
        name,
        ":",
        df["Date"].min().date(),
        "to",
        df["Date"].max().date()
    )


# ============================================================
# Merge datasets
# ============================================================

print("\nMerging datasets...")


master = datasets["fao"]


merge_order = [
    "hcp",
    "brent",
    "policy",
    "bdi"
]


for name in merge_order:

    master = master.merge(
        datasets[name],
        on="Date",
        how="left"
    )

    print(
        f"Merged {name}:",
        master.shape
    )


# ============================================================
# Sort
# ============================================================

master = master.sort_values(
    "Date"
).reset_index(drop=True)


# ============================================================
# Data Quality Report
# ============================================================

print("\nFINAL DATASET CHECK")

print(
    "Shape:",
    master.shape
)


print(
    "\nMissing values:"
)

missing = (
    master.isnull()
    .sum()
    .sort_values(
        ascending=False
    )
)


print(missing.head(15))


# ============================================================
# Save
# ============================================================

OUTPUT = (
    "data/processed/"
    "master_food_inflation_dataset.csv"
)


master.to_csv(
    OUTPUT,
    index=False
)


print("\nSaved successfully:")
print(OUTPUT)


print("=" * 90)
print("MASTER DATASET CREATION COMPLETE")
print("=" * 90)