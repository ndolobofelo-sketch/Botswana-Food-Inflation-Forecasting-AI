"""
Human Capital Project (HCP) Dataset Preprocessing

Purpose:
--------
Convert HCP dataset from long format into wide format.

Input:
------
data/raw/05_human_capital_project.csv

Output:
-------
data/processed/hcp_wide.csv

Processing steps:
-----------------
1. Load raw HCP data
2. Validate required columns
3. Check missing values
4. Detect duplicates
5. Create country-indicator features
6. Pivot into wide format
7. Save processed dataset
"""

import pandas as pd
import os


# ============================================================
# File Paths
# ============================================================

INPUT_FILE = "data/raw/05_human_capital_project.csv"
OUTPUT_FILE = "data/processed/hcp_wide.csv"


# ============================================================
# Load Dataset
# ============================================================

print("=" * 80)
print("LOADING HUMAN CAPITAL PROJECT DATASET")
print("=" * 80)


if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}"
    )


hcp = pd.read_csv(INPUT_FILE)


print(f"Rows loaded: {len(hcp)}")
print(f"Columns found: {list(hcp.columns)}")


# ============================================================
# Validate Columns
# ============================================================

print("\nChecking required columns...")


required_columns = [
    "Date",
    "REF_AREA",
    "INDICATOR",
    "Value"
]


missing_columns = [
    col for col in required_columns
    if col not in hcp.columns
]


if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


print("✓ All required columns available")


# ============================================================
# Data Quality Checks
# ============================================================

print("\nRunning data quality checks...")


# Missing values

missing_values = hcp[required_columns].isnull().sum()

print("\nMissing values:")
print(missing_values)


# Duplicate detection

duplicates = hcp.duplicated().sum()

print(
    f"\nDuplicate rows detected: {duplicates}"
)


if duplicates > 0:
    print("Removing duplicates...")
    hcp = hcp.drop_duplicates()


# ============================================================
# Create Country Indicator Feature Names
# ============================================================

print("\nCreating country-indicator features...")


hcp["Feature_Name"] = (
    hcp["REF_AREA"]
    + "_"
    + hcp["INDICATOR"]
)


print(
    "Example features:"
)

print(
    hcp["Feature_Name"]
    .head()
)


# ============================================================
# Pivot Dataset
# ============================================================

print("\nConverting long format to wide format...")


hcp_wide = hcp.pivot_table(
    index="Date",
    columns="Feature_Name",
    values="Value",
    aggfunc="mean"
)


# Reset index

hcp_wide = hcp_wide.reset_index()


# Sort columns

hcp_wide = hcp_wide.sort_index(axis=1)


# ============================================================
# Final Validation
# ============================================================

print("\nFinal dataset verification")

print(
    f"Rows: {len(hcp_wide)}"
)

print(
    f"Columns: {len(hcp_wide.columns)}"
)


print("\nFirst 5 rows:")

print(
    hcp_wide.head()
)


# ============================================================
# Save Output
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)


hcp_wide.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 80)
print("HCP PROCESSING COMPLETE")
print("=" * 80)

print(
    f"Saved successfully:\n{OUTPUT_FILE}"
)