import pandas as pd
import os

# Folder containing the raw datasets
DATA_FOLDER = "data/raw"

# List all CSV files
files = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")])

print("=" * 80)
print("RAW DATASET INSPECTION")
print("=" * 80)

for file in files:
    print(f"\n📂 FILE: {file}")

    path = os.path.join(DATA_FOLDER, file)

    df = pd.read_csv(path)

    print("-" * 80)
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("=" * 80)