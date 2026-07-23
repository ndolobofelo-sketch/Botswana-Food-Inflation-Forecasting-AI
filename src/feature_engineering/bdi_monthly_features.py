import os
import numpy as np
import pandas as pd


def load_bdi_data(file_path):
    """Load the daily Baltic Dry Index dataset."""
    print("=" * 80)
    print("LOADING BALTIC DRY INDEX DATA")
    print("=" * 80)

    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])

    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    return df


def create_monthly_features(df):
    """Aggregate daily BDI data into monthly features."""

    print("\nCreating monthly BDI features...")

    df["YearMonth"] = df["Date"].dt.to_period("M")

    monthly = (
        df.groupby("YearMonth")
        .agg(
            BDI_Close_Mean=("BDI_Close", "mean"),
            BDI_Close_Max=("BDI_Close", "max"),
            BDI_Close_Min=("BDI_Close", "min"),
            BDI_Close_STD=("BDI_Close", "std"),
            BDI_High_Max=("BDI_High", "max"),
            BDI_Low_Min=("BDI_Low", "min"),
            Trading_Days=("BDI_Close", "count"),
        )
        .reset_index()
    )

    monthly["BDI_Range"] = (
        monthly["BDI_Close_Max"] - monthly["BDI_Close_Min"]
    )

    monthly["BDI_CV"] = (
        monthly["BDI_Close_STD"] / monthly["BDI_Close_Mean"]
    )

    monthly["BDI_Monthly_Return"] = (
        monthly["BDI_Close_Mean"].pct_change() * 100
    )

    monthly["BDI_Rolling3"] = (
        monthly["BDI_Close_Mean"]
        .rolling(3)
        .mean()
    )

    monthly["BDI_Rolling6"] = (
        monthly["BDI_Close_Mean"]
        .rolling(6)
        .mean()
    )

    monthly["Date"] = monthly["YearMonth"].dt.to_timestamp()

    monthly = monthly.drop(columns=["YearMonth"])

    return monthly


def save_monthly_features(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print("\nMonthly BDI features saved to:")
    print(output_path)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nFinal shape:")
    print(df.shape)


def main():
    input_file = "data/raw/01_baltic_dry_index_daily.csv"

    output_file = "data/processed/bdi_monthly_features.csv"

    bdi = load_bdi_data(input_file)

    monthly = create_monthly_features(bdi)

    save_monthly_features(monthly, output_file)


if __name__ == "__main__":
    main()