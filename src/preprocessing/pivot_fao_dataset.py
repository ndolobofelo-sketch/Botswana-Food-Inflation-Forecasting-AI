import os
import pandas as pd


def load_fao_data(file_path):
    """Load the FAO Botswana prices dataset."""
    print("=" * 80)
    print("LOADING FAO DATASET")
    print("=" * 80)

    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])

    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    return df


def pivot_fao_data(df):
    """Convert the FAO dataset from long to wide format."""

    print("\nConverting long format to wide format...")

    wide = (
        df.pivot(
            index="Date",
            columns="Item Code",
            values="Value"
        )
        .reset_index()
    )

    # Rename columns
    wide.columns = [
        "Date" if col == "Date" else f"FAO_CP_{int(col)}"
        for col in wide.columns
    ]

    wide = wide.sort_values("Date")

    return wide


def save_dataset(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print("\nDataset saved successfully.")
    print(output_path)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)


def main():
    input_file = "data/raw/04_fao_botswana_prices.csv"
    output_file = "data/processed/fao_wide.csv"

    fao = load_fao_data(input_file)

    fao_wide = pivot_fao_data(fao)

    save_dataset(fao_wide, output_file)


if __name__ == "__main__":
    main()