import pandas as pd
import os


def load_data(path):
    """
    Load dataset
    """
    print("Loading dataset...")
    df = pd.read_csv(path)

    print("Dataset loaded successfully")
    print("Shape:", df.shape)

    return df


def inspect_data(df):
    """
    Basic data inspection
    """

    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Data Information ---")
    print(df.info())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())


def clean_data(df):
    """
    Cleaning operations
    """

    print("\nCleaning data...")

    # Remove duplicates
    df = df.drop_duplicates()

    # Sort by date if available
    date_columns = [
        col for col in df.columns
        if "date" in col.lower()
    ]

    if date_columns:
        date_col = date_columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

    print("Cleaning completed")

    return df


def save_processed_data(df, output_path):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("\nSaved processed dataset:")
    print(output_path)


if __name__ == "__main__":

    input_file = (
        "data/processed/"
        "hcp_bwa_wide.csv"
    )

    output_file = (
        "data/processed/"
        "ml_ready_dataset.csv"
    )


    data = load_data(input_file)

    inspect_data(data)

    cleaned_data = clean_data(data)

    save_processed_data(
        cleaned_data,
        output_file
    )