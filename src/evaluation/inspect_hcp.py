import pandas as pd


print("="*80)
print("INSPECTING HUMAN CAPITAL DATASET")
print("="*80)


df = pd.read_csv(
    "data/raw/05_human_capital_project.csv"
)


print("\nColumns:")
print(df.columns.tolist())


print("\nNumber of indicators:")
print(df["INDICATOR"].nunique())


print("\nFirst indicators:")
print(
    df["INDICATOR_LABEL"]
    .drop_duplicates()
    .head(50)
)


print("\nAll countries:")
print(
    df["REF_AREA_LABEL"]
    .unique()
)


print("="*80)