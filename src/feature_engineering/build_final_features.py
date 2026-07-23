import pandas as pd
import numpy as np
import os


print("="*80)
print("FINAL FEATURE ENGINEERING PIPELINE")
print("="*80)


# =========================================================
# PATHS
# =========================================================

RAW = "data/raw"

OUTPUT = "data/processed/final_feature_dataset.csv"



# =========================================================
# 1. LOAD DATASETS
# =========================================================

print("\nLoading datasets...")


bdi = pd.read_csv(
    f"{RAW}/01_baltic_dry_index_daily.csv"
)


brent = pd.read_csv(
    f"{RAW}/02_brent_crude_monthly.csv"
)


policy = pd.read_csv(
    f"{RAW}/03_botswana_policy_rate.csv"
)


fao = pd.read_csv(
    f"{RAW}/04_fao_botswana_prices.csv"
)


hcp = pd.read_csv(
    f"{RAW}/05_human_capital_project.csv"
)



print("Datasets loaded successfully")



# =========================================================
# 2. DATE FORMATTING
# =========================================================

for df in [bdi, brent, policy, fao, hcp]:

    df["Date"] = pd.to_datetime(
        df["Date"]
    )



# =========================================================
# 3. BDI DAILY -> MONTHLY FEATURES
# =========================================================

print("\nCreating BDI monthly features...")


bdi["YearMonth"] = (
    bdi["Date"]
    .dt
    .to_period("M")
)


bdi_monthly = bdi.groupby(
    "YearMonth"
).agg(

    BDI_average=("BDI_Close","mean"),

    BDI_volatility=("BDI_Close","std"),

    BDI_max=("BDI_High","max"),

    BDI_min=("BDI_Low","min")

).reset_index()



bdi_monthly["BDI_momentum"] = (
    bdi_monthly["BDI_average"]
    .pct_change()
)



bdi_monthly["Date"] = (
    bdi_monthly["YearMonth"]
    .dt
    .to_timestamp()
)



bdi_monthly.drop(
    columns=["YearMonth"],
    inplace=True
)



# =========================================================
# 4. BRENT FEATURES
# =========================================================

print("Processing Brent oil data...")


brent["Brent_change"] = (
    brent["Brent_USD_per_barrel"]
    .pct_change()
)


brent["Brent_lag1"] = (
    brent["Brent_USD_per_barrel"]
    .shift(1)
)



# =========================================================
# 5. POLICY RATE FEATURES
# =========================================================

print("Processing policy rate...")


policy["Policy_change"] = (
    policy["policy_rate"]
    .diff()
)



policy["Policy_lag1"] = (
    policy["policy_rate"]
    .shift(1)
)



# =========================================================
# 6. FAO FOOD PRICE TARGET
# =========================================================

print("Preparing food price target...")


fao = fao[
    fao["Item Code"] == 23013
]


fao = fao[
    ["Date","Value"]
]


fao.rename(

    columns={
        "Value":"Food_Price_Index"
    },

    inplace=True

)



# Create inflation rate

fao["Food_Inflation"] = (
    fao["Food_Price_Index"]
    .pct_change()
)



# =========================================================
# 7. MERGE DATASETS
# =========================================================

print("\nMerging datasets...")


dataset = fao.merge(
    bdi_monthly,
    on="Date",
    how="left"
)



dataset = dataset.merge(
    brent,
    on="Date",
    how="left"
)



dataset = dataset.merge(
    policy,
    on="Date",
    how="left"
)



dataset = dataset.sort_values(
    "Date"
)



# =========================================================
# 8. CREATE FORECAST LAGS
# =========================================================

print("Creating lag features...")


lag_columns = [

    "Food_Price_Index",

    "Food_Inflation",

    "BDI_average",

    "Brent_USD_per_barrel",

    "policy_rate"

]



for col in lag_columns:

    for lag in [1,3,6,12]:

        dataset[
            f"{col}_lag_{lag}"
        ] = dataset[col].shift(lag)



# =========================================================
# 9. CLEAN DATA
# =========================================================

print("Cleaning missing values...")


dataset = dataset.interpolate()


dataset = dataset.bfill()


dataset = dataset.ffill()



# =========================================================
# SAVE
# =========================================================

os.makedirs(

    "data/processed",

    exist_ok=True

)



dataset.to_csv(

    OUTPUT,

    index=False

)



print("\nFINAL DATASET CREATED")

print(
    OUTPUT
)


print("\nShape:")
print(
    dataset.shape
)


print("\nColumns:")
print(
    list(dataset.columns)
)


print("="*80)
print("FEATURE ENGINEERING COMPLETE")
print("="*80)