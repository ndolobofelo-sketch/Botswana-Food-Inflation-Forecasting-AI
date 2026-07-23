import pandas as pd
import numpy as np
import os

import statsmodels.api as sm


print("="*80)
print("HUMAN CAPITAL LINKAGE STATISTICAL ANALYSIS")
print("="*80)



# ==================================================
# LOAD HUMAN CAPITAL DATA
# ==================================================

DATA_PATH = "data/raw/05_human_capital_project.csv"


df = pd.read_csv(DATA_PATH)


print("\nDataset loaded")

print(df.head())

print("\nShape:")
print(df.shape)



# ==================================================
# FILTER BOTSWANA DATA
# ==================================================

bwa = df[
    df["REF_AREA"] == "BWA"
].copy()


print("\nBotswana dataset shape:")

print(bwa.shape)



# ==================================================
# PREPARE DATE
# ==================================================

bwa["Date"] = pd.to_datetime(
    bwa["Date"]
)


bwa = bwa.sort_values(
    "Date"
)



# ==================================================
# CREATE WIDE FORMAT
# ==================================================

hcp = bwa.pivot_table(

    index="Date",

    columns="INDICATOR",

    values="Value"

)


hcp = hcp.reset_index()



print("\nIndicators available:")

print(
    hcp.columns.tolist()
)



# ==================================================
# LOAD FOOD DATA
# ==================================================

food = pd.read_csv(

    "data/processed/final_feature_dataset.csv"

)


food["Date"] = pd.to_datetime(
    food["Date"]
)



# ==================================================
# MERGE FOOD + HCP
# ==================================================

merged = pd.merge(

    food[[
        "Date",
        "Food_Price_Index",
        "Food_Inflation"
    ]],

    hcp,

    on="Date",

    how="inner"

)



print("\nMerged dataset:")

print(
    merged.shape
)



# ==================================================
# CLEAN DATA
# ==================================================

merged = merged.replace(

    [np.inf, -np.inf],

    np.nan

)


merged = merged.dropna()



print("\nFinal analysis dataset:")

print(
    merged.shape
)



# ==================================================
# SELECT HUMAN CAPITAL INDICATORS
# ==================================================

indicators = [

    col for col in merged.columns

    if col not in [
        "Date",
        "Food_Price_Index",
        "Food_Inflation"
    ]

]


print("\nHuman capital indicators:")

print(indicators)



# ==================================================
# REGRESSION ANALYSIS
# ==================================================

results = []


for indicator in indicators:


    X = merged[
        [indicator]
    ]


    y = merged[
        "Food_Price_Index"
    ]


    X = sm.add_constant(
        X
    )


    model = sm.OLS(
        y,
        X
    ).fit()



    results.append({

        "Indicator": indicator,

        "Coefficient": model.params[indicator],

        "P_Value": model.pvalues[indicator],

        "R_squared": model.rsquared

    })



results_df = pd.DataFrame(
    results
)



results_df = results_df.sort_values(

    by="P_Value"

)



# ==================================================
# SAVE RESULTS
# ==================================================

os.makedirs(

    "reports",

    exist_ok=True

)



results_df.to_csv(

    "reports/hcp_regression_results.csv",

    index=False

)



print("\nSTATISTICAL RESULTS")

print(results_df)



print("\nSaved:")

print(
"reports/hcp_regression_results.csv"
)



print("="*80)

print("HCP ANALYSIS COMPLETE")

print("="*80)