import pandas as pd
import os


print("=" * 80)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 80)



# Create comparison table

results = pd.DataFrame({

    "Model": [
        "LSTM",
        "XGBoost",
        "Random Forest"
    ],

    "MAE": [
        0.04139380175124825,
        15.743924745518276,
        16.162013111234383
    ],

    "RMSE": [
        0.050494611079255056,
        20.216007395536636,
        20.54789946048432
    ],

    "R2": [
        0.7898508291005255,
        -1.5420116023448127,
        -1.6261626297418776
    ]

})



print("\nMODEL RESULTS")
print(results)



# Rank models

results = results.sort_values(
    by="R2",
    ascending=False
)



results["Rank"] = range(
    1,
    len(results)+1
)



print("\nRANKING")
print(results)



# Save report

os.makedirs(
    "reports",
    exist_ok=True
)



results.to_csv(
    "reports/model_comparison.csv",
    index=False
)



print("\nSaved:")
print("reports/model_comparison.csv")



print("\nBEST MODEL:")

print(
    results.iloc[0]["Model"]
)

print("="*80)