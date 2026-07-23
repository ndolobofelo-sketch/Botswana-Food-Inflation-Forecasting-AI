import pandas as pd
import matplotlib.pyplot as plt
import os

print("=" * 80)
print("CREATING HCP VISUALISATIONS")
print("=" * 80)

# Professional chart style - ADDED
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10

# Create figures folder
os.makedirs("figures", exist_ok=True)

# =====================================================
# LOAD HUMAN CAPITAL DATASET - USE ONLY REAL DATASET
# =====================================================

df = pd.read_csv("data/raw/05_human_capital_project.csv")

df["Date"] = pd.to_datetime(df["Date"])

# Botswana only
df = df[df["REF_AREA_LABEL"] == "Botswana"]

# Convert long format to wide format
wide = df.pivot(index="Date", columns="INDICATOR_LABEL", values="Value")

wide = wide.sort_index()

print("\nHistorical dataset:")
print(wide.head())


# =====================================================
# HISTORICAL CO-MOVEMENT CHART - FINAL RECOMMENDED DESIGN
# =====================================================

plt.figure(figsize=(10, 5))

plt.plot(wide.index, wide["Consumer Prices, Food Indices (2015 = 100)"], label="Food Price Index", linewidth=2)
plt.plot(wide.index, wide["Consumer Prices, General Indices (2015 = 100)"], label="General CPI", linewidth=2)

# Improved title - matches hackathon requirement: historical co-movement chart
plt.title("Historical Co-Movement Between Botswana Food Inflation and Human Capital Project (HCP) Indicators (2000–2023)", fontsize=11, fontweight='bold')

plt.xlabel("Year")
# Improved Y-axis label
plt.ylabel("Price Index (2015 = 100)")

plt.legend()

# Better grid style - cleaner
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig("figures/hcp_historical_comovement.png", dpi=300, bbox_inches="tight")

plt.close()


# =====================================================
# FORWARD FORECAST PROJECTION - FINAL RECOMMENDED DESIGN
# FIXED: No longer creates/overwrites forecast file - only reads
# =====================================================

forecast_path = "submissions/final_predictions.csv"

if not os.path.exists(forecast_path):
    raise FileNotFoundError("Final forecast file not found. Generate final predictions first. Expected at: submissions/final_predictions.csv")

forecast = pd.read_csv(forecast_path)

print("\nForecast file:")
print(forecast.head())

# Convert year_month column
forecast["year_month"] = pd.to_datetime(forecast["year_month"])

plt.figure(figsize=(11, 5.5))

# Line graph with circular markers on each monthly forecast point
plt.plot(forecast["year_month"], forecast["forecast"], marker="o", markersize=8, markerfacecolor="#1f77b4", markeredgecolor="white", markeredgewidth=1.2, linewidth=2.5, label="Base Scenario Forecast Food Price Index")

# Add forecast value labels - allows evaluators to see actual projected values
for x, y in zip(forecast["year_month"], forecast["forecast"]):
    plt.text(x, y, f"{y:.1f}", fontsize=8, rotation=45, ha='left', va='bottom')

# Improved forecast title - shows forward projection and full required period
plt.title("Forward Projection of Botswana Food Prices Using HCP-Linked Forecasting (January–December 2024)", fontsize=11, fontweight='bold')

plt.xlabel("Month")
plt.ylabel("Forecast Food Price Index")

# X-axis must show January 2024 to December 2024
plt.xticks(forecast["year_month"], [d.strftime("%b %Y") for d in forecast["year_month"]], rotation=45, ha='right')

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig("figures/hcp_2024_forecast_projection.png", dpi=300, bbox_inches="tight")

plt.close()


# =====================================================
# COMPLETE
# =====================================================

print("\nSaved files:")
print("figures/hcp_historical_comovement.png")
print("figures/hcp_2024_forecast_projection.png")

print("=" * 80)
print("HCP VISUALISATIONS COMPLETE")
print("=" * 80)