import pandas as pd
import matplotlib.pyplot as plt
import os

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4


print("="*80)
print("CREATING FEATURE IMPORTANCE REPORT")
print("="*80)


# Create folders

os.makedirs(
    "figures",
    exist_ok=True
)

os.makedirs(
    "reports/pdf",
    exist_ok=True
)



# =====================================================
# LOAD FEATURE IMPORTANCE
# =====================================================

importance_path = (
    "reports/final_xgboost_feature_importance.csv"
)


importance = pd.read_csv(
    importance_path
)


importance = importance.sort_values(
    "Importance",
    ascending=False
)


print("\nTop features:")
print(
    importance.head(10)
)



# =====================================================
# CREATE VISUALISATION
# =====================================================

top_features = importance.head(10)


plt.figure(
    figsize=(10,6)
)


plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)


plt.xlabel(
    "Feature Importance"
)


plt.ylabel(
    "Economic Feature"
)


plt.title(
    "Top 10 Factors Influencing Botswana Food Price Forecasts"
)


plt.tight_layout()


chart_path = (
    "figures/feature_importance_top10.png"
)


plt.savefig(
    chart_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()



# =====================================================
# CREATE PDF REPORT
# =====================================================

pdf_path = (
    "reports/pdf/Feature_Importance_Interpretability_Report.pdf"
)


doc = SimpleDocTemplate(
    pdf_path,
    pagesize=A4
)


styles = getSampleStyleSheet()


story = []


def add_text(text):

    story.append(
        Paragraph(
            text,
            styles["BodyText"]
        )
    )

    story.append(
        Spacer(1,12)
    )



story.append(
    Paragraph(
        "Feature Importance and Interpretability Report - Botswana Food Inflation Forecasting",
        styles["Title"]
    )
)


story.append(
    Spacer(1,20)
)



add_text("""
This report explains the main factors used by the XGBoost forecasting model
to predict Botswana food prices. Feature importance provides insight into
which economic variables contributed most to the model predictions.
""")


add_text("""
The model used multiple economic drivers:

- Historical food price behaviour
- Baltic Dry Index shipping conditions
- Brent crude oil prices
- Botswana policy rate
- Lagged economic indicators

These variables represent supply chain costs, energy prices, monetary
conditions, and previous food price movements.
""")


story.append(
    Image(
        chart_path,
        width=400,
        height=240
    )
)


story.append(
    Spacer(1,20)
)



add_text("""
Economic Interpretation:

Food price lag variables capture persistence because food inflation usually
does not immediately return to previous levels after a shock.

Shipping variables from the Baltic Dry Index represent international
transportation pressure. Higher shipping costs can increase imported food
costs.

Brent crude oil variables represent energy costs affecting transport,
production, and supply chains.

Policy rate variables represent monetary conditions that may influence
consumer prices through demand and financial conditions.
""")


add_text("""
Interpretation Limitation:

Feature importance indicates predictive contribution, not direct causation.
Economic relationships require additional statistical testing before making
causal claims.
""")


doc.build(
    story
)



print("\nSaved files:")

print(
    chart_path
)

print(
    pdf_path
)


print("="*80)
print("FEATURE IMPORTANCE REPORT COMPLETE")
print("="*80)