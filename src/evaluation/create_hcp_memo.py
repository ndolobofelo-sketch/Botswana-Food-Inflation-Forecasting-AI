from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle
)

from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
import os


print("="*70)
print("CREATING HCP LINKAGE MEMO")
print("="*70)


os.makedirs(
    "reports",
    exist_ok=True
)


output = "reports/HCP_Linkage_Memo.pdf"


doc = SimpleDocTemplate(output)


styles = getSampleStyleSheet()

story = []


# Title

story.append(
    Paragraph(
        "Human Capital Project (HCP) Linkage Memo: Botswana Food Inflation Analysis",
        styles["Title"]
    )
)

story.append(Spacer(1,20))


# Introduction

text = """
This memo evaluates the relationship between Botswana food prices and
macroeconomic indicators using Human Capital Project (HCP) data and
food price datasets.

The objective is to identify factors associated with food inflation and
provide evidence-based forecasting insights.
"""

story.append(
    Paragraph(text, styles["BodyText"])
)

story.append(Spacer(1,15))


# Regression Results

story.append(
    Paragraph(
        "Regression Analysis Results",
        styles["Heading2"]
    )
)


df = pd.read_csv(
    "reports/hcp_regression_results.csv"
)


table_data = [
    [
        "Indicator",
        "Coefficient",
        "P-value",
        "R-squared"
    ]
]


for _, row in df.iterrows():

    table_data.append(
        [
            row["Indicator"],
            round(row["Coefficient"],4),
            "{:.4e}".format(row["P_Value"]),
            round(row["R_squared"],4)
        ]
    )


table = Table(table_data)


table.setStyle(
    TableStyle(
        [
            ("GRID",(0,0),(-1,-1),0.5,None),
            ("VALIGN",(0,0),(-1,-1),"TOP")
        ]
    )
)


story.append(table)

story.append(Spacer(1,20))


# Interpretation

story.append(
    Paragraph(
"""
Key Findings:

1. FAO_CP_23013 shows a perfect positive relationship with food prices
(R² = 1.00), indicating strong alignment.

2. FAO_CP_23012 has a strong positive relationship with food prices
(coefficient = 0.992, p-value < 0.001).

3. FAO_CP_23014 shows a statistically significant negative relationship,
although the explanatory power is limited.

These findings demonstrate that external indicators are associated with
Botswana food price movements.
""",
styles["BodyText"]
))


story.append(Spacer(1,20))


# Add figures

for fig in [
    "figures/hcp_historical_comovement.png",
    "figures/hcp_2024_forecast_projection.png"
]:

    if os.path.exists(fig):

        story.append(
            Image(
                fig,
                width=400,
                height=220
            )
        )

        story.append(Spacer(1,15))


# Forecast section

story.append(
    Paragraph(
"""
Forward Projection:

The forecasting model provides monthly food price projections for 2024.
The projected values indicate continued monitoring is required because
food prices remain sensitive to economic and external market conditions.

The combination of machine learning forecasts and indicator linkage
provides an evidence-based framework for food inflation monitoring.
""",
styles["BodyText"]
))


doc.build(story)


print()
print("SUCCESS")
print(output)