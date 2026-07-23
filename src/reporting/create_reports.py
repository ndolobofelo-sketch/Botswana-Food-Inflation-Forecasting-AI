"""
Professional Hackathon Report Generator

Creates:
1. Feature Engineering Report
2. Model Comparison Report
3. Feature Importance Report
"""

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib.styles import getSampleStyleSheet

import pandas as pd
import os


# --------------------------------------------------
# Setup
# --------------------------------------------------

os.makedirs(
    "reports",
    exist_ok=True
)


styles = getSampleStyleSheet()


def create_pdf(filename, title, sections):

    path = f"reports/{filename}"

    doc = SimpleDocTemplate(path)

    content = []

    content.append(
        Paragraph(
            title,
            styles["Title"]
        )
    )

    content.append(
        Spacer(1,20)
    )


    for heading, text in sections:

        content.append(
            Paragraph(
                heading,
                styles["Heading2"]
            )
        )

        content.append(
            Paragraph(
                text,
                styles["BodyText"]
            )
        )

        content.append(
            Spacer(1,15)
        )


    doc.build(content)


    print("Created:")
    print(path)



# --------------------------------------------------
# 1. Feature Engineering Report
# --------------------------------------------------

feature_sections = [

(
"1. Dataset Integration",

"""
Five datasets were integrated into a unified monthly food inflation
forecasting dataset.

Datasets included:

- Baltic Dry Index shipping indicators
- Brent crude oil prices
- Botswana policy rate
- FAO food price indicators
- Human Capital Project indicators

All datasets were aligned using monthly timestamps.
"""
),


(
"2. Data Transformation",

"""
Raw datasets were cleaned and converted into machine learning
ready format.

Long format datasets were transformed into wide format using
country-indicator feature creation.
"""
),


(
"3. Feature Engineering",

"""
The following features were created:

- Monthly and quarterly time features
- Lag features (1, 3, 6 and 12 months)
- Rolling averages
- Economic change indicators
- Baltic Dry Index volatility features

These features allow the model to capture trends,
seasonality and delayed economic effects.
"""
),


(
"4. Final Dataset",

"""
The final machine learning dataset contains 288 monthly
observations from January 2000 to December 2023.

The target variable is FAO_CP_23012 food price index.
"""
)

]


create_pdf(
    "Feature_Engineering_Report.pdf",
    "Food Inflation Forecasting - Feature Engineering Report",
    feature_sections
)



# --------------------------------------------------
# 2. Model Comparison Report
# --------------------------------------------------

results = pd.read_csv(
    "reports/forecasting_model_results.csv"
)


model_sections = [

(
"1. Machine Learning Models",

"""
Three forecasting approaches were evaluated:

- Linear Regression baseline
- Random Forest Regressor
- Gradient Boosting Regressor

Models were trained using historical monthly observations.
"""
),

(
"2. Evaluation Metrics",

"""
Performance was measured using:

MAE:
Average prediction error.

RMSE:
Penalizes larger prediction errors.

MAPE:
Measures percentage prediction error.
"""
),


(
"3. Model Selection",

f"""
The final selected model was Gradient Boosting.

Evaluation results:

{results.to_string(index=False)}
"""
)

]


create_pdf(
    "Model_Comparison_Report.pdf",
    "Food Inflation Forecasting - Model Comparison Report",
    model_sections
)



# --------------------------------------------------
# 3. Feature Importance Report
# --------------------------------------------------

importance = pd.read_csv(
    "reports/feature_importance.csv"
)


top_features = importance.head(10).to_string(
    index=False
)


importance_sections = [

(
"1. Explainability Approach",

"""
Gradient Boosting feature importance was used
to identify the variables contributing most to
food price predictions.
"""
),


(
"2. Top Predictive Features",

top_features
),


(
"3. Interpretation",

"""
The analysis shows that regional food price indicators,
especially neighbouring country indicators,
provide strong predictive information.

Historical food price behaviour through lag and rolling
features also contributes significantly.
"""
)

]


create_pdf(
    "Feature_Importance_Report.pdf",
    "Food Inflation Forecasting - Feature Importance Report",
    importance_sections
)



print("="*80)
print("ALL REPORTS CREATED SUCCESSFULLY")
print("="*80)