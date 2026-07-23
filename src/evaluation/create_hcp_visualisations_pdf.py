from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

import os


print("="*70)
print("CREATING HCP VISUALISATIONS PDF")
print("="*70)


os.makedirs("reports", exist_ok=True)

output = "reports/HCP_Visualisations.pdf"


doc = SimpleDocTemplate(
    output,
    pagesize=A4
)


styles = getSampleStyleSheet()

story = []


# Title

story.append(
    Paragraph(
        "HCP Visualisations - Botswana Food Inflation Forecasting Challenge",
        styles["Title"]
    )
)

story.append(
    Spacer(1,20)
)


# Historical chart

story.append(
    Paragraph(
        "1. Historical Co-Movement Between Food Inflation and Human Capital Indicators",
        styles["Heading2"]
    )
)

story.append(
    Spacer(1,12)
)


story.append(
    Image(
        "figures/hcp_historical_comovement.png",
        width=420,
        height=280
    )
)


story.append(
    Spacer(1,20)
)


story.append(
    Paragraph(
        """
        This chart presents the historical relationship between food inflation
        and Human Capital Project indicators over time. It highlights how
        economic and human development factors have moved together with food
        price changes in Botswana.
        """,
        styles["BodyText"]
    )
)


story.append(
    PageBreak()
)


# Forecast chart

story.append(
    Paragraph(
        "2. Forward Projection of Food Inflation Using HCP-Linked Forecasting",
        styles["Heading2"]
    )
)


story.append(
    Spacer(1,12)
)


story.append(
    Image(
        "figures/hcp_2024_forecast_projection.png",
        width=420,
        height=280
    )
)


story.append(
    Spacer(1,20)
)


story.append(
    Paragraph(
        """
        This forward projection chart extends the historical relationship into
        the forecast period. The projection provides evidence of how Human
        Capital indicators can support future food inflation monitoring and
        decision-making.
        """,
        styles["BodyText"]
    )
)


doc.build(story)


print("Saved:")
print(output)

print("="*70)
print("HCP VISUALISATIONS COMPLETE")
print("="*70)