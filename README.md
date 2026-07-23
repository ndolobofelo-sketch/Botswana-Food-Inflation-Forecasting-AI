\# 🇧🇼 Botswana Food Inflation Forecasting Challenge



\## AI-Driven Food Price Inflation Forecasting System Using Economic, Commodity, and Human Capital Indicators



\### VenturePulse: Food Price Inflation Forecasting Challenge  

\### Deep Learning IndabaX Botswana 2026 Hackathon



\---



\# 1. Project Overview



Food price inflation is one of the most important economic challenges affecting households, businesses, and governments. Accurate forecasting of food inflation enables better policy decisions, improved supply chain planning, and stronger food security strategies.



This project develops an Artificial Intelligence-powered forecasting system to predict Botswana food inflation by integrating multiple economic and global indicators, including:



\- Food price indices

\- Human Capital Project indicators

\- Baltic Dry Index shipping activity

\- Brent crude oil prices

\- Botswana policy interest rates

\- Historical inflation trends



The objective is to build a robust forecasting framework capable of identifying inflation patterns, understanding economic drivers, and generating accurate future food inflation predictions.



\---



\# 2. Motivation and National Importance 🇧🇼



Botswana, like many economies, is exposed to external factors influencing food prices, including:



\- International commodity price changes

\- Global transportation costs

\- Energy price fluctuations

\- Monetary policy decisions

\- Supply chain disruptions

\- Climate-related pressures



Food inflation directly affects:



\- Household purchasing power

\- Food affordability

\- Economic stability

\- Government planning

\- Social welfare programs



This project provides an evidence-based AI forecasting approach to support proactive decision-making and improve resilience against future food price shocks.



\---



\# 3. Global Relevance 🌍



Food inflation is a global challenge affecting both developing and developed economies.



The developed forecasting framework demonstrates how Artificial Intelligence can combine:



\- Local economic information

\- International market signals

\- Alternative indicators

\- Time-series modelling



to improve understanding and prediction of complex economic systems.



The methodology can be adapted to other countries facing similar food security and inflation challenges.



\---



\# 4. Project Objectives



The main objectives are:



1\. Develop an accurate food inflation forecasting model for Botswana.



2\. Integrate multiple datasets from economic, commodity, and human development domains.



3\. Engineer meaningful time-series features while preventing data leakage.



4\. Compare classical machine learning and deep learning approaches.



5\. Identify the strongest economic drivers influencing food inflation.



6\. Generate reliable future predictions for decision support.



\---



\# 5. Data Integration Strategy



The project combines five major data sources:



\## Food Price Data



Provides historical food inflation behaviour and price movements.



\## Human Capital Project Indicators



Captures socio-economic and development-related factors.



\## Baltic Dry Index (BDI)



Represents global shipping activity and international supply chain conditions.



Daily BDI observations were aggregated into monthly indicators using statistically appropriate transformations, including:



\- Monthly average

\- Monthly volatility

\- Monthly changes



This converts high-frequency shipping information into usable economic signals.



\## Brent Crude Oil Prices



Captures energy cost pressures affecting:



\- Transportation

\- Agriculture inputs

\- Food production costs



\## Botswana Policy Rate



Represents monetary policy conditions affecting inflation dynamics.



\---



\# 6. Feature Engineering



A comprehensive feature engineering pipeline was developed.



Features include:



\- Historical inflation values

\- Lag variables

\- Rolling statistics

\- Growth rates

\- Commodity indicators

\- Shipping indicators

\- Monetary policy variables

\- Economic interaction features



All feature engineering was performed chronologically to avoid future information leakage.



\---



\# 7. Machine Learning Approach



Two forecasting approaches were developed and compared.



\---



\## Model 1: Classical Machine Learning Baseline



\### Random Forest Regression



Advantages:



\- Captures nonlinear relationships

\- Provides feature importance interpretation

\- Robust against complex economic interactions



\### XGBoost Regression



Advantages:



\- High predictive performance

\- Handles nonlinear patterns effectively

\- Strong performance on structured economic datasets



\---



\## Model 2: Deep Learning Approach



\### Long Short-Term Memory Network (LSTM)



The LSTM model was designed for sequential economic forecasting.



Advantages:



\- Learns long-term temporal dependencies

\- Captures changing inflation patterns

\- Suitable for time-series problems



\---



\# 8. Model Evaluation Strategy



Models were evaluated using time-series validation methods.



Evaluation metrics include:



\- RMSE (Root Mean Squared Error)

\- Prediction comparison

\- Residual analysis

\- Feature importance analysis



The evaluation approach respects chronological ordering to prevent data leakage.



\---



\# 9. Interpretability and Economic Analysis



Beyond prediction accuracy, this project investigates why inflation changes.



The analysis includes:



\- Feature importance ranking

\- Economic relationship analysis

\- Human Capital Project linkage analysis

\- Statistical evidence between indicators and food inflation



This improves transparency and supports responsible AI adoption.



\---



\# 10. Project Structure

Botswana\_Food\_Inflation\_Hackathon/



├── data/

│ ├── raw/

│ └── processed/



├── notebooks/

│ └── forecasting analysis notebooks



├── src/

│ ├── preprocessing/

│ ├── feature\_engineering/

│ ├── models/

│ ├── evaluation/

│ └── presentation/



├── models/

│ └── trained forecasting models



├── final\_submission/

│ ├── reports

│ ├── presentation

│ └── predictions



└── requirements.txt



\---



\# 11. Reproducibility



To reproduce the project:



\## Create environment



```bash

python -m venv venv

Botswana\_Food\_Inflation\_Hackathon/



├── data/

│ ├── raw/

│ └── processed/



├── notebooks/

│ └── forecasting analysis notebooks



├── src/

│ ├── preprocessing/

│ ├── feature\_engineering/

│ ├── models/

│ ├── evaluation/

│ └── presentation/



├── models/

│ └── trained forecasting models



├── final\_submission/

│ ├── reports

│ ├── presentation

│ └── predictions



└── requirements.txt





\---



\# 11. Reproducibility



To reproduce the project:



\## Create environment



```bash

python -m venv venv



Activate environment.



Install dependencies:



pip install -r requirements.txt



Run preprocessing:



python src/preprocessing/prepare\_data.py



Run feature engineering:



python src/feature\_engineering/build\_final\_features.py



Train models:



python src/models/train\_xgboost.py



python src/deep\_learning/train\_lstm.py



Generate predictions and reports using scripts inside:



src/evaluation/

12\. Key Deliverables



This repository includes:



Feature Engineering Report

Model Comparison Report

&#x20;Feature Importance Report

HCP Linkage Analysis

Forecast Visualisations

Final Presentation

&#x20;Best Model Predictions



13\. Technologies Used



Programming:



Python



Libraries:



Pandas

NumPy

Scikit-learn

XGBoost

TensorFlow/Keras

Matplotlib



Development:



VS Code

Git

Jupyter Notebook

14\. Responsible AI Considerations



The project follows responsible machine learning practices:



Avoiding data leakage

Chronological validation

Model comparison

Explainability analysis

Transparent reporting



Forecasts are designed as decision-support tools and should complement expert economic analysis.



15\. Conclusion



This project demonstrates how Artificial Intelligence can support economic forecasting by combining local and global indicators into a unified predictive framework.



By integrating machine learning, deep learning, economic analysis, and explainable AI, the system provides a scalable approach for understanding and forecasting food inflation challenges in Botswana and beyond.



Author

\## Author



Botswana Food Inflation Forecasting Team



Deep Learning IndabaX Botswana 2026 Hackathon



Project developed for the VenturePulse: Food Price Inflation Forecasting Challenge.



