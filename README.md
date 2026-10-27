# Explainable AI for Climate Change Prediction and Environmental Risk Analysis Using Big Data Analytics

An Explainable AI (XAI)-driven Big Data Analytics framework for environmental disaster risk prediction that integrates historical climate data, extreme-weather records, air-quality information, machine learning, and real-time environmental observations.

## Team Members

| Name | ID Number |
|---|---|
| L. Mrudani | 2420030591 |
| Sri Kruthi | 2420090003 |
| P. Hasitha | 2420090107 |

**Supervisor:** Dr. K. Swapnika

## Abstract

Climate change generates large and complex datasets involving temperature, rainfall, humidity, wind speed, atmospheric pressure, solar radiation, air quality, and extreme weather events. Traditional machine-learning approaches may produce predictions without clearly explaining the environmental factors responsible for their results.

This project presents a completed Explainable AI-driven environmental disaster risk assessment framework that combines Big Data Analytics, machine learning, historical environmental evidence, real-time weather data, and Explainable AI. Three heterogeneous environmental datasets are processed and analyzed to identify important patterns and develop a model for disaster-risk classification.

The proposed framework uses a hybrid Logistic Regression–XGBoost model with Platt Scaling for probability calibration. It further integrates historical environmental risk scoring, a weighted risk-fusion mechanism, Mahalanobis-distance-based out-of-distribution detection, and SHAP-based explainability. An interactive Streamlit dashboard provides location-based environmental risk assessment, live weather and air-quality information, analytical visualizations, model evaluation, and interpretable prediction results.

The final system demonstrates how machine-learning predictions and historical environmental evidence can be combined to support transparent, data-driven environmental monitoring and early-warning decision support.

## Problem Statement

Climate change creates complex environmental conditions that require the analysis of multiple datasets and environmental indicators. Although machine-learning models can identify patterns and predict risks, their predictions may be difficult to interpret without appropriate explainability techniques.

This project addresses the need for an integrated framework that processes environmental data, predicts disaster risks, incorporates historical evidence, analyzes live environmental conditions, and explains the factors influencing predictions.

## Project Objectives

- Integrate climate, extreme-weather, and air-quality datasets for comprehensive environmental analysis.
- Perform data cleaning, missing-value handling, feature selection, and feature engineering.
- Use Big Data Analytics and PySpark to support environmental data processing.
- Develop and evaluate machine-learning models for disaster-risk classification.
- Combine Logistic Regression and XGBoost into a hybrid prediction model.
- Apply Platt Scaling to improve the reliability of predicted probabilities.
- Incorporate historical environmental disaster evidence into risk assessment.
- Detect unusual live observations using Mahalanobis-distance-based OOD detection.
- Apply SHAP to explain model predictions and identify influential environmental factors.
- Develop an interactive Streamlit dashboard integrating live environmental information, predictions, risk assessment, and explainability.

## Dataset Description

The project integrates three cleaned environmental datasets.

| Dataset | Records | Columns | Purpose |
|---|---:|---:|---|
| Climate dataset | 2,425 | 13 | Historical climate analysis and machine-learning development |
| Extreme-weather dataset | 49 | 23 | Extreme-weather and environmental context |
| Air-quality dataset | 842,160 | 28 | Air-quality analysis, environmental context, and dashboard monitoring |

The datasets are processed to support environmental analysis, feature engineering, model development, and dashboard visualization.

Air-quality variables are retained for environmental context and monitoring but are not included in the final eight-feature machine-learning model.

## Technologies Used

- **Programming:** Python
- **Big Data Processing:** Apache Spark, PySpark
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, Logistic Regression, XGBoost, Random Forest, Support Vector Machine
- **Probability Calibration:** Platt Scaling
- **Explainable AI:** SHAP
- **Out-of-Distribution Detection:** Mahalanobis distance
- **Visualization:** Matplotlib and Python visualization libraries
- **Dashboard:** Streamlit
- **Live Data Integration:** Weather and air-quality APIs

## Methodology

The complete project follows these stages:

1. **Data acquisition and integration:** Collect and process climate, extreme-weather, and air-quality datasets.
2. **Data preprocessing:** Handle missing numerical values through median imputation, clean the datasets, select relevant features, and standardize inputs where required.
3. **Feature engineering:** Generate cyclic sine and cosine features from the day of the year to represent seasonal patterns.
4. **Exploratory Data Analysis:** Analyze environmental trends, relationships, distributions, and potential risk indicators.
5. **Model development:** Train and evaluate multiple machine-learning algorithms, including Logistic Regression, XGBoost, Random Forest, and Support Vector Machine.
6. **Hybrid model construction:** Combine Logistic Regression and XGBoost predictions at the probability level.
7. **Probability calibration:** Apply Platt Scaling to improve the reliability of the hybrid model's probabilities.
8. **Historical risk analysis:** Calculate environmental risk scores using disaster patterns in the historical dataset.
9. **Risk fusion:** Combine calibrated machine-learning probabilities with historical environmental risk evidence.
10. **OOD detection:** Identify unusual live observations using Mahalanobis distance.
11. **Explainable AI:** Use SHAP to identify the environmental features contributing to predictions.
12. **Dashboard integration:** Present environmental information, risk estimates, model performance, and explanations through an interactive Streamlit application.

## Final Machine-Learning Model

The final hybrid model combines Logistic Regression and XGBoost to estimate the probability of an environmental disaster.

The model uses eight predictors:

| Feature | Description |
|---|---|
| `ALLSKY_SFC_SW_DWN` | Surface solar/shortwave radiation |
| `T2M` | Temperature at 2 metres |
| `RH2M` | Relative humidity at 2 metres |
| `PS` | Surface pressure |
| `WS2M` | Wind speed at 2 metres |
| `PRECTOTCORR` | Corrected precipitation |
| `DOY_SIN` | Sine-transformed day of year |
| `DOY_COS` | Cosine-transformed day of year |

The target variable is `DISASTER`, formulated as a binary classification problem.

The model development and evaluation process includes preprocessing, class balancing, feature standardization where required, probability calibration, and temporal evaluation.

## Historical and Real-Time Risk Fusion

A major contribution of the project is the integration of machine-learning predictions with historical environmental evidence.

The final fusion equation is:

`P_fusion = 0.25 × P_ML,calibrated + 0.75 × R_H`

Where:

- `P_ML,calibrated` represents the calibrated probability from the hybrid machine-learning model.
- `R_H` represents the historical environmental risk score.
- `P_fusion` represents the final combined risk score.

The fusion weights are fixed at 25% for the calibrated machine-learning probability and 75% for historical environmental risk. The final fusion decision threshold is 0.43.

This approach combines learned environmental patterns with historical disaster evidence to support risk assessment.

## Explainable AI and OOD Detection

### SHAP-Based Explainability

SHAP is integrated to explain model predictions by identifying the contributions of individual environmental features.

The explainability component helps users understand which climate variables influence the predicted risk, improving the transparency and interpretability of the system.

### Out-of-Distribution Detection

Mahalanobis-distance-based OOD detection is used to assess whether live environmental observations resemble the distribution of the training data. This helps identify unusual observations that may require cautious interpretation.

## Interactive Dashboard

The Streamlit dashboard integrates the completed analytical and predictive components into a single application.

Its implemented capabilities include:

- Location-based environmental risk assessment.
- Live weather and air-quality data retrieval.
- Calibrated machine-learning predictions.
- Historical environmental risk scoring.
- Machine-learning and historical-risk fusion.
- Out-of-distribution detection.
- Environmental data visualization and exploratory analysis.
- Model comparison and performance evaluation.
- SHAP-based explanations.
- Risk assessment insights for environmental monitoring and decision support.

## Results and Performance Evaluation

The final fusion model was evaluated on an untouched 2025 test set containing 365 observations.

| Evaluation Metric | Final Result |
|---|---:|
| Accuracy | 65.75% |
| Precision | 41.57% |
| Recall | 77.89% |
| F1-score | 0.5421 |
| Matthews Correlation Coefficient (MCC) | 0.3457 |
| ROC-AUC | 0.7559 |
| PR-AUC | 0.4428 |
| Brier score | 0.1925 |

The confusion matrix for the final fusion model is:

| | Predicted Normal | Predicted Disaster |
|---|---:|---:|
| Actual Normal | 166 | 104 |
| Actual Disaster | 21 | 74 |

The model detected 74 out of 95 disaster observations in the test set, achieving approximately 77.9% recall. This emphasizes the model's ability to identify a substantial proportion of disaster cases, while the number of false positives highlights the trade-off involved in early-warning applications.

The results demonstrate the potential of combining machine-learning predictions with historical environmental evidence. However, predictions are decision-support estimates and do not guarantee the occurrence or absence of a disaster.

## Key Contributions and Innovation

- **Multi-source data integration:** Combines climate, extreme-weather, and air-quality information.
- **Big Data Analytics:** Applies PySpark-based environmental data processing.
- **Hybrid machine learning:** Integrates Logistic Regression and XGBoost.
- **Probability calibration:** Uses Platt Scaling to improve probability reliability.
- **Historical and real-time risk assessment:** Combines historical environmental patterns with live environmental observations.
- **Weighted risk fusion:** Incorporates historical risk evidence into the final prediction.
- **Explainable AI:** Uses SHAP to identify important factors influencing predictions.
- **OOD detection:** Identifies live observations that differ from the training distribution.
- **Interactive visualization:** Integrates analytical results, live data, risk estimates, and explanations in a Streamlit dashboard.

## Project Completion Status

**Overall Status: Completed**

All planned project stages have been completed, including:

- Literature review and research gap identification.
- Environmental dataset collection and integration.
- Data preprocessing and feature engineering.
- Big Data processing and exploratory data analysis.
- Machine-learning model development and evaluation.
- Hybrid Logistic Regression–XGBoost model implementation.
- Probability calibration using Platt Scaling.
- Historical environmental risk scoring and risk fusion.
- Mahalanobis-distance-based OOD detection.
- SHAP-based explainability.
- Live weather and air-quality API integration.
- Streamlit dashboard development and integration.
- Final model evaluation and project documentation.

## Conclusion

This project presents an integrated Explainable AI-driven framework for climate prediction and environmental disaster risk assessment. By combining environmental datasets, Big Data Analytics, hybrid machine learning, probability calibration, historical risk evidence, OOD detection, and SHAP-based explainability, the system provides a more transparent approach to environmental risk assessment.

The completed Streamlit dashboard connects live environmental observations with predictive analytics and interpretable insights. The final evaluation demonstrates the potential of the weighted fusion approach to identify disaster cases while incorporating historical environmental evidence.

The framework can support environmental monitoring, risk analysis, and early-warning decision support, with further validation and broader data coverage offering opportunities for future improvement.

## Team

**Batch 6 — Data Analytics and Visualization (24DSB3101)**

- L. Mrudani
- Sri Kruthi
- P. Hasitha

**Supervisor:** Dr. K. Swapnika

## License

This project is developed for academic and research purposes.
