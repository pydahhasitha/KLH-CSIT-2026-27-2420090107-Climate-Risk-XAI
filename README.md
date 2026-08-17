# Explainable AI for Climate Change Prediction and Environmental Risk Analysis Using Big Data Analytics

## Team Members

| Name       | ID Number  |
| ---------- | ---------- |
| L. Mrudani | 2420030591 |
| Sri Kruthi | 2420090003 |
| P. Hasitha | 2420090107 |

## Supervisor

**Dr. K. Swapnika**

---

## Abstract

Climate change produces large and complex datasets involving temperature, rainfall, greenhouse gases, sea levels, extreme weather events, and other environmental indicators. Existing AI and machine-learning models can provide accurate predictions, but many of them operate as black-box models, making it difficult to understand why a particular climate or environmental risk prediction was generated.

Therefore, there is a need for a scalable Big Data Analytics and Explainable AI (XAI) framework that can predict climate-related risks while providing understandable explanations of the factors influencing those predictions.

This project proposes a Multi-Source Climate Intelligence Framework that integrates weather, satellite, IoT, and historical disaster data using Big Data Analytics, Machine Learning, and Explainable AI for comprehensive environmental risk prediction. The system aims to provide clear, interactive, and data-driven insights that can support environmental monitoring, planning, and informed decision-making.

---

## Problem Statement

Climate change produces large and complex datasets involving temperature, rainfall, greenhouse gases, sea levels, extreme weather events, and other environmental indicators.

Existing AI and machine-learning models can provide accurate predictions, but many operate as black-box models, making it difficult to understand why a particular climate or environmental risk prediction was generated.

Therefore, there is a need for a scalable Big Data Analytics and Explainable AI framework that can predict climate-related risks while providing understandable explanations of the factors influencing those predictions.

---

## Project Objectives

1. To develop an interactive data analytics dashboard for analysing climate and environmental data from multiple sources.
2. To identify and visualize important climate and environmental trends, patterns, relationships, and risks through comprehensive data analysis.
3. To use predictive analytics and Explainable AI techniques to identify environmental risk levels and the factors influencing them.
4. To develop a Multi-Source Climate Intelligence Framework that integrates weather, satellite, IoT, and historical disaster data using Big Data Analytics, Machine Learning, and Explainable AI.
5. To provide clear, interactive, and data-driven insights that can support environmental monitoring, planning, and informed decision-making.

---

## Proposed Work and Uniqueness

The proposed system:

* Combines climate change prediction and environmental risk analysis in one system.
* Uses multiple environmental and climate factors instead of focusing on a single hazard.
* Integrates Big Data Analytics, Machine Learning, and Explainable AI (XAI).
* Provides explanations for predictions, showing which factors contributed to the risk level.
* Aims to provide risk classification and early-warning insights for real-world environmental decision-making.
* Integrates multiple environmental data sources through a Multi-Source Climate Intelligence Framework.

---

## System Architecture

```text
Climate & Environmental Data Sources
              |
              v
      Data Collection
              |
              v
    Data Cleaning & Integration
              |
              v
      Big Data Processing
        Spark / PySpark
              |
              v
   Exploratory Data Analysis
              |
              v
      ML / DL Models
              |
              v
    Climate Risk Prediction
              |
              v
     Risk Classification
      Low / Medium / High
              |
              v
        SHAP + LIME
              |
              v
 Feature Importance & Explanation
              |
              v
   Environmental Risk Dashboard
```

---

## Benchmark Datasets

The project uses a multi-source approach to collect climate and environmental information.

### Dataset 1 – NASA EarthData

**Source:** https://earthdata.nasa.gov/

**Purpose:**

* Land Surface Temperature
* Vegetation Index (NDVI)
* Soil Moisture
* Wildfire Data

### Dataset 2 – India Open Government Data (OGD)

**Source:** https://data.gov.in/

**Contains:**

* Rainfall
* Temperature
* Forest Cover
* Water Resources
* Groundwater
* Renewable Energy

### Dataset 3 – India Meteorological Department (IMD)

**Source:** https://mausam.imd.gov.in/

**Contains:**

* Daily Temperature
* Rainfall
* Humidity
* Wind Speed
* Cyclone Data
* Heat Wave Records

### Dataset Integration

The collected datasets will be cleaned, integrated, preprocessed, and analysed to identify climate and environmental patterns and support environmental risk prediction.

The project combines information from:

```text
NASA EarthData
       +
India Open Government Data
       +
India Meteorological Department
       |
       v
Multi-Source Climate Dataset
       |
       v
Data Cleaning & Integration
       |
       v
Climate Risk Analysis
```

---

## Technologies and Tools

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Random Forest
* XGBoost
* LightGBM
* CatBoost

### Deep Learning

* LSTM
* Other suitable deep-learning models when required

### Big Data Processing

* Apache Spark
* PySpark

### Explainable AI

* SHAP
* LIME

### Visualization and Dashboard

* Streamlit
* Power BI
* Python visualization libraries

### Development Environment

* Local system
* Google Colab

---

# Project Plan

## Phase 1 – Literature Review

* Study 10–15 recent research papers.
* Identify limitations and research gaps.

**Status:** Completed / Review-1

---

## Phase 2 – Dataset Collection

* Collect climate and environmental datasets.
* Perform data cleaning and integration.

**Status:** Next Phase

---

## Phase 3 – Big Data Processing

* Perform preprocessing and exploratory analysis.
* Use Spark/PySpark if required.

**Status:** Planned

---

## Phase 4 – Model Development

* Train multiple ML/DL models.
* Compare prediction performance.

**Status:** Planned

---

## Phase 5 – Explainable AI

* Implement SHAP/LIME.
* Identify important climate/environmental factors.

**Status:** Planned

---

## Phase 6 – Risk Analysis

* Develop risk categories such as **Low, Medium, High**.
* Analyze factors responsible for each risk.

**Status:** Planned

---

## Phase 7 – Visualization

* Build a dashboard showing predictions, trends, risk levels and explanations.

**Status:** Planned

---

## Phase 8 – Evaluation & Documentation

* Compare models.
* Validate results.
* Document findings and prepare research paper.

**Status:** Planned

---

## Project Phase Status

| Phase   | Description                | Status                   |
| ------- | -------------------------- | ------------------------ |
| Phase 1 | Literature Review          | **Completed / Review-1** |
| Phase 2 | Dataset Collection         | **Next Phase**           |
| Phase 3 | Big Data Processing        | **Planned**              |
| Phase 4 | Model Development          | **Planned**              |
| Phase 5 | Explainable AI             | **Planned**              |
| Phase 6 | Risk Analysis              | **Planned**              |
| Phase 7 | Visualization              | **Planned**              |
| Phase 8 | Evaluation & Documentation | **Planned**              |

---

## Research Gap

The project addresses the following research gaps:

* Many climate prediction models focus primarily on prediction accuracy rather than interpretability.
* Several AI-based climate models function as black boxes, making their decisions difficult for environmental researchers and policymakers to understand.
* Existing studies often focus on one climate variable or one type of environmental risk rather than combining multiple factors.
* Large-scale heterogeneous climate datasets create challenges in data processing and scalability.
* XAI techniques are not consistently integrated with Big Data-based climate analytics.
* Limited research combines climate prediction, environmental risk assessment, and explainability in a single framework.
* Many studies evaluate models technically but provide limited explanation of which environmental factors drive the predicted risk.

---

## Risk Analysis

The system will classify environmental risks into categories such as:

```text
Low Risk
Medium Risk
High Risk
```

The system will also analyse the factors responsible for each predicted risk level.

Explainable AI techniques such as SHAP and LIME will be used to identify important environmental and climate factors contributing to model predictions.

---

## Explainable AI Workflow

```text
Input Climate Data
       |
       v
Machine Learning Model
       |
       v
Risk Prediction
       |
       v
SHAP / LIME
       |
       v
Feature Importance
       |
       v
Explanation of Prediction
       |
       v
Dashboard
```

The XAI component aims to make the prediction process more transparent and understandable by showing which climate and environmental factors influenced the predicted risk.

---

## Dashboard

The proposed interactive dashboard will display:

* Climate trends
* Environmental indicators
* Risk predictions
* Risk levels
* Important contributing factors
* XAI explanations
* Environmental patterns
* Analytical insights

The dashboard may be implemented using Streamlit or Power BI.

---

## Setup and Execution

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

For Linux/macOS:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the major project dependencies:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost shap lime matplotlib streamlit
```

For Big Data processing:

```bash
pip install pyspark
```

### 5. Add Datasets

Place the downloaded datasets inside the appropriate directories:

```text
data/
├── nasa_earthdata/
├── india_ogd/
├── imd/
└── processed/
```

### 6. Perform Preprocessing

Run the preprocessing scripts:

```bash
python <preprocessing_script>.py
```

### 7. Run the Machine Learning Models

```bash
python <model_script>.py
```

### 8. Run Explainable AI

Run the XAI implementation for SHAP/LIME:

```bash
python <xai_script>.py
```

### 9. Run the Dashboard

For Streamlit:

```bash
streamlit run app.py
```

If Power BI is used, load the processed dataset into Power BI and open the dashboard/report.

---

## Repository Structure

```text
project-root/
│
├── README.md
├── data/
│   ├── nasa_earthdata/
│   │   ├── land_surface_temperature/
│   │   ├── ndvi/
│   │   ├── soil_moisture/
│   │   └── wildfire/
│   │
│   ├── india_ogd/
│   │   ├── rainfall/
│   │   ├── temperature/
│   │   ├── forest_cover/
│   │   ├── water_resources/
│   │   ├── groundwater/
│   │   └── renewable_energy/
│   │
│   ├── imd/
│   │   ├── temperature/
│   │   ├── rainfall/
│   │   ├── humidity/
│   │   ├── wind_speed/
│   │   ├── cyclone/
│   │   └── heat_wave/
│   │
│   └── processed/
│
├── notebooks/
│   ├── data_analysis/
│   ├── eda/
│   └── experiments/
│
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── xai/
│   └── risk_analysis/
│
├── dashboard/
├── results/
├── docs/
├── requirements.txt
└── app.py
```

---

## Literature Review

The literature review focuses on the following areas:

* Climate Prediction using AI/ML
* Big Data for Environmental Analysis
* Explainable AI (XAI)
* Environmental Risk Prediction
* Extreme Weather Prediction
* Flood Risk Prediction
* Wildfire Severity Prediction
* Climate Time-Series Forecasting
* IoT and Remote Sensing for Climate Applications

The literature review identifies the limitation that many existing approaches focus on prediction accuracy while providing limited interpretability and limited integration of climate prediction, environmental risk assessment, Big Data Analytics, and Explainable AI.

---

## References

The project review includes research related to:

1. Comparative Analysis of Various Machine Learning Models That Aim at Analyzing Climate Change and Extreme Weather Patterns.
2. Prediction of Flooding Due to Heavy Rainfall in India Using Machine Learning Algorithms: Providing Advanced Warning.
3. Predicting and Mapping Flood Susceptibility: Leveraging Explainable AI and GIS Techniques.
4. Flood Risk Assessment and Interpretation Using Explainable AI.
5. Explainability of Machine Learning Models for Hydrological Time Series Forecasting: The Case of Neuro-Fuzzy Approaches.
6. Maximizing Prediction Accuracy in Wildfire Severity: A Comprehensive Analysis of Machine Learning Models Using Environmental Features.
7. Flood Forecasting with Classical and Quantum Machine Learning Models: A Comparative Analysis.
8. Federated Learning-Based Flood Forecasting Model Enhancing Predictive Accuracy and Privacy in Flood Prediction.
9. Flood Prediction and Adaptive Farming Solutions Using IoT, Machine Learning, and Remote Sensing for Climate-Resilient Agriculture.
10. AI and Climate Protection: Research Gaps and Needs to Align Machine Learning with Greenhouse Gas Reductions.

---

## Current Project Status

**Current Phase:** Phase 1 – Literature Review / Review-1

### Completed

* Problem statement
* Project objectives
* Initial literature review
* Research gap identification
* Proposed work and uniqueness
* Initial system architecture
* Benchmark dataset identification
* Project phase planning

### Next Step

**Phase 2 – Dataset Collection**

The next stage will focus on collecting the NASA EarthData, India Open Government Data, and IMD datasets, followed by data cleaning and integration.

---

## License

This project is developed for academic and research purposes.
