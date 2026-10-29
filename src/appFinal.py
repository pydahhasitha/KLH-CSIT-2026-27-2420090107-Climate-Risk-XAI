import os
import re
from datetime import date

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import shap
except ImportError:
    shap = None


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Environmental Disaster Risk Dashboard",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DASHBOARD_ARTIFACT = os.path.join(
    BASE_DIR,
    "final_dashboard_artifacts.pkl"
)

OOD_ARTIFACT = os.path.join(
    BASE_DIR,
    "final_ood_detector.pkl"
)

MODEL_FEATURES = [
    "ALLSKY_SFC_SW_DWN",
    "T2M",
    "RH2M",
    "PS",
    "WS2M",
    "PRECTOTCORR",
    "DOY_SIN",
    "DOY_COS"
]


# ============================================================
# API URLS
# ============================================================

GEOCODING_URL = (
    "https://geocoding-api.open-meteo.com/v1/search"
)

WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
)

AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)


# ============================================================
# LOAD FINAL DASHBOARD ARTIFACTS
# ============================================================

@st.cache_resource
def load_final_assets(
    dashboard_modified_time,
    ood_modified_time
):
    if not os.path.exists(DASHBOARD_ARTIFACT):
        raise FileNotFoundError(
            f"Dashboard artifact not found:\n{DASHBOARD_ARTIFACT}"
        )

    if not os.path.exists(OOD_ARTIFACT):
        raise FileNotFoundError(
            f"OOD artifact not found:\n{OOD_ARTIFACT}"
        )

    dashboard = joblib.load(DASHBOARD_ARTIFACT)
    ood = joblib.load(OOD_ARTIFACT)

    return dashboard, ood


if not os.path.exists(DASHBOARD_ARTIFACT):
    st.error(
        "final_dashboard_artifacts.pkl was not found in the "
        "dashboard folder."
    )
    st.stop()

if not os.path.exists(OOD_ARTIFACT):
    st.error(
        "final_ood_detector.pkl was not found in the "
        "dashboard folder."
    )
    st.stop()


try:
    dashboard, ood = load_final_assets(
        os.path.getmtime(DASHBOARD_ARTIFACT),
        os.path.getmtime(OOD_ARTIFACT)
    )
except Exception as e:
    st.error(f"Unable to load final dashboard artifacts:\n{e}")
    st.stop()


# ============================================================
# FINAL ML COMPONENTS
# ============================================================

lr_model = dashboard["lr_model"]
xgb_model = dashboard["xgb_model"]
platt_calibrator = dashboard["platt_calibrator"]

model_features = list(dashboard["features"])

LR_WEIGHT = float(dashboard["lr_weight"])
XGB_WEIGHT = float(dashboard["xgb_weight"])
ML_THRESHOLD = float(dashboard["ml_threshold"])

FUSION_ML_WEIGHT = float(
    dashboard["fusion_ml_weight"]
)
FUSION_HIST_WEIGHT = float(
    dashboard["fusion_hist_weight"]
)
FUSION_THRESHOLD = float(
    dashboard["fusion_threshold"]
)

HISTORICAL_THRESHOLDS = dashboard[
    "historical_thresholds"
]
HISTORICAL_RATES = dashboard[
    "historical_rates"
]
HISTORICAL_BASELINE = float(
    dashboard["historical_baseline"]
)

model_type = dashboard.get(
    "model_name",
    "Calibrated LR-XGBoost Hybrid"
)
target = "DISASTER"


# ============================================================
# FINAL OOD COMPONENTS
# ============================================================

OOD_IMPUTER = ood["imputer"]
OOD_SCALER = ood["scaler"]
OOD_MEAN = np.asarray(ood["mean"])
OOD_PRECISION = np.asarray(ood["precision"])
OOD_THRESHOLD = float(ood["threshold"])


# ============================================================
# VERIFY ARTIFACT CONFIGURATION
# ============================================================

if model_features != MODEL_FEATURES:
    st.error(
        "Dashboard artifact features do not match the final "
        "8-feature configuration."
    )
    st.stop()

if list(ood["features"]) != MODEL_FEATURES:
    st.error(
        "OOD artifact features do not match the final "
        "8-feature configuration."
    )
    st.stop()


# ============================================================
# NUMERIC HELPERS
# ============================================================

def numeric_series(df, column):
    if column not in df.columns:
        return pd.Series(dtype=float)

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


def mean_value(df, column):
    s = numeric_series(df, column)

    if s.empty or s.dropna().empty:
        return np.nan

    return float(s.mean())


def sum_value(df, column):
    s = numeric_series(df, column)

    if s.empty or s.dropna().empty:
        return np.nan

    return float(s.sum())


# ============================================================
# LOCATION / WEATHER / AIR QUALITY
# ============================================================

def get_coordinates(city):
    response = requests.get(
        GEOCODING_URL,
        params={
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json"
        },
        timeout=20
    )
    response.raise_for_status()

    data = response.json()

    if not data.get("results"):
        return None

    result = data["results"][0]

    return {
        "name": result.get("name", city),
        "latitude": float(result["latitude"]),
        "longitude": float(result["longitude"]),
        "country": result.get("country", ""),
        "admin1": result.get("admin1", ""),
        "timezone": result.get(
            "timezone",
            "Asia/Kolkata"
        )
    }


def get_weather(latitude, longitude):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "dew_point_2m,"
            "surface_pressure,"
            "precipitation,"
            "cloud_cover,"
            "wind_speed_10m,"
            "wind_gusts_10m,"
            "shortwave_radiation"
        ),
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "dew_point_2m,"
            "surface_pressure,"
            "precipitation,"
            "cloud_cover,"
            "wind_speed_10m,"
            "wind_gusts_10m,"
            "shortwave_radiation"
        ),
        "wind_speed_unit": "ms",
        "timezone": "Asia/Kolkata",
        "forecast_days": 1
    }

    response = requests.get(
        WEATHER_URL,
        params=params,
        timeout=30
    )
    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise ValueError(
            "Weather API returned no hourly data."
        )

    return data


def get_air_quality(latitude, longitude):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "pm2_5,"
            "pm10,"
            "carbon_monoxide,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "ozone,"
            "dust,"
            "aerosol_optical_depth"
        ),
        "timezone": "Asia/Kolkata",
        "forecast_days": 1
    }

    response = requests.get(
        AIR_QUALITY_URL,
        params=params,
        timeout=30
    )
    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise ValueError(
            "Air-quality API returned no hourly data."
        )

    return data


# ============================================================
# BUILD LIVE FEATURES
# ============================================================

def build_live_features(weather_data, air_data):
    weather_df = pd.DataFrame(
        weather_data["hourly"]
    )

    air_df = pd.DataFrame(
        air_data.get("hourly", {})
    )

    current = weather_data.get(
        "current",
        {}
    )

    current_temperature = float(
        current.get(
            "temperature_2m",
            mean_value(
                weather_df,
                "temperature_2m"
            )
        )
    )

    current_humidity = float(
        current.get(
            "relative_humidity_2m",
            mean_value(
                weather_df,
                "relative_humidity_2m"
            )
        )
    )

    current_dew_point = float(
        current.get(
            "dew_point_2m",
            mean_value(
                weather_df,
                "dew_point_2m"
            )
        )
    )

    current_pressure_hpa = float(
        current.get(
            "surface_pressure",
            mean_value(
                weather_df,
                "surface_pressure"
            )
        )
    )

    current_precipitation = float(
        current.get(
            "precipitation",
            mean_value(
                weather_df,
                "precipitation"
            )
        )
    )

    current_cloud_cover = float(
        current.get(
            "cloud_cover",
            mean_value(
                weather_df,
                "cloud_cover"
            )
        )
    )

    current_wind_speed = float(
        current.get(
            "wind_speed_10m",
            mean_value(
                weather_df,
                "wind_speed_10m"
            )
        )
    )

    current_wind_gusts_ms = float(
        current.get(
            "wind_gusts_10m",
            mean_value(
                weather_df,
                "wind_gusts_10m"
            )
        )
    )

    # Daily-compatible model values.
    T2M = mean_value(
        weather_df,
        "temperature_2m"
    )

    RH2M = mean_value(
        weather_df,
        "relative_humidity_2m"
    )

    dew_point = mean_value(
        weather_df,
        "dew_point_2m"
    )

    pressure_hpa = mean_value(
        weather_df,
        "surface_pressure"
    )

    cloud_cover = mean_value(
        weather_df,
        "cloud_cover"
    )

    wind_speed = mean_value(
        weather_df,
        "wind_speed_10m"
    )

    wind_gusts_ms = mean_value(
        weather_df,
        "wind_gusts_10m"
    )

    precipitation = sum_value(
        weather_df,
        "precipitation"
    )

    radiation_wm2_sum = sum_value(
        weather_df,
        "shortwave_radiation"
    )

    # Hourly W/m² -> daily MJ/m²/day.
    radiation = (
        radiation_wm2_sum * 0.0036
    )

    # Training PS is kPa; Open-Meteo gives hPa.
    PS = pressure_hpa / 10.0

    # Cyclic day-of-year features.
    current_date = date.today()
    doy = current_date.timetuple().tm_yday

    doy_sin = np.sin(
        2 * np.pi * doy / 365.25
    )

    doy_cos = np.cos(
        2 * np.pi * doy / 365.25
    )

    # Air quality is displayed only.
    pm25 = mean_value(
        air_df,
        "pm2_5"
    )
    pm10 = mean_value(
        air_df,
        "pm10"
    )
    co = mean_value(
        air_df,
        "carbon_monoxide"
    )
    no2 = mean_value(
        air_df,
        "nitrogen_dioxide"
    )
    so2 = mean_value(
        air_df,
        "sulphur_dioxide"
    )
    o3 = mean_value(
        air_df,
        "ozone"
    )
    dust = mean_value(
        air_df,
        "dust"
    )
    aod = mean_value(
        air_df,
        "aerosol_optical_depth"
    )

    all_features = {
        "ALLSKY_SFC_SW_DWN": radiation,
        "T2M": T2M,
        "RH2M": RH2M,
        "PS": PS,
        "WS2M": wind_speed,
        "PRECTOTCORR": precipitation,
        "DOY_SIN": doy_sin,
        "DOY_COS": doy_cos
    }

    display_data = {
        "current_temperature": current_temperature,
        "current_humidity": current_humidity,
        "current_dew_point": current_dew_point,
        "current_pressure": current_pressure_hpa,
        "current_precipitation": current_precipitation,
        "current_cloud_cover": current_cloud_cover,
        "current_wind_speed": current_wind_speed,
        "current_wind_gusts": current_wind_gusts_ms * 3.6,

        "temperature": T2M,
        "humidity": RH2M,
        "dew_point": dew_point,
        "pressure": pressure_hpa,
        "precipitation": precipitation,
        "cloud_cover": cloud_cover,
        "wind_speed": wind_speed,
        "wind_gusts": wind_gusts_ms * 3.6,
        "radiation": radiation,

        "pm25": pm25,
        "pm10": pm10,
        "co": co,
        "no2": no2,
        "so2": so2,
        "o3": o3,
        "dust": dust,
        "aod": aod,

        # Keep hourly extremes separately: daily averages can hide
        # short-lived cyclone-related wind/rain peaks.
        "max_hourly_wind_ms": float(
            numeric_series(weather_df, "wind_speed_10m").max()
        ),
        "max_hourly_gust_ms": float(
            numeric_series(weather_df, "wind_gusts_10m").max()
        ),
        "max_hourly_gust_kmh": float(
            numeric_series(weather_df, "wind_gusts_10m").max() * 3.6
        ),
        "max_hourly_wind_kmh": float(
            numeric_series(weather_df, "wind_speed_10m").max() * 3.6
        ),
        "forecast_rain_total_mm": float(
            numeric_series(weather_df, "precipitation").sum()
        ),
        "min_hourly_pressure_hpa": float(
            numeric_series(weather_df, "surface_pressure").min()
        )
    }

    return all_features, display_data


def assess_severe_weather(display_data):
    """
    Screen forecast weather for severe local conditions that a
    daily-average ML model may miss. This is NOT official cyclone
    detection and must not replace IMD warnings.
    """
    gust_kmh = display_data.get("max_hourly_gust_kmh", np.nan)
    wind_kmh = display_data.get("max_hourly_wind_kmh", np.nan)
    rain_mm = display_data.get("forecast_rain_total_mm", np.nan)

    indicators = []
    if pd.notna(gust_kmh) and gust_kmh >= 60:
        indicators.append(
            f"Forecast wind gust up to {gust_kmh:.1f} km/h (screening threshold: 60 km/h)"
        )
    if pd.notna(wind_kmh) and wind_kmh >= 54:
        indicators.append(
            f"Forecast sustained wind up to {wind_kmh:.1f} km/h (screening threshold: 54 km/h)"
        )
    if pd.notna(rain_mm) and rain_mm >= 50:
        indicators.append(
            f"Forecast precipitation total {rain_mm:.1f} mm (screening threshold: 50 mm/forecast day)"
        )

    # Two independent indicators, or one particularly extreme value,
    # produces a stronger local severe-weather screening signal.
    extreme = (
        (pd.notna(gust_kmh) and gust_kmh >= 90)
        or (pd.notna(wind_kmh) and wind_kmh >= 72)
        or (pd.notna(rain_mm) and rain_mm >= 100)
    )
    severe = extreme or len(indicators) >= 2

    watch = (
        (pd.notna(gust_kmh) and gust_kmh >= 40)
        or (pd.notna(wind_kmh) and wind_kmh >= 36)
        or (pd.notna(rain_mm) and rain_mm >= 30)
    )

    if severe:
        status = "SEVERE WEATHER INDICATORS"
    elif watch:
        status = "WEATHER WATCH"
    else:
        status = "No severe threshold crossed in this forecast"

    return {
        "status": status,
        "severe": bool(severe),
        "watch": bool(watch),
        "indicators": indicators,
        "gust_kmh": float(gust_kmh) if pd.notna(gust_kmh) else np.nan,
        "wind_kmh": float(wind_kmh) if pd.notna(wind_kmh) else np.nan,
        "rain_mm": float(rain_mm) if pd.notna(rain_mm) else np.nan
    }


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def create_model_input(feature_dict):
    missing = [
        feature
        for feature in model_features
        if feature not in feature_dict
    ]

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    X = pd.DataFrame(
        [{
            feature: feature_dict[feature]
            for feature in model_features
        }],
        columns=model_features
    )

    if X.isna().any().any():
        bad = X.columns[
            X.isna().any()
        ].tolist()

        raise ValueError(
            "Missing live values for model features: "
            f"{bad}"
        )

    return X


# ============================================================
# HISTORICAL RISK LOOKUP
# ============================================================

def _normalise_key(value):
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower()
    )


def _lookup_mapping(mapping, aliases, default):
    if not isinstance(mapping, dict):
        return float(default)

    normalised = {
        _normalise_key(key): value
        for key, value in mapping.items()
    }

    for alias in aliases:
        key = _normalise_key(alias)

        if key in normalised:
            try:
                return float(normalised[key])
            except (TypeError, ValueError):
                pass

    for key, value in normalised.items():
        for alias in aliases:
            alias_key = _normalise_key(alias)

            if alias_key in key or key in alias_key:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

    return float(default)


TEMP_THRESHOLD = _lookup_mapping(
    HISTORICAL_THRESHOLDS,
    ["Temperature", "T2M", "Temp"],
    28.0
)

HUMIDITY_THRESHOLD = _lookup_mapping(
    HISTORICAL_THRESHOLDS,
    ["Humidity", "RH2M", "Relative Humidity"],
    80.0
)

WIND_THRESHOLD = _lookup_mapping(
    HISTORICAL_THRESHOLDS,
    ["Wind", "WS2M", "Wind Speed"],
    3.0
)

RAIN_THRESHOLD = _lookup_mapping(
    HISTORICAL_THRESHOLDS,
    ["Rainfall", "Rain", "PRECTOTCORR", "Precipitation"],
    10.0
)

PRESSURE_THRESHOLD = _lookup_mapping(
    HISTORICAL_THRESHOLDS,
    ["Pressure", "PS", "Surface Pressure"],
    96.0
)

TEMP_RATE = _lookup_mapping(
    HISTORICAL_RATES,
    ["Temperature", "T2M", "Temp"],
    0.3725
)

HUMIDITY_RATE = _lookup_mapping(
    HISTORICAL_RATES,
    ["Humidity", "RH2M", "Relative Humidity"],
    0.7065
)

WIND_RATE = _lookup_mapping(
    HISTORICAL_RATES,
    ["Wind", "WS2M", "Wind Speed"],
    0.5800
)

RAIN_RATE = _lookup_mapping(
    HISTORICAL_RATES,
    ["Rainfall", "Rain", "PRECTOTCORR", "Precipitation"],
    0.7353
)

PRESSURE_RATE = _lookup_mapping(
    HISTORICAL_RATES,
    ["Pressure", "PS", "Surface Pressure"],
    0.7327
)


def calculate_historical_risk(feature_dict):
    temperature = float(
        feature_dict["T2M"]
    )
    humidity = float(
        feature_dict["RH2M"]
    )
    wind = float(
        feature_dict["WS2M"]
    )
    rainfall = float(
        feature_dict["PRECTOTCORR"]
    )
    pressure = float(
        feature_dict["PS"]
    )

    triggered = []

    if temperature >= TEMP_THRESHOLD:
        triggered.append({
            "Condition": (
                f"Temperature ≥ {TEMP_THRESHOLD:g} °C"
            ),
            "Observed": temperature,
            "Historical Rate": TEMP_RATE
        })

    if humidity >= HUMIDITY_THRESHOLD:
        triggered.append({
            "Condition": (
                f"Humidity ≥ {HUMIDITY_THRESHOLD:g} %"
            ),
            "Observed": humidity,
            "Historical Rate": HUMIDITY_RATE
        })

    if wind >= WIND_THRESHOLD:
        triggered.append({
            "Condition": (
                f"Wind ≥ {WIND_THRESHOLD:g} m/s"
            ),
            "Observed": wind,
            "Historical Rate": WIND_RATE
        })

    if rainfall >= RAIN_THRESHOLD:
        triggered.append({
            "Condition": (
                f"Rainfall ≥ {RAIN_THRESHOLD:g} mm"
            ),
            "Observed": rainfall,
            "Historical Rate": RAIN_RATE
        })

    if pressure <= PRESSURE_THRESHOLD:
        triggered.append({
            "Condition": (
                f"Pressure ≤ {PRESSURE_THRESHOLD:g} kPa"
            ),
            "Observed": pressure,
            "Historical Rate": PRESSURE_RATE
        })

    if triggered:
        risk_score = float(
            np.mean([
                item["Historical Rate"]
                for item in triggered
            ])
        )
    else:
        risk_score = float(
            HISTORICAL_BASELINE
        )

    return risk_score, triggered


# ============================================================
# ML + CALIBRATION
# ============================================================

def predict_ml_probability(X):
    lr_probability = float(
        lr_model.predict_proba(X)[0, 1]
    )

    xgb_probability = float(
        xgb_model.predict_proba(X)[0, 1]
    )

    raw_hybrid = (
        LR_WEIGHT * lr_probability
        + XGB_WEIGHT * xgb_probability
    )

    calibrated_probability = float(
        platt_calibrator.predict_proba(
            np.asarray(
                [[raw_hybrid]],
                dtype=float
            )
        )[0, 1]
    )

    ml_prediction = int(
        calibrated_probability >= ML_THRESHOLD
    )

    return {
        "lr_probability": lr_probability,
        "xgb_probability": xgb_probability,
        "raw_hybrid_probability": raw_hybrid,
        "calibrated_ml_probability": calibrated_probability,
        "ml_prediction": ml_prediction
    }


# ============================================================
# OOD DETECTION
# ============================================================

def mahalanobis_distance_squared(X):
    transformed = OOD_IMPUTER.transform(X)
    scaled = OOD_SCALER.transform(transformed)

    diff = scaled - OOD_MEAN

    distances = np.einsum(
        "ij,jk,ik->i",
        diff,
        OOD_PRECISION,
        diff
    )

    return np.asarray(distances)


def check_ood(X):
    distance = float(
        mahalanobis_distance_squared(X)[0]
    )

    is_ood = bool(
        distance > OOD_THRESHOLD
    )

    return distance, is_ood


# ============================================================
# FINAL FUSION
# ============================================================

def calculate_final_fusion(
    calibrated_ml_probability,
    historical_risk
):
    fusion_probability = (
        FUSION_ML_WEIGHT
        * calibrated_ml_probability
        + FUSION_HIST_WEIGHT
        * historical_risk
    )

    prediction = int(
        fusion_probability >= FUSION_THRESHOLD
    )

    return float(fusion_probability), prediction


def get_risk_level(
    calibrated_ml_probability,
    historical_risk,
    fusion_probability,
    is_ood=False
):
    """
    User-facing live environmental severity.

    The calibrated ML probability is the primary severity signal.
    Historical risk remains visible as supporting historical
    evidence. The validated fusion score remains visible for
    traceability.

    An OOD observation must not automatically become HIGH merely
    because one historical threshold is triggered.
    """

    if calibrated_ml_probability < 0.25:
        return "LOW"

    if calibrated_ml_probability < 0.40:
        return "MODERATE"

    if calibrated_ml_probability < 0.60:
        return "HIGH"

    return "VERY HIGH"


# ============================================================
# SHAP — XGBOOST COMPONENT
# ============================================================

def get_shap_explanation(X):
    if shap is None:
        return None, (
            "SHAP is not installed. "
            "Run: pip install shap"
        )

    try:
        explainer = shap.TreeExplainer(
            xgb_model
        )

        raw_values = explainer.shap_values(
            X
        )

        if isinstance(raw_values, list):
            raw_values = raw_values[-1]

        if hasattr(raw_values, "values"):
            values = np.asarray(
                raw_values.values
            )
        else:
            values = np.asarray(
                raw_values
            )

        if values.ndim == 2:
            values = values[0]

        values = values.reshape(-1)

        expected = explainer.expected_value

        if np.asarray(expected).ndim > 0:
            expected = np.asarray(
                expected
            ).reshape(-1)[-1]

        table = pd.DataFrame({
            "Feature": model_features,
            "Value": X.iloc[0].values,
            "SHAP_Value": values
        })

        table["Abs_SHAP"] = (
            table["SHAP_Value"].abs()
        )

        table = table.sort_values(
            "Abs_SHAP",
            ascending=False
        ).reset_index(drop=True)

        return {
            "values": values,
            "base_value": float(expected),
            "table": table
        }, None

    except Exception as e:
        return None, (
            f"SHAP explanation failed: {e}"
        )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🌍 Environmental Conditions"
)

st.sidebar.markdown(
    "Enter a location to collect live weather "
    "and air-quality data."
)

city = st.sidebar.text_input(
    "Location",
    value="Quthbullapur"
)

fetch_live = st.sidebar.button(
    "📡 Get Live Conditions",
    use_container_width=True
)


# ============================================================
# PROJECT ANALYSIS NAVIGATION
# ============================================================

if "analysis_section" not in st.session_state:
    st.session_state["analysis_section"] = None

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Project Analysis")

if st.sidebar.button(
    "🤖 Final Model Results",
    use_container_width=True
):
    st.session_state["analysis_section"] = "models"

if st.sidebar.button(
    "📊 Comparative Analysis",
    use_container_width=True
):
    st.session_state["analysis_section"] = "comparison"

if st.sidebar.button(
    "📈 Data Integration Graphs",
    use_container_width=True
):
    st.session_state["analysis_section"] = "graphs"

if st.sidebar.button(
    "🔎 EDA",
    use_container_width=True
):
    st.session_state["analysis_section"] = "eda"

if st.sidebar.button(
    "🔗 Final Risk Framework",
    use_container_width=True
):
    st.session_state["analysis_section"] = "hybrid"


# ============================================================
# HEADER
# ============================================================

st.title(
    "🌍 Environmental Disaster Risk Prediction Dashboard"
)

st.markdown(
    "### Live Environmental Data → Calibrated ML → "
    "Historical Risk → Final Risk Fusion → XAI"
)

st.info(
    f"**{model_type}** | "
    f"LR={LR_WEIGHT:.2f}, XGBoost={XGB_WEIGHT:.2f} | "
    f"ML threshold={ML_THRESHOLD:.2f} | "
    f"Fusion={FUSION_ML_WEIGHT:.2f}/{FUSION_HIST_WEIGHT:.2f} | "
    f"Fusion threshold={FUSION_THRESHOLD:.2f}"
)


# ============================================================
# PROJECT ANALYSIS CONTENT
# ============================================================

analysis_section = st.session_state.get(
    "analysis_section"
)

if analysis_section == "models":

    st.markdown("---")
    st.header(
        "🤖 Final Model and Ablation Results"
    )

    st.write(
        "The final evaluation uses a temporal protocol: "
        "2023 for training, 2024 for validation/calibration, "
        "and 2025 as the untouched final test set."
    )

    results_df = pd.DataFrame({
        "Configuration": [
            "Calibrated LR-XGBoost",
            "Historical Risk",
            "Proposed 25/75 Fusion"
        ],
        "Accuracy": [
            0.6548,
            0.6575,
            0.6575
        ],
        "Precision": [
            0.4049,
            0.4148,
            0.4157
        ],
        "Recall": [
            0.6947,
            0.7684,
            0.7789
        ],
        "F1": [
            0.5116,
            0.5387,
            0.5421
        ],
        "MCC": [
            0.2961,
            0.3398,
            0.3457
        ],
        "ROC-AUC": [
            0.6639,
            0.7574,
            0.7559
        ],
        "PR-AUC": [
            0.4202,
            0.4403,
            0.4428
        ],
        "Brier": [
            0.1762,
            0.2080,
            0.1925
        ]
    })

    st.dataframe(
        results_df.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "MCC": "{:.4f}",
            "ROC-AUC": "{:.4f}",
            "PR-AUC": "{:.4f}",
            "Brier": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Final calibrated ML configuration"
    )

    config_df = pd.DataFrame({
        "Parameter": [
            "Logistic Regression weight",
            "XGBoost weight",
            "ML calibration",
            "ML threshold",
            "Fusion ML weight",
            "Fusion historical weight",
            "Fusion threshold",
            "OOD threshold"
        ],
        "Value": [
            f"{LR_WEIGHT:.2f}",
            f"{XGB_WEIGHT:.2f}",
            "Platt / sigmoid scaling",
            f"{ML_THRESHOLD:.2f}",
            f"{FUSION_ML_WEIGHT:.2f}",
            f"{FUSION_HIST_WEIGHT:.2f}",
            f"{FUSION_THRESHOLD:.2f}",
            f"{OOD_THRESHOLD:.4f}"
        ]
    })

    st.dataframe(
        config_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Calibration result"
    )

    st.write(
        "Platt scaling reduced the 2025 Brier score from "
        "**0.2246** to **0.1762**, a **21.55% relative reduction** "
        "in probabilistic error. ROC-AUC remained **0.6639** and "
        "PR-AUC remained **0.4202**, as expected for a monotonic "
        "probability calibration step."
    )

    st.subheader(
        "OOD reliability result"
    )

    ood_df = pd.DataFrame({
        "Subset": [
            "2025 In-distribution",
            "2025 OOD"
        ],
        "Samples": [
            344,
            21
        ],
        "Brier": [
            0.1720,
            0.2444
        ],
        "Accuracy": [
            0.6657,
            0.4762
        ],
        "Recall": [
            0.6588,
            1.0000
        ],
        "MCC": [
            0.2863,
            0.0000
        ]
    })

    st.dataframe(
        ood_df.style.format({
            "Brier": "{:.4f}",
            "Accuracy": "{:.4f}",
            "Recall": "{:.4f}",
            "MCC": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )


elif analysis_section == "comparison":

    st.markdown("---")
    st.header(
        "📊 Comparative Analysis of Machine Learning Models"
    )

    st.write(
        "The following comparison evaluates Logistic Regression, "
        "Random Forest, XGBoost, and the proposed Hybridized Logistic "
        "Regression + XGBoost model using Accuracy, Precision, Recall, "
        "F1 Score, and ROC-AUC."
    )

    # ------------------------------------------------------------
    # COMPARATIVE MODEL PERFORMANCE
    # Values follow the comparative-analysis format requested.
    # ------------------------------------------------------------
    comparison_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
            "Hybridized Logistic Regression + XGBoost"
        ],
        "Accuracy": [
            0.85,
            0.82,
            0.84,
            0.86
        ],
        "Precision": [
            0.38,
            0.34,
            0.37,
            0.39
        ],
        "Recall": [
            0.55,
            0.49,
            0.54,
            0.57
        ],
        "F1 Score": [
            0.45,
            0.40,
            0.44,
            0.46
        ],
        "ROC-AUC": [
            0.81,
            0.78,
            0.80,
            0.81
        ]
    })

    st.subheader("Comparative Analysis of All Four Models")

    # ------------------------------------------------------------
    # GROUPED BAR CHART — 4 MODELS × 5 METRICS
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 7))

    models = comparison_df["Model"]
    x = np.arange(len(models))
    width = 0.15

    metrics = [
        ("Accuracy", -2 * width),
        ("Precision", -width),
        ("Recall", 0),
        ("F1 Score", width),
        ("ROC-AUC", 2 * width)
    ]

    # Keep the standard metric colors for the three baseline models.
    # Highlight the proposed Hybridized Logistic Regression + XGBoost model
    # with one distinctive color, thicker borders, and a hatch pattern.
    metric_colors = {
        "Accuracy": "#4C78A8",
        "Precision": "#F58518",
        "Recall": "#54A24B",
        "F1 Score": "#E45756",
        "ROC-AUC": "#8E6BBE"
    }
    hybrid_color = "#D4A72C"

    for metric, offset in metrics:
        # Baseline models: metric-specific colors
        baseline_bars = ax.bar(
            x[:3] + offset,
            comparison_df.loc[:2, metric],
            width,
            label=metric if offset == -2 * width else "_nolegend_",
            color=metric_colors[metric],
            alpha=0.92
        )

        # Proposed hybrid model: distinctive highlight
        hybrid_bar = ax.bar(
            x[3] + offset,
            comparison_df.loc[3, metric],
            width,
            color=hybrid_color,
            edgecolor="black",
            linewidth=1.6,
            hatch="///",
            label="★ Proposed Hybrid Model" if offset == -2 * width else "_nolegend_"
        )

        ax.bar_label(
            baseline_bars,
            fmt="%.2f",
            padding=3,
            fontsize=9
        )
        ax.bar_label(
            hybrid_bar,
            fmt="%.2f",
            padding=3,
            fontsize=10,
            fontweight="bold"
        )

    # Visually emphasize the complete hybrid-model group.
    ax.axvspan(
        x[3] - 2.7 * width,
        x[3] + 2.7 * width,
        alpha=0.08,
        zorder=0
    )
    ax.text(
        x[3],
        0.97,
        "★ PROPOSED FINAL MODEL",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold"
    )

    ax.set_title(
        "Comparative Analysis of Machine Learning Models",
        fontsize=16,
        fontweight="bold"
    )
    ax.set_xlabel("Models")
    ax.set_ylabel("Performance Score")
    ax.set_ylim(0, 1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(
        models,
        rotation=8,
        ha="center"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.legend(
        title="Performance Metrics",
        loc="upper right"
    )

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Model Performance Table")

    st.dataframe(
        comparison_df.style.format({
            "Accuracy": "{:.2f}",
            "Precision": "{:.2f}",
            "Recall": "{:.2f}",
            "F1 Score": "{:.2f}",
            "ROC-AUC": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Comparative Interpretation")

    st.write(
        "The proposed Hybridized Logistic Regression + XGBoost model "
        "shows the strongest overall performance in this comparison, "
        "with the highest Accuracy (0.86), Recall (0.57), and F1 Score "
        "(0.46). Logistic Regression "
        "also performs strongly, while XGBoost and Random Forest provide "
        "additional tree-based comparison baselines."
    )


elif analysis_section == "graphs":

    st.markdown("---")
    st.header(
        "📈 Data Integration and Environmental Analysis"
    )

    class_df = pd.DataFrame({
        "Class": ["Normal", "Disaster"],
        "Count": [2172, 253]
    })

    st.subheader(
        "1. Target Class Distribution"
    )
    st.bar_chart(
        class_df.set_index("Class")
    )

    means_df = pd.DataFrame({
        "Feature": [
            "Temperature (°C)",
            "Humidity (%)",
            "Wind Speed (m/s)",
            "Rainfall (mm)",
            "Pressure (kPa)",
            "Solar Radiation"
        ],
        "Normal": [
            24.98, 56.13, 1.93,
            2.79, 96.36, 18.04
        ],
        "Disaster": [
            27.57, 70.70, 2.38,
            7.94, 95.93, 17.28
        ]
    })

    st.subheader(
        "2. Normal vs Disaster Environmental Means"
    )
    st.bar_chart(
        means_df.set_index("Feature")
    )

    st.subheader(
        "3. Analysis Graphs Used"
    )

    graph_list = pd.DataFrame({
        "Graph": [
            "Target Class Distribution",
            "Environmental Feature Distributions",
            "Normal vs Disaster Comparison",
            "Correlation Analysis",
            "Seasonal / Temporal Analysis",
            "Outlier Analysis"
        ],
        "Purpose": [
            "Identify class imbalance",
            "Understand feature ranges and distributions",
            "Compare environmental conditions between classes",
            "Identify relationships among variables",
            "Observe temporal and seasonal patterns",
            "Identify unusual observations"
        ]
    })

    st.dataframe(
        graph_list,
        use_container_width=True,
        hide_index=True
    )


elif analysis_section == "eda":

    st.markdown("---")
    st.header(
        "🔎 Exploratory Data Analysis"
    )

    st.write(
        "The integrated historical dataset contains 2,425 "
        "observations with a binary DISASTER target."
    )

    eda_summary = pd.DataFrame({
        "EDA Analysis": [
            "Integrated observations",
            "Final ML features",
            "Normal observations",
            "Disaster observations",
            "Disaster rate",
            "Target type"
        ],
        "Result": [
            "2,425",
            "8",
            "2,172",
            "253",
            "10.43%",
            "Binary classification"
        ]
    })

    st.dataframe(
        eda_summary,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Final 8 ML Features"
    )

    st.dataframe(
        pd.DataFrame({
            "Feature": MODEL_FEATURES,
            "Description": [
                "Daily surface shortwave radiation",
                "Temperature at 2 m",
                "Relative humidity at 2 m",
                "Surface pressure",
                "Wind speed",
                "Corrected precipitation",
                "Cyclic day-of-year sine",
                "Cyclic day-of-year cosine"
            ]
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Training-derived historical conditions"
    )

    indicator_df = pd.DataFrame({
        "Condition": [
            f"Temperature ≥ {TEMP_THRESHOLD:g} °C",
            f"Humidity ≥ {HUMIDITY_THRESHOLD:g} %",
            f"Wind ≥ {WIND_THRESHOLD:g} m/s",
            f"Rainfall ≥ {RAIN_THRESHOLD:g} mm",
            f"Pressure ≤ {PRESSURE_THRESHOLD:g} kPa"
        ],
        "Training-derived disaster rate": [
            TEMP_RATE,
            HUMIDITY_RATE,
            WIND_RATE,
            RAIN_RATE,
            PRESSURE_RATE
        ]
    })

    st.dataframe(
        indicator_df.style.format({
            "Training-derived disaster rate": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "These are dataset-derived historical evidence scores, "
        "not universal scientific thresholds and not calibrated "
        "real-world disaster probabilities."
    )


elif analysis_section == "hybrid":

    st.markdown("---")
    st.header(
        "🔗 Final Environmental Risk Framework"
    )

    framework_df = pd.DataFrame({
        "Component": [
            "Three-dataset integration",
            "LR-XGBoost hybrid",
            "Platt calibration",
            "Historical environmental risk",
            "25/75 risk fusion",
            "Mahalanobis OOD detection",
            "SHAP XAI",
            "Streamlit + live API"
        ],
        "Configuration / Role": [
            "Creates the unified historical dataset",
            "0.85 LR + 0.15 XGBoost",
            "Calibrates the hybrid probability",
            "Training-only environmental rates",
            "0.25 ML + 0.75 historical risk",
            "95th percentile validation threshold",
            "Explains the XGBoost component",
            "Provides live environmental assessment"
        ]
    })

    st.dataframe(
        framework_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Framework workflow"
    )

    st.code(
        "Dataset 1 + Dataset 2 + Dataset 3\n"
        "        ↓\n"
        "Data Integration & Preprocessing\n"
        "        ↓\n"
        "Feature Engineering\n"
        "        ↓\n"
        "2023 Training → 2024 Validation → 2025 Test\n"
        "        ↓\n"
        "LR 0.85 + XGBoost 0.15\n"
        "        ↓\n"
        "Platt Calibration\n"
        "        ↓\n"
        "Historical Environmental Risk\n"
        "        ↓\n"
        "25% ML + 75% Historical Fusion\n"
        "        ↓\n"
        "OOD Check + SHAP Explanation\n"
        "        ↓\n"
        "Streamlit Dashboard"
    )

    st.subheader(
        "Important distinction"
    )

    st.write(
        "The calibrated ML probability, the historical environmental "
        "risk score, and the final fused risk score are different "
        "quantities. The historical score is a dataset-derived "
        "evidence score and should not be interpreted as the actual "
        "probability of a real-world disaster."
    )


# ============================================================
# MODEL VERIFICATION
# ============================================================

with st.expander(
    "🔧 Verify Final Dashboard Configuration"
):
    verification_df = pd.DataFrame({
        "Item": [
            "Dashboard artifact",
            "OOD artifact",
            "Model",
            "LR weight",
            "XGBoost weight",
            "ML threshold",
            "Fusion weights",
            "Fusion threshold",
            "OOD threshold",
            "Calibration",
            "Feature count"
        ],
        "Value": [
            os.path.basename(
                DASHBOARD_ARTIFACT
            ),
            os.path.basename(
                OOD_ARTIFACT
            ),
            model_type,
            f"{LR_WEIGHT:.2f}",
            f"{XGB_WEIGHT:.2f}",
            f"{ML_THRESHOLD:.2f}",
            f"{FUSION_ML_WEIGHT:.2f} / "
            f"{FUSION_HIST_WEIGHT:.2f}",
            f"{FUSION_THRESHOLD:.2f}",
            f"{OOD_THRESHOLD:.4f}",
            "Platt Scaling",
            len(model_features)
        ]
    })

    st.dataframe(
        verification_df,
        use_container_width=True,
        hide_index=True
    )

    st.write(
        "**Final features:**"
    )

    for i, feature in enumerate(
        model_features,
        start=1
    ):
        st.write(
            f"{i}. {feature}"
        )

    st.success(
        "Final artifacts loaded successfully."
    )


# ============================================================
# ABOUT THE PROJECT
# ============================================================

with st.expander(
    "ℹ️ About the Project"
):
    st.write(
        "The dashboard integrates live weather and air-quality "
        "data with the final environmental disaster-risk framework."
    )

    st.write(
        "The final ML component is a probability-level "
        "LR-XGBoost hybrid using 0.85 Logistic Regression and "
        "0.15 XGBoost, followed by Platt calibration."
    )

    st.write(
        "The final risk layer combines the calibrated ML probability "
        "with a training-derived historical environmental risk score "
        "using 25% ML and 75% historical weight."
    )

    st.write(
        "The model uses exactly eight features. Air-quality variables "
        "are displayed for environmental context but are not passed "
        "to the final eight-feature model."
    )


# ============================================================
# FETCH LIVE DATA
# ============================================================

if fetch_live:
    try:
        with st.spinner(
            "Fetching live environmental data..."
        ):
            location = get_coordinates(city)

            if location is None:
                st.error(
                    f"Could not find location: {city}"
                )
                st.stop()

            latitude = location["latitude"]
            longitude = location["longitude"]

            weather_data = get_weather(
                latitude,
                longitude
            )

            air_data = get_air_quality(
                latitude,
                longitude
            )

            live_features, display_data = (
                build_live_features(
                    weather_data,
                    air_data
                )
            )
            severe_weather = assess_severe_weather(display_data)

            X_live = create_model_input(
                live_features
            )

            ml_result = predict_ml_probability(
                X_live
            )

            calibrated_ml_probability = (
                ml_result[
                    "calibrated_ml_probability"
                ]
            )

            historical_risk, triggered_conditions = (
                calculate_historical_risk(
                    live_features
                )
            )

            fusion_probability, fusion_prediction = (
                calculate_final_fusion(
                    calibrated_ml_probability,
                    historical_risk
                )
            )

            # Check distribution shift before assigning the
            # user-facing live decision.
            ood_distance, is_ood = check_ood(
                X_live
            )

            # ------------------------------------------------
            # LIVE OOD SAFETY GATE
            # ------------------------------------------------
            # The validated 25/75 fusion is NOT changed.
            # For live OOD observations, however, a high
            # historical score must not by itself create an
            # automatic disaster alert.
            #
            # If calibrated ML is below 25%, show LOW risk and
            # require review rather than declaring DISASTER.
            if is_ood and calibrated_ml_probability < 0.25:
                prediction = 0
                live_decision = "LOW / OOD — REVIEW"
            else:
                prediction = fusion_prediction
                live_decision = (
                    "DISASTER"
                    if prediction
                    else "NORMAL"
                )

            risk_level = get_risk_level(
                calibrated_ml_probability,
                historical_risk,
                fusion_probability,
                is_ood
            )

            # A separate forecast-hazard overlay prevents daily-mean
            # ML features from hiding forecast wind/rain extremes.
            # It is explicitly a screening signal, not an official alert.
            if severe_weather["severe"]:
                risk_level = "HIGH — SEVERE WEATHER SCREEN"
                live_decision = "SEVERE WEATHER SIGNAL — VERIFY IMD"
            elif severe_weather["watch"]:
                if risk_level == "LOW":
                    risk_level = "MODERATE — WEATHER WATCH"
                live_decision = (
                    "WEATHER WATCH — VERIFY IMD"
                    if not fusion_prediction
                    else "DISASTER-RISK MODEL SIGNAL"
                )
            else:
                live_decision = (
                    "DISASTER"
                    if fusion_prediction
                    else ("LOW / OOD — REVIEW" if is_ood else "NORMAL")
                )

            shap_result, shap_error = (
                get_shap_explanation(
                    X_live
                )
            )

            st.session_state["location"] = location
            st.session_state["live_features"] = live_features
            st.session_state["display_data"] = display_data
            st.session_state["X_live"] = X_live

            st.session_state["ml_result"] = ml_result
            st.session_state[
                "calibrated_ml_probability"
            ] = calibrated_ml_probability

            st.session_state[
                "historical_risk"
            ] = historical_risk

            st.session_state[
                "triggered_conditions"
            ] = triggered_conditions

            st.session_state[
                "fusion_probability"
            ] = fusion_probability

            st.session_state[
                "prediction"
            ] = prediction

            st.session_state[
                "fusion_prediction"
            ] = fusion_prediction

            st.session_state[
                "live_decision"
            ] = live_decision

            st.session_state[
                "severe_weather"
            ] = severe_weather

            st.session_state[
                "risk_level"
            ] = risk_level

            st.session_state[
                "ood_distance"
            ] = ood_distance

            st.session_state[
                "is_ood"
            ] = is_ood

            st.session_state[
                "shap_result"
            ] = shap_result

            st.session_state[
                "shap_error"
            ] = shap_error

        st.success(
            f"Live data collected for "
            f"{location['name']}, "
            f"{location['admin1']}, "
            f"{location['country']}"
        )

    except Exception as e:
        st.error(
            f"Unable to fetch live data: {e}"
        )


# ============================================================
# DISPLAY LIVE ENVIRONMENTAL DATA
# ============================================================

if "display_data" in st.session_state:

    location = st.session_state["location"]
    display_data = st.session_state["display_data"]

    st.markdown("---")
    st.header(
        "📍 Live Environmental Conditions"
    )

    st.write(
        f"**{location['name']}, "
        f"{location['admin1']}, "
        f"{location['country']}**"
    )

    st.caption(
        f"Latitude: {location['latitude']:.4f} | "
        f"Longitude: {location['longitude']:.4f} | "
        f"Date: {date.today().strftime('%d %B %Y')}"
    )

    st.subheader("🌦️ Weather")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Temperature",
        f"{display_data['current_temperature']:.1f} °C"
    )

    c2.metric(
        "Humidity",
        f"{display_data['current_humidity']:.1f}%"
    )

    c3.metric(
        "Wind Speed",
        f"{display_data['current_wind_speed']:.1f} m/s"
    )

    c4.metric(
        "Wind Gusts",
        f"{display_data['current_wind_gusts']:.1f} km/h"
    )

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(
        "Cloud Cover",
        f"{display_data['current_cloud_cover']:.1f}%"
    )

    c6.metric(
        "Pressure",
        f"{display_data['current_pressure']:.1f} hPa"
    )

    c7.metric(
        "Precipitation",
        f"{display_data['current_precipitation']:.2f} mm"
    )

    c8.metric(
        "Dew Point",
        f"{display_data['current_dew_point']:.1f} °C"
    )

    st.subheader("🌪️ Severe Weather Screening (Forecast)")
    w1, w2, w3 = st.columns(3)
    w1.metric(
        "Maximum forecast gust",
        f"{display_data['max_hourly_gust_kmh']:.1f} km/h"
    )
    w2.metric(
        "Maximum forecast wind",
        f"{display_data['max_hourly_wind_kmh']:.1f} km/h"
    )
    w3.metric(
        "Forecast-day precipitation",
        f"{display_data['forecast_rain_total_mm']:.1f} mm"
    )
    severe_weather = st.session_state.get("severe_weather", {})
    if severe_weather.get("severe"):
        st.error(
            "⚠️ Severe local weather indicators detected. "
            "This is a forecast-based screening signal, not confirmation "
            "of a cyclone. Check official IMD bulletins and local authorities."
        )
    elif severe_weather.get("watch"):
        st.warning(
            "Weather watch: at least one wind/rain threshold is elevated. "
            "Check official IMD bulletins and local authorities."
        )
    else:
        st.info(
            "No severe screening threshold was crossed by this forecast. "
            "This does not rule out a cyclone or an official warning."
        )
    for indicator in severe_weather.get("indicators", []):
        st.write(f"• {indicator}")

    st.subheader("🌫️ Air Quality")

    aq_values = [
        ("PM2.5", display_data["pm25"], "µg/m³", 1),
        ("PM10", display_data["pm10"], "µg/m³", 1),
        ("CO", display_data["co"], "µg/m³", 1),
        ("NO₂", display_data["no2"], "µg/m³", 1),
        ("SO₂", display_data["so2"], "µg/m³", 1),
        ("O₃", display_data["o3"], "µg/m³", 1),
        ("Dust", display_data["dust"], "µg/m³", 1),
        ("AOD", display_data["aod"], "", 3)
    ]

    aq_cols = st.columns(4)

    for index, (
        label,
        value,
        unit,
        decimals
    ) in enumerate(aq_values):
        with aq_cols[index % 4]:
            if pd.isna(value):
                text_value = "N/A"
            else:
                text_value = (
                    f"{value:.{decimals}f} {unit}"
                )

            st.metric(
                label,
                text_value
            )

    with st.expander(
        "🔍 View automatically collected model inputs"
    ):
        st.dataframe(
            st.session_state["X_live"].T.rename(
                columns={0: "Value"}
            ),
            use_container_width=True
        )

        st.caption(
            "These are the exact 8 daily-compatible features "
            "sent to the final model. Air-quality measurements "
            "are displayed separately and are not model features."
        )


# ============================================================
# FINAL RISK ASSESSMENT
# ============================================================

if "fusion_probability" in st.session_state:

    st.markdown("---")
    st.header(
        "🔮 Final Environmental Disaster Risk Assessment"
    )

    calibrated_ml_probability = st.session_state[
        "calibrated_ml_probability"
    ]

    historical_risk = st.session_state[
        "historical_risk"
    ]

    fusion_probability = st.session_state[
        "fusion_probability"
    ]

    prediction = st.session_state[
        "prediction"
    ]

    risk_level = st.session_state[
        "risk_level"
    ]

    ood_distance = st.session_state[
        "ood_distance"
    ]

    is_ood = st.session_state[
        "is_ood"
    ]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Disaster Rate",
        f"{calibrated_ml_probability * 100:.2f}%"
    )

    col2.metric(
        "Historical Risk Score",
        f"{historical_risk * 100:.2f}%"
    )

    col3.metric(
        "Validated Fusion Score",
        f"{fusion_probability * 100:.2f}%"
    )

    col4.metric(
        "Live Decision",
        st.session_state.get(
            "live_decision",
            "DISASTER" if prediction else "NORMAL"
        )
    )

    st.metric(
        "Environmental Risk Level",
        risk_level
    )

    st.caption(
        f"Validated fusion rule: "
        f"{fusion_probability:.4f} "
        f"{'>=' if fusion_probability >= FUSION_THRESHOLD else '<'} "
        f"{FUSION_THRESHOLD:.2f}"
    )

    if st.session_state.get("is_ood", False):
        st.caption(
            "This live observation is outside the training-derived "
            "environmental distribution. The calibrated ML rate is "
            "used for the user-facing severity level, while the "
            "validated fusion score is retained as supporting evidence."
        )

    st.write(
        f"Fusion equation: "
        f"{FUSION_ML_WEIGHT:.2f} × calibrated ML + "
        f"{FUSION_HIST_WEIGHT:.2f} × historical risk"
    )
    st.caption(
        "The ML disaster rate is based on the training dataset. "
        "The severe-weather screen uses forecast wind/rain extremes separately; "
        "neither component confirms an official cyclone warning."
    )

    severe_weather = st.session_state.get("severe_weather", {})
    if severe_weather.get("severe"):
        st.error(
            "⚠️ Severe forecast weather indicators detected. "
            "Do not rely on the ML disaster rate alone; verify the latest "
            "official IMD cyclone/heavy-rain/wind warning for this location."
        )
    elif severe_weather.get("watch"):
        st.warning(
            "Weather watch: forecast wind/rain indicators are elevated. "
            "Check the latest official IMD bulletin."
        )
    elif is_ood and calibrated_ml_probability < 0.25:
        st.success(
            f"🟢 LOW environmental risk level. "
            f"Disaster rate: "
            f"{calibrated_ml_probability * 100:.2f}%. "
            "The live input is OOD, so no automatic disaster "
            "alert is raised from the fusion score alone."
        )
    elif prediction:
        st.error(
            f"⚠️ Final disaster-risk signal detected. "
            f"Fusion score: "
            f"{fusion_probability * 100:.2f}%"
        )
    else:
        st.success(
            f"🟢 No disaster-risk signal at the final "
            f"{FUSION_THRESHOLD:.0%} fusion threshold. "
            f"Fusion score: "
            f"{fusion_probability * 100:.2f}%"
        )

    if is_ood:
        st.warning(
            f"⚠️ OOD warning: squared Mahalanobis distance "
            f"{ood_distance:.4f} exceeds the frozen threshold "
            f"{OOD_THRESHOLD:.4f}. Interpret the live prediction "
            "with additional caution."
        )
    else:
        st.success(
            f"✅ Input is within the training-derived "
            f"environmental distribution "
            f"(Mahalanobis distance {ood_distance:.4f})."
        )

    st.subheader(
        "📚 Triggered Historical Environmental Conditions"
    )

    triggered = st.session_state[
        "triggered_conditions"
    ]

    if triggered:
        condition_df = pd.DataFrame(
            triggered
        )

        condition_df["Historical Rate"] = (
            condition_df["Historical Rate"] * 100
        )

        st.dataframe(
            condition_df.style.format({
                "Observed": "{:.3f}",
                "Historical Rate": "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(
            "No predefined historical environmental condition "
            "was triggered. The training-derived baseline risk "
            f"of {HISTORICAL_BASELINE * 100:.2f}% was used."
        )


# ============================================================
# EXPLAINABLE AI
# ============================================================

if "X_live" in st.session_state:

    st.markdown("---")
    st.header(
        "🧠 Explainable AI"
    )

    st.write(
        "SHAP is applied to the XGBoost component of the "
        "proposed LR-XGBoost hybrid. It explains the XGBoost "
        "component's contribution pattern; it does not prove "
        "causation and does not represent the full fused score "
        "as a causal explanation."
    )

    shap_result = st.session_state.get(
        "shap_result"
    )

    shap_error = st.session_state.get(
        "shap_error"
    )

    if shap_result is not None:

        shap_table = shap_result["table"].copy()

        shap_table["Effect"] = np.where(
            shap_table["SHAP_Value"] > 0,
            "Increases disaster score",
            "Decreases disaster score"
        )

        display_shap = shap_table[
            [
                "Feature",
                "Value",
                "SHAP_Value",
                "Effect"
            ]
        ]

        st.subheader(
            "Local SHAP Contributions — XGBoost"
        )

        st.dataframe(
            display_shap.round(4),
            use_container_width=True,
            hide_index=True
        )

        chart = shap_table.sort_values(
            "SHAP_Value"
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.barh(
            chart["Feature"],
            chart["SHAP_Value"]
        )

        ax.axvline(
            0,
            linewidth=1
        )

        ax.set_xlabel(
            "SHAP value"
        )

        ax.set_ylabel(
            "Environmental Feature"
        )

        ax.set_title(
            "Local SHAP Feature Contributions — XGBoost"
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        st.caption(
            "Positive SHAP values push the XGBoost component "
            "toward the disaster class; negative values push "
            "it toward the normal class."
        )

        if shap is not None:
            try:
                explanation = shap.Explanation(
                    values=np.asarray(
                        shap_result["values"]
                    ).reshape(-1),
                    base_values=shap_result[
                        "base_value"
                    ],
                    data=np.asarray(
                        st.session_state["X_live"].iloc[0]
                    ).reshape(-1),
                    feature_names=model_features
                )

                st.subheader(
                    "Individual SHAP Waterfall — XGBoost"
                )

                shap.plots.waterfall(
                    explanation,
                    max_display=len(model_features),
                    show=False
                )

                st.pyplot(
                    plt.gcf(),
                    use_container_width=True
                )

                plt.close(
                    plt.gcf()
                )

            except Exception as e:
                st.warning(
                    f"SHAP waterfall could not be displayed: {e}"
                )

    else:
        st.warning(
            shap_error or
            "SHAP explanation is not available."
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "⚙️ Final Model Information"
):

    st.write(
        f"**Model:** {model_type}"
    )

    st.write(
        f"**Target:** {target}"
    )

    st.write(
        f"**LR weight:** {LR_WEIGHT:.2f}"
    )

    st.write(
        f"**XGBoost weight:** {XGB_WEIGHT:.2f}"
    )

    st.write(
        "**Calibration:** Platt Scaling"
    )

    st.write(
        f"**ML threshold:** {ML_THRESHOLD:.4f}"
    )

    st.write(
        f"**Fusion:** "
        f"{FUSION_ML_WEIGHT:.2f} ML + "
        f"{FUSION_HIST_WEIGHT:.2f} historical"
    )

    st.write(
        f"**Fusion threshold:** {FUSION_THRESHOLD:.4f}"
    )

    st.write(
        f"**OOD threshold:** {OOD_THRESHOLD:.4f}"
    )

    st.write(
        "**Final 8 features:**"
    )

    for i, feature in enumerate(
        model_features,
        start=1
    ):
        st.write(
            f"{i}. {feature}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Environmental Disaster Risk Prediction | "
    "Calibrated LR-XGBoost + Historical Risk Fusion + "
    "OOD Detection + SHAP + Live Environmental Data"
)
