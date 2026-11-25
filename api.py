"""
SmogLens prediction API
-----------------------
Place this file beside:
  M1_PM25_Regression.joblib
  M2_Weather.joblib
  M3_Fire_Weather_Regime_Clustering.joblib

Run:
  pip install -r requirements.txt
  uvicorn api:app --reload

Interactive API documentation:
  http://127.0.0.1:8000/docs

All prediction inputs are sent as JSON dictionaries. GET /metadata tells the
frontend which feature names each saved model expects.
"""

from pathlib import Path
from datetime import date
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent

M1_PATH = BASE_DIR / "M1_PM25_Regression.joblib"
M2_PATH = BASE_DIR / "M2_Weather.joblib"
M3_PATH = BASE_DIR / "M3_Fire_Weather_Regime_Clustering.joblib"


def load_artifact(path: Path, name: str):
    if not path.exists():
        raise RuntimeError(
            f"{name} model file not found: {path}. "
            "Keep api.py and the .joblib files in the same folder."
        )
    try:
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {name} model. Check that scikit-learn/joblib "
            f"versions match the versions used to train it. Details: {exc}"
        ) from exc


# Load once when the API starts, not on every prediction request.
M1 = load_artifact(M1_PATH, "M1")
M2 = load_artifact(M2_PATH, "M2")
M3 = load_artifact(M3_PATH, "M3")

M1_FEATURES = list(M1["features"])
M2_W_FEATURES = list(M2["features"]["W_features"])
M2_WF_FEATURES = list(M2["features"]["WF_features"])
M2_WF_EXTRA = list(M2["features"]["WF_extra"])
M3_FEATURES = [
    "log_fire_frp_3d",
    "delhi_wind_kmh",
    "delhi_pblh_m",
    "delhi_humidity_pct",
    "delhi_temperature_c",
]

app = FastAPI(
    title="SmogLens Prediction API",
    description=(
        "Prediction endpoints for the uploaded M1 PM2.5 regression, "
        "M2 weather-adjusted pollution ledger, and M3 fire-weather regimes."
    ),
    version="1.0.0",
)

# Allows a separately hosted frontend to call this API during development.
# Restrict allow_origins to your frontend URL before public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FeatureRequest(BaseModel):
    # Example: {"features": {"pm25": 80, "pm10": 140}}
    features: Dict[str, Optional[float]] = Field(default_factory=dict)


class M2Request(BaseModel):
    # date is used for the month-specific low-fire reference in M2.
    date: date
    features: Dict[str, Optional[float]] = Field(default_factory=dict)


class CombinedRequest(BaseModel):
    date: date
    m1_features: Dict[str, Optional[float]] = Field(default_factory=dict)
    m2_features: Dict[str, Optional[float]] = Field(default_factory=dict)
    m3_features: Dict[str, Optional[float]] = Field(default_factory=dict)


def make_feature_frame(
    supplied: Dict[str, Optional[float]],
    expected: list[str],
    model_name: str,
) -> pd.DataFrame:
    """Create a one-row frame in the exact training feature order."""
    unknown = sorted(set(supplied) - set(expected))
    if unknown:
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"Unknown feature(s) supplied for {model_name}.",
                "unknown_features": unknown,
                "hint": "Call GET /metadata for the exact accepted feature names.",
            },
        )

    # Missing values are left as NaN. The saved sklearn pipelines contain
    # SimpleImputer steps, so they apply the training-set median imputation.
    row = {
        feature: (
            np.nan if supplied.get(feature) is None
            else float(supplied[feature])
        )
        for feature in expected
    }
    return pd.DataFrame([row], columns=expected)


def check_minimum_inputs(
    supplied: Dict[str, Optional[float]],
    required_groups: list[list[str]],
    model_name: str,
):
    """
    Avoid returning a prediction when the user supplied almost no context.
    The trained imputer can handle some missing values, but not an empty form.
    """
    provided = {k for k, v in supplied.items() if v is not None}
    if not any(provided.intersection(group) for group in required_groups):
        raise HTTPException(
            status_code=422,
            detail=f"Provide at least one valid input feature for {model_name}.",
        )


def clean_number(value: Any) -> Optional[float]:
    value = float(value)
    return value if np.isfinite(value) else None


def predict_m1(features: Dict[str, Optional[float]]) -> dict:
    check_minimum_inputs(
        features,
        [
            ["pm25", "pm25_today", "pm25_lag1", "pm25_lag2"],
            ["pm10", "no2", "o3", "so2", "co"],
        ],
        "M1",
    )
    frame = make_feature_frame(features, M1_FEATURES, "M1")
    prediction = float(M1["model"].predict(frame)[0])

    lower = max(0.0, prediction + float(M1["interval_lower_offset"]))
    upper = max(lower, prediction + float(M1["interval_upper_offset"]))

    return {
        "model": "M1 - Next-day PM2.5 regression",
        "algorithm": M1.get("model_name", "Saved regression model"),
        "predicted_next_day_pm25_ug_m3": round(max(0.0, prediction), 2),
        "empirical_prediction_interval": {
            "lower_ug_m3": round(lower, 2),
            "upper_ug_m3": round(upper, 2),
            "note": (
                "Offsets are the empirical residual bounds saved with the "
                "trained model; this is an approximate interval, not a guarantee."
            ),
        },
        "target": "Next-day PM2.5 concentration (µg/m³)",
    }


def predict_m2(features: Dict[str, Optional[float]], target_date: date) -> dict:
    check_minimum_inputs(
        features,
        [
            ["ws_mean", "ws_max", "blh_mean", "temp_mean", "rh_mean"],
            ["fire_count_lag0", "frp_sum_lag0", "fire_count_lag1", "frp_sum_lag1"],
        ],
        "M2",
    )

    # WF input contains both the weather features and fire-history features.
    # Pass only the weather subset to W; otherwise its strict schema would
    # correctly reject the additional fire columns.
    w_values = {key: value for key, value in features.items() if key in M2_W_FEATURES}
    wf_values = {key: value for key, value in features.items() if key in M2_WF_FEATURES}
    w_frame = make_feature_frame(w_values, M2_W_FEATURES, "M2 weather model")
    wf_frame = make_feature_frame(wf_values, M2_WF_FEATURES, "M2 weather + fire model")

    weather_expected = float(M2["models"]["W_model"].predict(w_frame)[0])
    weather_fire_expected = float(M2["models"]["WF_model"].predict(wf_frame)[0])

    month_key = int(target_date.month)
    low_fire_ref = M2["parameters"]["low_fire_ref"]
    month_reference = low_fire_ref.get(month_key, low_fire_ref.get(str(month_key)))

    low_fire_prediction = None
    fire_excess = None
    if month_reference is not None:
        low_fire_row = dict(features)
        low_fire_row.update(month_reference)
        low_fire_frame = make_feature_frame(
            low_fire_row, M2_WF_FEATURES, "M2 low-fire reference"
        )
        low_fire_prediction = float(
            M2["models"]["WF_model"].predict(low_fire_frame)[0]
        )
        fire_excess = max(0.0, weather_fire_expected - low_fire_prediction)

    bands = M2["parameters"]["residual_bands"]
    p10 = float(bands["p10"])
    p90 = float(bands["p90"])
    p05 = float(bands["p05"])
    p95 = float(bands["p95"])

    return {
        "model": "M2 - Weather-adjusted pollution ledger",
        "date": target_date.isoformat(),
        "weather_expected_pm25_ug_m3": round(max(0.0, weather_expected), 2),
        "weather_and_fire_expected_pm25_ug_m3": round(
            max(0.0, weather_fire_expected), 2
        ),
        "low_fire_reference_pm25_ug_m3": (
            round(max(0.0, low_fire_prediction), 2)
            if low_fire_prediction is not None else None
        ),
        "estimated_fire_associated_excess_ug_m3": (
            round(fire_excess, 2) if fire_excess is not None else None
        ),
        "weather_model_empirical_bands": {
            "80_percent": {
                "lower_ug_m3": round(max(0.0, weather_expected + p10), 2),
                "upper_ug_m3": round(max(0.0, weather_expected + p90), 2),
            },
            "90_percent": {
                "lower_ug_m3": round(max(0.0, weather_expected + p05), 2),
                "upper_ug_m3": round(max(0.0, weather_expected + p95), 2),
            },
        },
        "algorithms": {
            "weather_model": M2["models"].get("W_algorithm"),
            "weather_fire_model": M2["models"].get("WF_algorithm"),
        },
        "interpretation": (
            "The weather model estimates the meteorological baseline. "
            "The weather + fire model includes recent fire indicators. "
            "Fire-associated excess is the weather + fire estimate minus "
            "the month-specific low-fire reference, clipped at zero. "
            "It is an exploratory model estimate, not proof of source attribution."
        ),
        "limitations": [
            "The API does not calculate the notebook's historical weather-severity percentile from one input row.",
            "The episode quadrant also needs an observed PM2.5 value and the calibrated historical percentile.",
        ],
    }


def predict_m3(features: Dict[str, Optional[float]]) -> dict:
    check_minimum_inputs(
        features,
        [[
            "log_fire_frp_3d",
            "delhi_wind_kmh",
            "delhi_pblh_m",
            "delhi_humidity_pct",
            "delhi_temperature_c",
        ]],
        "M3",
    )
    frame = make_feature_frame(features, M3_FEATURES, "M3")
    cluster_id = int(M3.predict(frame)[0])

    # The notebook labels the cluster with the highest fire-intensity profile
    # as fire-active. The saved artifact contains the K-Means pipeline but
    # does not contain the notebook's separately generated label dictionary.
    centers = M3.named_steps["kmeans"].cluster_centers_
    fire_active_cluster = int(np.argmax(centers[:, 0]))

    if cluster_id == fire_active_cluster:
        regime_label = "Warmer / fire-active / better mixed"
    else:
        regime_label = "Cooler / humid / lower ventilation"

    return {
        "model": "M3 - Fire-weather regime clustering",
        "algorithm": "K-Means",
        "cluster_id": cluster_id,
        "regime": regime_label,
        "input_features": M3_FEATURES,
        "note": (
            "Clustering uses environmental features only; PM2.5 is not an input. "
            "The regime is an exploratory grouping, not source attribution or "
            "a direct pollution forecast."
        ),
    }


@app.get("/")
def root():
    return {
        "name": "SmogLens Prediction API",
        "status": "running",
        "docs": "/docs",
        "available_predictions": ["/predict/m1", "/predict/m2", "/predict/m3"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": {"M1": True, "M2": True, "M3": True},
    }


@app.get("/metadata")
def metadata():
    """Frontend can use this endpoint to build forms dynamically."""
    return {
        "M1": {
            "endpoint": "/predict/m1",
            "description": "Predict next-day PM2.5 using the saved 61-feature model.",
            "features": M1_FEATURES,
            "required_json_shape": {"features": {"pm25": 80, "pm10": 140}},
            "note": "Other omitted features are imputed using training medians.",
        },
        "M2": {
            "endpoint": "/predict/m2",
            "description": "Estimate weather baseline and weather-plus-fire PM2.5.",
            "weather_features": M2_W_FEATURES,
            "weather_fire_features": M2_WF_FEATURES,
            "required_json_shape": {
                "date": "2026-11-25",
                "features": {"ws_mean": 8, "blh_mean": 500, "temp_mean": 20},
            },
            "note": "For a meaningful result, supply the relevant weather and fire inputs.",
        },
        "M3": {
            "endpoint": "/predict/m3",
            "description": "Assign a day to a fire-weather regime.",
            "features": M3_FEATURES,
            "required_json_shape": {
                "features": {
                    "log_fire_frp_3d": 1.2,
                    "delhi_wind_kmh": 7,
                    "delhi_pblh_m": 450,
                    "delhi_humidity_pct": 65,
                    "delhi_temperature_c": 18,
                }
            },
        },
    }


@app.post("/predict/m1")
def predict_m1_endpoint(request: FeatureRequest):
    return predict_m1(request.features)


@app.post("/predict/m2")
def predict_m2_endpoint(request: M2Request):
    return predict_m2(request.features, request.date)


@app.post("/predict/m3")
def predict_m3_endpoint(request: FeatureRequest):
    return predict_m3(request.features)


@app.post("/predict/all")
def predict_all(request: CombinedRequest):
    """Run all three saved models in one request."""
    return {
        "M1": predict_m1(request.m1_features),
        "M2": predict_m2(request.m2_features, request.date),
        "M3": predict_m3(request.m3_features),
    }
