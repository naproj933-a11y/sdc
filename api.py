"""
SmogLens Prediction API
=======================

Run:
    pip install -r requirements.txt
    uvicorn api:app --host 0.0.0.0 --port $PORT

The Swagger UI (/docs) includes real-data example request bodies for
22 November 2026. These examples are built from the CSV data bundled with
the project. Fire-derived values are null because the FIRMS CSV ends on
31 August 2026; null means unavailable, not zero fires.

Keep this file beside:
    M1_PM25_Regression.joblib
    M2_Weather.joblib
    M3_Fire_Weather_Regime_Clustering.joblib
"""

from pathlib import Path
from datetime import date as DateType
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
            "Keep api.py and all three .joblib files in the same directory."
        )
    try:
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {name}. Install the same scikit-learn version "
            f"used during training. Original error: {exc}"
        ) from exc


# Load the trained models once when the service starts.
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


# ---------------------------------------------------------------------------
# Example values calculated from the supplied CSVs for 22 November 2026.
# ---------------------------------------------------------------------------

M1_EXAMPLE = {
    "pm25": 112.9,
    "pm10": 203.5,
    "no2": 49.3,
    "o3": 20.9,
    "so2": 12.6,
    "co": 2.1,
    "delhi_temperature_2m_mean": 19.8,
    "delhi_wind_speed_10m_max": 6.4,
    "delhi_wind_gusts_10m_max": 13.7,
    "delhi_wind_direction_10m_dominant": 321.0,
    "delhi_hourly_temperature_2m": 19.8,
    "delhi_hourly_relative_humidity_2m": 71.1667,
    "delhi_hourly_dew_point_2m": 14.025,
    "delhi_hourly_apparent_temperature": 20.6167,
    "delhi_hourly_pressure_msl": 1019.3833,
    "delhi_hourly_surface_pressure": 992.3625,
    "delhi_hourly_cloud_cover": 2.625,
    "delhi_hourly_vapour_pressure_deficit": 0.74375,
    "delhi_hourly_wind_speed_10m": 3.3042,
    "delhi_hourly_wind_direction_10m": 249.7917,
    "delhi_hourly_wind_gusts_10m": 8.825,
    "delhi_hourly_boundary_layer_height": 353.3333,
    "delhi_hourly_shortwave_radiation": 149.5417,
    "region_temperature_2m_mean": 17.9,
    "region_wind_speed_10m_max": 8.9,
    "region_wind_gusts_10m_max": 19.4,
    "region_wind_direction_10m_dominant": 342.0,
    "region_hourly_temperature_2m": 17.9542,
    "region_hourly_relative_humidity_2m": 69.2083,
    "region_hourly_dew_point_2m": 11.5792,
    "region_hourly_apparent_temperature": 17.575,
    "region_hourly_pressure_msl": 1019.8542,
    "region_hourly_surface_pressure": 993.0042,
    "region_hourly_cloud_cover": 0.0,
    "region_hourly_vapour_pressure_deficit": 0.7521,
    "region_hourly_wind_speed_10m": 5.8833,
    "region_hourly_wind_direction_10m": 226.3333,
    "region_hourly_wind_gusts_10m": 11.9792,
    "region_hourly_boundary_layer_height": 286.875,
    "region_hourly_shortwave_radiation": 148.0417,
    "fire_count": None,
    "fire_frp": None,
    "pm25_today": 112.9,
    "pm25_lag1": 125.9,
    "pm25_lag2": 134.6,
    "pm25_lag3": 139.7,
    "pm25_lag7": 112.3,
    "pm25_roll3": 124.4667,
    "pm25_roll7": 127.6143,
    "pm25_trend3": -26.8,
    "pm10_lag1": 231.7,
    "no2_lag1": 53.0,
    "fire_count_lag1": None,
    "fire_count_lag2": None,
    "fire_frp_lag1": None,
    "fire_frp_lag2": None,
    "month": 11,
    "day_of_week": 6,
    "day_of_year": 326,
    "day_sin": -0.62505,
    "day_cos": 0.780585,
}

M2_EXAMPLE = {
    "ws_mean": 3.3042,
    "ws_max": 6.4,
    "wd_sin": -0.938443,
    "wd_cos": -0.345435,
    "wd_dom_sin": -0.62932,
    "wd_dom_cos": 0.777146,
    "blh_mean": 353.3333,
    "blh_daytime": 602.6923,
    "ventilation": 1167.4722,
    "temp_mean": 19.8,
    "temp_range": 9.3,
    "rh_mean": 71.1667,
    "dewpoint_mean": 14.025,
    "dewpoint_depression": 5.775,
    "vpd_mean": 0.74375,
    "pressure_mean": 1019.3833,
    "cloud_mean": 2.625,
    "upwind_ws_mean": 5.8833,
    "upwind_blh_mean": 286.875,
    "upwind_temp_mean": 17.9542,
    "upwind_rh_mean": 69.2083,
    "upwind_wd_sin": -0.723369,
    "upwind_wd_cos": -0.690462,
    "nw_flow": 0.0,
    "sin_doy": -0.62505,
    "cos_doy": 0.780585,
    "sin_doy2": -0.975809,
    "cos_doy2": 0.218626,
    "trend_norm": 1.0,
    # The fire archive ends on 31 August 2026. These are unavailable,
    # so null is passed and the model's fitted imputer handles them.
    "fire_count_lag0": None,
    "fire_count_lag1": None,
    "fire_count_lag2": None,
    "fire_count_lag3": None,
    "frp_sum_lag0": None,
    "frp_sum_lag1": None,
    "frp_sum_lag2": None,
    "frp_sum_lag3": None,
    "frp_roll3": None,
    "frp_roll5": None,
    "frp_trend": None,
    "transport_align": None,
}

M3_EXAMPLE = {
    "log_fire_frp_3d": None,
    "delhi_wind_kmh": 3.3042,
    "delhi_pblh_m": 353.3333,
    "delhi_humidity_pct": 71.1667,
    "delhi_temperature_c": 19.8,
}


app = FastAPI(
    title="SmogLens Prediction API",
    description=(
        "Prediction API for M1 next-day PM2.5 regression, "
        "M2 weather-adjusted pollution analysis, and M3 fire-weather "
        "regime clustering. Open /docs and use the example request bodies."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend URL before production.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request schemas. OpenAPI examples are added below so Swagger displays them.
# ---------------------------------------------------------------------------

class FeatureRequest(BaseModel):
    features: Dict[str, Optional[float]] = Field(default_factory=dict)


class M2Request(BaseModel):
    date: DateType = Field(description="Input/observation date, YYYY-MM-DD.")
    features: Dict[str, Optional[float]] = Field(default_factory=dict)


class CombinedRequest(BaseModel):
    date: DateType = Field(description="Input/observation date, YYYY-MM-DD.")
    m1_features: Dict[str, Optional[float]] = Field(default_factory=dict)
    m2_features: Dict[str, Optional[float]] = Field(default_factory=dict)
    m3_features: Dict[str, Optional[float]] = Field(default_factory=dict)


def make_feature_frame(
    supplied: Dict[str, Optional[float]],
    expected: list[str],
    model_name: str,
) -> pd.DataFrame:
    """Build a one-row DataFrame in the exact order used during training."""
    unknown = sorted(set(supplied) - set(expected))
    if unknown:
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"Unknown feature(s) supplied for {model_name}.",
                "unknown_features": unknown,
                "hint": "Use GET /metadata to see the exact feature names.",
            },
        )

    row = {}
    for feature in expected:
        value = supplied.get(feature)
        row[feature] = np.nan if value is None else float(value)

    return pd.DataFrame([row], columns=expected)


def check_minimum_inputs(
    supplied: Dict[str, Optional[float]],
    required_groups: list[list[str]],
    model_name: str,
):
    """Reject an entirely empty or irrelevant input dictionary."""
    provided = {key for key, value in supplied.items() if value is not None}
    if not any(provided.intersection(group) for group in required_groups):
        raise HTTPException(
            status_code=422,
            detail=f"Provide at least one valid, non-null input feature for {model_name}.",
        )


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
                "Approximate interval based on saved training residual offsets; "
                "not a guarantee."
            ),
        },
        "target": "Next-day PM2.5 concentration (µg/m³)",
    }


def predict_m2(features: Dict[str, Optional[float]], input_date: DateType) -> dict:
    check_minimum_inputs(
        features,
        [
            ["ws_mean", "ws_max", "blh_mean", "temp_mean", "rh_mean"],
            ["fire_count_lag0", "frp_sum_lag0", "fire_count_lag1", "frp_sum_lag1"],
        ],
        "M2",
    )

    # The W model receives weather columns only. WF receives weather + fire.
    w_values = {key: value for key, value in features.items() if key in M2_W_FEATURES}
    wf_values = {key: value for key, value in features.items() if key in M2_WF_FEATURES}

    w_frame = make_feature_frame(w_values, M2_W_FEATURES, "M2 weather model")
    wf_frame = make_feature_frame(wf_values, M2_WF_FEATURES, "M2 weather + fire model")

    weather_expected = float(M2["models"]["W_model"].predict(w_frame)[0])
    weather_fire_expected = float(M2["models"]["WF_model"].predict(wf_frame)[0])

    month_key = int(input_date.month)
    low_fire_ref = M2["parameters"]["low_fire_ref"]
    month_reference = low_fire_ref.get(month_key, low_fire_ref.get(str(month_key)))

    # Do not calculate a fire-associated excess when the current fire history
    # is unavailable. A median-imputed WF prediction is not evidence of a
    # measured fire contribution.
    fire_inputs_complete = all(
        features.get(name) is not None for name in M2_WF_EXTRA
    )
    low_fire_prediction = None
    fire_excess = None

    if month_reference is not None and fire_inputs_complete:
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
        "model": "M2 - Weather-adjusted pollution analysis",
        "input_date": input_date.isoformat(),
        "weather_expected_pm25_ug_m3": round(max(0.0, weather_expected), 2),
        "weather_and_fire_model_estimate_pm25_ug_m3": round(
            max(0.0, weather_fire_expected), 2
        ),
        "fire_features_complete": fire_inputs_complete,
        "weather_fire_estimate_note": (
            "All fire-history inputs were supplied."
            if fire_inputs_complete else
            "Some or all fire-history inputs were unavailable and median-imputed; "
            "do not interpret this as a measured fire-adjusted estimate."
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
            "The weather model estimates a meteorological baseline. "
            "The weather + fire model includes recent fire-history features. "
            "The fire-associated excess is an exploratory model difference, "
            "not proof of source attribution."
        ),
        "data_quality_note": (
            "The supplied fire CSV ends on 2026-08-31. Null fire inputs are "
            "imputed by the saved pipeline and should not be interpreted as "
            "zero observed fires."
        ),
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

    # The first training feature is log_fire_frp_3d. The K-Means centers
    # are in standardized feature space, so the largest first coordinate
    # identifies the more fire-active cluster for this two-cluster model.
    kmeans = M3.named_steps.get("kmeans")
    if kmeans is None:
        raise HTTPException(
            status_code=500,
            detail="M3 artifact does not contain the expected 'kmeans' pipeline step.",
        )

    fire_feature_available = features.get("log_fire_frp_3d") is not None
    if kmeans.n_clusters == 2 and fire_feature_available:
        fire_active_cluster = int(np.argmax(kmeans.cluster_centers_[:, 0]))
        regime_label = (
            "Warmer / fire-active / better mixed"
            if cluster_id == fire_active_cluster
            else "Cooler / humid / lower ventilation"
        )
    elif kmeans.n_clusters == 2:
        regime_label = f"Cluster {cluster_id} (fire feature imputed; interpretation limited)"
    else:
        regime_label = f"Regime cluster {cluster_id}"

    return {
        "model": "M3 - Fire-weather regime clustering",
        "algorithm": "K-Means",
        "cluster_id": cluster_id,
        "regime": regime_label,
        "input_features": M3_FEATURES,
        "note": (
            "This is an environmental grouping, not a PM2.5 forecast or "
            "proof that fires caused pollution."
        ),
        "data_quality_note": (
            "The fire CSV ends on 2026-08-31. The 3-day fire feature is null "
            "for 2026-11-22 and is imputed by the saved pipeline."
        ),
    }


@app.get("/")
def root():
    return {
        "name": "SmogLens Prediction API",
        "status": "running",
        "docs": "/docs",
        "metadata": "/metadata",
        "example_inputs": "/example-inputs",
        "available_predictions": [
            "/predict/m1",
            "/predict/m2",
            "/predict/m3",
            "/predict/all",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": {"M1": True, "M2": True, "M3": True},
    }


@app.get("/metadata")
def metadata():
    return {
        "example_date": "2026-11-22",
        "data_coverage": {
            "air_quality_latest_date": "2026-11-22",
            "weather_latest_date": "2026-11-22",
            "fire_latest_date": "2026-08-31",
            "fire_note": "Fire features after 2026-08-31 are unavailable, not zero.",
        },
        "M1": {
            "endpoint": "/predict/m1",
            "description": "Predict next-day PM2.5 from the 61 training features.",
            "feature_count": len(M1_FEATURES),
            "features": M1_FEATURES,
        },
        "M2": {
            "endpoint": "/predict/m2",
            "description": "Estimate weather baseline and weather-plus-fire PM2.5.",
            "weather_feature_count": len(M2_W_FEATURES),
            "weather_features": M2_W_FEATURES,
            "weather_fire_feature_count": len(M2_WF_FEATURES),
            "weather_fire_features": M2_WF_FEATURES,
            "fire_features": M2_WF_EXTRA,
        },
        "M3": {
            "endpoint": "/predict/m3",
            "description": "Assign a day to a fire-weather regime.",
            "features": M3_FEATURES,
        },
    }


@app.get("/example-inputs")
def example_inputs():
    """Return copy-pasteable examples for all prediction routes."""
    return {
        "example_date": "2026-11-22",
        "note": (
            "Examples use the 22 November 2026 CSV observations. Fire inputs "
            "are null because the fire archive ends on 31 August 2026."
        ),
        "m1": {"features": M1_EXAMPLE},
        "m2": {"date": "2026-11-22", "features": M2_EXAMPLE},
        "m3": {"features": M3_EXAMPLE},
        "all": {
            "date": "2026-11-22",
            "m1_features": M1_EXAMPLE,
            "m2_features": M2_EXAMPLE,
            "m3_features": M3_EXAMPLE,
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
    return {
        "M1": predict_m1(request.m1_features),
        "M2": predict_m2(request.m2_features, request.date),
        "M3": predict_m3(request.m3_features),
    }


# Add concrete request examples to OpenAPI. Swagger UI displays these in
# "Example Value" instead of showing only a generic empty dictionary.
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    examples = {
        "/predict/m1": {
            "summary": "M1 example — CSV values for 22 Nov 2026",
            "description": "Predicts next-day PM2.5. Fire fields are null because the fire CSV does not cover this date.",
            "value": {"features": M1_EXAMPLE},
        },
        "/predict/m2": {
            "summary": "M2 example — CSV values for 22 Nov 2026",
            "description": "Weather inputs are from 22 Nov 2026. Fire-history inputs are null because the archive ends 31 Aug 2026.",
            "value": {"date": "2026-11-22", "features": M2_EXAMPLE},
        },
        "/predict/m3": {
            "summary": "M3 example — CSV values for 22 Nov 2026",
            "description": "Weather values are from 22 Nov 2026; the 3-day fire feature is unavailable.",
            "value": {"features": M3_EXAMPLE},
        },
        "/predict/all": {
            "summary": "All models — CSV values for 22 Nov 2026",
            "description": "Runs M1, M2 and M3 together using the same input date.",
            "value": {
                "date": "2026-11-22",
                "m1_features": M1_EXAMPLE,
                "m2_features": M2_EXAMPLE,
                "m3_features": M3_EXAMPLE,
            },
        },
    }

    for path, example in examples.items():
        operation = schema.get("paths", {}).get(path, {}).get("post")
        if operation:
            operation.setdefault("requestBody", {}).setdefault(
                "content", {}
            ).setdefault("application/json", {})["examples"] = {
                "csv_2026_11_22": example
            }

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi
