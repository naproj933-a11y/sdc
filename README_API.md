# SDC FastAPI model testing

## Contents
- `api.py`: FastAPI service for M1 regression, M2 classification and M4 station clustering.
- Existing `.joblib` model artifacts and original `data/` CSV files are required and are included in this project.

## Setup (Windows PowerShell)
```powershell
cd SDC
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-api.txt
uvicorn api:app --reload
```
Open `http://127.0.0.1:8000/docs` for interactive API testing.

## Endpoints
- `GET /health` — confirm all models loaded.
- `GET /schemas` — exact feature names required by each model.
- `POST /test/m1` — individual M1 prediction. JSON: `{"features": {"pm25_lag0": 120, ...}}`. Include other M1 fields where known; pipeline median imputation handles remaining optional missing values.
- `POST /test/m2` — individual M2 prediction. JSON: `{"features": {"pm25": 120, ...}}`. Use `/schemas` for all 88 feature names.
- `POST /test/m4` — individual M4 station clustering. JSON: `{"stations": [{"Station Name": "Station A", "Mean PM2.5 Anomaly (vs other stations)": 1.2, "PM2.5 Exceedance Share (%)": 20, "Weekday-Weekend Gap (ug/m3)": 3, "Mean High-Pollution Episode (days)": 2}]}`.
- `POST /test/combined` — uses the latest available engineered dataset rows for M1 and M2 and full station history for M4.
- `POST /test/random-day?seed=42` — randomly chooses an eligible historical origin day, predicts the following day and compares with actual recorded PM2.5/class.

## Notes
1. M1 needs `pm25_lag0`; its model bundle supplies the trained model, feature list and interval calibration.
2. M2 uses the 88-column feature order embedded in the saved sklearn pipeline. `/schemas` returns it.
3. M4 uses four station-level features, not daily weather fields.
4. Random-day is a retrospective backtest, not a real future forecast. The M1 notebook constructs next-day weather inputs from observed next-day weather, so the backtest labels that limitation explicitly.
5. The notebook artifacts were serialized with scikit-learn 1.9.1. The requirements pin that version to reduce pickle compatibility problems. Use a compatible Python version supported by that release.

## Live date-based prediction (Delhi four zones)

The new endpoint is `POST /predict/live`. It accepts a date and automatically retrieves weather and CAMS air-quality model estimates for the four Delhi zone coordinates used by this project.

Example Postman request:

```http
POST http://127.0.0.1:8000/predict/live
Content-Type: application/json
```

```json
{
  "date": "2026-11-20"
}
```

The date must be within the provider's available window: yesterday through four days ahead (Delhi local date). The response includes the requested date's weather, M1 next-day PM2.5 prediction, M2 air-quality class prediction, and M4 historical station-profile clusters.

**Important model limitation:** the trained models were fitted on the project's historical CPCB/weather/fire data. The live endpoint uses Open-Meteo weather forecasts and CAMS gridded air-quality estimates, which are not CPCB station observations. Several training features (for example, some pollutants, Punjab upwind-wind features and fire-history features) are unavailable from this public feed and are filled by the trained model's imputer. The response lists these fields. Treat the result as an experimental integration/demo, not a validated operational forecast. For reliable live accuracy, retrain and validate the models using the same live data source and feature definitions.

M4 remains a historical station-profile clustering output; it does not change by date.
