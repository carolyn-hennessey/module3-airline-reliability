# Predicting Airline Reliability 

## Navigating the Repo
`/data` contains the original source data in 5 files:
1. `data/airlines.csv` aggregates delay and cancellation data at the airline level
2. `data/airports.csv` aggregates delay and cancellation data at the airport level
3. `data/daily_airport_stats.csv` provides a daily snapshot of flight delay/cancellation/departure aggregates by day by airport
4. `data/flights_sample.csv` contains the dataset of departure/delay/cancellatio data aggregated at the level of the individual flights


## Source Data Analysis
Initial exploratory analysis lives in `preprocessing/preprocess.py`, which loads each raw file in `/data` into a pandas DataFrame and prints summaries (run `python preprocessing/preprocess.py` from the repo root).

### Dataset overview
| File | Rows x Cols | Grain | Notes |
|---|---|---|---|
| `airlines.csv` | 14 x 7 | airline | ~11.7M total flights; no nulls or duplicates |
| `airports.csv` | 361 x 11 | airport | 99 large, 259 medium, 3 small; 52 states/territories; no nulls |
| `daily_airport_stats.csv` | 202,872 x 13 | airport-day | 361 airports; (date, airport) is unique |
| `flights_sample.csv` | 500,000 x 40 | flight | ~4% sample of the full flight-level data |

All flight-level data covers **2025-01-01 to 2026-08-31**. Keys are consistent across files: every airline code and every origin/destination airport in the flight sample and daily stats appears in `airlines.csv` / `airports.csv`.

### Key findings
- **Overall reliability (flight sample):** 22.7% of flights depart 15+ minutes late, 22.8% arrive 15+ minutes late, 1.8% are cancelled and 0.3% are diverted.
- **Airline differences are meaningful.** Arrival-delay rates range from 17.3% (Hawaiian) to ~28% (Frontier, PSA). Cancellation rates range from 0.6% (Allegiant) to 4.9% (PSA) and 4.2% (Republic). Large carriers like Delta, United and Alaska sit near 20-22% late arrivals.
- **Delays have strong time patterns.**
  - Time of day: 9% late for departures before 6am, rising steadily to ~33% in the 8-9pm block.
  - Season: peaks in June/July (27-29%) and December (27%); lowest in September (17%).
  - Day of week: Sunday (26%) and Monday (25%) are worst; Tuesday (19%) and Wednesday (20%) are best.
- **Delay causes** (share of delay minutes on delayed flights): late-arriving aircraft 39%, carrier 33%, NAS 21%, weather 7%, security <1%. Weather, however, drives most cancellations (5,720 of 8,884 cancelled flights in the sample).
- **Daily airport stats** are highly skewed: median airport-day has 9 scheduled departures (max 1,161), and `dep_delay_mean` ranges from -65 to 1,614 minutes, so small-airport days are noisy.

### Data quality notes and caveats
- **Structural nulls, not errors.** ~1.8-2.1% of flight-level timing/delay fields are null because the flight was cancelled or diverted. The five delay-cause columns (`carrier_delay`, etc.) are null for 77.7% of rows - i.e. any flight that was not 15+ minutes late - and should be filled with 0 or handled explicitly. `cancellation_code`/`cancellation_reason` are null for the 98.2% of flights that were not cancelled.
- **Daily stats nulls:** ~0.4% of airport-days have null delay rates (all flights cancelled that day).
- **Possible label oddities:** 398 cancelled flights in the sample still have a `dep_time` (likely cancelled after pushback), and one flight has a non-positive `sched_elapsed_time`.
- **Heavy-tailed delays:** `dep_delay` and `arr_delay` reach 3,400+ minutes, so use robust statistics or clip when modeling.
- **Time fields are HHMM integers** (e.g. `25` = 00:25, `2400` appears in actuals) rather than timestamps and need conversion.
- **Incomplete airline coverage:** Hawaiian's last flight is 2025-12-31 and Spirit's is 2026-05-01 (the other 12 run through 2026-08-31), so these carriers have shorter histories.
- **Non-contiguous US airports:** `airports.csv` includes territories (latitudes down to -14, longitudes up to +145), which may need handling for geographic features.
- **Leakage risk:** `arr_delay`, `arr_del15`, the delay-cause columns, `cancelled` and `diverted` are only known after a flight, so they must be excluded from features when predicting reliability in advance.



## Training Set Preprocessing
`preprocessing/preprocess.py` also builds the modeling table (`build_training_set`) and writes `data/processed/train.parquet` and `data/processed/test.parquet` (git-ignored; regenerate with `python preprocessing/preprocess.py`). It currently reads `flights_sample.csv` and will use the full `flights` data if it is loaded.

- **Unit and filtering:** one row per flight. Cancelled and diverted flights are dropped (they have no delay minutes), leaving 489,636 rows.
- **Target:** `delay_minutes` is the worse of departure and arrival delay, clipped at 0, so any delay cause counts. 52% of flights have zero delay minutes. `is_delayed` flags 15+ minutes (27.2% of flights).
- **Log transform:** the raw target is extremely right-skewed (skewness 10.9). `log_delay_minutes = log1p(delay_minutes)` brings skewness to 0.76 and is the recommended regression target. Predictions can be mapped back with `expm1`. `distance` (1.4 to -0.35) and airport volume are also log-transformed.
- **Features (all known before departure):**
  - Airline and route: `airline_code`, `origin`, `dest`, `route`
  - Date: `month`, `day_of_week`, `day_of_month`, `week_of_year`, `is_weekend`, `days_to_nearest_holiday`, `near_holiday`
  - Scheduled departure time: `sched_dep_hour`, `sched_dep_minutes` (since midnight), and `sched_dep_sin`/`sched_dep_cos` (cyclical, so 23:59 sits next to 00:00)
  - Distance: `distance`, `log_distance`
  - Airport attributes from `airports.csv`: latitude, longitude, elevation, airport type, log departure volume
- **Excluded to avoid leakage:** actual times, taxi/air time, arrival fields and the delay-cause columns.
- **Split:** time-based, so the model is tested on data later than anything it trained on. Train is 2025-01-01 to 2026-05-31 (413,112 rows) and test is 2026-06-01 to 2026-08-31 (76,524 rows).
- **Still to do at modeling time:** one-hot or target-encode the categorical columns (`route` has high cardinality). Four test-set origins (ERI, LNS, DJT, HEZ) never appear in train, so the encoder must handle unseen categories.
- **Upload to GCS:** the last step of `preprocess.py` uploads `train.parquet` and `test.parquet` to `gs://aipi50_module3_chennessey_training_data/`. It uses Application Default Credentials, so set up access first:
  1. Install the Google Cloud CLI (`gcloud`): on macOS run `brew install --cask gcloud-cli`, or follow https://cloud.google.com/sdk/docs/install for other platforms. Confirm with `gcloud --version`.
  2. Run `gcloud auth application-default login` and sign in with an account that can write to the bucket. (Alternatively, set `GOOGLE_APPLICATION_CREDENTIALS` to a service-account key file.)
  3. Re-run `python preprocessing/preprocess.py`. Without credentials the script still writes the local files, then exits with a message.


## Model Pipeline
`pipeline/pipeline.py` trains and evaluates the delay model. Run it from the repo root with `python pipeline/pipeline.py` (optionally `--config <path>`). Every tunable parameter lives in `pipeline/config.yaml`.

1. **Ingestion:** downloads `train.parquet` and `test.parquet` from `gs://aipi50_module3_chennessey_training_data/` into `data/gcs/` (git-ignored; files already present are reused unless `force_download: true`). This needs the same `gcloud` setup described under Training Set Preprocessing.
2. **Feature engineering:** done in the preprocessing step (`preprocessing/preprocess.py`): date and holiday features, log transforms and the `log_delay_minutes` target. The pipeline only adds model-specific encoding: one-hot for airline and airport type, and cross-fitted target encoding for the high-cardinality `origin`, `dest` and `route`. Unseen airports and routes fall back to the global mean.
3. **Split:** the provided test set is the later time period. The validation set is the most recent `split.validation_fraction` (default 15%) of the training data, so train, validation and test run in chronological order.
4. **Model:** scikit-learn `HistGradientBoostingRegressor` on `log_delay_minutes`, with `learning_rate`, `max_iter`, tree size and regularization set in `config.yaml`.
5. **Evaluation:** RMSE/MAE/R² on the log scale, MAE in real minutes (after `expm1`), and ROC AUC of the predicted delay against `is_delayed`, on train, validation and test, plus a predict-the-mean baseline. Metrics are saved to `pipeline/artifacts/metrics.json` and the model to `pipeline/artifacts/model.joblib`.

Setup: `pip install pandas pyarrow scikit-learn pyyaml joblib google-cloud-storage`.

**Results** (data built from `flights_sample.csv`; test = Jun-Aug 2026): test RMSE on the log scale is 1.68 vs 1.81 for the predict-the-mean baseline, R² is 0.13, MAE is about 24 minutes (median absolute error 5 minutes), and ROC AUC for delay likelihood is 0.70.

| Feature set | Test AUC | Test R² (log) |
|---|---|---|
| Airline, date, airports | 0.60 | 0.04 |
| + scheduled departure time | 0.70 | 0.13 |

Airport delay history from `daily_airport_stats.csv` (yesterday and trailing-7-day delay and cancel rates for origin and destination, 30 extra columns) was tried and removed: it lifted test AUC only from 0.70 to 0.72 and R² from 0.13 to 0.16, not enough to justify the added dimensionality.

Delays remain hard to predict from schedule information alone (day-of weather and aircraft rotation are not in the features), so treat the model as a risk ranking rather than a precise delay estimate.


## Experiment Tracking (MLflow)
Experiments are tracked with MLflow in `experiments/`, separately from the pipeline: `pipeline/pipeline.py` is unchanged and has no MLflow dependency.

- `experiments/track_experiments.py` imports the pipeline's building blocks (GCS download, chronological split, model, evaluation), so each tracked run is identical to a normal pipeline run. It then logs, per run: the model parameters, split and data settings, train/validation/test metrics (log-scale RMSE/MAE/R², MAE in minutes, ROC AUC), the baseline RMSE, the fitted model, and the resolved config.
- `experiments/experiments.yaml` lists the runs. Each run is a name plus dotted-path overrides of `pipeline/config.yaml` (e.g. `model.learning_rate: 0.05`), so adding an experiment means adding a few lines of YAML. It also sets the MLflow experiment name and storage locations.
- Run metadata is stored in `experiments/mlflow.db` (SQLite) and models in `experiments/mlartifacts/`; both are git-ignored.

```
pip install mlflow
python experiments/track_experiments.py
mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db   # then open http://localhost:5000
```

The six default runs (learning rate, tree size and regularization variations) all land within 0.004 of each other on test ROC AUC (0.701-0.704), so the model is insensitive to these hyperparameters and the baseline config is fine.


## Prediction API (FastAPI)
`api/main.py` serves the trained model. It loads `pipeline/artifacts/model.joblib` once at startup (override the location with the `MODEL_PATH` environment variable), so run `python pipeline/pipeline.py` first.

```
pip install fastapi uvicorn
uvicorn api.main:app --reload      # interactive docs at http://127.0.0.1:8000/docs
```

**`POST /predict`** takes raw flight details and applies the same feature engineering used in training (`engineer_features` in `preprocessing/preprocess.py`, shared so there is no train/serve skew):

```
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "airline_code": "DL", "flight_date": "2026-07-04",
  "origin": "ATL", "dest": "JFK", "sched_dep_time": 1745
}'
```
```
{"predicted_delay_minutes": 7.6, "predicted_log_delay_minutes": 2.1492, "airline": "Delta Air Lines"}
```

| Field | Notes |
|---|---|
| `airline_code` | IATA code, must appear in `airlines.csv` (case-insensitive) |
| `flight_date` | `YYYY-MM-DD` |
| `origin`, `dest` | IATA codes from `airports.csv`; must differ |
| `sched_dep_time` | local time as HHMM, 0-2359 (e.g. `1745`) |
| `distance` | optional miles; estimated from airport coordinates if omitted |

Invalid or unknown inputs return HTTP 422 with an explanation. `GET /health` reports whether the model is loaded.

The response is the model's expected delay in minutes (`expm1` of the log-scale prediction, floored at 0). It is a risk-ranking estimate, not a calibrated probability, and predictions are typically small because most flights have little or no delay.
