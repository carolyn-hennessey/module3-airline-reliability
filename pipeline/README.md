# Model Pipeline

Trains and evaluates the delay model from the training and test sets in GCS. Part of the project described in the [root README](../README.md). It consumes the output of [preprocessing](../preprocessing/README.md).

## Reproduce this module
Run from the repo root, after preprocessing has uploaded the data (and with `gcloud` set up as described in the [preprocessing README](../preprocessing/README.md)).
```
pip install -r requirements.txt
python pipeline/pipeline.py
```
This writes `pipeline/artifacts/model.joblib` and `pipeline/artifacts/metrics.json`. The [API](../api/README.md) serves that model file.

### Train inside Docker
The root `Dockerfile` can run training, so the pipeline is reproducible without a local Python setup. The container needs Google credentials to download the data; the simplest way is to mount your local application-default credentials (created by `gcloud auth application-default login`) and the artifacts folder so the model lands on your machine:
```
docker build -t airline-delay .
docker run --rm \
  -v "$HOME/.config/gcloud/application_default_credentials.json:/tmp/adc.json:ro" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/adc.json \
  -v "$PWD/pipeline/artifacts:/app/pipeline/artifacts" \
  airline-delay train
```
Rebuild the image afterwards so the new `model.joblib` is baked in for serving. Other commands are `serve`, `preprocess` and `experiments` (see `docker/entrypoint.sh`).

## How it works
`pipeline/pipeline.py` trains and evaluates the delay model. Run it from the repo root with `python pipeline/pipeline.py` (optionally `--config <path>`). Every tunable parameter lives in `pipeline/config.yaml`.

1. **Ingestion:** downloads `train.parquet` and `test.parquet` from `gs://aipi50_module3_chennessey_training_data/` into `data/gcs/` (git-ignored; files already present are reused unless `force_download: true`). This needs the same `gcloud` setup described under Training Set Preprocessing.
2. **Feature engineering:** done in the preprocessing step (`preprocessing/preprocess.py`): date and holiday features, log transforms and the `log_delay_minutes` target. The pipeline only adds model-specific encoding: one-hot for airline and airport type, and cross-fitted target encoding for the high-cardinality `origin`, `dest` and `route`. Unseen airports and routes fall back to the global mean.
3. **Split:** the provided test set is the later time period. The validation set is the most recent `split.validation_fraction` (default 15%) of the training data, so train, validation and test run in chronological order.
4. **Model:** scikit-learn `HistGradientBoostingRegressor` on `log_delay_minutes`, with `learning_rate`, `max_iter`, tree size and regularization set in `config.yaml`.
5. **Evaluation:** RMSE/MAE/R² on the log scale, MAE in real minutes (after `expm1`), and ROC AUC of the predicted delay against `is_delayed`, on train, validation and test, plus a predict-the-mean baseline. Metrics are saved to `pipeline/artifacts/metrics.json` and the model to `pipeline/artifacts/model.joblib`.

**Results** (data built from `flights_sample.csv`; test = Jun-Aug 2026): test RMSE on the log scale is 1.68 vs 1.81 for the predict-the-mean baseline, R² is 0.13, MAE is about 24 minutes (median absolute error 5 minutes), and ROC AUC for delay likelihood is 0.70.

| Feature set | Test AUC | Test R² (log) |
|---|---|---|
| Airline, date, airports | 0.60 | 0.04 |
| + scheduled departure time | 0.70 | 0.13 |

Airport delay history from `daily_airport_stats.csv` (yesterday and trailing-7-day delay and cancel rates for origin and destination, 30 extra columns) was tried and removed: it lifted test AUC only from 0.70 to 0.72 and R² from 0.13 to 0.16, not enough to justify the added dimensionality.

Delays remain hard to predict from schedule information alone (day-of weather and aircraft rotation are not in the features), so treat the model as a risk ranking rather than a precise delay estimate.
