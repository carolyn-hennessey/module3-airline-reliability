# Predicting Airline Reliability

How late could my flight be? This project trains a regression model on roughly 500,000 US domestic flights (Jan 2025 - Aug 2026) to predict the delay of a flight using pre-departure data: the airline, the date, the scheduled departure time and the two airports. The model is served through a FastAPI HTTP backend hosted in GCP Cloud Run and easily interfaced using a Streamlit Community Cloud hosted frontend with a small web form.

## Live services
- [API SwaggerUI](https://airline-delay-api-170279949235.us-east1.run.app/docs)
- [Front end](https://aipi51-module3-airline-reliability-carolynh.streamlit.app/) ([deployment steps](frontend/README.md))

**Model Type:** regression. Due to the large number of row, the ScikitLearn `HistGradientBoostingRegressor` was selected, specifically.  **Target:** `log1p(delay minutes)`, where delay is the larger of a flight's departure and arrival delay, floored at 0. **Evaluation:** RMSE on the log scale (against a predict-the-mean baseline), plus ROC AUC of the predicted delay against "delayed 15+ minutes" as an easier-to-read ranking metric. Ultimately, this model performs only subtly better than always esitmating the mean value of `log1p(delay minutes)` due to the low frequency of this event (roughly 23% of all flights were delayed), but since the goal of this project was not to train the best model, this poor performance was accepted.

## Project Overview

```
data/*.csv ──► preprocessing ──► GCS bucket ──► pipeline ──► model.joblib ──► API (Cloud Run) ◄── Streamlit app
 (raw)        EDA + features     train/test     train +        in the          FastAPI          (Community Cloud)
                                  parquet       evaluate       Docker image
                                                    │
                                          experiments (MLflow tracking)
```

## Reproducing each module
Each module has its own README with setup and run instructions. Run them in this order, always from the repo root.

| Step | Module | What it does | Instructions |
|---|---|---|---|
| 1 | `preprocessing/` | Explores the raw data, engineers features, builds train/test sets and uploads them to GCS | [preprocessing/README.md](preprocessing/README.md) |
| 2 | `pipeline/` | Downloads the data from GCS, trains and evaluates the model | [pipeline/README.md](pipeline/README.md) |
| 3 | `experiments/` | Tracks and compares model variations with MLflow | [experiments/README.md](experiments/README.md) |
| 4 | `api/` (+ `Dockerfile`) | FastAPI backend serving the model; Docker and Cloud Run deployment | [api/README.md](api/README.md) |
| 5 | `frontend/` | Streamlit form that calls the API | [frontend/README.md](frontend/README.md) |

Cloud setup (installing `gcloud` and signing in) is described once, in the [preprocessing README](preprocessing/README.md). Install the Python dependencies with `pip install -r requirements.txt` (Python 3.10). Alternatively, the root `Dockerfile` runs training and serving in a container (`docker run IMAGE train` / `serve`; see the [pipeline](pipeline/README.md#train-inside-docker) and [API](api/README.md) READMEs).

## The data
**Source:** [Kaggle: US Flight Delays & Cancellations 2025-2026](https://www.kaggle.com/datasets/samuelpombourcq/us-flight-delays-and-cancellations-2025-2026)

The raw files in `data/` cover US domestic flights from 2025-01-01 to 2026-08-31:

| File | Grain | Rows | Used for |
|---|---|---|---|
| `flights_sample.csv` | one row per flight | 500,000 (~4% sample of ~11.7M flights) | Training and test data |
| `airports.csv` | one row per airport | 361 | Airport attributes (location, elevation, type, volume) |
| `airlines.csv` | one row per airline | 14 | Airline names and validation |
| `daily_airport_stats.csv` | one row per airport per day | 202,872 | Explored; airport history features were tried and dropped due to large increase in dimensionality and low AUC lift |


What the exploration found, in brief (full detail in the [preprocessing README](preprocessing/README.md)):
- About 23% of flights arrive 15+ minutes late, 1.8% are cancelled and 0.3% are diverted.
- Delays vary a lot by airline (17% to 28% late), by time of day (9% before 6am vs 33% at 8pm), by season (summer and December worst) and by day of week.
- Delay minutes are extremely right-skewed, and 52% of flights have no delay at all.
- Late-arriving aircraft (39%) and carrier problems (33%) cause most delay minutes; weather causes most cancellations.

## The model
- **Task:** regression. The target is the larger of a flight's departure and arrival delay, floored at zero, so any cause of delay counts. Cancelled and diverted flights are excluded because they have no delay minutes.
- **Target transform:** the target is modeled as `log1p(delay minutes)`, which cuts skewness from 10.9 to 0.76. Predictions are converted back to minutes with `expm1`.
- **Features (all known before departure):** airline, origin, destination, route, date and holiday features, scheduled departure time (encoded cyclically), distance and airport attributes. Post-flight columns are excluded to avoid leakage.
- **Algorithm:** scikit-learn `HistGradientBoostingRegressor`: a simple, fast tree ensemble that copes with skewed data, mixed feature types and missing values. High-cardinality categoricals (origin, destination, route) use cross-fitted target encoding; airline and airport type are one-hot encoded.
- **Evaluation:** a time-based split. Train is Jan 2025 - May 2026, with the last 15% of it held out for validation, and test is Jun - Aug 2026, so the model is always scored on flights later than anything it trained on.

| Feature set | Test ROC AUC (delayed 15+ min) | Test R² (log scale) |
|---|---|---|
| Airline, date, airports | 0.60 | 0.04 |
| + scheduled departure time (final model) | **0.70** | **0.13** |
| + 30 airport delay-history features (not kept) | 0.72 | 0.16 |

The final model has a log-scale test RMSE of 1.68 against 1.81 for always predicting the mean, with a median absolute error of about 5 minutes. Delays are hard to predict from schedule information alone, since weather and aircraft rotation aren't known in advance.

## Decision Record

Here I would note that most of my decisions were driven by past experience and ergonomics where relevant (GCP, FastAPI and Streamlit).

| Decision | Choice | Why |
|---|---|---|
| Cloud provider | **Google Cloud Platform** | One provider for the whole path: **Cloud Storage** hands the training and test sets from preprocessing to the pipeline, **Cloud Build** builds the container, and **Cloud Run** serves the API. Cloud Run scales to zero, so the API costs nothing while idle. |
| Experiment tracking | **MLflow** | Open source and runs locally with no server (SQLite plus a local artifact folder). It records the parameters, metrics and model of every run so variations can be compared. It lives in `experiments/`, separate from the pipeline, so training has no dependency on it. |
| Backend | **FastAPI** | Typed request validation (bad airlines, airports and times return clear 422 errors), automatic interactive docs at `/docs`, and a very small amount of code. It reuses the training feature code, so there's no train/serve skew. |
| Front end | **Streamlit on Community Cloud** | A form page is a few dozen lines of Python, and Community Cloud hosts it free straight from the GitHub repo. It calls the Cloud Run API over HTTP, so the two deploy independently. |
| Model choice | Large dataset with regression problem-type | I did not spend much time on this since it was noted as not the point of the assignment. Simple, fast to train. |
| Airport delay-history features | Tried, then dropped | They raised test AUC only from 0.70 to 0.72 while adding 30 columns, which didn't justify the extra complexity per our previous Module. |
| Containerized model | Model baked into the Docker image | The simplest deployment: the image is self-contained and starts fast. |

## Cloud services used
| Service | Role |
|---|---|
| **Google Cloud Storage** | Hosts the cleaned training and test sets (`gs://aipi50_module3_chennessey_training_data`). Preprocessing uploads them and the pipeline downloads them. |
| **Google Cloud Build** | Builds the container image from the `Dockerfile` during `gcloud run deploy --source .`. |
| **Google Artifact Registry** | Stores the built image (the `cloud-run-source-deploy` repository, created automatically). |
| **Google Cloud Run** | Hosts the FastAPI service as a public HTTPS endpoint that scales to zero. |
| **Streamlit Community Cloud** | Hosts the web front end, deployed from this GitHub repo. |

## Ethics and limitations
- **Not an airline rating.** The model learns from airline, airport, date and time. Differences between airlines partly reflect which airports and hours they serve and their hub structure, so predictions should not be read as a judgement of an airline's quality or used to rank carriers.
- **US Domestic Flights Only** The dataset covers US domestic flights from January 2025 to August 2026, trained on a 4% sample (about 490,000 flights). Dates outside that range, international routes and brand-new airports or airlines are outside what the model has seen. Four airports in the test period never appeared in training.
- **No Conditions Outside of Airline/Airport Considered** Weather, air-traffic programs and the inbound aircraft's status drive a large share of real delays and are not inputs, so the model gives a typical-conditions estimate, not a forecast for a particular day.
- **Low reliability/accuracy.** The output is a rough risk estimate. It should not be relied on for decisions such as booking a tight connection.
- **Data manipulation.** Delay minutes are heavy-tailed and over half of flights have no delay, so errors are largest on rare, very long delays. Cancelled and diverted flights are excluded, so the model says nothing about cancellation risk.

## Repository layout
```
data/            raw source CSVs (processed/ and gcs/ copies are git-ignored)
preprocessing/   EDA, feature engineering, training set build, GCS upload
pipeline/        config-driven training and evaluation (config.yaml)
experiments/     MLflow experiment tracking
api/             FastAPI backend
frontend/        Streamlit app
docker/          entrypoint script for the container (serve | train | preprocess | experiments)
Dockerfile       one image for training and serving; used for the Cloud Run deployment
requirements.txt pinned Python dependencies for the backend (the front end has its own)
.dockerignore, .gcloudignore   what goes into the image build / the Cloud Build upload
```
## AI Use Disclaimer:
The choice of dataset, planning of the pipeline, all architectural decisions and cloud configuration were carried out without AI-assistance. The code implementation was generated using Claude Code (Opus 5.5) and manually reviewed.