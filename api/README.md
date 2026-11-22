# Prediction API (FastAPI) and Deployment

Serves the trained model over HTTP and packages it for Google Cloud Run. Part of the project described in the [root README](../README.md). It serves the model produced by the [pipeline](../pipeline/README.md); the [front end](../frontend/README.md) calls it.

## Reproduce this module
Run locally from the repo root, after `python pipeline/pipeline.py` has created the model file.
```
pip install -r requirements.txt
uvicorn api.main:app --reload      # interactive docs at http://127.0.0.1:8000/docs
```
To package and deploy it, see [Deploying](#deploying-the-api-docker-and-google-cloud-run).

## The API
`api/main.py` serves the trained model. It loads `pipeline/artifacts/model.joblib` once at startup (override the location with the `MODEL_PATH` environment variable), so run `python pipeline/pipeline.py` first.

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

The response is the model's expected delay in minutes. It is a rough estimate, and predictions are typically small because most flights have little or no delay. All diverted/cancelled flights were removed from the dataset.

## Deploying the API (Docker and Google Cloud Run)
The root `Dockerfile` builds one image for the whole backend: it installs the pinned `requirements.txt` (versions match the environment the model was pickled in), copies the code, configs and reference data, and runs as a non-root user. Its entry point (`docker/entrypoint.sh`) takes a command: `serve` (the default, which listens on the `PORT` Cloud Run provides, default 8080), `train`, `preprocess` or `experiments`. The trained model is baked into the image when `pipeline/artifacts/model.joblib` exists at build time, which is what lets the deployed service start without training. `.dockerignore` keeps the build context to the files the image needs.

Train first so the model is baked into the image (`pipeline/artifacts/model.joblib` is git-ignored). Either run `python pipeline/pipeline.py`, or train inside the container as described in the [pipeline README](../pipeline/README.md#train-inside-docker).

Test locally:
```
docker build -t airline-delay .
docker run --rm -p 8080:8080 airline-delay          # same as: ... airline-delay serve
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" \
  -d '{"airline_code":"DL","flight_date":"2026-07-04","origin":"ATL","dest":"JFK","sched_dep_time":1745}'
```

Deploy to Cloud Run (requires the `gcloud` CLI; see the [preprocessing README](../preprocessing/README.md) for installing it and signing in):
```
gcloud run deploy airline-delay-api --source . --project project-4364f816-1535-4389-aef --region us-east1 --memory 1Gi --allow-unauthenticated
```
`--source .` builds the image with Cloud Build using `.gcloudignore`, which lists what to leave out of the upload so that the git-ignored model file is included. Do not copy `.dockerignore` into it: `gcloud` uses git-style rules where a leading `*` would also drop the `Dockerfile`. `--allow-unauthenticated` makes the API publicly callable, which the Streamlit front end needs because it sends no credentials. Leave it off if callers should present an identity token. To retrain and redeploy, re-run the pipeline and the deploy command.

**First-time project setup and troubleshooting.** Enable the APIs once:
```
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com --project project-4364f816-1535-4389-aef
```
If the build fails almost immediately with an empty log, the default Compute Engine service account that runs builds is likely missing roles. Grant them (this changes IAM on the project):
```
PROJECT=project-4364f816-1535-4389-aef
SA=170279949235-compute@developer.gserviceaccount.com   # shown by: gcloud builds describe BUILD_ID --region us-east1
for role in roles/cloudbuild.builds.builder roles/artifactregistry.writer roles/storage.objectViewer roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding $PROJECT --member="serviceAccount:$SA" --role="$role" --condition=None
done
```
Deploy it as a Cloud Run **Service**. If you deploy a prebuilt image instead, the image URL needs the full path including the image name.

The deployed service: https://airline-delay-api-170279949235.us-east1.run.app (`/health`, `/docs`, `POST /predict`).

**Test the deployed endpoint**:
```
curl https://airline-delay-api-170279949235.us-east1.run.app/health
curl -X POST https://airline-delay-api-170279949235.us-east1.run.app/predict -H "Content-Type: application/json" \
  -d '{"airline_code":"DL","flight_date":"2026-07-04","origin":"ATL","dest":"JFK","sched_dep_time":1745}'
```
