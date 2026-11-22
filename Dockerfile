# Container image for the prediction API, built to run on Google Cloud Run.
# Build from the repo root AFTER training the model (pipeline/artifacts/model.joblib must exist):
#   docker build -t airline-delay-api .
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies first so this layer is cached when only code changes
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Only what the API needs at runtime: code, shared feature engineering, config, model, reference data
COPY api/ api/
COPY preprocessing/preprocess.py preprocessing/preprocess.py
COPY pipeline/config.yaml pipeline/config.yaml
COPY pipeline/artifacts/model.joblib pipeline/artifacts/model.joblib
COPY data/airports.csv data/airlines.csv data/

# Run as a non-root user
RUN useradd --create-home appuser
USER appuser

# Cloud Run injects PORT (default 8080); listen on all interfaces
ENV PORT=8080
EXPOSE 8080
CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
