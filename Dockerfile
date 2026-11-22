# One image for the whole backend project: training AND serving.
#   docker build -t airline-delay .
#   docker run --rm -p 8080:8080 airline-delay                 # serve the API (default)
#   docker run --rm <credentials, see pipeline/README.md> airline-delay train
# See docker/entrypoint.sh for all commands (serve | train | preprocess | experiments).
# Cloud Run runs the default command (serve) and injects PORT.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies first so this layer is cached when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code, configs and reference data. pipeline/artifacts/model.joblib is included when it exists
# (the .dockerignore allows it), which is what lets the deployed service start without training.
COPY api/ api/
COPY preprocessing/preprocess.py preprocessing/preprocess.py
COPY pipeline/ pipeline/
COPY experiments/track_experiments.py experiments/experiments.yaml experiments/
COPY data/airports.csv data/airlines.csv data/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

# Run as a non-root user that owns the app directory (training writes artifacts here)
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

ENV PORT=8080
EXPOSE 8080
ENTRYPOINT ["entrypoint.sh"]
CMD ["serve"]
