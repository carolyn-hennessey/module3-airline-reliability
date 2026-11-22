#!/bin/sh
# Entry point for the project image. Usage: docker run IMAGE <command>
#   serve        start the FastAPI prediction service (default; honours $PORT for Cloud Run)
#   train        run the training pipeline (downloads data from GCS, writes pipeline/artifacts/)
#   preprocess   build the train/test sets and upload them to GCS (needs data/ mounted)
#   experiments  run the MLflow experiment sweep
set -e
cmd="${1:-serve}"
[ $# -gt 0 ] && shift

case "$cmd" in
  serve)
    if [ ! -f "${MODEL_PATH:-pipeline/artifacts/model.joblib}" ]; then
      echo "No trained model found. Run 'train' first (and rebuild the image, or mount pipeline/artifacts)." >&2
      exit 1
    fi
    exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8080}" "$@" ;;
  train)       exec python pipeline/pipeline.py "$@" ;;
  preprocess)  exec python preprocessing/preprocess.py "$@" ;;
  experiments) exec python experiments/track_experiments.py "$@" ;;
  *) echo "Unknown command '$cmd'. Use: serve | train | preprocess | experiments" >&2; exit 2 ;;
esac
