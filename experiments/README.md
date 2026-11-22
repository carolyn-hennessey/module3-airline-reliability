# Experiment Tracking (MLflow)

Tracks model experiments with MLflow, separately from the training pipeline. Part of the project described in the [root README](../README.md).

## Reproduce this module
Run from the repo root, after the [preprocessing](../preprocessing/README.md) step has uploaded the data.
```
pip install -r requirements.txt
python experiments/track_experiments.py
mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db   # then open http://localhost:5000
```

## How it works
Experiments are tracked with MLflow in `experiments/`, separately from the pipeline: `pipeline/pipeline.py` is unchanged and has no MLflow dependency.

- `experiments/track_experiments.py` imports the pipeline's building blocks (GCS download, chronological split, model, evaluation), so each tracked run is identical to a normal pipeline run. It then logs, per run: the model parameters, split and data settings, train/validation/test metrics, the baseline RMSE, two diagnostic plots (test-set predicted vs actual, and residuals, under `plots/`), the fitted model, and the resolved config.
- `experiments/experiments.yaml` lists the runs. Each run is a name plus dotted-path overrides of `pipeline/config.yaml` (e.g. `model.learning_rate: 0.05`), so adding an experiment means adding a few lines of YAML. It also sets the MLflow experiment name and storage locations.
- Run metadata is stored in `experiments/mlflow.db` (SQLite) and models in `experiments/mlartifacts/`; both are git-ignored.

```
python experiments/track_experiments.py
mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db   # then open http://localhost:5000
```
