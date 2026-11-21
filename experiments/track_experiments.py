"""Track model experiments with MLflow, separately from the training pipeline.

Run from the repo root:  python experiments/track_experiments.py
View results:            mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db

pipeline/pipeline.py is unchanged and has no MLflow dependency. This script imports its
building blocks (GCS download, split, model, evaluation) so runs are identical to a normal
pipeline run, applies the parameter variations from experiments/experiments.yaml, and logs
each one to MLflow: parameters, metrics for train/validation/test, and the fitted model.
"""
import argparse
import copy
import sys
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.pipeline import build_model, download_from_gcs, evaluate, load_config, split_train_validation  # noqa: E402


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Return a copy of cfg with dotted-path overrides applied (e.g. 'model.learning_rate')."""
    cfg = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        *parents, leaf = dotted.split(".")
        node = cfg
        for key in parents:
            node = node[key]
        if leaf not in node:
            raise KeyError(f"override '{dotted}' does not exist in pipeline/config.yaml")
        node[leaf] = value
    return cfg


def log_run(name: str, cfg: dict, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> None:
    f = cfg["features"]
    with mlflow.start_run(run_name=name):
        mlflow.log_params({f"model.{k}": v for k, v in cfg["model"].items()})
        mlflow.log_params({
            "split.validation_fraction": cfg["split"]["validation_fraction"],
            "features.target": f["target"],
            "features.n_features": len(columns),
            "data.train_rows": len(train), "data.val_rows": len(val), "data.test_rows": len(test),
            "data.gcs_bucket": cfg["data"]["gcs_bucket"],
        })
        mlflow.set_tags({"feature_engineering": "preprocessing/preprocess.py", "model_family": cfg["model"]["type"]})

        model = build_model(cfg)
        model.fit(train[columns], train[f["target"]])
        mlflow.log_metric("n_iter", model.named_steps["model"].n_iter_)

        for split_name, df in [("train", train), ("val", val), ("test", test)]:
            for metric, value in evaluate(model, df, cfg, columns).items():
                if metric != "rows":
                    mlflow.log_metric(f"{split_name}_{metric}", value)
        baseline = np.full(len(test), train[f["target"]].mean())
        mlflow.log_metric("baseline_test_rmse_log", float(np.sqrt(mean_squared_error(test[f["target"]], baseline))))

        # MLflow's safe (skops) serializer flags this sklearn type; we trained the model
        # ourselves, so trusting exactly this one type is safe.
        mlflow.sklearn.log_model(
            model, name="model",
            skops_trusted_types=["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resolved_config.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            mlflow.log_artifact(str(path))


def main(args: argparse.Namespace) -> None:
    base = load_config(args.pipeline_config)
    exp = load_config(args.experiments_config)
    ml = exp["mlflow"]

    artifact_root = (ROOT / ml["artifact_root"]).resolve()
    mlflow.set_tracking_uri(ml["tracking_uri"])
    if mlflow.get_experiment_by_name(ml["experiment_name"]) is None:
        mlflow.create_experiment(ml["experiment_name"], artifact_location=artifact_root.as_uri())
    mlflow.set_experiment(ml["experiment_name"])

    train_path, test_path = download_from_gcs(base)
    train_full, test = pd.read_parquet(train_path), pd.read_parquet(test_path)
    f = base["features"]
    columns = f["onehot"] + f["target_encode"] + f["numeric"]

    for run in exp["runs"]:
        cfg = apply_overrides(base, run.get("overrides") or {})
        train, val = split_train_validation(train_full, cfg["split"]["validation_fraction"])
        print(f"run '{run['name']}' ...")
        log_run(run["name"], cfg, train, val, test, columns)
    print(f"done. View with: mlflow ui --backend-store-uri {ml['tracking_uri']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pipeline-config", type=Path, default=ROOT / "pipeline" / "config.yaml")
    parser.add_argument("--experiments-config", type=Path, default=ROOT / "experiments" / "experiments.yaml")
    main(parser.parse_args())
