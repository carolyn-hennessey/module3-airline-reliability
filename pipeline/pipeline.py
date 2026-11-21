"""Model training pipeline: ingest -> prepare -> split -> train -> evaluate.

Run from the repo root:  python pipeline/pipeline.py [--config pipeline/config.yaml]

Feature engineering (date features, log transforms, holiday proximity, airport
attributes, the log-delay target) was already done in preprocessing/preprocess.py,
which also wrote train.parquet / test.parquet to the GCS bucket. This pipeline only
does model-specific preparation (categorical encoding) on top of those tables.

All tunable parameters live in config.yaml.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# 1. Data ingestion (from the cloud)
# --------------------------------------------------------------------------- #
def gcs_client():
    """GCS client using Application Default Credentials.

    `gcloud auth application-default login` creds carry a quota project but no default
    project, which storage.Client() rejects, so fall back to the credentials' quota project.
    """
    import google.auth
    from google.cloud import storage

    creds, project = google.auth.default()
    project = project or getattr(creds, "quota_project_id", None)
    return storage.Client(project=project, credentials=creds)


def download_from_gcs(cfg: dict) -> tuple[Path, Path]:
    """Download train/test parquet files from the GCS bucket (skips files already local).

    Requires Application Default Credentials: `gcloud auth application-default login`.
    """
    d = cfg["data"]
    local_dir = ROOT / d["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    client = None
    for blob_name in (d["train_blob"], d["test_blob"]):
        dest = local_dir / blob_name
        if dest.exists() and not d["force_download"]:
            print(f"using existing {dest}")
        else:
            if client is None:
                client = gcs_client()  # created lazily: not needed if files are cached
            client.bucket(d["gcs_bucket"]).blob(blob_name).download_to_filename(dest)
            print(f"downloaded gs://{d['gcs_bucket']}/{blob_name} -> {dest}")
        paths.append(dest)
    return paths[0], paths[1]


# --------------------------------------------------------------------------- #
# 2. Preparation + 3. Train/validation/test split
# --------------------------------------------------------------------------- #
def split_train_validation(train: pd.DataFrame, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: the latest `validation_fraction` of training rows become validation."""
    train = train.sort_values("flight_date", kind="stable")
    cutoff = int(len(train) * (1 - validation_fraction))
    return train.iloc[:cutoff], train.iloc[cutoff:]


def build_model(cfg: dict) -> Pipeline:
    f, m = cfg["features"], cfg["model"]
    encoder = ColumnTransformer(
        [
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), f["onehot"]),
            # Unseen airports/routes (e.g. in the test set) fall back to the global target mean.
            ("target_enc", TargetEncoder(target_type="continuous", random_state=m["random_state"]), f["target_encode"]),
            ("numeric", "passthrough", f["numeric"]),
        ]
    )
    if m["type"] != "hist_gradient_boosting":
        raise ValueError(f"unsupported model type: {m['type']}")
    regressor = HistGradientBoostingRegressor(
        learning_rate=m["learning_rate"],
        max_iter=m["max_iter"],
        max_depth=m["max_depth"],
        max_leaf_nodes=m["max_leaf_nodes"],
        min_samples_leaf=m["min_samples_leaf"],
        l2_regularization=m["l2_regularization"],
        early_stopping=m["early_stopping"],
        random_state=m["random_state"],
    )
    return Pipeline([("encode", encoder), ("model", regressor)])


# --------------------------------------------------------------------------- #
# 4. Evaluation
# --------------------------------------------------------------------------- #
def evaluate(model: Pipeline, df: pd.DataFrame, cfg: dict, columns: list[str]) -> dict:
    """Metrics on the log scale (what the model optimizes) and in real minutes."""
    f = cfg["features"]
    pred_log = model.predict(df[columns])
    true_log = df[f["target"]].to_numpy()
    pred_min, true_min = np.expm1(pred_log).clip(min=0), np.expm1(true_log)
    return {
        "rows": int(len(df)),
        "rmse_log": float(np.sqrt(mean_squared_error(true_log, pred_log))),
        "mae_log": float(mean_absolute_error(true_log, pred_log)),
        "r2_log": float(r2_score(true_log, pred_log)),
        "mae_minutes": float(mean_absolute_error(true_min, pred_min)),
        "median_abs_error_minutes": float(np.median(np.abs(true_min - pred_min))),
        # How well the predicted delay ranks flights that end up delayed 15+ minutes
        "roc_auc_is_delayed": float(roc_auc_score(df[f["binary_target"]], pred_log)),
    }


def main(config_path: Path) -> None:
    cfg = load_config(config_path)
    f = cfg["features"]
    columns = f["onehot"] + f["target_encode"] + f["numeric"]

    # 1. Ingest
    train_path, test_path = download_from_gcs(cfg)
    train_full, test = pd.read_parquet(train_path), pd.read_parquet(test_path)

    # 2/3. Features were engineered in preprocessing; split train into train/validation
    train, val = split_train_validation(train_full, cfg["split"]["validation_fraction"])
    print(f"train {len(train):,} | validation {len(val):,} | test {len(test):,}")

    # 4. Train
    model = build_model(cfg)
    model.fit(train[columns], train[f["target"]])
    print(f"trained for {model.named_steps['model'].n_iter_} iterations")

    # 5. Evaluate
    metrics = {name: evaluate(model, df, cfg, columns) for name, df in [("train", train), ("validation", val), ("test", test)]}
    # Baseline: always predict the training mean of the target
    baseline = np.full(len(test), train[f["target"]].mean())
    metrics["baseline_test_rmse_log"] = float(np.sqrt(mean_squared_error(test[f["target"]], baseline)))
    print(json.dumps(metrics, indent=2))

    # 6. Save
    out = cfg["output"]
    for key in ("model_path", "metrics_path"):
        (ROOT / out[key]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ROOT / out["model_path"])
    (ROOT / out["metrics_path"]).write_text(json.dumps(metrics, indent=2))
    print(f"saved model and metrics to {(ROOT / out['model_path']).parent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "pipeline" / "config.yaml")
    main(parser.parse_args().config)
