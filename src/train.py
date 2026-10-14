"""Train and compare regression models, then save the selected pipeline.

Usage:
    python -m src.train                          # California Housing (needs internet once)
    python -m src.train --data my.csv            # CSV with the California columns
    python -m src.train --dataset synthetic      # DEMO fallback (original synthetic generator)

Methodology:
1. Validate/clean data; one 80/20 random train/test split with a fixed seed.
2. Each model is a Pipeline(preprocessing -> regressor); imputer/scaler are fitted on
   training data only (also inside every CV fold).
3. 5-fold CV on the TRAINING set (RMSE, MAE, R2) for every model, including a
   predict-the-mean baseline for context.
4. The model with the lowest CV RMSE is selected. The test set is used once, for reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.data_loader import DatasetError, DatasetSpec, load_dataset
from src.evaluation import regression_metrics
from src.preprocessing import build_preprocessor

logger = logging.getLogger("train")

ARTIFACT_FORMAT_VERSION = 1
MODEL_FILE = "house_model.joblib"
METRICS_FILE = "metrics.json"
PREDICTIONS_FILE = "test_predictions.csv"
SIZE_WARN_MB = 50


def candidate_models(features: list[str], seed: int) -> dict[str, Pipeline]:
    def pipe(reg, scale):
        return Pipeline([("prep", build_preprocessor(features, scale)), ("model", reg)])

    return {
        "Mean baseline (reference)": pipe(DummyRegressor(strategy="mean"), False),
        "Linear Regression": pipe(LinearRegression(), True),
        "Random Forest": pipe(
            RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_depth=18,
                                  n_jobs=-1, random_state=seed), False),
        "Hist Gradient Boosting": pipe(HistGradientBoostingRegressor(random_state=seed), False),
    }


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def feature_metadata(spec: DatasetSpec, X_train: pd.DataFrame) -> list[dict]:
    """Per-feature info used by the app form; ranges come from the TRAINING data."""
    out = []
    for f in spec.features:
        s = X_train[f.name]
        out.append({
            "name": f.name, "label": f.label, "help": f.help, "integer": f.integer,
            "min_allowed": f.min_allowed, "max_allowed": f.max_allowed,
            "train_min": float(s.min()), "train_max": float(s.max()),
            "train_median": float(s.median()),
        })
    return out


def run_training(spec_kind: str, data_path: Path | None, models_dir: Path, seed: int = 42,
                 test_size: float = 0.2, cv_folds: int = 5) -> dict:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    df, report, spec = load_dataset(spec_kind, data_path, seed=seed)
    features = spec.feature_names
    X, y = df[features], df[spec.target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
    results, fitted = {}, {}
    for name, pipe in candidate_models(features, seed).items():
        logger.info("Cross-validating %s ...", name)
        cvres = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {
            "cv": {
                "rmse_mean": float(-cvres["test_rmse"].mean()), "rmse_std": float(cvres["test_rmse"].std()),
                "mae_mean": float(-cvres["test_mae"].mean()), "r2_mean": float(cvres["test_r2"].mean()),
            },
            "test": regression_metrics(y_test, pred),
        }
        fitted[name] = pipe

    # selection: lowest CV RMSE among real models (exclude the reference baseline)
    real = {n: r for n, r in results.items() if "baseline" not in n}
    selected = min(real, key=lambda n: real[n]["cv"]["rmse_mean"])
    best = fitted[selected]
    logger.info("Selected by CV RMSE: %s", selected)

    # --- feature analysis (kept separate: coefficients are NOT feature importances) ---
    lin = fitted["Linear Regression"]
    coefs = lin.named_steps["model"].coef_
    linear_analysis = {
        "note": "Coefficients of the linear model on STANDARDISED features: change in price (USD) per +1 standard deviation of the feature, others held fixed.",
        "coefficients": {f: float(c) for f, c in zip(features, coefs)},
    }
    perm = permutation_importance(best, X_test, y_test, scoring="neg_root_mean_squared_error",
                                  n_repeats=5, random_state=seed, n_jobs=1)
    importance_analysis = {
        "model": selected,
        "note": "Permutation importance on the test set: increase in RMSE (USD) when a feature's values are shuffled.",
        "rmse_increase_mean": {f: float(m) for f, m in zip(features, perm.importances_mean)},
        "rmse_increase_std": {f: float(s) for f, s in zip(features, perm.importances_std)},
    }

    model_path = models_dir / MODEL_FILE
    joblib.dump(best, model_path, compress=3)
    size_mb = model_path.stat().st_size / 1e6
    if size_mb > SIZE_WARN_MB:
        logger.warning("Model file is %.1f MB; GitHub rejects files over 100 MB.", size_mb)

    pd.DataFrame({"actual": y_test.to_numpy(), "predicted": best.predict(X_test)}).to_csv(
        models_dir / PREDICTIONS_FILE, index=False)

    metrics = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "key": spec.key, "title": spec.title, "target": spec.target,
            "target_description": spec.target_description, "currency": spec.currency,
            "is_demo": spec.is_demo, "notes": spec.notes, "report": report.to_dict(),
        },
        "features": feature_metadata(spec, X_train),
        "split": {"test_size": test_size, "seed": seed, "cv_folds": cv_folds,
                  "n_train": int(len(X_train)), "n_test": int(len(X_test))},
        "selection": {"criterion": f"lowest mean {cv_folds}-fold CV RMSE on the training set",
                      "selected_model": selected},
        "models": results,
        "linear_analysis": linear_analysis,
        "importance_analysis": importance_analysis,
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__},
        "model_file": MODEL_FILE, "predictions_file": PREDICTIONS_FILE,
        "model_sha256": sha256_of(model_path), "model_size_bytes": model_path.stat().st_size,
    }
    (models_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def print_summary(metrics: dict) -> None:
    ds = metrics["dataset"]
    print(f"\nDataset: {ds['title']}" + ("  [DEMO DATA - not a real valuation model]" if ds["is_demo"] else ""))
    rows = []
    for name, r in metrics["models"].items():
        rows.append({"model": name, "cv_rmse": round(r["cv"]["rmse_mean"], 1), "test_MAE": round(r["test"]["mae"], 1),
                     "test_RMSE": round(r["test"]["rmse"], 1), "test_R2": round(r["test"]["r2"], 4)})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSelected (lowest CV RMSE): {metrics['selection']['selected_model']}")
    print(f"Model file: {metrics['model_file']} ({metrics['model_size_bytes'] / 1e6:.1f} MB)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["california", "synthetic"], default="california")
    ap.add_argument("--data", type=Path, default=None, help="Optional CSV matching the chosen dataset's columns.")
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--cv-folds", type=int, default=5)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        metrics = run_training(args.dataset, args.data, args.models_dir, args.seed, args.test_size, args.cv_folds)
    except DatasetError as exc:
        print(f"\nDataset problem: {exc}", file=sys.stderr)
        return 1
    print_summary(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
