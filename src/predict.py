"""Load the trained pipeline safely and estimate a price.

Security note: joblib/pickle files can execute code when loaded. Only load artifacts you
generated yourself with `python -m src.train`. The SHA-256 check catches corruption or
mix-ups, not a malicious replacement of both files.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.preprocessing import validate_inputs

logger = logging.getLogger(__name__)

ARTIFACT_FORMAT_VERSION = 1


class ModelNotAvailableError(RuntimeError):
    """Artifacts are missing, corrupted or incompatible (message is user-friendly)."""


@dataclass(frozen=True)
class PriceEstimate:
    price: float
    warnings: list[str] = field(default_factory=list)


class HousePredictor:
    def __init__(self, pipeline, metrics: dict):
        self.pipeline = pipeline
        self.metrics = metrics
        self.features: list[dict] = metrics["features"]

    @property
    def model_name(self) -> str:
        return self.metrics["selection"]["selected_model"]

    def predict(self, values: Mapping[str, object]) -> PriceEstimate:
        """Validate inputs and return an estimated price (never negative).

        Raises ValueError for invalid inputs.
        """
        frame, warnings = validate_inputs(values, self.features)
        pred = float(np.asarray(self.pipeline.predict(frame)).ravel()[0])
        if not np.isfinite(pred):
            raise ValueError("The model produced an invalid estimate for these inputs.")
        return PriceEstimate(price=max(pred, 0.0), warnings=warnings)

    def predict_many(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline.predict(frame[[f["name"] for f in self.features]]), dtype=float)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_metrics(models_dir: Path | str = "models") -> dict:
    path = Path(models_dir) / "metrics.json"
    if not path.exists():
        raise ModelNotAvailableError("No trained model found. Run `python -m src.train` first (see README).")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelNotAvailableError("metrics.json is corrupted; retrain the model.") from exc


def load_test_predictions(models_dir: Path | str = "models") -> pd.DataFrame:
    metrics = load_metrics(models_dir)
    path = Path(models_dir) / metrics.get("predictions_file", "test_predictions.csv")
    if not path.exists():
        raise ModelNotAvailableError("test_predictions.csv is missing; retrain the model.")
    return pd.read_csv(path)


def load_predictor(models_dir: Path | str = "models") -> HousePredictor:
    """Load model + metadata, verifying format version, checksum and scikit-learn version."""
    models_dir = Path(models_dir)
    metrics = load_metrics(models_dir)
    if metrics.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION:
        raise ModelNotAvailableError("Model artifact format is outdated; retrain with `python -m src.train`.")
    model_path = models_dir / metrics.get("model_file", "house_model.joblib")
    if not model_path.exists():
        raise ModelNotAvailableError(f"Model file {model_path.name} is missing; run `python -m src.train`.")
    if _sha256(model_path) != metrics.get("model_sha256"):
        raise ModelNotAvailableError("Model file does not match its recorded checksum. Retrain with `python -m src.train`.")
    trained_with = metrics.get("environment", {}).get("scikit_learn")
    if trained_with and trained_with.split(".")[:2] != sklearn.__version__.split(".")[:2]:
        raise ModelNotAvailableError(
            f"Model was trained with scikit-learn {trained_with} but {sklearn.__version__} is installed. "
            "Retrain with `python -m src.train`."
        )
    try:
        pipeline = joblib.load(model_path)  # trusted, self-generated artifact only
    except Exception as exc:
        raise ModelNotAvailableError(f"Could not load model ({exc}). Retrain with `python -m src.train`.") from exc
    return HousePredictor(pipeline, metrics)
