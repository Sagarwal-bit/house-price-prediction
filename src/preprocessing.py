"""Preprocessing (fitted only on training data, inside the Pipeline) and input validation."""

from __future__ import annotations

import math
from typing import Mapping

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessor(feature_names: list[str], scale: bool) -> ColumnTransformer:
    """Median imputation for numeric features, plus standard scaling when `scale` is True.

    Scaling is used for the linear model (comparable coefficients, stable fitting) and
    skipped for tree models, which do not need it. All features in the supported datasets
    are numeric, so no categorical encoder is needed; imputation is a safety net only.
    """
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer([("num", Pipeline(steps), feature_names)], remainder="drop")


def validate_inputs(values: Mapping[str, object], features: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    """Validate one property's inputs against the model's feature metadata.

    `features` is the list stored in metrics.json (name, label, min_allowed, max_allowed,
    train_min, train_max). Returns a one-row DataFrame and a list of soft warnings (value
    outside the range seen in training). Raises ValueError for missing, non-numeric,
    non-finite or hard-out-of-bounds values.
    """
    row, warnings = {}, []
    for f in features:
        name, label = f["name"], f["label"]
        if name not in values or values[name] is None or values[name] == "":
            raise ValueError(f"'{label}' is required.")
        try:
            x = float(values[name])
        except (TypeError, ValueError):
            raise ValueError(f"'{label}' must be a number.") from None
        if not math.isfinite(x):
            raise ValueError(f"'{label}' must be a finite number.")
        lo, hi = f.get("min_allowed"), f.get("max_allowed")
        if lo is not None and x < lo:
            raise ValueError(f"'{label}' must be at least {lo:g}.")
        if hi is not None and x > hi:
            raise ValueError(f"'{label}' must be at most {hi:g}.")
        if x < f["train_min"] or x > f["train_max"]:
            warnings.append(
                f"'{label}' = {x:g} is outside the range seen in training "
                f"({f['train_min']:g} to {f['train_max']:g}); the estimate is an extrapolation."
            )
        row[name] = x
    unknown = set(values) - {f["name"] for f in features}
    if unknown:
        raise ValueError(f"Unknown input field(s): {sorted(unknown)}")
    return pd.DataFrame([row], columns=[f["name"] for f in features]), warnings
