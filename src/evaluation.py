"""Regression metrics and plots (all computed from real predictions)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def r2_interpretation(r2: float) -> str:
    """Plain-language reading of R^2 (honest about weak / negative values)."""
    if r2 < 0:
        return "Negative R2: the model predicts worse than simply guessing the average price."
    if r2 < 0.3:
        return "Weak R2: the model explains only a small part of the variation in price."
    if r2 < 0.7:
        return "Moderate R2: useful signal, but a large share of price variation is unexplained."
    return "Strong R2 on this dataset - that still does not guarantee accuracy for individual properties."


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs predicted (test set)"):
    plt = _plt()
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(y_true, y_pred, s=8, alpha=0.35)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1, label="Perfect prediction")
    ax.set_xlabel("Actual price (USD)")
    ax.set_ylabel("Predicted price (USD)")
    ax.set_title(title, fontsize=10)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, title="Residuals (actual - predicted)"):
    plt = _plt()
    resid = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].scatter(y_pred, resid, s=8, alpha=0.35)
    axes[0].axhline(0, color="r", ls="--", lw=1)
    axes[0].set_xlabel("Predicted price (USD)")
    axes[0].set_ylabel("Residual (USD)")
    axes[0].set_title("Residuals vs predicted", fontsize=10)
    axes[1].hist(resid, bins=40)
    axes[1].set_xlabel("Residual (USD)")
    axes[1].set_title("Residual distribution", fontsize=10)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_bars(labels, values, title, xlabel):
    plt = _plt()
    order = np.argsort(np.abs(values))
    fig, ax = plt.subplots(figsize=(5.5, 0.45 * len(labels) + 1.2))
    ax.barh(np.asarray(labels)[order], np.asarray(values)[order])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.axvline(0, color="k", lw=0.6)
    fig.tight_layout()
    return fig
