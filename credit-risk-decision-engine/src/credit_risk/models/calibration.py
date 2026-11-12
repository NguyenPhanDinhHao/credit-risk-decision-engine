"""Probability calibration (leakage-safe).

Procedure: the base model is fitted on the TRAIN split; calibrators are fitted on the VALIDATION
split (never on training predictions, which are optimistic); the calibration method is selected by
cross-validated Brier score *within* the validation split; the TEST split is used only for the final
report. Calibration matters in credit risk because probabilities feed pricing, provisioning and
threshold/cost decisions, not just rankings. Class weighting and boosting both distort probabilities.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Calibrator:
    """method in {'none', 'sigmoid' (Platt), 'isotonic'}."""

    def __init__(self, method: str = "none"):
        if method not in ("none", "sigmoid", "isotonic"):
            raise ValueError(f"Unknown calibration method: {method}")
        self.method = method

    def fit(self, p: np.ndarray, y: np.ndarray) -> Calibrator:
        if self.method == "sigmoid":
            self.model_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.method == "none":
            return np.clip(p, 0.0, 1.0)
        if self.method == "sigmoid":
            return self.model_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        return np.clip(self.model_.predict(p), 0.0, 1.0)


def select_calibration_method(
    p_val: np.ndarray, y_val: np.ndarray, methods: list[str], seed: int = 42, n_splits: int = 5
) -> tuple[str, dict[str, float]]:
    """Cross-validated Brier score on the validation split for each method (lower is better)."""
    y_val = np.asarray(y_val)
    n_splits = max(2, min(n_splits, int(y_val.sum()), int((1 - y_val).sum())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores: dict[str, float] = {}
    for m in methods:
        oof = np.zeros(len(y_val))
        for tr, te in skf.split(p_val.reshape(-1, 1), y_val):
            oof[te] = Calibrator(m).fit(p_val[tr], y_val[tr]).predict(p_val[te])
        scores[m] = float(brier_score_loss(y_val, oof))
    return min(scores, key=scores.get), scores


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return [{"mean_predicted": float(a), "observed_rate": float(b)} for a, b in zip(mean_pred, frac_pos, strict=True)]


def plot_calibration(curves: dict[str, tuple[np.ndarray, np.ndarray]], out_path: Path, title: str = "Calibration curve") -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    hi = 0.0
    for name, (y, p) in curves.items():
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", ms=4, label=f"{name} (Brier={brier_score_loss(y, p):.4f})")
        hi = max(hi, mean.max(), frac.max())
    ax.plot([0, hi], [0, hi], "k--", lw=0.8, label="perfect calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed default rate", title=title)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)
    return out_path
