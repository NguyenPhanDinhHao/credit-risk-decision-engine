"""Candidate model factories and a trivial reference baseline."""
from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

MODEL_NAMES = ("logistic_regression", "random_forest", "lightgbm")
# Lower rank = preferred when predictive performance is statistically indistinguishable.
INTERPRETABILITY_RANK = {"logistic_regression": 0, "lightgbm": 1, "random_forest": 2}
NEEDS_SCALING = {"logistic_regression": True, "random_forest": False, "lightgbm": False}


def make_estimator(name: str, params: dict[str, Any] | None = None, seed: int = 42, n_jobs: int = -1) -> Any:
    params = dict(params or {})
    if name == "logistic_regression":
        return LogisticRegression(random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **params)
    if name == "lightgbm":
        return LGBMClassifier(random_state=seed, n_jobs=n_jobs, verbose=-1, **params)
    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}")


def make_prior_baseline() -> DummyClassifier:
    """Predicts the training default rate for everyone (ROC-AUC 0.5). Sanity reference for all metrics."""
    return DummyClassifier(strategy="prior")
