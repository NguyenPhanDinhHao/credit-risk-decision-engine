"""Serialisable model bundle: preprocessing + model + calibrator + metadata, used identically by
training evaluation, the CLI and the API."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from credit_risk.decision.policy import RiskBands, categorize
from credit_risk.models.calibration import Calibrator

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Educational portfolio model trained on public Home Credit data. Output is a statistical estimate, "
    "not a credit decision, and must not be used for real lending."
)


@dataclass
class ModelBundle:
    pipeline: Any                       # sklearn Pipeline: features -> prep -> model (fitted)
    calibrator: Calibrator
    raw_columns: list[str]              # raw input schema seen at training time
    numeric_raw_columns: list[str]
    model_name: str
    model_version: str
    risk_bands: RiskBands
    metadata: dict[str, Any] = field(default_factory=dict)
    _explainer: Any = field(default=None, repr=False, compare=False)

    # ---------------------------------------------------------------- input handling
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align an arbitrary input frame to the training schema: unknown columns are ignored,
        missing ones become NaN (handled by the fitted imputers), numeric columns are coerced."""
        X = df.reindex(columns=self.raw_columns).copy()
        for c in self.numeric_raw_columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        for c in X.columns:
            if c not in self.numeric_raw_columns:
                X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
        return X

    # ---------------------------------------------------------------- prediction
    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(self.prepare(df))[:, 1]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Calibrated probability of default."""
        return np.clip(self.calibrator.predict(self.predict_raw(df)), 0.0, 1.0)

    def risk_category(self, p: np.ndarray) -> np.ndarray:
        return categorize(p, self.risk_bands)

    def predict_records(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame.from_records(records)
        p = self.predict_proba(df)
        return pd.DataFrame({"probability_of_default": p, "risk_category": self.risk_category(p)})

    # ---------------------------------------------------------------- explanation
    def explain(self, df: pd.DataFrame, top_k: int = 5) -> list[list[dict[str, Any]]]:
        from credit_risk.explainability.shap_analysis import explain_rows

        return explain_rows(self, df, top_k=top_k)

    # ---------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = {k: v for k, v in self.__dict__.items() if k != "_explainer"}
        joblib.dump(obj, path, compress=3)
        return path

    @classmethod
    def load(cls, path: str | Path) -> ModelBundle:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model bundle not found: {path}. Train a model first (python scripts/train.py).")
        return cls(**joblib.load(path))
