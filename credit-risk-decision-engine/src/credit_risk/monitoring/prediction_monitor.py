"""Prediction logging, prediction-drift, performance monitoring and the retraining trigger policy."""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import brier_score_loss, roc_auc_score

from credit_risk.monitoring.data_drift import drift_level, psi_numeric

logger = logging.getLogger(__name__)
_lock = threading.Lock()


class PredictionLogger:
    """Append-only JSONL log of served predictions. Stores NO raw applicant data: only the score,
    category, model version, and which core inputs were missing."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def log_batch(self, model_version: str, probs: Any, cats: Any, core_inputs: pd.DataFrame, core_fields: list[str]) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for i, (p, c) in enumerate(zip(np.asarray(probs), np.asarray(cats), strict=True)):
                missing = [f for f in core_fields if f not in core_inputs.columns or pd.isna(core_inputs.iloc[i][f])] if len(core_inputs) > i else core_fields
                lines.append(json.dumps({"ts": ts, "model_version": model_version, "p": round(float(p), 6), "category": str(c), "missing_core": missing}))
            with _lock, open(self.path, "a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except OSError as exc:  # logging must never break serving
            logger.warning("Prediction log write failed: %s", exc)


def load_prediction_log(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["ts", "model_version", "p", "category", "missing_core"])
    return pd.read_json(path, lines=True)


def prediction_drift(ref_scores: np.ndarray, cur_scores: np.ndarray, moderate: float = 0.10, high: float = 0.25) -> dict[str, Any]:
    psi = psi_numeric(pd.Series(ref_scores), pd.Series(cur_scores))
    ks = ks_2samp(ref_scores, cur_scores)
    return {
        "psi": psi, "level": drift_level(psi, moderate, high), "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
        "ref_mean": float(np.mean(ref_scores)), "cur_mean": float(np.mean(cur_scores)),
        "note": "Prediction drift is a symptom: it can come from population change, pipeline bugs or concept change. It does not prove degradation.",
    }


def performance_report(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Performance on labelled outcomes (available only after labels mature)."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Need both classes to compute ROC-AUC")
    return {
        "n": int(len(y_true)), "roc_auc": float(roc_auc_score(y_true, y_prob)), "brier": float(brier_score_loss(y_true, y_prob)),
        "mean_predicted": float(np.mean(y_prob)), "observed_rate": float(np.mean(y_true)),
    }


@dataclass
class RetrainingPolicy:
    """Documented, human-in-the-loop policy. The function only RECOMMENDS; it never triggers retraining."""
    min_labeled_samples: int = 1000
    max_auc_drop: float = 0.03
    max_calibration_gap_rel: float = 0.25
    drift_feature_fraction: float = 0.10
    consecutive_windows_for_priority: int = 3
    pred_psi_high: float = 0.25

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> RetrainingPolicy:
        r = cfg["monitoring"]["retraining"]
        return cls(r["min_labeled_samples"], r["max_auc_drop"], r["max_calibration_gap_rel"], r["drift_feature_fraction"],
                   r["consecutive_windows_for_priority"], cfg["monitoring"]["psi_high"])


def evaluate_retraining_trigger(
    policy: RetrainingPolicy, baseline_auc: float, performance: dict[str, float] | None,
    dataset_drift_flag: bool, prediction_psi: float, consecutive_drift_windows: int = 0,
) -> dict[str, Any]:
    """Return NO_ACTION | INVESTIGATE | INVESTIGATE_PRIORITY | RETRAIN_RECOMMENDED with reasons.

    RETRAIN_RECOMMENDED requires labelled evidence of degradation. Drift signals alone only escalate to
    investigation (data-pipeline checks, segment analysis, label collection), because drift can be benign.
    """
    reasons: list[str] = []
    degraded = False
    if performance is not None and performance["n"] >= policy.min_labeled_samples:
        drop = baseline_auc - performance["roc_auc"]
        gap = abs(performance["mean_predicted"] - performance["observed_rate"]) / max(performance["observed_rate"], 1e-9)
        if drop >= policy.max_auc_drop:
            degraded = True
            reasons.append(f"ROC-AUC dropped by {drop:.3f} (>= {policy.max_auc_drop}) on {performance['n']} labelled samples")
        if gap >= policy.max_calibration_gap_rel:
            degraded = True
            reasons.append(f"Calibration gap {gap:.0%} of observed rate (>= {policy.max_calibration_gap_rel:.0%}); consider recalibration first")
    elif performance is not None:
        reasons.append(f"Only {performance['n']} labelled samples (< {policy.min_labeled_samples}); performance evidence insufficient")

    drift_signal = dataset_drift_flag or (prediction_psi >= policy.pred_psi_high)
    if dataset_drift_flag:
        reasons.append("Dataset-level data drift flagged")
    if prediction_psi >= policy.pred_psi_high:
        reasons.append(f"Prediction PSI {prediction_psi:.3f} >= {policy.pred_psi_high}")

    if degraded:
        decision = "RETRAIN_RECOMMENDED"
    elif drift_signal and consecutive_drift_windows >= policy.consecutive_windows_for_priority:
        decision = "INVESTIGATE_PRIORITY"
        reasons.append(f"Drift persisted for {consecutive_drift_windows} consecutive windows; obtain labels / segment analysis")
    elif drift_signal:
        decision = "INVESTIGATE"
    else:
        decision = "NO_ACTION"
    return {"decision": decision, "reasons": reasons, "automatic_retraining": False,
            "next_step": "Human review required; retraining is never triggered automatically."}
