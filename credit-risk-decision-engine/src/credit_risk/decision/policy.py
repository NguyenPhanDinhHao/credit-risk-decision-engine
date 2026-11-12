"""Decision-support layer: risk bands, policy scenarios, threshold sensitivity and expected cost.

All thresholds and costs are ILLUSTRATIVE configuration values. They are not industry standards and
a risk category is NOT a credit approval decision: "flagged" means "would be routed to review/decline
under this illustrative policy".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskBands:
    low_max: float = 0.05
    medium_max: float = 0.15

    def __post_init__(self) -> None:
        if not 0 < self.low_max < self.medium_max < 1:
            raise ValueError("Require 0 < low_max < medium_max < 1")

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> RiskBands:
        b = cfg["decision"]["risk_bands"]
        return cls(float(b["low_max"]), float(b["medium_max"]))

    def label(self, p: float) -> str:
        return str(categorize(np.array([p]), self)[0])


def categorize(p: np.ndarray, bands: RiskBands) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.select([p < bands.low_max, p < bands.medium_max], ["Low Risk", "Medium Risk"], default="High Risk")


def expected_cost(fn: int, fp: int, n: int, cost_fn: float, cost_fp: float) -> float:
    """Average illustrative cost per applicant: FN = missed defaulter, FP = good applicant flagged."""
    return (fn * cost_fn + fp * cost_fp) / max(n, 1)


def threshold_sweep(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray, cost_fn: float, cost_fp: float) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    n_pos = int(y.sum())
    for t in thresholds:
        flag = p >= t
        tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
        fn = n_pos - tp
        rows.append({
            "threshold": float(t),
            "flagged_rate": float(flag.mean()),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(n_pos, 1),
            "false_positives": fp, "false_negatives": fn,
            "expected_cost_per_applicant": expected_cost(fn, fp, len(y), cost_fn, cost_fp),
        })
    return pd.DataFrame(rows)


def optimal_cost_threshold(y: np.ndarray, p: np.ndarray, cost_fn: float, cost_fp: float, n_grid: int = 200) -> float:
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, n_grid)))
    sweep = threshold_sweep(y, p, grid, cost_fn, cost_fp)
    return float(sweep.loc[sweep["expected_cost_per_applicant"].idxmin(), "threshold"])


def tier_summary(y: np.ndarray, p: np.ndarray, bands: RiskBands) -> pd.DataFrame:
    cat = categorize(p, bands)
    df = pd.DataFrame({"y": np.asarray(y), "cat": cat})
    g = df.groupby("cat")["y"].agg(applicants="size", defaults="sum", observed_default_rate="mean")
    g["share_of_applicants"] = g["applicants"] / len(df)
    return g.reindex(["Low Risk", "Medium Risk", "High Risk"]).reset_index().rename(columns={"cat": "risk_category"})


def policy_comparison(y: np.ndarray, p: np.ndarray, policies: dict[str, dict[str, float]], costs: dict[str, float]) -> pd.DataFrame:
    rows = []
    for name, spec in policies.items():
        s = threshold_sweep(y, p, np.array([spec["flag_threshold"]]), costs["false_negative"], costs["false_positive"]).iloc[0]
        rows.append({"policy": name, **s.to_dict()})
    return pd.DataFrame(rows)


def no_model_cost(y: np.ndarray, costs: dict[str, float]) -> dict[str, float]:
    """Reference: flag nobody (all defaulters missed) vs flag everybody (all good applicants flagged)."""
    y = np.asarray(y)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    return {
        "flag_nobody_cost": expected_cost(n_pos, 0, len(y), costs["false_negative"], costs["false_positive"]),
        "flag_everybody_cost": expected_cost(0, n_neg, len(y), costs["false_negative"], costs["false_positive"]),
    }
