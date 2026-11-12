"""Data drift and input-quality monitoring.

Definitions (see docs/methodology.md):
  data (covariate) drift     P(X) changes.                    Measurable without labels.
  concept drift              P(y|X) changes.                  Needs labels; can occur with NO data drift.
  prediction drift           P(model score) changes.          Symptom only; may reflect population change or a pipeline bug.
  performance degradation    metric(y, score) worsens.        Needs labels (often delayed for credit).
Drift statistics here are screening signals, not proof of model failure.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from credit_risk.data.validation import find_invalid_values

EPS = 1e-4


def psi_from_proportions(ref_p: np.ndarray, cur_p: np.ndarray) -> float:
    ref_p, cur_p = np.clip(ref_p, EPS, None), np.clip(cur_p, EPS, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Population Stability Index with reference-quantile bins (non-missing values only)."""
    r, c = pd.to_numeric(ref, errors="coerce").dropna().to_numpy(), pd.to_numeric(cur, errors="coerce").dropna().to_numpy()
    if len(r) == 0 or len(c) == 0:
        return float("nan")
    edges = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                       # (near-)constant reference
        return 0.0 if np.allclose(np.unique(c), np.unique(r)) else float(psi_from_proportions(np.array([1.0]), np.array([np.mean(c == r[0])])))
    edges[0], edges[-1] = -np.inf, np.inf
    rp = np.histogram(r, edges)[0] / len(r)
    cp = np.histogram(c, edges)[0] / len(c)
    return psi_from_proportions(rp, cp)


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    r = ref.astype(object).where(ref.notna(), "MISSING").value_counts(normalize=True)
    c = cur.astype(object).where(cur.notna(), "MISSING").value_counts(normalize=True)
    cats = r.index.union(c.index)
    return psi_from_proportions(r.reindex(cats, fill_value=0).to_numpy(), c.reindex(cats, fill_value=0).to_numpy())


def drift_level(psi: float, moderate: float = 0.10, high: float = 0.25) -> str:
    if np.isnan(psi):
        return "unknown"
    return "high" if psi >= high else "moderate" if psi >= moderate else "none"


def compare_datasets(
    ref: pd.DataFrame, cur: pd.DataFrame, features: list[str] | None = None,
    psi_moderate: float = 0.10, psi_high: float = 0.25, ks_pvalue: float = 0.01,
) -> pd.DataFrame:
    """Per-feature drift table: PSI, KS test (numeric), missing-rate change."""
    features = features or [c for c in ref.columns if c in cur.columns and c not in ("SK_ID_CURR", "TARGET")]
    rows = []
    for f in features:
        numeric = pd.api.types.is_numeric_dtype(ref[f])
        psi = psi_numeric(ref[f], cur[f]) if numeric else psi_categorical(ref[f], cur[f])
        ks_stat = ks_p = np.nan
        if numeric:
            a, b = pd.to_numeric(ref[f], errors="coerce").dropna(), pd.to_numeric(cur[f], errors="coerce").dropna()
            if len(a) > 1 and len(b) > 1:
                ks_stat, ks_p = ks_2samp(a, b)
        rm, cm = float(ref[f].isna().mean()), float(cur[f].isna().mean())
        level = drift_level(psi, psi_moderate, psi_high)
        rows.append({
            "feature": f, "type": "numeric" if numeric else "categorical", "psi": psi, "drift_level": level,
            "ks_stat": ks_stat, "ks_pvalue": ks_p, "ks_significant": bool(ks_p < ks_pvalue) if not np.isnan(ks_p) else False,
            "ref_missing_rate": rm, "cur_missing_rate": cm, "missing_rate_delta": cm - rm,
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def input_quality_report(cur: pd.DataFrame, core_fields: list[str] | None = None) -> dict[str, Any]:
    """Missing inputs and invalid (rule-violating) values in a batch of incoming requests."""
    core = [c for c in (core_fields or []) if c in cur.columns]
    return {
        "n_rows": int(len(cur)),
        "missing_rate_core_fields": {c: float(cur[c].isna().mean()) for c in core},
        "invalid_value_counts": find_invalid_values(cur),
        "duplicate_rows": int(cur.duplicated().sum()),
    }


def drift_summary(table: pd.DataFrame, feature_fraction_threshold: float = 0.10) -> dict[str, Any]:
    n_high = int((table["drift_level"] == "high").sum())
    return {
        "n_features": int(len(table)), "n_high_drift": n_high,
        "n_moderate_drift": int((table["drift_level"] == "moderate").sum()),
        "high_drift_fraction": n_high / max(len(table), 1),
        "dataset_drift_flag": n_high / max(len(table), 1) >= feature_fraction_threshold,
        "top_drifted_features": table.head(5)["feature"].tolist(),
    }


def simulate_drift(df: pd.DataFrame, seed: int = 0, income_scale: float = 1.35, ext1_extra_missing: float = 0.15) -> pd.DataFrame:
    """SIMULATED covariate shift for the monitoring demo: a higher-income applicant population (income and the
    loan amounts that scale with it) and more missing EXT_SOURCE_1."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for col in ("AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"):
        if col in out:
            out[col] = out[col] * income_scale
    if "EXT_SOURCE_1" in out:
        out.loc[rng.random(len(out)) < ext1_extra_missing, "EXT_SOURCE_1"] = np.nan
    return out


def drift_report_markdown(table: pd.DataFrame, summary: dict[str, Any], quality: dict[str, Any], simulated: bool, title: str = "Data Drift Report") -> str:
    head = ("> **SIMULATED monitoring scenario.** The 'current' data was produced by perturbing a reference sample "
            "(see `simulate_drift`). It demonstrates the tooling, not real production behaviour.\n\n") if simulated else ""
    return "\n".join([
        f"# {title}\n", head,
        f"- Features compared: {summary['n_features']} | high drift: {summary['n_high_drift']} | moderate: {summary['n_moderate_drift']}",
        f"- Dataset-level drift flag: **{summary['dataset_drift_flag']}** (high-drift share {summary['high_drift_fraction']:.1%})",
        f"- Top drifted: {', '.join(summary['top_drifted_features'])}\n",
        "A drift flag means the input population changed. It does NOT by itself show that model performance degraded.\n",
        "## Feature drift (top 15)\n", table.head(15).to_markdown(index=False, floatfmt=".4f"), "",
        "## Input quality\n", f"```\n{quality}\n```",
    ])
