"""Data audit: summary statistics, leakage screen and EDA figures."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from credit_risk.data.validation import find_invalid_values

logger = logging.getLogger(__name__)


def _cat_cols(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    return [c for c in df.columns if c not in exclude and not pd.api.types.is_numeric_dtype(df[c])]


def _num_cols(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def outlier_summary(df: pd.DataFrame, num_cols: list[str], top: int = 15) -> pd.DataFrame:
    rows = []
    for c in num_cols:
        s = df[c].dropna()
        if s.nunique() <= 2 or s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        n_out = int(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum())
        rows.append({"feature": c, "extreme_outliers_3iqr": n_out, "share": n_out / len(s), "min": s.min(), "max": s.max()})
    return pd.DataFrame(rows).sort_values("share", ascending=False).head(top).reset_index(drop=True)


def single_feature_auc(df: pd.DataFrame, target: str, id_col: str, max_rows: int = 60000, seed: int = 0) -> pd.DataFrame:
    """Univariate AUC per feature as a crude leakage / signal screen.

    Numeric: AUC of the (median-imputed) feature itself. Categorical: out-of-fold target-mean encoding.
    An AUC far from 0.5 (e.g. > 0.85) deserves manual inspection for leakage; it is not proof of leakage.
    """
    d = df.sample(min(max_rows, len(df)), random_state=seed) if len(df) > max_rows else df
    y = d[target].to_numpy()
    rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for c in [c for c in d.columns if c not in (target,)]:
        s = d[c]
        try:
            if pd.api.types.is_numeric_dtype(s):
                v = s.fillna(s.median()).to_numpy(dtype=float)
                auc = roc_auc_score(y, v) if np.nanstd(v) > 0 else 0.5
            else:
                s = s.astype(object).where(s.notna(), "MISSING")
                oof = np.zeros(len(d))
                for tr, va in skf.split(d, y):
                    means = pd.Series(y[tr]).groupby(s.iloc[tr].to_numpy()).mean()
                    oof[va] = s.iloc[va].map(means).fillna(y[tr].mean()).to_numpy()
                auc = roc_auc_score(y, oof)
        except Exception:  # noqa: BLE001 - screening must never crash the audit
            continue
        rows.append({"feature": c, "univariate_auc": auc, "abs_signal": abs(auc - 0.5)})
    out = pd.DataFrame(rows).sort_values("abs_signal", ascending=False).reset_index(drop=True)
    out["flag"] = np.where(out["univariate_auc"].gt(0.85) | out["univariate_auc"].lt(0.15), "REVIEW_FOR_LEAKAGE", "")
    return out


def audit_summary(df: pd.DataFrame, target: str, id_col: str) -> dict[str, Any]:
    num, cat = _num_cols(df, [target, id_col]), _cat_cols(df, [target, id_col])
    miss = df.isna().mean().sort_values(ascending=False)
    card = {c: int(df[c].nunique()) for c in cat}
    vc = df[target].value_counts()
    summary = {
        "shape": list(df.shape),
        "n_numeric": len(num), "n_categorical": len(cat),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1e6), 1),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df[id_col].duplicated().sum()) if id_col in df else None,
        "target_distribution": {str(k): int(v) for k, v in vc.items()},
        "default_rate": float(df[target].mean()),
        "imbalance_ratio_neg_to_pos": float(vc.get(0, 0) / max(vc.get(1, 1), 1)),
        "columns_with_missing": int((miss > 0).sum()),
        "columns_over_50pct_missing": miss[miss > 0.5].round(3).to_dict(),
        "constant_columns": [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
        "categorical_cardinality": dict(sorted(card.items(), key=lambda kv: -kv[1])),
        "invalid_values": find_invalid_values(df),
    }
    if "DAYS_EMPLOYED" in df:
        summary["days_employed_sentinel_365243"] = int((df["DAYS_EMPLOYED"] == 365243).sum())
    return summary


def make_eda_figures(df: pd.DataFrame, out_dir: Path, target: str, id_col: str, seed: int = 0) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    sns.set_theme(style="whitegrid", context="notebook")
    num, cat = _num_cols(df, [target, id_col]), _cat_cols(df, [target, id_col])

    def save(fig: plt.Figure, name: str) -> None:
        p = out_dir / name
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    vc = df[target].value_counts().sort_index()
    ax.bar(["0 (no difficulty)", "1 (difficulty)"], vc.values, color=["#4c78a8", "#e45756"])
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{v:,}\n{v / len(df):.1%}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Target distribution")
    save(fig, "eda_target_distribution.png")

    miss = df.isna().mean().sort_values(ascending=False).head(25)
    if miss.iloc[0] > 0:
        fig, ax = plt.subplots(figsize=(7, 6))
        miss[::-1].plot.barh(ax=ax, color="#f58518")
        ax.set_title("Top missing-value rates"); ax.set_xlabel("share missing")
        save(fig, "eda_missing_values.png")

    sig = single_feature_auc(df, target, id_col, seed=seed)
    top = sig.head(20)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["feature"][::-1], top["univariate_auc"][::-1] - 0.5, color="#54a24b", left=0.5)
    ax.axvline(0.5, color="k", lw=0.8)
    ax.set_title("Univariate AUC (signal / leakage screen)"); ax.set_xlabel("AUC")
    save(fig, "eda_univariate_auc.png")

    key_num = [c for c in sig["feature"] if c in num][:6]
    if key_num:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        for ax, c in zip(axes.ravel(), key_num, strict=False):
            lo, hi = df[c].quantile([0.01, 0.99])
            for t, colr in ((0, "#4c78a8"), (1, "#e45756")):
                s = df.loc[df[target] == t, c].dropna().clip(lo, hi)
                ax.hist(s, bins=40, alpha=0.5, density=True, color=colr, label=f"TARGET={t}")
            ax.set_title(c, fontsize=9)
        axes.ravel()[0].legend(fontsize=7)
        save(fig, "eda_numeric_by_target.png")

    top_corr = [c for c in sig["feature"] if c in num][:15]
    if len(top_corr) > 2:
        fig, ax = plt.subplots(figsize=(8, 6.5))
        sns.heatmap(df[top_corr].corr(), cmap="coolwarm", center=0, ax=ax, annot=False)
        ax.set_title("Correlation among top-signal numeric features")
        save(fig, "eda_correlation_heatmap.png")

    low_card = [c for c in cat if df[c].nunique() <= 20][:6]
    if low_card:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for ax, c in zip(axes.ravel(), low_card, strict=False):
            g = df.assign(_c=df[c].astype(object).where(df[c].notna(), "MISSING")).groupby("_c")[target].agg(["mean", "size"])
            g = g[g["size"] >= 30].sort_values("mean")
            ax.barh(g.index.astype(str), g["mean"], color="#b279a2")
            ax.axvline(df[target].mean(), color="k", ls="--", lw=0.8)
            ax.set_title(f"Default rate by {c}", fontsize=9)
        save(fig, "eda_categorical_default_rate.png")
    return paths


def run_audit(df: pd.DataFrame, out_dir: Path, target: str = "TARGET", id_col: str = "SK_ID_CURR", seed: int = 0) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = audit_summary(df, target, id_col)
    num = _num_cols(df, [target, id_col])
    summary["outliers_top"] = outlier_summary(df, num).to_dict("records")
    leak = single_feature_auc(df, target, id_col, seed=seed)
    summary["leakage_screen_top"] = leak.head(15).to_dict("records")
    summary["leakage_flags"] = leak.loc[leak["flag"] != "", "feature"].tolist()
    summary["figures"] = [str(p.name) for p in make_eda_figures(df, out_dir, target, id_col, seed)]
    with open(out_dir / "data_audit_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("Audit written to %s", out_dir)
    return summary
