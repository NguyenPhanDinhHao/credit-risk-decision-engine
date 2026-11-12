"""Markdown experiment report generated from actual training output (never hand-written numbers)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _fmt(df: pd.DataFrame, cols: list[str]) -> str:
    return df[cols].to_markdown(index=False, floatfmt=".4f") if hasattr(df, "to_markdown") else df[cols].to_string(index=False)


def write_experiment_report(result: dict[str, Any], out_path: Path) -> Path:
    synthetic = result["data_source"] != "home_credit"
    banner = (
        "> **WARNING: SYNTHETIC DATA.** This run used generated demo data (`--demo`). The numbers below only prove "
        "that the pipeline executes; they say nothing about real credit risk and must not be quoted as results.\n\n"
        if synthetic else
        "> Results below were produced from the Home Credit Default Risk `application_train.csv` using the split and "
        "protocol described in `docs/methodology.md`.\n\n"
    )
    comp = pd.DataFrame(result["comparison"])
    lines = [
        "# Experiment Report\n",
        banner,
        f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Data source: `{result['data_source']}` | rows: {result['n_rows']:,} | default rate: {result['default_rate']:.4f}",
        f"- Split (stratified, seed {result['seed']}): train {result['splits']['train']:,} / validation {result['splits']['val']:,} / test {result['splits']['test']:,}",
        f"- Model version: `{result['model_version']}`  | selected model: **{result['selected_model']}**",
        f"- Selection rule: {result['selection_rule']}",
        f"- Calibration: `{result['calibration_method']}` (selected by cross-validated Brier on the validation split)",
        f"- Excluded sensitive features: {result['excluded_sensitive'] or 'none'}",
        "",
        "## Model comparison\n",
        "Validation metrics are used for tuning and selection. Test metrics are computed once, after selection.\n",
        _fmt(comp, ["model", "val_roc_auc", "test_roc_auc", "test_pr_auc", "test_ks", "test_brier_uncalibrated", "test_brier_calibrated", "test_precision", "test_recall", "test_f1", "calibration", "best_params"]),
        "",
        "Precision/recall/F1 use a threshold chosen on the validation split (max F1), not the test split.\n",
        "## Decision-support analysis (test split, calibrated probabilities)\n",
        "Illustrative policies and costs from `configs/config.yaml`; not industry standards.\n",
        pd.DataFrame(result["policy_comparison"]).to_markdown(index=False, floatfmt=".4f"),
        "",
        f"Reference expected cost per applicant: {result['no_model_cost']}\n",
        "### Risk tiers\n",
        pd.DataFrame(result["tier_summary"]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Baseline reference\n",
        f"Prior-rate baseline (constant prediction): ROC-AUC {result['baseline']['roc_auc']:.3f}, Brier {result['baseline']['brier']:.4f}.\n",
        "## Figures and artifacts\n",
        "See `reports/figures/` (ROC/PR, calibration, confusion matrices, threshold sensitivity, SHAP) and `artifacts/`.\n",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
