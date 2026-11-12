#!/usr/bin/env python
"""Train, calibrate, evaluate and save the Credit Risk model.

Examples:
    python scripts/train.py                       # real Home Credit data in data/raw (fails with instructions if absent)
    python scripts/train.py --lightweight         # 40k-row stratified subsample, fewer trials
    python scripts/train.py --demo --lightweight  # SYNTHETIC data for a smoke run (NOT real results)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.config import apply_lightweight, load_config  # noqa: E402
from credit_risk.data.loader import DatasetNotFoundError, load_dataset  # noqa: E402
from credit_risk.logging_config import setup_logging  # noqa: E402
from credit_risk.models.train import run_training  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config"); ap.add_argument("--data-dir"); ap.add_argument("--artifacts-dir"); ap.add_argument("--reports-dir")
    ap.add_argument("--lightweight", action="store_true", help="fast mode for 8GB machines")
    ap.add_argument("--demo", action="store_true", help="allow SYNTHETIC data if real data is missing")
    ap.add_argument("--no-mlflow", action="store_true"); ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--supplementary", action="store_true", help="add bureau/previous_application aggregates if files exist")
    a = ap.parse_args()
    setup_logging()
    cfg = load_config(a.config)
    if a.data_dir:
        cfg["paths"]["data_dir"] = a.data_dir
    if a.lightweight:
        cfg = apply_lightweight(cfg)
    if a.supplementary:
        cfg["data"]["use_supplementary_tables"] = True
    try:
        df, source = load_dataset(cfg, demo=a.demo)
    except DatasetNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    res = run_training(cfg, df, source, a.artifacts_dir, a.reports_dir, mlflow_enabled=False if a.no_mlflow else None, run_shap=not a.no_shap)
    m = res["selected_test_metrics"]
    print(f"\nSelected: {res['selected_model']} ({res['model_version']}) | data source: {source}")
    print(f"Test ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} KS={m['ks']:.4f} Brier={m['brier']:.4f}")
    if source != "home_credit":
        print("NOTE: metrics are from SYNTHETIC data and are not real results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
