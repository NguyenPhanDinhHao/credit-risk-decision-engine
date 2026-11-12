#!/usr/bin/env python
"""Run the data audit and write EDA figures to reports/figures/ and a JSON summary."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.config import load_config, resolve_path  # noqa: E402
from credit_risk.data.audit import run_audit  # noqa: E402
from credit_risk.data.loader import DatasetNotFoundError, dataset_status_report, load_dataset  # noqa: E402
from credit_risk.data.validation import validate_application_data  # noqa: E402
from credit_risk.logging_config import setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true"); ap.add_argument("--out-dir")
    a = ap.parse_args()
    setup_logging()
    cfg = load_config()
    print(dataset_status_report(cfg))
    try:
        df, source = load_dataset(cfg, demo=a.demo)
    except DatasetNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    v = validate_application_data(df, cfg["data"]["id_column"], cfg["data"]["target_column"])
    print("Validation passed:", v.passed, "| errors:", v.errors, "| warnings:", v.warnings)
    out = Path(a.out_dir) if a.out_dir else resolve_path(cfg["paths"]["figures_dir"])
    summary = run_audit(df, out, cfg["data"]["target_column"], cfg["data"]["id_column"], cfg["project"]["seed"])
    print(f"data source={source} shape={summary['shape']} default_rate={summary['default_rate']:.4f}")
    print("Leakage screen flags:", summary["leakage_flags"] or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
