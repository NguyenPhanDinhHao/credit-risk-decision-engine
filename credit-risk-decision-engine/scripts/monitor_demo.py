#!/usr/bin/env python
"""SIMULATED monitoring demo: perturb the held-out test split and compare it with the training-time reference.

Produces reports/monitoring_demo_report.md. Every number is from a simulated shift, not production traffic.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from credit_risk.config import apply_lightweight, load_config, resolve_path  # noqa: E402
from credit_risk.data.loader import DatasetNotFoundError, load_dataset  # noqa: E402
from credit_risk.features.preprocessing import split_data  # noqa: E402
from credit_risk.logging_config import setup_logging  # noqa: E402
from credit_risk.models.predict import ModelBundle  # noqa: E402
from credit_risk.monitoring.data_drift import (  # noqa: E402
    compare_datasets,
    drift_report_markdown,
    drift_summary,
    input_quality_report,
    simulate_drift,
)
from credit_risk.monitoring.prediction_monitor import (  # noqa: E402
    RetrainingPolicy,
    evaluate_retraining_trigger,
    performance_report,
    prediction_drift,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true"); ap.add_argument("--lightweight", action="store_true")
    ap.add_argument("--artifacts-dir"); ap.add_argument("--reports-dir"); ap.add_argument("--data-dir")
    a = ap.parse_args()
    setup_logging()
    cfg = apply_lightweight(load_config()) if a.lightweight else load_config()
    if a.data_dir:
        cfg["paths"]["data_dir"] = a.data_dir
    art = Path(a.artifacts_dir or resolve_path(cfg["paths"]["artifacts_dir"]))
    rep = Path(a.reports_dir or resolve_path(cfg["paths"]["reports_dir"]))
    try:
        df, source = load_dataset(cfg, demo=a.demo)
    except DatasetNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    bundle = ModelBundle.load(art / "model_bundle.joblib")
    ref = pd.read_csv(art / "reference_features.csv.gz")
    ref_scores = np.load(art / "reference_scores.npy")
    s, t = cfg["split"], cfg["data"]["target_column"]
    *_, X_te, _, _, y_te = split_data(df, t, s["test_size"], s["val_size"], cfg["project"]["seed"])

    m = cfg["monitoring"]
    baseline_auc = json.loads((art / "metrics.json").read_text())["selected_test_metrics"]["roc_auc"]
    md = ""
    for scenario, scale in (("moderate shift (x1.35)", 1.35), ("severe shift (x1.60)", 1.60)):
        cur = simulate_drift(X_te, income_scale=scale)                       # SIMULATED shift
        table = compare_datasets(ref, cur.reset_index(drop=True), psi_moderate=m["psi_moderate"], psi_high=m["psi_high"], ks_pvalue=m["ks_pvalue"])
        summ = drift_summary(table, m["retraining"]["drift_feature_fraction"])
        quality = input_quality_report(cur, ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"])
        cur_scores = bundle.predict_proba(cur)
        pdrift = prediction_drift(ref_scores, cur_scores, m["psi_moderate"], m["psi_high"])
        perf = performance_report(y_te.to_numpy(), cur_scores)
        decision = evaluate_retraining_trigger(RetrainingPolicy.from_cfg(cfg), baseline_auc, perf, summ["dataset_drift_flag"], pdrift["psi"], 1)
        md += drift_report_markdown(table, summ, quality, simulated=True, title=f"Monitoring Demo - {scenario} (SIMULATED)")
        md += f"\n\n### Prediction drift\n\n```\n{json.dumps(pdrift, indent=2)}\n```\n\n### Performance on labelled outcomes (labels from the held-out split; the simulation changes inputs, not outcomes)\n\n```\n{json.dumps(perf, indent=2)}\n```\nBaseline test ROC-AUC: {baseline_auc:.4f}\n\n### Retraining policy decision (recommendation only)\n\n```\n{json.dumps(decision, indent=2)}\n```\n\n---\n\n"
    if source != "home_credit":
        md = "> **Model and data are SYNTHETIC.**\n\n" + md
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "monitoring_demo_report.md").write_text(md, encoding="utf-8")
    print(md[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
