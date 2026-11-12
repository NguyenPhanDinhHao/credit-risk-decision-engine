#!/usr/bin/env python
"""Re-evaluate the saved model bundle on the deterministic test split (same seed/config as training)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from credit_risk.config import apply_lightweight, load_config, resolve_path  # noqa: E402
from credit_risk.data.loader import DatasetNotFoundError, load_dataset  # noqa: E402
from credit_risk.decision.policy import policy_comparison, tier_summary  # noqa: E402
from credit_risk.features.preprocessing import split_data  # noqa: E402
from credit_risk.logging_config import setup_logging  # noqa: E402
from credit_risk.models.evaluate import full_metrics  # noqa: E402
from credit_risk.models.predict import ModelBundle  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None); ap.add_argument("--data-dir"); ap.add_argument("--lightweight", action="store_true")
    ap.add_argument("--demo", action="store_true", help="use SYNTHETIC data if real data is missing")
    a = ap.parse_args()
    setup_logging()
    cfg = apply_lightweight(load_config()) if a.lightweight else load_config()
    if a.data_dir:
        cfg["paths"]["data_dir"] = a.data_dir
    try:
        df, source = load_dataset(cfg, demo=a.demo)
    except DatasetNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    bundle = ModelBundle.load(a.model or resolve_path(cfg["api"]["model_path"]))
    s, t = cfg["split"], cfg["data"]["target_column"]
    *_, X_te, _, _, y_te = split_data(df, t, s["test_size"], s["val_size"], cfg["project"]["seed"])
    p = bundle.predict_proba(X_te)
    thr = cfg["decision"]["policies"]["balanced"]["flag_threshold"]
    out = {"data_source": source, "model_version": bundle.model_version, "metrics": full_metrics(y_te.to_numpy(), p, thr),
           "policies": policy_comparison(y_te.to_numpy(), p, cfg["decision"]["policies"], cfg["decision"]["costs"]).to_dict("records"),
           "tiers": tier_summary(y_te.to_numpy(), p, bundle.risk_bands).to_dict("records")}
    print(json.dumps(out, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    if source != "home_credit":
        print("NOTE: SYNTHETIC data; not real results.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
