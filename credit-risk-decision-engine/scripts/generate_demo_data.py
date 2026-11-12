#!/usr/bin/env python
"""Write SYNTHETIC application_train.csv (artificial schema-compatible data) to data/synthetic/.

Only for smoke tests and demos. Models trained on it have no real-world meaning.
Use:  python scripts/train.py --data-dir data/synthetic --lightweight
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.config import load_config, resolve_path  # noqa: E402
from credit_risk.data.synthetic import generate_synthetic_applications  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    cfg = load_config()
    out = Path(a.out_dir) if a.out_dir else resolve_path(cfg["paths"]["synthetic_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_applications(a.n, a.seed)
    df.to_csv(out / cfg["data"]["application_train"], index=False)
    print(f"Wrote {len(df):,} SYNTHETIC rows to {out / cfg['data']['application_train']} (default rate {df.TARGET.mean():.3f})")


if __name__ == "__main__":
    main()
