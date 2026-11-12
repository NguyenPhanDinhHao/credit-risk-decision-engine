"""Dataset discovery and loading.

The raw Home Credit files are never downloaded or committed by this project. If they are missing,
the loader raises ``DatasetNotFoundError`` with manual download instructions.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk.config import resolve_path

logger = logging.getLogger(__name__)

DOWNLOAD_INSTRUCTIONS = """\
Home Credit Default Risk data was not found.

Manual download (no credentials are stored by this project):
  1. Create/sign in to a Kaggle account and accept the competition rules at
     https://www.kaggle.com/competitions/home-credit-default-risk/data
  2. Download "home-credit-default-risk.zip" through the website
     (or use your own Kaggle CLI setup: kaggle competitions download -c home-credit-default-risk).
  3. Unzip it so that at least application_train.csv is located in: {data_dir}
     (configurable via configs/config.yaml -> paths.data_dir or CREDIT_RISK_DATA_DIR).

To only exercise the code path with SYNTHETIC data (not real results), run with --demo.
"""

SYNTHETIC_SOURCE = "synthetic_demo"
REAL_SOURCE = "home_credit"


class DatasetNotFoundError(FileNotFoundError):
    """Raised when required dataset files are missing."""


def data_dir_from_cfg(cfg: dict[str, Any], synthetic: bool = False) -> Path:
    return resolve_path(cfg["paths"]["synthetic_dir" if synthetic else "data_dir"])


def discover_files(cfg: dict[str, Any], data_dir: Path | None = None) -> dict[str, Path | None]:
    """Map logical table names to existing file paths (None if absent)."""
    data_dir = data_dir or data_dir_from_cfg(cfg)
    names = {"application_train": cfg["data"]["application_train"], "application_test": cfg["data"]["application_test"]}
    names.update(cfg["data"].get("optional_tables", {}))
    return {key: (data_dir / fn if (data_dir / fn).exists() else None) for key, fn in names.items()}


def dataset_status_report(cfg: dict[str, Any], data_dir: Path | None = None) -> str:
    data_dir = data_dir or data_dir_from_cfg(cfg)
    found = discover_files(cfg, data_dir)
    lines = [f"Data directory: {data_dir}"]
    for key, path in found.items():
        required = "required" if key == "application_train" else "optional"
        lines.append(f"  [{'OK' if path else 'MISSING'}] {key:<24} ({required}) {path or ''}")
    return "\n".join(lines)


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    return df


def stratified_sample(df: pd.DataFrame, n: int, target: str, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df
    sampled, _ = train_test_split(df, train_size=n, stratify=df[target], random_state=seed)
    return sampled.reset_index(drop=True)


def load_application_train(cfg: dict[str, Any], data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or data_dir_from_cfg(cfg)
    path = data_dir / cfg["data"]["application_train"]
    if not path.exists():
        raise DatasetNotFoundError(
            f"{path} not found.\n{dataset_status_report(cfg, data_dir)}\n\n"
            + DOWNLOAD_INSTRUCTIONS.format(data_dir=data_dir)
        )
    logger.info("Loading %s", path)
    df = _downcast(pd.read_csv(path))
    n = cfg["data"].get("sample_rows")
    if n:
        df = stratified_sample(df, int(n), cfg["data"]["target_column"], cfg["project"]["seed"])
        logger.info("Using stratified subsample of %d rows", len(df))
    return df


def load_dataset(cfg: dict[str, Any], demo: bool = False) -> tuple[pd.DataFrame, str]:
    """Load real data if available. ``demo=True`` explicitly allows synthetic data when real data is absent.

    Returns (dataframe, data_source). The synthetic fallback is never silent.
    """
    real_dir = data_dir_from_cfg(cfg)
    if discover_files(cfg, real_dir)["application_train"] is not None:
        df = load_application_train(cfg, real_dir)
        if cfg["data"].get("use_supplementary_tables"):
            from credit_risk.features.engineering import attach_supplementary_features

            df = attach_supplementary_features(df, cfg, real_dir)
        return df, REAL_SOURCE
    if not demo:
        load_application_train(cfg, real_dir)  # raises DatasetNotFoundError with instructions
    from credit_risk.data.synthetic import generate_synthetic_applications

    logger.warning("REAL DATA NOT FOUND - generating SYNTHETIC demo data. Results are NOT real model results.")
    n = int(cfg["data"].get("sample_rows") or 30000)
    return generate_synthetic_applications(n=n, seed=cfg["project"]["seed"]), SYNTHETIC_SOURCE


def load_optional_table(cfg: dict[str, Any], name: str, data_dir: Path | None = None) -> pd.DataFrame | None:
    path = discover_files(cfg, data_dir).get(name)
    if path is None:
        logger.info("Optional table %s not available; skipping.", name)
        return None
    return _downcast(pd.read_csv(path))
