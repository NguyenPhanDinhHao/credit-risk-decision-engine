"""Configuration loading. Paths resolve relative to the repository root."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.environ.get("CREDIT_RISK_HOME", Path(__file__).resolve().parents[2]))
CONFIG_DIR = PROJECT_ROOT / "configs"


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(config_path: str | Path | None = None, model_config_path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml, attach model search spaces under ``cfg['models']`` and apply env overrides."""
    cfg = _read_yaml(Path(config_path) if config_path else CONFIG_DIR / "config.yaml")
    cfg["models"] = _read_yaml(Path(model_config_path) if model_config_path else CONFIG_DIR / "model_config.yaml")
    env_map = {
        "CREDIT_RISK_DATA_DIR": ("paths", "data_dir"),
        "CREDIT_RISK_ARTIFACTS_DIR": ("paths", "artifacts_dir"),
        "CREDIT_RISK_MLFLOW_URI": ("mlflow", "tracking_uri"),
        "CREDIT_RISK_MODEL_PATH": ("api", "model_path"),
    }
    for env, (section, key) in env_map.items():
        if os.environ.get(env):
            cfg.setdefault(section, {})[key] = os.environ[env]
    return cfg


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def apply_lightweight(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of cfg configured for a fast run on a laptop."""
    cfg = copy.deepcopy(cfg)
    lw = cfg.get("lightweight", {})
    cfg["data"]["sample_rows"] = lw.get("sample_rows", 40000)
    cfg["training"]["n_iter_tuning"] = lw.get("n_iter_tuning", 3)
    cfg["training"]["shap_sample_size"] = lw.get("shap_sample_size", 500)
    cfg["models"]["random_forest"]["fixed"]["n_estimators"] = lw.get("rf_n_estimators", 100)
    cfg["models"]["lightgbm"]["fixed"]["n_estimators"] = lw.get("lgbm_n_estimators", 200)
    return cfg
