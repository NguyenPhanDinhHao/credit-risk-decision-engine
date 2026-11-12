"""Model loading and request-scoped dependencies."""
from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException

from credit_risk.config import load_config, resolve_path
from credit_risk.models.predict import ModelBundle

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_state: dict[str, object] = {"bundle": None, "error": None}


@lru_cache(maxsize=1)
def _cfg() -> dict:
    return load_config()


def model_path() -> Path:
    env = os.environ.get("CREDIT_RISK_MODEL_PATH")
    return resolve_path(env) if env else resolve_path(_cfg()["api"]["model_path"])


def load_bundle(force: bool = False) -> ModelBundle | None:
    """Load the model once; returns None (and remembers the error) if unavailable."""
    with _lock:
        if _state["bundle"] is None or force:
            try:
                _state["bundle"] = ModelBundle.load(model_path())
                _state["error"] = None
                logger.info("Model loaded: %s", _state["bundle"].model_version)  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001
                _state["bundle"], _state["error"] = None, str(exc)
                logger.warning("Model not loaded: %s", exc)
        return _state["bundle"]  # type: ignore[return-value]


def get_bundle() -> ModelBundle:
    bundle = load_bundle()
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is not available. Train a model first.")
    return bundle


def settings() -> dict:
    cfg = _cfg()["api"]
    return {
        "max_batch_size": int(cfg["max_batch_size"]),
        "log_predictions": os.environ.get("CREDIT_RISK_LOG_PREDICTIONS", str(cfg["log_predictions"])).lower() == "true",
        "prediction_log": resolve_path(os.environ.get("CREDIT_RISK_PREDICTION_LOG", cfg["prediction_log"])),
    }
