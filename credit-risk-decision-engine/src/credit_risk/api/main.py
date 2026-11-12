"""FastAPI application. Run: uvicorn credit_risk.api.main:app --port 8000  (docs at /docs)."""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from credit_risk import __version__
from credit_risk.api import dependencies as deps
from credit_risk.api.schemas import (
    ApplicantInput,
    BatchPrediction,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ModelInfo,
    PredictionResponse,
)
from credit_risk.logging_config import setup_logging
from credit_risk.models.predict import DISCLAIMER, ModelBundle
from credit_risk.monitoring.prediction_monitor import PredictionLogger

logger = logging.getLogger("credit_risk.api")

CORE_FIELDS = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_EMPLOYED", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    deps.load_bundle()
    yield


app = FastAPI(
    title="Credit Risk Decision Engine API",
    version=__version__,
    description="Educational credit default prediction service. NOT a lending decision system.",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    # Return field locations and messages only; never echo submitted values.
    errors = [{"field": ".".join(str(x) for x in e["loc"][1:]) or "body", "message": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": "Invalid input", "errors": errors})


@app.exception_handler(Exception)
async def unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
    ref = uuid.uuid4().hex[:8]
    logger.exception("Unhandled error ref=%s", ref)
    return JSONResponse(status_code=500, content={"detail": "Internal server error", "reference": ref})


def _log(bundle: ModelBundle, df: pd.DataFrame, probs, cats) -> None:
    s = deps.settings()
    if s["log_predictions"]:
        PredictionLogger(s["prediction_log"]).log_batch(bundle.model_version, probs, cats, df[[c for c in CORE_FIELDS if c in df]] if len(df.columns) else df, CORE_FIELDS)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    loaded = deps.load_bundle() is not None
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded)


@app.get("/model-info", response_model=ModelInfo, tags=["system"])
def model_info(bundle: ModelBundle = Depends(deps.get_bundle)) -> ModelInfo:
    md = bundle.metadata
    m = md.get("selected_test_metrics", {})
    keep = ("roc_auc", "pr_auc", "ks", "brier")
    return ModelInfo(
        model_name=bundle.model_name, model_version=bundle.model_version, data_source=md.get("data_source"),
        trained_at=md.get("trained_at"), calibration_method=md.get("calibration_method"),
        n_features_after_preprocessing=md.get("n_features_after_preprocessing"),
        risk_bands={"low_max": bundle.risk_bands.low_max, "medium_max": bundle.risk_bands.medium_max},
        excluded_sensitive_features=list(md.get("excluded_sensitive", [])),
        test_metrics={k: float(m[k]) for k in keep if k in m}, disclaimer=DISCLAIMER,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(
    applicant: ApplicantInput,
    explain: bool = Query(False, description="Include top SHAP contributions (model attribution, not causal)"),
    bundle: ModelBundle = Depends(deps.get_bundle),
) -> PredictionResponse:
    df = pd.DataFrame([applicant.to_record()])
    p = bundle.predict_proba(df)
    cat = bundle.risk_category(p)
    factors: list[dict[str, Any]] | None = None
    if explain:
        try:
            factors = [{k: f[k] for k in ("feature", "contribution", "direction")} for f in bundle.explain(df, top_k=5)[0]]
        except Exception:  # noqa: BLE001
            logger.exception("Explanation failed; returning prediction without it")
    _log(bundle, df, p, cat)
    return PredictionResponse(
        probability_of_default=float(p[0]), risk_category=str(cat[0]), model_version=bundle.model_version,
        top_factors=factors, disclaimer=DISCLAIMER,
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(req: BatchRequest, bundle: ModelBundle = Depends(deps.get_bundle)) -> BatchResponse:
    limit = deps.settings()["max_batch_size"]
    if len(req.applicants) > limit:
        raise HTTPException(status_code=413, detail=f"Batch too large: {len(req.applicants)} > {limit}")
    df = pd.DataFrame([a.to_record() for a in req.applicants])
    p = bundle.predict_proba(df)
    cats = bundle.risk_category(p)
    _log(bundle, df, p, cats)
    return BatchResponse(
        count=len(p), model_version=bundle.model_version, disclaimer=DISCLAIMER,
        predictions=[BatchPrediction(index=i, probability_of_default=float(x), risk_category=str(c)) for i, (x, c) in enumerate(zip(p, cats, strict=True))],
    )
