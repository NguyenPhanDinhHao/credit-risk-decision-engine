# API reference

Start: `uvicorn credit_risk.api.main:app --port 8000` (or `docker compose up --build`). Interactive docs: <http://localhost:8000/docs>.
The model bundle is read from `CREDIT_RISK_MODEL_PATH` (default `artifacts/model_bundle.joblib`). Without it, `/health` reports `degraded` and prediction endpoints return `503`.

All fields of an applicant are optional; omitted fields are treated as missing and imputed with training statistics, which reduces quality. Extra Home Credit columns are accepted (scalars only). Responses always carry a disclaimer.

## GET /health
```json
{"status": "ok", "model_loaded": true}
```

## GET /model-info
Model name/version, data source, calibration method, risk bands, excluded sensitive features and headline test metrics from training (`data_source` tells you whether it was trained on synthetic data).

## POST /predict?explain=false
Request:
```json
{"AMT_INCOME_TOTAL": 150000, "AMT_CREDIT": 450000, "AMT_ANNUITY": 18000, "NAME_CONTRACT_TYPE": "Cash loans",
 "DAYS_EMPLOYED": -1500, "EXT_SOURCE_1": 0.5, "EXT_SOURCE_2": 0.6, "EXT_SOURCE_3": 0.4}
```
Response (values illustrative):
```json
{"probability_of_default": 0.0713, "risk_category": "Medium Risk",
 "model_version": "<model>-<yyyymmdd>-<hash>", "top_factors": null, "disclaimer": "..."}
```
With `?explain=true`, `top_factors` lists the 5 largest SHAP contributions (`feature`, `contribution`, `direction`). These describe the model's behaviour, not causes.

Core fields and constraints: `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE` > 0; `EXT_SOURCE_1/2/3` in [0,1]; `NAME_CONTRACT_TYPE` in {Cash loans, Revolving loans}; `FLAG_OWN_CAR/REALTY` in {Y,N}; `REGION_RATING_CLIENT` in 1..3; `CNT_CHILDREN` integer 0..25; `DAYS_*` <= 0 (`DAYS_EMPLOYED` may be 365243 placeholder).

## POST /predict/batch
Request: `{"applicants": [ {...}, {...} ]}` (max `api.max_batch_size`, default 500; larger -> `413`).
```json
{"count": 2, "model_version": "...", "predictions": [{"index": 0, "probability_of_default": 0.07, "risk_category": "Medium Risk"},
                                                  {"index": 1, "probability_of_default": 0.02, "risk_category": "Low Risk"}], "disclaimer": "..."}
```

## Errors
| Code | Meaning | Body |
|---|---|---|
| 422 | Invalid input | `{"detail": "Invalid input", "errors": [{"field": "AMT_INCOME_TOTAL", "message": "..."}]}` (submitted values are not echoed) |
| 413 | Batch too large | `{"detail": "..."}` |
| 503 | Model not available | `{"detail": "Model is not available. Train a model first."}` |
| 500 | Unexpected error | `{"detail": "Internal server error", "reference": "<id>"}`; the stack trace is only in server logs |

## curl
```bash
curl -s localhost:8000/health
curl -s -X POST "localhost:8000/predict?explain=true" -H 'content-type: application/json' \
     -d '{"AMT_INCOME_TOTAL":150000,"AMT_CREDIT":450000,"EXT_SOURCE_2":0.6}'
```

## Prediction log
Served predictions append to `artifacts/monitoring/predictions.jsonl` (`CREDIT_RISK_LOG_PREDICTIONS=false` disables): timestamp, model version, score, category and the names of missing core fields. No raw applicant values are stored.
