# Architecture

## Components

```mermaid
flowchart LR
    subgraph Offline["Offline: training & evaluation"]
        D[(application_train.csv<br/>user-provided)] --> L[data.loader<br/>discovery + instructions]
        L --> V[data.validation]
        V --> S[Stratified split<br/>train / val / test]
        S --> FE[features.engineering<br/>FeatureEngineer]
        FE --> P[features.preprocessing<br/>ColumnTransformer]
        P --> T[models.train<br/>tune on validation<br/>LR / RF / LightGBM]
        T --> C[models.calibration<br/>fit on validation]
        C --> SEL[Select model<br/>val AUC + interpretability]
        SEL --> TE[Single test evaluation<br/>+ decision analysis]
        SEL --> SH[explainability<br/>SHAP]
        T -. params / metrics .-> ML[(MLflow)]
        SEL --> B[(model_bundle.joblib<br/>pipeline + calibrator + metadata)]
        S --> REF[(reference_features.csv.gz<br/>reference_scores.npy)]
    end
    subgraph Online["Online: serving"]
        B --> API[FastAPI<br/>/predict /predict/batch<br/>/model-info /health]
        CL[Client] --> API
        API --> LOG[(predictions.jsonl<br/>scores only, no raw inputs)]
    end
    subgraph Mon["Monitoring"]
        REF --> DR[data_drift<br/>PSI / KS / missingness]
        LOG --> PM[prediction_monitor<br/>prediction drift]
        LAB[(delayed labels)] --> PERF[performance report]
        DR --> POL[Retraining policy<br/>recommendation only]
        PM --> POL
        PERF --> POL
    end
```

## Training sequence and leakage controls

```mermaid
sequenceDiagram
    participant Tr as Train split
    participant Va as Validation split
    participant Te as Test split
    Tr->>Tr: fit FeatureEngineer schema, imputers, scalers, encoders (cross-fitted target encoding)
    Tr->>Tr: fit candidate models
    Va->>Va: tune hyper-parameters (random search), pick calibration method (CV Brier)
    Va->>Va: fit calibrator, choose max-F1 and cost-optimal thresholds
    Va->>Va: select final model
    Te->>Te: ONE final evaluation, policy and cost analysis
```

## Module map

| Package | Responsibility |
|---|---|
| `credit_risk.config` | YAML configuration, env overrides, lightweight mode |
| `credit_risk.data` | `loader` (discovery, manual-download instructions), `validation`, `audit`, `synthetic` (tests/demo only) |
| `credit_risk.features` | `engineering.FeatureEngineer` (serialised in the pipeline), `preprocessing` (column groups, ColumnTransformer, split) |
| `credit_risk.models` | `baseline` (factories), `train`, `evaluate`, `calibration`, `predict.ModelBundle` |
| `credit_risk.decision` | risk bands, threshold sweep, expected-cost, policy comparison |
| `credit_risk.explainability` | SHAP global/local explanations |
| `credit_risk.api` | FastAPI app, schemas, model loading |
| `credit_risk.monitoring` | drift statistics, prediction logging, retraining policy |

## Design decisions

- **One artifact for inference.** `ModelBundle` serialises feature engineering + preprocessing + model + calibrator + schema, so API and offline evaluation share a single code path. `scripts/evaluate.py` reproduces training-time test metrics from raw data to confirm this.
- **Config-driven.** Paths, split, tuning budget, risk bands, policies, costs and monitoring thresholds live in `configs/`.
- **Local-first deployment.** Docker Compose separates inference (`api`), a one-off `trainer` job and `mlflow`; no cloud credentials.
- **No raw applicant data in logs.** The prediction log stores score, category, model version and the *names* of missing core fields only.
