# Model Card - Credit Risk Decision Engine

> Fill the *Results* section from `artifacts/metrics.json` and `reports/experiment_report.md` after training on the real dataset. Nothing below is a measured result.

## Model details
- Task: estimate the probability that a loan applicant will experience payment difficulties (`TARGET = 1`).
- Candidates: logistic regression, random forest, LightGBM (+ prior-rate reference). The selected model, version string and calibration method are recorded in `artifacts/model_bundle.joblib` metadata and exposed at `GET /model-info`.
- Preprocessing, features and calibrator are part of the same serialised bundle.
- Author/contact: <fill in>. Version: see `/model-info`.

## Intended use
Demonstrating an end-to-end credit-risk modelling workflow (training, calibration, explanation, serving, monitoring) in a portfolio setting.

## Out-of-scope use
Real credit approval, pricing, collections, or any automated decision affecting individuals. Use outside the Home Credit population.

## Data
Kaggle *Home Credit Default Risk*, `application_train.csv` (user-provided), stratified 60/20/20 split, seed 42. See `docs/methodology.md`.

## Features
Raw application fields plus engineered ratios (credit/income, annuity/income, credit term, credit/goods, income per person, mean external score, employment-placeholder flag). Excluded by default as sensitive: `CODE_GENDER`, `DAYS_BIRTH`, `NAME_FAMILY_STATUS`, `DAYS_EMPLOYED_PERC`.

## Evaluation protocol
Tuning, calibration choice, thresholds and selection on validation; single test evaluation. Metrics: ROC-AUC, PR-AUC, KS, Brier, calibration curve, precision/recall/F1 at a validation-chosen threshold.

## Results
| Item | Value |
|---|---|
| Data source | _to be filled_ (`home_credit` expected) |
| Test ROC-AUC / PR-AUC / KS | _to be filled_ |
| Brier (uncalibrated -> calibrated) | _to be filled_ |
| Selected model and why | _to be filled; see selection rule in methodology_ |

## Ethical considerations
Sensitive attributes excluded but proxies remain; no fairness metrics computed; risk categories are illustrative and not decisions. See `reports/limitations.md`.

## Caveats and recommendations
Do not deploy. For any real use: out-of-time validation, reject inference, fairness audit with appropriate attributes, legal review, challenger models, ongoing monitoring with delayed labels.
