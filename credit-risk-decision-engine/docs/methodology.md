# Methodology

## 1. Problem framing
Binary classification: `TARGET = 1` marks clients with payment difficulties in the Home Credit Default Risk data. The model estimates a probability of default (PD); a decision-support layer maps PD to illustrative risk categories. This is an educational project and not a lending system.

## 2. Data
- Required: `application_train.csv` (user-supplied; never committed or auto-downloaded). Optional `bureau.csv` and `previous_application.csv` are aggregated per applicant when `--supplementary` is used. Other supplementary tables are not used yet.
- `application_test.csv` has no labels and is not used for evaluation.
- Validation (`data/validation.py`): required columns, numeric dtypes, target presence and values in {0,1}, both classes present, duplicate IDs, nulls in critical fields; plausibility rules (e.g. EXT_SOURCE in [0,1]) are reported as warnings.
- Audit (`data/audit.py`, `scripts/run_eda.py`): shape, types, missingness, duplicates, target balance, cardinality, IQR outliers, correlations and a univariate-AUC leakage screen. Findings on real data must be reviewed by a human; the screen flags suspicious features, it cannot certify absence of leakage.

## 3. Splits and protocol
Stratified train 60% / validation 20% / test 20% (seed 42). Hyper-parameter tuning (random search, `training.n_iter_tuning` trials from `configs/model_config.yaml`), calibration-method choice, threshold choice and model selection use **validation only**. The test split is evaluated once after selection. The final model is trained on the training split only (not refit on train+validation) so the calibrator sees genuinely out-of-sample predictions.

## 4. Preprocessing and features
All statistics are learned on the training split.
- Numeric: median imputation; winsorisation at training 1%/99% and standardisation for logistic regression only. Explicit missing flags only for columns with >= 1% missing in training (flags for near-never-missing columns carry no signal and, when scaled, become extreme values if a request omits the field - this was found and fixed during API testing, see regression test `test_omitted_fields_do_not_produce_extreme_predictions`).
- Categorical: columns with <= 15 levels are one-hot encoded (unseen levels ignored); high-cardinality columns (e.g. `ORGANIZATION_TYPE`) use cross-fitted target encoding. Missing is its own category only where training data has missing values; otherwise omitted fields are imputed with the training mode.
- Engineered features (zero/inf-safe): `CREDIT_INCOME_RATIO` (leverage), `ANNUITY_INCOME_RATIO` (repayment burden), `CREDIT_TERM` (annuity/credit, inverse term), `CREDIT_GOODS_RATIO`, `INCOME_PER_PERSON`, `EXT_SOURCE_MEAN`, `DAYS_EMPLOYED_ANOM`, `DAYS_EMPLOYED_PERC`. `EMPLOYMENT_TO_AGE_RATIO` is mathematically identical to `DAYS_EMPLOYED_PERC` and is therefore implemented once.
- `DAYS_EMPLOYED = 365243` is a placeholder; it becomes NaN with an anomaly flag.

### Fairness and sensitive attributes
Excluded from the model by default (`features.sensitive_columns`): `CODE_GENDER`, `DAYS_BIRTH`, `NAME_FAMILY_STATUS`, and the age-derived `DAYS_EMPLOYED_PERC`. Removing columns does **not** remove proxy effects: fields such as `ORGANIZATION_TYPE`, `OCCUPATION_TYPE`, `NAME_EDUCATION_TYPE`, region features, `CNT_CHILDREN`/`CNT_FAM_MEMBERS` (and `INCOME_PER_PERSON`) may correlate with protected characteristics and are retained. No fairness metrics are computed in this project; a real assessment needs legally appropriate attributes, group-wise error analysis and regulatory review. Whether use of any of these fields is lawful depends on jurisdiction.

## 5. Models
Logistic regression, random forest, LightGBM, plus a prior-rate reference (ROC-AUC 0.5). Class weighting (`balanced`) is a *searched hyper-parameter*, not assumed: it often improves ranking for imbalanced data but distorts probabilities, which calibration then repairs. SMOTE is not used (no evidence it helps tree models on tabular credit data, and it complicates calibration).

## 6. Metrics
ROC-AUC, Gini, PR-AUC (average precision), KS, precision/recall/F1 and confusion matrix (threshold chosen on validation by max-F1), Brier score, log-loss, calibration curves. Accuracy is reported but not used: with a low default rate, "approve everyone" is highly accurate and useless.

**Selection rule:** highest validation ROC-AUC; among models within `selection_auc_tolerance` (0.003) of the best, choose the most interpretable (logistic regression < LightGBM < random forest). Calibration quality, operational cost and explainability are reported alongside and inform the written decision in the model card.

## 7. Calibration
Methods: none, sigmoid (Platt), isotonic. Fitted on validation predictions of the training-fitted model; the method is chosen by 5-fold cross-validated Brier score *inside* the validation split; test predictions are never used. Calibration matters because PDs feed thresholds, pricing, provisioning and expected-loss estimates, not only rankings.

## 8. Decision analysis
- Risk bands (`decision.risk_bands`): Low < 0.05 <= Medium < 0.15 <= High. **Illustrative**, not industry standards.
- Policy scenarios: conservative / balanced / higher_acceptance flag thresholds.
- Expected cost per applicant = (FN x c_FN + FP x c_FP) / n with arbitrary relative costs (5 : 1 by default). FN = a defaulter not flagged; FP = a good applicant flagged. Replace with business estimates (exposure, LGD, margin).
- Threshold sweep, policy comparison, tier summary, and a cost-optimal threshold chosen on validation then evaluated on test.
- A risk category is not an approval decision.

## 9. Explainability
SHAP (TreeExplainer for tree models, LinearExplainer for logistic regression): global importance, beeswarm summary, dependence plots, local top-k explanations (also available at `POST /predict?explain=true`). SHAP explains the *model*, not causal effects; correlated features split attribution; proxies of sensitive attributes can appear important. Values are in the uncalibrated margin space.

## 10. Monitoring
| Concept | Definition | Needs labels? | Tool here |
|---|---|---|---|
| Data (covariate) drift | P(X) changes | no | PSI, KS, missing-rate change |
| Concept drift | P(y given X) changes | yes | only visible via performance monitoring |
| Prediction drift | distribution of scores changes | no | PSI/KS on scores |
| Performance degradation | metrics on labelled outcomes worsen | yes (delayed) | ROC-AUC, Brier, calibration gap |

Prediction drift is a symptom with several causes and does not prove degradation. PSI rules of thumb (0.10 moderate, 0.25 high) are conventions, configurable in `configs/config.yaml`.

**Retraining trigger policy** (`evaluate_retraining_trigger`, recommendation only - never automatic):
1. `RETRAIN_RECOMMENDED` only with labelled evidence (>= 1000 labelled samples): ROC-AUC drop >= 0.03 vs baseline, or mean predicted vs observed default rate gap >= 25% (consider recalibration first).
2. Drift without labelled evidence -> `INVESTIGATE`; persisting for >= 3 consecutive windows -> `INVESTIGATE_PRIORITY` (check upstream pipeline, segment analysis, accelerate label collection).
3. Otherwise `NO_ACTION`. Credit labels mature slowly, so monitoring must combine drift signals now with performance checks later.

The demo (`scripts/monitor_demo.py`) perturbs held-out data (higher incomes/loan amounts and more missing `EXT_SOURCE_1`) in two severities. It is **simulated**.

## 11. Environment
Developed and tested with Python 3.12.3, pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1, LightGBM 4.7.0, SHAP 0.54.1, MLflow 3.18.0, FastAPI 0.162.4. `requirements.txt` uses lower bounds only; the code contains one version switch (target-encoder cross-validation arguments moved in scikit-learn 1.9). Docker targets Python 3.11 and installs the latest compatible releases; this combination has not been built or tested here (see README "Verification status").
