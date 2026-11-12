# Limitations

**Scope.** Educational portfolio project on public data. It is not a lending system, has no regulatory validation, and must not be used for real credit decisions.

## Data
- Home Credit Default Risk is a historical competition dataset (one lender, specific markets and period). Its default definition ("payment difficulties") and population do not transfer to other lenders, products or markets (including Vietnam).
- Rejected applicants are absent: models learn only from approved loans (selection bias / reject inference problem).
- No timestamps in `application_train.csv`; the split is random stratified, not out-of-time, so performance under temporal shift is untested and likely optimistic.
- Supplementary tables are only partly used (bureau and previous applications as simple aggregates, off by default).

## Modelling
- Hyper-parameter search is a small random search; results depend on the budget (`training.n_iter_tuning`) and are not exhaustively optimised.
- The final model is trained on the training split only; refitting on train+validation would use more data but remove clean out-of-sample calibration.
- Calibration is checked on a single random split; no confidence intervals or repeated-split variance are reported.
- Risk bands, policy thresholds and costs are illustrative configuration values, not industry standards and not derived from real business economics.

## Fairness and ethics
- Sensitive attributes (`CODE_GENDER`, age, `NAME_FAMILY_STATUS`) are excluded by default, but proxies remain (organisation, occupation, education, region, family size). No group fairness metrics are computed. Disparate impact could exist even without the explicit attributes.
- The legal acceptability of any feature depends on jurisdiction and use case.

## Explainability
- SHAP describes model behaviour, not causal effects. Correlated features share attribution unpredictably; explanations are in the uncalibrated margin space; attributions for omitted (imputed) inputs are not meaningful business explanations.

## Serving and monitoring
- The API accepts a subset (about 20) of the roughly 120 raw columns; omitted fields are imputed, which can reduce accuracy. A production service would require the full feature contract.
- Monitoring uses PSI/KS screening and a simple policy. Thresholds are conventions, not tuned. The drift demo is simulated. Performance monitoring needs labels that, for credit, arrive months later.
- No authentication, rate limiting, request tracing or model registry promotion workflow. MLflow tracks runs and artifacts; the model version is a string embedded in the bundle, not an MLflow Model Registry stage.
- Model bundles are `joblib` pickles: load only files you trust.

## Not verified in the development environment
The Docker image and Compose stack were written but **not built or run** (no Docker daemon available). Real-data training was not run (dataset unavailable offline). See README "Verification status".
