# GitHub publishing checklist

1. Replace `<Your Name>` in `LICENSE` and `<fill in>` in `reports/model_card.md`.
2. Train on the real dataset, then commit the generated `reports/experiment_report.md` and selected figures you want to showcase from `reports/figures/` (remove the `.gitignore` exception for those files or `git add -f`). Fill the README *Results* table and the model card from `artifacts/metrics.json`.
3. Run `ruff check src tests scripts` and `pytest`; run `docker compose build` yourself before claiming Docker support.
4. Make sure `git status` shows no data files, `.env`, `mlflow.db`, `mlruns/` or `artifacts/*`.
5. Rewrite "What I Learned" in README in your own words based on your own run.

## Suggested commit sequence
```
chore: project skeleton, configs, gitignore, license
feat(data): dataset discovery, validation, audit, synthetic fixtures for tests
feat(features): FeatureEngineer and leakage-safe preprocessing
feat(models): LR/RF/LightGBM training with validation-only tuning
feat(models): leakage-safe probability calibration
feat(decision): risk bands, policy scenarios, threshold and cost analysis
feat(explainability): SHAP global and local explanations
feat(tracking): MLflow integration (optional)
feat(api): FastAPI service with validation and safe errors
feat(monitoring): drift metrics, prediction log, retraining policy
test: unit, API and monitoring tests
build: Dockerfile and docker-compose
docs: README, methodology, architecture, model card, limitations
```
