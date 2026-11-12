"""Training orchestration: split -> feature engineering -> tune (validation) -> calibrate (validation)
-> select -> single final test evaluation -> decision analysis -> SHAP -> artifacts -> MLflow."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline

from credit_risk.config import resolve_path
from credit_risk.data.validation import validate_application_data
from credit_risk.decision.policy import (
    RiskBands,
    no_model_cost,
    optimal_cost_threshold,
    policy_comparison,
    threshold_sweep,
    tier_summary,
)
from credit_risk.features.engineering import FeatureEngineer
from credit_risk.features.preprocessing import build_preprocessor, infer_column_groups, split_data
from credit_risk.models.baseline import INTERPRETABILITY_RANK, NEEDS_SCALING, make_estimator
from credit_risk.models.calibration import Calibrator, plot_calibration, select_calibration_method
from credit_risk.models.evaluate import (
    best_f1_threshold,
    full_metrics,
    plot_confusion_matrix,
    plot_roc_pr,
    probability_metrics,
)
from credit_risk.models.predict import ModelBundle
from credit_risk.reporting import write_experiment_report

logger = logging.getLogger(__name__)


def _grid_size(space: dict[str, list]) -> int:
    n = 1
    for v in space.values():
        n *= len(v)
    return n


def fit_candidate(name: str, cfg: dict[str, Any], groups, Xtr_fe, y_tr, Xva_fe, y_va, n_iter: int, seed: int) -> dict[str, Any]:
    """Random-search tuning on the validation split. Preprocessing is fitted once on training data."""
    f = cfg["features"]
    prep = build_preprocessor(groups, NEEDS_SCALING[name], tuple(f["winsorize_quantiles"]) if name == "logistic_regression" else None, seed)
    Mtr, Mva = prep.fit_transform(Xtr_fe, y_tr), prep.transform(Xva_fe)
    spec = cfg["models"][name]
    n_jobs = cfg["training"].get("n_jobs", -1)
    combos = list(ParameterSampler(spec["space"], n_iter=min(n_iter, _grid_size(spec["space"])), random_state=seed))
    best: dict[str, Any] | None = None
    t0 = time.time()
    for params in combos:
        est = make_estimator(name, {**spec["fixed"], **params}, seed, n_jobs)
        est.fit(Mtr, y_tr)
        auc = roc_auc_score(y_va, est.predict_proba(Mva)[:, 1])
        logger.info("  %-20s %s -> val AUC %.4f", name, params, auc)
        if best is None or auc > best["val_auc"]:
            best = {"est": est, "params": params, "val_auc": float(auc)}
    assert best is not None
    best.update(prep=prep, Mva=Mva, seconds=time.time() - t0, n_trials=len(combos))
    return best


def select_final_model(candidates: dict[str, dict[str, Any]], tolerance: float) -> tuple[str, str]:
    best_auc = max(c["val_auc"] for c in candidates.values())
    eligible = [n for n, c in candidates.items() if c["val_auc"] >= best_auc - tolerance]
    chosen = sorted(eligible, key=lambda n: (INTERPRETABILITY_RANK[n], -candidates[n]["val_auc"]))[0]
    rule = (f"highest validation ROC-AUC, but among models within {tolerance} AUC of the best, prefer the most interpretable "
            f"(logistic regression < LightGBM < random forest); eligible: {sorted(eligible)}")
    return chosen, rule


def _plot_threshold_sensitivity(sweep: pd.DataFrame, policies: dict[str, dict[str, float]], out: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(sweep["threshold"], sweep["precision"], label="precision")
    ax[0].plot(sweep["threshold"], sweep["recall"], label="recall")
    ax[0].plot(sweep["threshold"], sweep["flagged_rate"], label="flagged share", ls=":")
    ax[1].plot(sweep["threshold"], sweep["expected_cost_per_applicant"], color="#e45756")
    for name, spec in policies.items():
        for a in ax:
            a.axvline(spec["flag_threshold"], ls="--", lw=0.7, color="gray")
        ax[1].text(spec["flag_threshold"], ax[1].get_ylim()[1], name, rotation=90, va="top", fontsize=7)
    ax[0].set(xlabel="probability threshold", title="Threshold sensitivity (test)"); ax[0].legend(fontsize=8)
    ax[1].set(xlabel="probability threshold", ylabel="illustrative cost / applicant", title="Expected cost (illustrative costs)")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def run_training(
    cfg: dict[str, Any], df: pd.DataFrame, data_source: str,
    artifacts_dir: Path | None = None, reports_dir: Path | None = None, mlflow_enabled: bool | None = None,
    run_shap: bool = True,
) -> dict[str, Any]:
    seed, tcfg, fcfg = cfg["project"]["seed"], cfg["training"], cfg["features"]
    target, id_col = cfg["data"]["target_column"], cfg["data"]["id_column"]
    artifacts_dir = Path(artifacts_dir or resolve_path(cfg["paths"]["artifacts_dir"]))
    reports_dir = Path(reports_dir or resolve_path(cfg["paths"]["reports_dir"]))
    fig_dir = reports_dir / "figures"
    for d in (artifacts_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    validate_application_data(df, id_col, target).raise_if_invalid()
    s = cfg["split"]
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(df, target, s["test_size"], s["val_size"], seed)
    logger.info("Split sizes: train=%d val=%d test=%d | default rate train=%.4f", len(X_tr), len(X_va), len(X_te), y_tr.mean())

    fe = FeatureEngineer(
        exclude_sensitive=fcfg["exclude_sensitive"], sensitive_columns=fcfg["sensitive_columns"],
        drop_columns=(id_col, target), sentinel=fcfg["days_employed_sentinel"],
    ).fit(X_tr)
    Xtr_fe, Xva_fe = fe.transform(X_tr), fe.transform(X_va)
    groups = infer_column_groups(Xtr_fe, fcfg["high_cardinality_threshold"])
    logger.info("Features: %d numeric, %d low-card cat, %d high-card cat", len(groups.numeric), len(groups.low_card), len(groups.high_card))
    numeric_raw = [c for c in X_tr.columns if pd.api.types.is_numeric_dtype(X_tr[c])]
    bands = RiskBands.from_cfg(cfg)
    costs = cfg["decision"]["costs"]

    candidates: dict[str, dict[str, Any]] = {}
    for name in tcfg["models"]:
        logger.info("Tuning %s", name)
        candidates[name] = fit_candidate(name, cfg, groups, Xtr_fe, y_tr, Xva_fe, y_va, tcfg["n_iter_tuning"], seed)

    selected, rule = select_final_model(candidates, tcfg["selection_auc_tolerance"])
    logger.info("Selected model: %s", selected)

    version = f"{selected}-{datetime.now(timezone.utc):%Y%m%d}-" + hashlib.sha1(
        json.dumps({"p": candidates[selected]["params"], "s": seed, "n": len(df), "src": data_source}, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]

    bundles: dict[str, ModelBundle] = {}
    comparison, test_curves_cal = [], {}
    for name, c in candidates.items():
        p_va_raw = c["est"].predict_proba(c["Mva"])[:, 1]
        method, cal_scores = select_calibration_method(p_va_raw, y_va.to_numpy(), tcfg["calibration_methods"], seed)
        cal = Calibrator(method).fit(p_va_raw, y_va.to_numpy())
        pipe = Pipeline([("features", copy.deepcopy(fe)), ("prep", c["prep"]), ("model", c["est"])])
        b = ModelBundle(pipe, cal, list(X_tr.columns), numeric_raw, name, version if name == selected else f"{name}-candidate",
                        bands, metadata={})
        p_va_cal = cal.predict(p_va_raw)
        thr = best_f1_threshold(y_va.to_numpy(), p_va_cal)
        p_te_raw, p_te_cal = b.predict_raw(X_te), b.predict_proba(X_te)   # end-to-end path on RAW test features
        m_te, m_te_raw = full_metrics(y_te.to_numpy(), p_te_cal, thr), probability_metrics(y_te.to_numpy(), p_te_raw)
        bundles[name] = b
        test_curves_cal[name] = (y_te.to_numpy(), p_te_cal)
        comparison.append({
            "model": name, "val_roc_auc": c["val_auc"], "test_roc_auc": m_te["roc_auc"], "test_pr_auc": m_te["pr_auc"],
            "test_ks": m_te["ks"], "test_brier_uncalibrated": m_te_raw["brier"], "test_brier_calibrated": m_te["brier"],
            "test_precision": m_te["precision"], "test_recall": m_te["recall"], "test_f1": m_te["f1"],
            "calibration": method, "calibration_cv_brier": cal_scores, "best_params": json.dumps(c["params"], default=str),
            "train_seconds": round(c["seconds"], 1), "val_threshold_maxf1": thr, "_test_metrics": m_te,
            "_p_te_raw": p_te_raw, "_p_te_cal": p_te_cal, "_p_va_cal": p_va_cal,
        })
        logger.info("%s | val AUC %.4f | test AUC %.4f PR-AUC %.4f | Brier raw %.4f -> cal %.4f (%s)",
                    name, c["val_auc"], m_te["roc_auc"], m_te["pr_auc"], m_te_raw["brier"], m_te["brier"], method)

    sel = next(r for r in comparison if r["model"] == selected)
    bundle = bundles[selected]
    y_te_np, p_te = y_te.to_numpy(), sel["_p_te_cal"]

    # ---------------- decision analysis (selected model, test split, calibrated probabilities)
    thresholds = np.unique(np.concatenate([np.linspace(0.01, 0.5, 50), [v["flag_threshold"] for v in cfg["decision"]["policies"].values()]]))
    sweep = threshold_sweep(y_te_np, p_te, thresholds, costs["false_negative"], costs["false_positive"])
    pol = policy_comparison(y_te_np, p_te, cfg["decision"]["policies"], costs)
    tiers = tier_summary(y_te_np, p_te, bands)
    cost_thr = optimal_cost_threshold(y_va.to_numpy(), sel["_p_va_cal"], costs["false_negative"], costs["false_positive"])
    cost_thr_test = threshold_sweep(y_te_np, p_te, np.array([cost_thr]), costs["false_negative"], costs["false_positive"]).iloc[0].to_dict()
    sweep.to_csv(artifacts_dir / "threshold_sensitivity.csv", index=False)
    pol.to_csv(artifacts_dir / "policy_comparison.csv", index=False)
    tiers.to_csv(artifacts_dir / "risk_tiers.csv", index=False)

    # ---------------- figures
    plot_roc_pr(test_curves_cal, fig_dir / "roc_pr_test.png", "(test, calibrated)")
    plot_calibration({"uncalibrated": (y_te_np, sel["_p_te_raw"]), "calibrated": (y_te_np, p_te)}, fig_dir / "calibration_selected_model.png", f"Calibration - {selected} (test)")
    plot_calibration({n: (y_te_np, r["_p_te_cal"]) for n, r in ((r["model"], r) for r in comparison)}, fig_dir / "calibration_all_models.png", "Calibration - all models, calibrated (test)")
    plot_confusion_matrix(sel["_test_metrics"], fig_dir / "confusion_matrix_selected.png", f"{selected} (test)")
    _plot_threshold_sensitivity(sweep, cfg["decision"]["policies"], fig_dir / "threshold_sensitivity.png")

    # ---------------- explainability
    shap_summary = None
    if run_shap:
        try:
            from credit_risk.explainability.shap_analysis import run_shap_analysis

            n = min(tcfg["shap_sample_size"], len(X_va))
            shap_summary = run_shap_analysis(bundle, X_va.sample(n, random_state=seed), fig_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SHAP analysis failed (training continues): %s", exc)

    # ---------------- monitoring references (validation split)
    ref_n = min(cfg["monitoring"]["reference_sample_size"], len(X_va))
    X_va.sample(ref_n, random_state=seed).to_csv(artifacts_dir / "reference_features.csv.gz", index=False)
    np.save(artifacts_dir / "reference_scores.npy", bundle.predict_proba(X_va))

    result = {
        "data_source": data_source, "n_rows": int(len(df)), "default_rate": float(df[target].mean()), "seed": seed,
        "splits": {"train": len(X_tr), "val": len(X_va), "test": len(X_te)},
        "selected_model": selected, "model_version": version, "selection_rule": rule,
        "calibration_method": sel["calibration"],
        "excluded_sensitive": fcfg["sensitive_columns"] if fcfg["exclude_sensitive"] else [],
        "comparison": [{k: v for k, v in r.items() if not k.startswith("_")} for r in comparison],
        "selected_test_metrics": sel["_test_metrics"],
        "policy_comparison": pol.to_dict("records"), "tier_summary": tiers.to_dict("records"),
        "cost_optimal_threshold_val": cost_thr, "cost_optimal_threshold_test_result": cost_thr_test,
        "no_model_cost": no_model_cost(y_te_np, costs),
        "baseline": {"roc_auc": 0.5, "brier": float(brier_score_loss(y_te_np, np.full(len(y_te_np), y_tr.mean())))},
        "n_features_after_preprocessing": int(len(bundle.pipeline.named_steps["prep"].get_feature_names_out())),
        "shap_top_features": shap_summary["importance"][:10] if shap_summary else None,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    bundle.metadata.update({k: result[k] for k in ("data_source", "trained_at", "seed", "selected_test_metrics", "calibration_method",
                                                    "excluded_sensitive", "selection_rule", "splits", "n_features_after_preprocessing")})
    bundle_path = bundle.save(artifacts_dir / "model_bundle.joblib")
    with open(artifacts_dir / "metrics.json", "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    pd.DataFrame(result["comparison"]).drop(columns=["calibration_cv_brier"]).to_csv(artifacts_dir / "model_comparison.csv", index=False)
    write_experiment_report(result, reports_dir / "experiment_report.md")

    use_mlflow = cfg["mlflow"]["enabled"] if mlflow_enabled is None else mlflow_enabled
    if use_mlflow:
        log_to_mlflow(cfg, result, bundle_path, fig_dir, artifacts_dir)
    logger.info("Done. Bundle: %s", bundle_path)
    return result


def log_to_mlflow(cfg: dict[str, Any], result: dict[str, Any], bundle_path: Path, fig_dir: Path, artifacts_dir: Path) -> None:
    """Track parameters, metrics, config and artifacts. Failures are logged and never break training."""
    try:
        import mlflow

        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
        with mlflow.start_run(run_name=f"train-{result['model_version']}"):
            mlflow.set_tags({"data_source": result["data_source"], "model_version": result["model_version"], "selected_model": result["selected_model"]})
            mlflow.log_params({
                "seed": result["seed"], "n_rows": result["n_rows"], "exclude_sensitive": cfg["features"]["exclude_sensitive"],
                "sensitive_columns": ",".join(cfg["features"]["sensitive_columns"]), "calibration": result["calibration_method"],
                "n_iter_tuning": cfg["training"]["n_iter_tuning"], "n_features": result["n_features_after_preprocessing"],
            })
            for k, v in result["selected_test_metrics"].items():
                if isinstance(v, float):
                    mlflow.log_metric(f"final_test_{k}", v)
            for row in result["comparison"]:
                with mlflow.start_run(run_name=row["model"], nested=True):
                    mlflow.log_params({"model": row["model"], "best_params": row["best_params"][:490], "calibration": row["calibration"]})
                    mlflow.log_metrics({k: float(v) for k, v in row.items() if k.startswith(("val_", "test_")) and isinstance(v, float)})
            mlflow.log_dict({"features": cfg["features"], "decision": cfg["decision"]}, "feature_and_decision_config.json")
            mlflow.log_artifact(str(bundle_path), "model")
            for f in ("metrics.json", "model_comparison.csv", "policy_comparison.csv", "threshold_sensitivity.csv"):
                if (artifacts_dir / f).exists():
                    mlflow.log_artifact(str(artifacts_dir / f), "results")
            if fig_dir.exists():
                mlflow.log_artifacts(str(fig_dir), "figures")
        logger.info("MLflow run logged to %s", cfg["mlflow"]["tracking_uri"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow tracking skipped due to error: %s", exc)
