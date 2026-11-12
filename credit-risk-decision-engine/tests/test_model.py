import numpy as np
import pandas as pd
import pytest

from credit_risk.decision.policy import (
    RiskBands,
    categorize,
    expected_cost,
    no_model_cost,
    optimal_cost_threshold,
    policy_comparison,
    threshold_sweep,
    tier_summary,
)
from credit_risk.models.baseline import make_estimator
from credit_risk.models.calibration import Calibrator, select_calibration_method
from credit_risk.models.predict import ModelBundle
from credit_risk.models.train import select_final_model


def test_three_models_were_trained_and_compared(trained):
    names = {r["model"] for r in trained["result"]["comparison"]}
    assert names == {"logistic_regression", "random_forest", "lightgbm"}
    assert all(0.5 < r["val_roc_auc"] <= 1 for r in trained["result"]["comparison"])


def test_artifacts_written(trained):
    for f in ("model_bundle.joblib", "metrics.json", "model_comparison.csv", "policy_comparison.csv", "threshold_sensitivity.csv", "reference_features.csv.gz", "reference_scores.npy"):
        assert (trained["art"] / f).exists(), f


def test_bundle_loading_and_probability_range(trained, synthetic_df):
    b = ModelBundle.load(trained["art"] / "model_bundle.joblib")
    p = b.predict_proba(synthetic_df.head(300).drop(columns=["TARGET"]))
    assert p.shape == (300,) and ((p >= 0) & (p <= 1)).all() and np.isfinite(p).all()
    assert set(b.risk_category(p)) <= {"Low Risk", "Medium Risk", "High Risk"}


def test_bundle_missing_file_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train a model first"):
        ModelBundle.load(tmp_path / "nope.joblib")


def test_prediction_independent_of_column_order_and_extra_columns(trained, synthetic_df):
    b, sub = trained["bundle"], synthetic_df.head(50).drop(columns=["TARGET"])
    p1 = b.predict_proba(sub)
    shuffled = sub[list(reversed(sub.columns))].assign(UNKNOWN_EXTRA=1)
    np.testing.assert_allclose(p1, b.predict_proba(shuffled))


def test_prediction_with_missing_fields_and_strings_in_numeric(trained):
    out = trained["bundle"].predict_proba(pd.DataFrame([{"AMT_CREDIT": "oops", "EXT_SOURCE_1": None}]))
    assert 0 <= out[0] <= 1


def test_model_has_signal_vs_prior_baseline(trained):
    m = trained["result"]["selected_test_metrics"]
    assert m["roc_auc"] > 0.55 and m["brier"] <= trained["result"]["baseline"]["brier"] * 1.02


def test_calibration_reduces_brier_for_distorted_probabilities():
    rng = np.random.default_rng(0)
    p_true = rng.beta(1, 12, 20000); y = (rng.random(20000) < p_true).astype(int)
    distorted = np.clip(p_true * 4, 0, 1)                      # over-confident scores
    method, scores = select_calibration_method(distorted, y, ["none", "sigmoid", "isotonic"])
    assert method != "none" and scores[method] < scores["none"]
    cal = Calibrator(method).fit(distorted, y).predict(distorted)
    assert ((cal >= 0) & (cal <= 1)).all()


def test_calibrator_unknown_method():
    with pytest.raises(ValueError):
        Calibrator("magic")


def test_selection_rule_prefers_simpler_model_within_tolerance():
    cands = {"logistic_regression": {"val_auc": 0.700}, "lightgbm": {"val_auc": 0.702}, "random_forest": {"val_auc": 0.690}}
    assert select_final_model(cands, 0.003)[0] == "logistic_regression"
    cands["lightgbm"]["val_auc"] = 0.72
    assert select_final_model(cands, 0.003)[0] == "lightgbm"


def test_unknown_model_name():
    with pytest.raises(ValueError):
        make_estimator("xgboost")


def test_risk_bands_and_validation():
    b = RiskBands(0.05, 0.15)
    assert list(categorize(np.array([0.01, 0.05, 0.1, 0.15, 0.9]), b)) == ["Low Risk", "Medium Risk", "Medium Risk", "High Risk", "High Risk"]
    with pytest.raises(ValueError):
        RiskBands(0.2, 0.1)


def test_threshold_sweep_and_cost_logic():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]); p = np.array([0.9, 0.4, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    s = threshold_sweep(y, p, np.array([0.0, 0.45, 1.01]), cost_fn=5, cost_fp=1)
    assert s.loc[0, "recall"] == 1 and s.loc[0, "false_positives"] == 8        # flag everyone
    assert s.loc[1, "false_negatives"] == 1 and s.loc[1, "false_positives"] == 1
    assert s.loc[2, "false_negatives"] == 2 and s.loc[2, "flagged_rate"] == 0   # flag nobody
    assert expected_cost(1, 1, 10, 5, 1) == pytest.approx(0.6)
    assert no_model_cost(y, {"false_negative": 5, "false_positive": 1}) == {"flag_nobody_cost": 1.0, "flag_everybody_cost": 0.8}
    assert 0 < optimal_cost_threshold(y, p, 5, 1) <= 0.9


def test_policy_ordering_more_conservative_flags_more():
    rng = np.random.default_rng(1); p = rng.beta(1, 10, 5000); y = (rng.random(5000) < p).astype(int)
    pol = policy_comparison(y, p, {"conservative": {"flag_threshold": 0.05}, "balanced": {"flag_threshold": 0.1}, "higher_acceptance": {"flag_threshold": 0.2}}, {"false_negative": 5, "false_positive": 1}).set_index("policy")
    assert pol.loc["conservative", "flagged_rate"] > pol.loc["balanced", "flagged_rate"] > pol.loc["higher_acceptance", "flagged_rate"]
    assert pol.loc["conservative", "recall"] >= pol.loc["higher_acceptance", "recall"]
    t = tier_summary(y, p, RiskBands()); assert t["share_of_applicants"].sum() == pytest.approx(1.0)


def test_omitted_fields_do_not_produce_extreme_predictions(trained):
    """Regression test: scaled rare missing-indicators once drove partial requests to ~0 probability."""
    b, base = trained["bundle"], trained["result"]["default_rate"]
    for payload in ({}, {"AMT_CREDIT": 300000}, {"AMT_INCOME_TOTAL": 120000, "AMT_CREDIT": 300000, "AMT_ANNUITY": 15000}):
        p = float(b.predict_proba(pd.DataFrame([payload]))[0])
        assert base / 3 < p < base * 3, (payload, p, base)


def test_better_external_scores_lower_risk_and_empty_request_is_typical(trained):
    b = trained["bundle"]
    good = {"AMT_INCOME_TOTAL": 150000, "AMT_CREDIT": 300000, "EXT_SOURCE_1": 0.8, "EXT_SOURCE_2": 0.85, "EXT_SOURCE_3": 0.8}
    bad = {**good, "EXT_SOURCE_1": 0.1, "EXT_SOURCE_2": 0.1, "EXT_SOURCE_3": 0.1}
    p_good, p_bad, p_empty = (float(b.predict_proba(pd.DataFrame([x]))[0]) for x in (good, bad, {}))
    assert p_good < p_empty < p_bad
