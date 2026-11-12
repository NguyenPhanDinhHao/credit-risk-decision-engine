import numpy as np
import pandas as pd
import pytest

from credit_risk.data.synthetic import generate_synthetic_applications
from credit_risk.monitoring.data_drift import (
    compare_datasets,
    drift_summary,
    input_quality_report,
    psi_categorical,
    psi_numeric,
    simulate_drift,
)
from credit_risk.monitoring.prediction_monitor import (
    PredictionLogger,
    RetrainingPolicy,
    evaluate_retraining_trigger,
    load_prediction_log,
    performance_report,
    prediction_drift,
)


def test_psi_zero_for_identical_and_large_for_shift():
    rng = np.random.default_rng(0); a = pd.Series(rng.normal(0, 1, 5000))
    assert psi_numeric(a, a) == pytest.approx(0, abs=1e-9)
    assert psi_numeric(a, pd.Series(rng.normal(0, 1, 5000))) < 0.05
    assert psi_numeric(a, pd.Series(rng.normal(1.5, 1, 5000))) > 0.25


def test_psi_handles_constant_and_all_missing():
    assert psi_numeric(pd.Series([1.0] * 50), pd.Series([1.0] * 50)) == 0.0
    assert np.isnan(psi_numeric(pd.Series([np.nan] * 5), pd.Series([1.0, 2.0])))


def test_psi_categorical_detects_new_category():
    ref = pd.Series(["a"] * 90 + ["b"] * 10); cur = pd.Series(["a"] * 50 + ["b"] * 10 + ["c"] * 40)
    assert psi_categorical(ref, ref) == pytest.approx(0) and psi_categorical(ref, cur) > 0.25


def test_no_drift_between_same_distribution_and_drift_when_simulated():
    ref = generate_synthetic_applications(6000, seed=1, with_target=False)
    same = generate_synthetic_applications(6000, seed=2, with_target=False)
    t0 = compare_datasets(ref, same)
    assert drift_summary(t0)["n_high_drift"] == 0
    shifted = simulate_drift(same)
    t1 = compare_datasets(ref, shifted).set_index("feature")
    assert t1.loc["AMT_INCOME_TOTAL", "drift_level"] == "high" and t1.loc["AMT_INCOME_TOTAL", "ks_significant"]
    assert 0.04 < t1.loc["EXT_SOURCE_1", "missing_rate_delta"] < 0.10   # 15% of the ~44% non-missing rows blanked
    assert drift_summary(t1.reset_index())["top_drifted_features"][0] in ("AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "INCOME_PER_PERSON")


def test_input_quality_report(synthetic_df):
    df = synthetic_df.head(100).copy(); df.loc[df.index[:5], "AMT_CREDIT"] = np.nan; df.loc[df.index[5], "EXT_SOURCE_2"] = 3
    q = input_quality_report(df, ["AMT_CREDIT"])
    assert q["missing_rate_core_fields"]["AMT_CREDIT"] == 0.05 and q["invalid_value_counts"] == {"EXT_SOURCE_2": 1}


def test_prediction_drift_and_performance():
    rng = np.random.default_rng(3); ref = rng.beta(1, 12, 5000); cur = rng.beta(2, 8, 5000)
    d = prediction_drift(ref, cur)
    assert d["level"] == "high" and d["cur_mean"] > d["ref_mean"] and "does not prove" in d["note"]
    y = (rng.random(5000) < ref).astype(int)
    perf = performance_report(y, ref); assert perf["roc_auc"] > 0.6 and perf["n"] == 5000
    with pytest.raises(ValueError):
        performance_report(np.zeros(10), np.zeros(10))


def test_retraining_policy_never_retrains_on_drift_alone():
    pol = RetrainingPolicy()
    ok = {"n": 5000, "roc_auc": 0.74, "mean_predicted": 0.08, "observed_rate": 0.08}
    assert evaluate_retraining_trigger(pol, 0.75, ok, False, 0.02)["decision"] == "NO_ACTION"
    d = evaluate_retraining_trigger(pol, 0.75, ok, True, 0.40)
    assert d["decision"] == "INVESTIGATE" and d["automatic_retraining"] is False
    assert evaluate_retraining_trigger(pol, 0.75, ok, True, 0.40, consecutive_drift_windows=3)["decision"] == "INVESTIGATE_PRIORITY"
    bad = {**ok, "roc_auc": 0.68}
    assert evaluate_retraining_trigger(pol, 0.75, bad, False, 0.0)["decision"] == "RETRAIN_RECOMMENDED"
    drift_gap = {**ok, "mean_predicted": 0.14}
    assert evaluate_retraining_trigger(pol, 0.75, drift_gap, False, 0.0)["decision"] == "RETRAIN_RECOMMENDED"
    few = {**bad, "n": 200}
    assert evaluate_retraining_trigger(pol, 0.75, few, False, 0.0)["decision"] == "NO_ACTION"


def test_prediction_logger_roundtrip_and_privacy(tmp_path):
    path = tmp_path / "log.jsonl"
    core = pd.DataFrame({"AMT_CREDIT": [100.0, np.nan], "AMT_INCOME_TOTAL": [5.0, 6.0]})
    PredictionLogger(path).log_batch("v1", np.array([0.1, 0.2]), np.array(["Low Risk", "Medium Risk"]), core, ["AMT_CREDIT", "AMT_INCOME_TOTAL"])
    log = load_prediction_log(path)
    assert len(log) == 2 and log.loc[1, "missing_core"] == ["AMT_CREDIT"]
    assert "AMT_INCOME_TOTAL" not in path.read_text().replace('"missing_core": []', "")   # no raw values stored
    assert load_prediction_log(tmp_path / "none.jsonl").empty
