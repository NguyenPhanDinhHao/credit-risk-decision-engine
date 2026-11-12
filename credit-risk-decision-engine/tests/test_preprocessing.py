import numpy as np
import pandas as pd

from credit_risk.features.engineering import (
    FeatureEngineer,
    aggregate_bureau,
    aggregate_previous_applications,
    safe_divide,
)
from credit_risk.features.preprocessing import build_preprocessor, infer_column_groups, split_data


def test_safe_divide_handles_zero_and_inf():
    out = safe_divide(pd.Series([1.0, 2.0, np.inf, 4.0]), pd.Series([0.0, 2.0, 1.0, np.nan]))
    assert np.isnan(out[0]) and out[1] == 1.0 and np.isnan(out[2]) and np.isnan(out[3])


def test_engineered_features_and_sentinel(synthetic_df):
    df = synthetic_df.head(200).copy(); df.loc[df.index[0], "AMT_INCOME_TOTAL"] = 0
    out = FeatureEngineer(exclude_sensitive=False).fit(df).transform(df)
    assert np.isnan(out["CREDIT_INCOME_RATIO"].iloc[0])
    assert not np.isinf(out.select_dtypes("number").to_numpy()).any()
    sentinel_rows = df["DAYS_EMPLOYED"] == 365243
    assert out.loc[sentinel_rows, "DAYS_EMPLOYED"].isna().all() and (out.loc[sentinel_rows, "DAYS_EMPLOYED_ANOM"] == 1).all()
    ok = ~sentinel_rows & (df["AMT_INCOME_TOTAL"] > 0)
    np.testing.assert_allclose(out.loc[ok, "CREDIT_INCOME_RATIO"], df.loc[ok, "AMT_CREDIT"] / df.loc[ok, "AMT_INCOME_TOTAL"], rtol=1e-9)


def test_sensitive_columns_excluded_by_default(synthetic_df):
    cols = FeatureEngineer(sensitive_columns=["CODE_GENDER", "DAYS_BIRTH", "NAME_FAMILY_STATUS", "DAYS_EMPLOYED_PERC"]).fit(synthetic_df).transform(synthetic_df).columns
    for c in ("CODE_GENDER", "DAYS_BIRTH", "NAME_FAMILY_STATUS", "DAYS_EMPLOYED_PERC", "SK_ID_CURR", "TARGET"):
        assert c not in cols
    kept = FeatureEngineer(exclude_sensitive=False).fit(synthetic_df).transform(synthetic_df).columns
    assert "CODE_GENDER" in kept and "DAYS_EMPLOYED_PERC" in kept


def test_inference_schema_matches_training_even_with_missing_columns(synthetic_df):
    fe = FeatureEngineer().fit(synthetic_df)
    partial = synthetic_df[["AMT_INCOME_TOTAL", "AMT_CREDIT"]].head(5)
    out = fe.transform(partial)
    assert list(out.columns) == fe.columns_out_ and len(out) == 5


def test_single_row_equals_batch_row(synthetic_df):
    fe = FeatureEngineer().fit(synthetic_df)
    batch = fe.transform(synthetic_df.head(20))
    single = fe.transform(synthetic_df.iloc[[7]])
    pd.testing.assert_frame_equal(batch.iloc[[7]].reset_index(drop=True), single.reset_index(drop=True), check_dtype=False)


def test_preprocessor_is_fitted_on_training_data_only(synthetic_df):
    X_tr, X_va, X_te, y_tr, *_ = split_data(synthetic_df, "TARGET", 0.2, 0.2, 42)
    fe = FeatureEngineer().fit(X_tr)
    Xtr, Xva = fe.transform(X_tr), fe.transform(X_va)
    prep = build_preprocessor(infer_column_groups(Xtr, 15), scale_numeric=True, winsorize=(0.01, 0.99))
    M = prep.fit_transform(Xtr, y_tr)
    num = prep.named_transformers_["num"]
    col = "AMT_INCOME_TOTAL"
    i = infer_column_groups(Xtr).numeric.index(col)
    lo, hi = np.nanquantile(Xtr[col], [0.01, 0.99])
    np.testing.assert_allclose(num.named_steps["winsor"].lo_[i], lo)           # bounds come from TRAIN
    before = num.named_steps["winsor"].lo_.copy()
    prep.transform(Xva)                                                          # transforming new data must not refit
    np.testing.assert_array_equal(before, num.named_steps["winsor"].lo_)
    assert np.isfinite(M).all() and np.isfinite(prep.transform(Xva)).all()


def test_unseen_categories_do_not_break_transform(synthetic_df):
    X_tr, _, _, y_tr, *_ = split_data(synthetic_df, "TARGET", 0.2, 0.2, 42)
    fe = FeatureEngineer().fit(X_tr)
    prep = build_preprocessor(infer_column_groups(fe.transform(X_tr), 15), False)
    prep.fit(fe.transform(X_tr), y_tr)
    new = X_tr.head(3).copy(); new["NAME_INCOME_TYPE"] = "Never seen"; new["ORGANIZATION_TYPE"] = "Never seen"
    assert np.isfinite(prep.transform(fe.transform(new))).all()


def test_split_is_stratified_and_disjoint(synthetic_df):
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(synthetic_df, "TARGET", 0.2, 0.2, 1)
    assert not (set(X_tr.SK_ID_CURR) & set(X_va.SK_ID_CURR)) and not (set(X_tr.SK_ID_CURR) & set(X_te.SK_ID_CURR)) and not (set(X_va.SK_ID_CURR) & set(X_te.SK_ID_CURR))
    assert len(X_tr) + len(X_va) + len(X_te) == len(synthetic_df)
    assert abs(y_tr.mean() - y_va.mean()) < 0.01 and abs(y_tr.mean() - y_te.mean()) < 0.01
    assert abs(len(X_va) / len(synthetic_df) - 0.2) < 0.01


def test_supplementary_aggregation():
    bureau = pd.DataFrame({"SK_ID_CURR": [1, 1, 2], "CREDIT_ACTIVE": ["Active", "Closed", "Active"], "AMT_CREDIT_SUM": [100.0, 300.0, 50.0], "AMT_CREDIT_SUM_DEBT": [50.0, 0.0, 0.0], "CREDIT_DAY_OVERDUE": [0, 10, 0]})
    b = aggregate_bureau(bureau).set_index("SK_ID_CURR")
    assert b.loc[1, "BUREAU_N_CREDITS"] == 2 and b.loc[1, "BUREAU_ACTIVE_SHARE"] == 0.5 and b.loc[1, "BUREAU_MAX_DAYS_OVERDUE"] == 10
    prev = pd.DataFrame({"SK_ID_CURR": [1, 1, 2], "NAME_CONTRACT_STATUS": ["Refused", "Approved", "Approved"], "AMT_APPLICATION": [10.0, 30.0, 5.0]})
    p = aggregate_previous_applications(prev).set_index("SK_ID_CURR")
    assert p.loc[1, "PREV_REFUSED_SHARE"] == 0.5 and p.loc[2, "PREV_AMT_APPLICATION_MEAN"] == 5.0


def test_missing_indicators_only_for_columns_with_enough_missingness(synthetic_df):
    X_tr, _, _, y_tr, *_ = split_data(synthetic_df, "TARGET", 0.2, 0.2, 42)
    fe = FeatureEngineer().fit(X_tr)
    Xfe = fe.transform(X_tr)
    g = infer_column_groups(Xfe, 15, 0.01)
    assert "EXT_SOURCE_1" in g.indicator and "AMT_ANNUITY" not in g.indicator and "AMT_INCOME_TOTAL" not in g.indicator
    prep = build_preprocessor(g, scale_numeric=True); M = prep.fit_transform(Xfe, y_tr)
    names = list(prep.get_feature_names_out())
    ind = [i for i, n in enumerate(names) if n.startswith("missingindicator_")]
    assert ind and set(np.unique(M[:, ind])) <= {0.0, 1.0}          # flags are NOT standardised
    absent = fe.transform(pd.DataFrame([{}]))                         # a request with no fields at all
    assert np.abs(prep.transform(absent)).max() < 20                  # all-missing input stays in a sane range
