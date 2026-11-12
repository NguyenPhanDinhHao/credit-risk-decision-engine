import numpy as np
import pandas as pd
import pytest

from credit_risk.config import load_config
from credit_risk.data.loader import DatasetNotFoundError, discover_files, load_application_train, load_dataset
from credit_risk.data.validation import DataValidationError, find_invalid_values, validate_application_data


def test_valid_data_passes(synthetic_df):
    res = validate_application_data(synthetic_df)
    assert res.passed and 0 < res.stats["default_rate"] < 1


def test_missing_required_columns(synthetic_df):
    res = validate_application_data(synthetic_df.drop(columns=["AMT_CREDIT", "TARGET"]))
    assert not res.passed and "AMT_CREDIT" in res.errors[0]
    with pytest.raises(DataValidationError):
        res.raise_if_invalid()


@pytest.mark.parametrize("bad", [2, -1, 0.5])
def test_invalid_target_values(synthetic_df, bad):
    df = synthetic_df.copy().astype({"TARGET": float}); df.loc[df.index[0], "TARGET"] = bad
    res = validate_application_data(df)
    assert not res.passed and any("TARGET" in e for e in res.errors)


def test_missing_target_values(synthetic_df):
    df = synthetic_df.copy().astype({"TARGET": float}); df.loc[df.index[:3], "TARGET"] = np.nan
    assert not validate_application_data(df).passed


def test_single_class_target(synthetic_df):
    df = synthetic_df.copy(); df["TARGET"] = 0
    assert any("single class" in e for e in validate_application_data(df).errors)


def test_duplicate_ids(synthetic_df):
    df = pd.concat([synthetic_df, synthetic_df.head(3)])
    assert any("duplicate" in e for e in validate_application_data(df).errors)


def test_non_numeric_amount(synthetic_df):
    df = synthetic_df.copy(); df["AMT_CREDIT"] = df["AMT_CREDIT"].astype(str)
    assert any("numeric" in e for e in validate_application_data(df).errors)


def test_critical_nulls_and_empty(synthetic_df):
    df = synthetic_df.copy(); df.loc[df.index[:2], "AMT_INCOME_TOTAL"] = np.nan
    assert not validate_application_data(df).passed
    assert not validate_application_data(synthetic_df.iloc[0:0]).passed


def test_no_target_required_for_inference(synthetic_df):
    assert validate_application_data(synthetic_df.drop(columns=["TARGET"]), require_target=False).passed


def test_invalid_values_detected_as_warning(synthetic_df):
    df = synthetic_df.copy(); df.loc[df.index[0], "EXT_SOURCE_2"] = 1.7; df.loc[df.index[1], "AMT_INCOME_TOTAL"] = -5
    assert find_invalid_values(df) == {"AMT_INCOME_TOTAL": 1, "EXT_SOURCE_2": 1}
    assert validate_application_data(df).warnings


def test_loader_reports_missing_dataset(tmp_path):
    cfg = load_config(); cfg["paths"]["data_dir"] = str(tmp_path)
    assert discover_files(cfg)["application_train"] is None
    with pytest.raises(DatasetNotFoundError) as e:
        load_application_train(cfg)
    assert "kaggle.com/competitions/home-credit-default-risk" in str(e.value) and "[MISSING]" in str(e.value)
    with pytest.raises(DatasetNotFoundError):
        load_dataset(cfg, demo=False)          # never silently falls back to synthetic data


def test_demo_mode_is_explicit_and_labelled(tmp_path):
    cfg = load_config(); cfg["paths"]["data_dir"] = str(tmp_path); cfg["data"]["sample_rows"] = 500
    df, source = load_dataset(cfg, demo=True)
    assert source == "synthetic_demo" and len(df) == 500


def test_loader_reads_real_layout(tmp_path, synthetic_df):
    cfg = load_config(); cfg["paths"]["data_dir"] = str(tmp_path)
    synthetic_df.head(300).to_csv(tmp_path / "application_train.csv", index=False)
    df, source = load_dataset(cfg)
    assert source == "home_credit" and len(df) == 300
