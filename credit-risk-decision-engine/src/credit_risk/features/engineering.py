"""Domain feature engineering. Implemented as an sklearn transformer so the exact same code path runs
in training and at inference time (it is serialised inside the model pipeline)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

DEFAULT_SENSITIVE = ["CODE_GENDER", "DAYS_BIRTH", "NAME_FAMILY_STATUS", "DAYS_EMPLOYED_PERC"]
ENGINEERED_FEATURES = [
    "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "CREDIT_GOODS_RATIO",
    "DAYS_EMPLOYED_PERC", "INCOME_PER_PERSON", "EXT_SOURCE_MEAN", "DAYS_EMPLOYED_ANOM",
]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").astype("float64")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    """num/den with zero denominators and infinities mapped to NaN (never raises, never emits inf)."""
    out = num / den.where(den != 0)
    return out.replace([np.inf, -np.inf], np.nan)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds credit-risk ratio features, cleans sentinels, drops identifiers/target/sensitive fields.

    Stateless apart from the recorded output schema (``columns_out_``), which guarantees that
    inference frames are aligned to the training schema.

    Purpose of each feature:
      CREDIT_INCOME_RATIO   loan size relative to annual income (leverage)
      ANNUITY_INCOME_RATIO  periodic repayment burden relative to income
      CREDIT_TERM           annuity / credit ~ inverse of repayment term (shorter term -> larger value)
      CREDIT_GOODS_RATIO    credit relative to financed goods price (extra financing beyond goods)
      DAYS_EMPLOYED_PERC    share of life employed (age-derived -> treated as sensitive by default)
      INCOME_PER_PERSON     income per family member
      EXT_SOURCE_MEAN       mean of external scores (robust to one missing source)
      DAYS_EMPLOYED_ANOM    flag for the 365243 placeholder in DAYS_EMPLOYED
    """

    def __init__(
        self,
        exclude_sensitive: bool = True,
        sensitive_columns: list[str] | None = None,
        drop_columns: tuple[str, ...] = ("SK_ID_CURR", "TARGET"),
        sentinel: int = 365243,
        add_features: bool = True,
    ):
        self.exclude_sensitive = exclude_sensitive
        self.sensitive_columns = sensitive_columns
        self.drop_columns = drop_columns
        self.sentinel = sentinel
        self.add_features = add_features

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for c in [c for c in df.columns if df[c].dtype == bool]:
            df[c] = df[c].astype("float64")
        if "DAYS_EMPLOYED" in df.columns:
            de = _num(df, "DAYS_EMPLOYED")
            df["DAYS_EMPLOYED_ANOM"] = (de == self.sentinel).astype("float64")
            df["DAYS_EMPLOYED"] = de.where(de != self.sentinel)
        else:
            df["DAYS_EMPLOYED_ANOM"] = np.nan

        if self.add_features:
            income, credit = _num(df, "AMT_INCOME_TOTAL"), _num(df, "AMT_CREDIT")
            annuity, goods = _num(df, "AMT_ANNUITY"), _num(df, "AMT_GOODS_PRICE")
            df["CREDIT_INCOME_RATIO"] = safe_divide(credit, income)
            df["ANNUITY_INCOME_RATIO"] = safe_divide(annuity, income)
            df["CREDIT_TERM"] = safe_divide(annuity, credit)
            df["CREDIT_GOODS_RATIO"] = safe_divide(credit, goods)
            df["DAYS_EMPLOYED_PERC"] = safe_divide(_num(df, "DAYS_EMPLOYED"), _num(df, "DAYS_BIRTH"))
            df["INCOME_PER_PERSON"] = safe_divide(income, _num(df, "CNT_FAM_MEMBERS"))
            ext = pd.concat([_num(df, f"EXT_SOURCE_{i}") for i in (1, 2, 3)], axis=1)
            df["EXT_SOURCE_MEAN"] = ext.mean(axis=1, skipna=True)

        drop = list(self.drop_columns)
        if self.exclude_sensitive:
            drop += list(self.sensitive_columns if self.sensitive_columns is not None else DEFAULT_SENSITIVE)
        df = df.drop(columns=[c for c in drop if c in df.columns])
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].astype(object)
        return df

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureEngineer:
        self.columns_out_ = list(self._transform(X.head(50)).columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self._transform(X)
        if hasattr(self, "columns_out_"):
            out = out.reindex(columns=self.columns_out_)
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.columns_out_, dtype=object)


# ---------------------------------------------------------------- supplementary tables (optional)
def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """Per-applicant aggregates of external credit history (bureau.csv). Uses no target information."""
    b = bureau.copy()
    b["IS_ACTIVE"] = (b.get("CREDIT_ACTIVE") == "Active").astype("float64") if "CREDIT_ACTIVE" in b else np.nan
    g = b.groupby("SK_ID_CURR")
    out = pd.DataFrame({"BUREAU_N_CREDITS": g.size()})
    if "IS_ACTIVE" in b:
        out["BUREAU_ACTIVE_SHARE"] = g["IS_ACTIVE"].mean()
    for col, name in (("AMT_CREDIT_SUM", "BUREAU_CREDIT_SUM"), ("AMT_CREDIT_SUM_DEBT", "BUREAU_DEBT_SUM"), ("CREDIT_DAY_OVERDUE", "BUREAU_MAX_DAYS_OVERDUE")):
        if col in b:
            out[name] = g[col].max() if "OVERDUE" in col else g[col].sum()
    if {"BUREAU_DEBT_SUM", "BUREAU_CREDIT_SUM"} <= set(out.columns):
        out["BUREAU_DEBT_CREDIT_RATIO"] = safe_divide(out["BUREAU_DEBT_SUM"], out["BUREAU_CREDIT_SUM"])
    return out.reset_index()


def aggregate_previous_applications(prev: pd.DataFrame) -> pd.DataFrame:
    """Per-applicant aggregates of previous Home Credit applications."""
    p = prev.copy()
    g = p.groupby("SK_ID_CURR")
    out = pd.DataFrame({"PREV_N_APPLICATIONS": g.size()})
    if "NAME_CONTRACT_STATUS" in p:
        p["REFUSED"] = (p["NAME_CONTRACT_STATUS"] == "Refused").astype("float64")
        p["APPROVED"] = (p["NAME_CONTRACT_STATUS"] == "Approved").astype("float64")
        out["PREV_REFUSED_SHARE"] = p.groupby("SK_ID_CURR")["REFUSED"].mean()
        out["PREV_APPROVED_SHARE"] = p.groupby("SK_ID_CURR")["APPROVED"].mean()
    if "AMT_APPLICATION" in p:
        out["PREV_AMT_APPLICATION_MEAN"] = g["AMT_APPLICATION"].mean()
    return out.reset_index()


def attach_supplementary_features(df: pd.DataFrame, cfg: dict[str, Any], data_dir: Path) -> pd.DataFrame:
    """Left-join aggregates from optional tables when present; silently skip missing tables (with a log)."""
    from credit_risk.data.loader import load_optional_table

    n0 = df.shape[1]
    for name, fn in (("bureau", aggregate_bureau), ("previous_application", aggregate_previous_applications)):
        table = load_optional_table(cfg, name, data_dir)
        if table is not None:
            df = df.merge(fn(table), on="SK_ID_CURR", how="left")
    logger.info("Supplementary features added: %d columns", df.shape[1] - n0)
    return df
