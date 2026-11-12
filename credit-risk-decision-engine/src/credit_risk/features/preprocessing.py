"""Preprocessing pipeline construction and data splitting.

Every fitted statistic (imputation medians, scaling, winsorisation bounds, one-hot vocabularies,
target encodings) is learned from the TRAINING split only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder


class Winsorizer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Clip each column to training-set quantiles (NaN preserved). Used for the linear model."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X: Any, y: Any = None) -> Winsorizer:
        A = np.asarray(X, dtype="float64")
        with np.errstate(all="ignore"):
            self.lo_ = np.nanquantile(A, self.lower, axis=0)
            self.hi_ = np.nanquantile(A, self.upper, axis=0)
        self.n_features_in_ = A.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: Any) -> np.ndarray:
        A = np.asarray(X, dtype="float64")
        return np.clip(A, self.lo_, self.hi_)


@dataclass
class ColumnGroups:
    numeric: list[str]
    low_card: list[str]
    high_card: list[str]
    indicator: list[str] = field(default_factory=list)   # numeric columns that get an explicit "was missing" flag
    cat_missing_informative: list[str] = field(default_factory=list)   # categoricals where "missing" is its own category


def infer_column_groups(X_fe: pd.DataFrame, high_card_threshold: int = 15, min_missing_indicator_rate: float = 0.01) -> ColumnGroups:
    """Split columns by type. Missing-value flags are only created for numeric columns that are missing in at
    least ``min_missing_indicator_rate`` of TRAINING rows: flags for almost-never-missing columns carry no
    learnable signal and become extreme out-of-distribution values when a request omits that field."""
    numeric, low, high, indicator, cat_marked = [], [], [], [], []
    for c in X_fe.columns:
        if pd.api.types.is_numeric_dtype(X_fe[c]):
            numeric.append(c)
            if min_missing_indicator_rate <= X_fe[c].isna().mean() < 1.0:
                indicator.append(c)
        else:
            (high if X_fe[c].nunique(dropna=True) > high_card_threshold else low).append(c)
            if min_missing_indicator_rate <= X_fe[c].isna().mean():
                cat_marked.append(c)
    return ColumnGroups(numeric, low, high, indicator, cat_marked)


def _target_encoder(seed: int) -> TargetEncoder:
    """Cross-fitted target encoder (fit_transform uses out-of-fold encodings to avoid target leakage).
    The shuffling arguments moved into the ``cv`` splitter in scikit-learn 1.9."""
    major_minor = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if major_minor >= (1, 9):
        return TargetEncoder(target_type="binary", cv=KFold(5, shuffle=True, random_state=seed))
    return TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=seed)


def build_preprocessor(
    groups: ColumnGroups, scale_numeric: bool, winsorize: tuple[float, float] | None = None, seed: int = 42
) -> ColumnTransformer:
    """ColumnTransformer: numeric (median impute + missing indicators [+ winsorise + scale]),
    low-cardinality categoricals (one-hot), high-cardinality categoricals (cross-fitted target encoding)."""
    num_steps: list[tuple[str, Any]] = []
    if winsorize:
        num_steps.append(("winsor", Winsorizer(*winsorize)))
    num_steps.append(("impute", SimpleImputer(strategy="median")))
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    transformers: list[tuple[str, Any, list[str]]] = []
    if groups.numeric:
        transformers.append(("num", Pipeline(num_steps), groups.numeric))
    if groups.indicator:   # unscaled 0/1 flags, appended after the (scaled) numeric block
        transformers.append(("num_missing", MissingIndicator(features="all"), groups.indicator))
    # Categorical imputation: "MISSING" is a real category only for columns that are missing in training data.
    # For all others, an omitted field at inference time is imputed with the training mode instead of an
    # out-of-vocabulary label (which would give the model an all-zero, never-seen one-hot vector).
    def _split(cols: list[str]) -> tuple[list[str], list[str]]:
        return [c for c in cols if c in groups.cat_missing_informative], [c for c in cols if c not in groups.cat_missing_informative]

    for tag, cols in (("low", groups.low_card), ("high", groups.high_card)):
        marked, plain = _split(cols)
        for kind, sub, imputer in (
            ("marked", marked, SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("plain", plain, SimpleImputer(strategy="most_frequent")),
        ):
            if not sub:
                continue
            enc = (OneHotEncoder(handle_unknown="ignore", min_frequency=0.005, sparse_output=False)
                   if tag == "low" else _target_encoder(seed))
            transformers.append((f"cat_{tag}_{kind}", Pipeline([("impute", imputer), ("enc", enc)]), sub))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def split_data(
    df: pd.DataFrame, target: str, test_size: float, val_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified train / validation / test split. ``val_size`` and ``test_size`` are fractions of the full data."""
    X, y = df.drop(columns=[target]), df[target].astype(int)
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    rel_val = val_size / (1.0 - test_size)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=rel_val, stratify=y_tmp, random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te
