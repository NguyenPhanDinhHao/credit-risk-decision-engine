"""Input validation for training data and inference frames."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = [
    "SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "NAME_CONTRACT_TYPE",
]
CRITICAL_NOT_NULL = ["SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT"]
NUMERIC_REQUIRED = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED"]

# Plausibility rules used for invalid-value detection (training audit and production monitoring).
# (column, predicate that is True for VALID values among non-null entries, description)
VALUE_RULES: list[tuple[str, Any, str]] = [
    ("AMT_INCOME_TOTAL", lambda s: s > 0, "income must be > 0"),
    ("AMT_CREDIT", lambda s: s > 0, "credit amount must be > 0"),
    ("AMT_ANNUITY", lambda s: s > 0, "annuity must be > 0"),
    ("AMT_GOODS_PRICE", lambda s: s > 0, "goods price must be > 0"),
    ("CNT_CHILDREN", lambda s: (s >= 0) & (s < 30), "children count in [0, 30)"),
    ("CNT_FAM_MEMBERS", lambda s: (s >= 1) & (s < 40), "family members in [1, 40)"),
    ("EXT_SOURCE_1", lambda s: (s >= 0) & (s <= 1), "external score in [0, 1]"),
    ("EXT_SOURCE_2", lambda s: (s >= 0) & (s <= 1), "external score in [0, 1]"),
    ("EXT_SOURCE_3", lambda s: (s >= 0) & (s <= 1), "external score in [0, 1]"),
    ("DAYS_BIRTH", lambda s: s < 0, "DAYS_BIRTH is expected to be negative"),
]


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if self.errors:
            raise DataValidationError("; ".join(self.errors))


class DataValidationError(ValueError):
    """Raised when a dataset fails validation."""


def find_invalid_values(df: pd.DataFrame) -> dict[str, int]:
    """Count rule violations among non-null values, per column present in df."""
    out: dict[str, int] = {}
    for col, ok, _desc in VALUE_RULES:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            valid = ok(s.dropna())
            n_bad = int((~valid).sum())
            non_numeric = int(df[col].notna().sum() - s.notna().sum())
            if n_bad + non_numeric:
                out[col] = n_bad + non_numeric
    return out


def validate_application_data(
    df: pd.DataFrame,
    id_column: str = "SK_ID_CURR",
    target_column: str = "TARGET",
    require_target: bool = True,
    required_columns: list[str] | None = None,
) -> ValidationResult:
    res = ValidationResult()
    required = list(required_columns or REQUIRED_COLUMNS)
    if not require_target and target_column in required:
        required.remove(target_column)

    missing = [c for c in required if c not in df.columns]
    if missing:
        res.errors.append(f"Missing required columns: {missing}")
        return res  # later checks depend on these columns
    if len(df) == 0:
        res.errors.append("Dataset is empty")
        return res

    if id_column in df.columns:
        n_dup = int(df[id_column].duplicated().sum())
        res.stats["duplicate_ids"] = n_dup
        if n_dup:
            res.errors.append(f"{n_dup} duplicate values in {id_column}")
    res.stats["duplicate_rows"] = int(df.duplicated().sum())

    if require_target:
        t = df[target_column]
        if t.isna().any():
            res.errors.append(f"{int(t.isna().sum())} missing values in {target_column}")
        bad = set(pd.unique(t.dropna())) - {0, 1}
        if bad:
            res.errors.append(f"Unexpected {target_column} values: {sorted(bad, key=str)[:5]} (expected 0/1)")
        elif t.nunique() < 2:
            res.errors.append(f"{target_column} has a single class; cannot train a classifier")
        else:
            res.stats["default_rate"] = float(t.mean())

    for col in NUMERIC_REQUIRED:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            res.errors.append(f"Column {col} must be numeric, found dtype {df[col].dtype}")

    for col in CRITICAL_NOT_NULL:
        if col in df.columns and (col != target_column or require_target):
            n_null = int(df[col].isna().sum())
            if n_null:
                res.errors.append(f"{n_null} missing values in critical column {col}")

    invalid = find_invalid_values(df)
    if invalid:
        res.warnings.append(f"Values violating plausibility rules: {invalid}")
    res.stats["invalid_values"] = invalid
    res.stats["n_rows"], res.stats["n_columns"] = df.shape
    return res
