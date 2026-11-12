"""SYNTHETIC data generator for tests, smoke runs and the --demo mode.

!! The generated data is entirely artificial. It mimics the *schema* of Home Credit's
application_train.csv (a subset of columns) so the pipeline can be exercised when the real data is
unavailable. Metrics computed on it carry no information about real-world credit risk. !!
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ORG_TYPES = [f"Organization type {i}" for i in range(1, 26)]
OCCUPATIONS = ["Laborers", "Core staff", "Sales staff", "Managers", "Drivers", "High skill tech staff", "Accountants", "Medicine staff"]


def generate_synthetic_applications(
    n: int = 20000, seed: int = 42, drift: dict | None = None, with_target: bool = True
) -> pd.DataFrame:
    """Generate n synthetic applications. ``drift`` optionally perturbs distributions (monitoring demo)."""
    rng = np.random.default_rng(seed)
    drift = drift or {}
    age_years = rng.uniform(21, 68, n)
    days_birth = -(age_years * 365.25).astype(int)
    income = np.exp(rng.normal(11.9, 0.55, n)) * drift.get("income_scale", 1.0)
    credit = income * rng.uniform(1.0, 6.0, n)
    annuity = credit / rng.uniform(8, 40, n)
    goods = credit * rng.uniform(0.85, 1.0, n)
    pensioner = rng.random(n) < 0.15
    employed_years = np.minimum(rng.exponential(6, n), age_years - 18)
    days_employed = np.where(pensioner, 365243, -(employed_years * 365.25).astype(int))
    ext1 = np.clip(rng.beta(4, 3, n), 0, 1)
    ext2 = np.clip(rng.beta(5, 3, n), 0, 1)
    ext3 = np.clip(rng.beta(4, 3, n), 0, 1)
    region = rng.choice([1, 2, 3], n, p=[0.12, 0.74, 0.14])
    education = rng.choice(["Secondary", "Higher education", "Incomplete higher", "Lower secondary"], n, p=[0.7, 0.2, 0.07, 0.03])
    income_type = rng.choice(["Working", "Commercial associate", "Pensioner", "State servant"], n, p=[0.5, 0.23, 0.18, 0.09])
    gender = rng.choice(["F", "M"], n, p=[0.65, 0.35])
    n_children = rng.choice([0, 1, 2, 3], n, p=[0.7, 0.2, 0.08, 0.02])
    org = rng.choice(ORG_TYPES, n)
    org_risk = dict(zip(ORG_TYPES, rng.normal(0, 0.15, len(ORG_TYPES)), strict=True))

    # Latent logit with known structure (synthetic ground truth).
    z = (
        -3.5
        - 2.2 * (ext1 - 0.57) - 3.0 * (ext2 - 0.62) - 2.5 * (ext3 - 0.57)
        + 0.35 * np.log(credit / income)
        + 0.25 * (region - 2)
        - 0.25 * (education == "Higher education")
        + np.array([org_risk[o] for o in org])
        + 0.5 * (annuity / income > 0.25)
        + 0.1 * n_children
        + 0.15 * (gender == "M")                    # present in the synthetic truth to test sensitive-feature handling
        + rng.normal(0, 0.4, n)
    )
    prob = 1 / (1 + np.exp(-z))

    def with_nan(a: np.ndarray, p: float) -> np.ndarray:
        a = a.astype("float64").copy()
        a[rng.random(n) < p] = np.nan
        return a

    ext1_miss = 0.56 + drift.get("ext1_missing_extra", 0.0)
    df = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100000, 100000 + n),
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]),
            "CODE_GENDER": gender,
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], n, p=[0.34, 0.66]),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n, p=[0.69, 0.31]),
            "CNT_CHILDREN": n_children,
            "AMT_INCOME_TOTAL": income.round(0),
            "AMT_CREDIT": credit.round(0),
            "AMT_ANNUITY": with_nan(annuity.round(0), 0.0004),
            "AMT_GOODS_PRICE": with_nan(goods.round(0), 0.001),
            "NAME_INCOME_TYPE": income_type,
            "NAME_EDUCATION_TYPE": education,
            "NAME_FAMILY_STATUS": rng.choice(["Married", "Single", "Civil marriage", "Separated"], n, p=[0.63, 0.15, 0.1, 0.12]),
            "NAME_HOUSING_TYPE": rng.choice(["House / apartment", "With parents", "Municipal apartment"], n, p=[0.88, 0.05, 0.07]),
            "DAYS_BIRTH": days_birth,
            "DAYS_EMPLOYED": days_employed,
            "DAYS_REGISTRATION": -rng.integers(0, 20000, n),
            "DAYS_ID_PUBLISH": -rng.integers(0, 6500, n),
            "DAYS_LAST_PHONE_CHANGE": with_nan(-rng.integers(0, 4000, n), 0.0001),
            "OCCUPATION_TYPE": np.where(rng.random(n) < 0.31, None, rng.choice(OCCUPATIONS, n)),
            "CNT_FAM_MEMBERS": n_children + rng.choice([1, 2], n, p=[0.3, 0.7]),
            "REGION_RATING_CLIENT": region,
            "ORGANIZATION_TYPE": org,
            "EXT_SOURCE_1": np.where(rng.random(n) < ext1_miss, np.nan, ext1),
            "EXT_SOURCE_2": with_nan(ext2, 0.002),
            "EXT_SOURCE_3": with_nan(ext3, 0.19),
            "OBS_30_CNT_SOCIAL_CIRCLE": with_nan(rng.poisson(1.4, n), 0.003),
            "AMT_REQ_CREDIT_BUREAU_YEAR": with_nan(rng.poisson(1.9, n), 0.135),
            "FLAG_DOCUMENT_3": rng.choice([0, 1], n, p=[0.29, 0.71]),
        }
    )
    if with_target:
        df.insert(1, "TARGET", (rng.random(n) < prob).astype(int))
    return df
