import pytest

from credit_risk.config import load_config
from credit_risk.data.synthetic import generate_synthetic_applications
from credit_risk.models.predict import ModelBundle
from credit_risk.models.train import run_training


@pytest.fixture(scope="session")
def synthetic_df():
    """SYNTHETIC fixture data - for tests only."""
    return generate_synthetic_applications(4000, seed=7)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, synthetic_df):
    cfg = load_config()
    cfg["training"]["n_iter_tuning"] = 2
    cfg["models"]["random_forest"]["fixed"]["n_estimators"] = 20
    cfg["models"]["lightgbm"]["fixed"]["n_estimators"] = 40
    cfg["mlflow"]["enabled"] = False
    art, rep = tmp_path_factory.mktemp("artifacts"), tmp_path_factory.mktemp("reports")
    result = run_training(cfg, synthetic_df, "synthetic_test", art, rep, mlflow_enabled=False, run_shap=False)
    return {"result": result, "bundle": ModelBundle.load(art / "model_bundle.joblib"), "art": art, "cfg": cfg}


@pytest.fixture()
def valid_payload():
    return {"AMT_INCOME_TOTAL": 150000, "AMT_CREDIT": 450000, "AMT_ANNUITY": 18000, "AMT_GOODS_PRICE": 400000,
            "NAME_CONTRACT_TYPE": "Cash loans", "DAYS_EMPLOYED": -1500, "CNT_CHILDREN": 1, "CNT_FAM_MEMBERS": 3,
            "EXT_SOURCE_1": 0.5, "EXT_SOURCE_2": 0.6, "EXT_SOURCE_3": 0.4, "REGION_RATING_CLIENT": 2,
            "NAME_INCOME_TYPE": "Working", "ORGANIZATION_TYPE": "Organization type 3"}
