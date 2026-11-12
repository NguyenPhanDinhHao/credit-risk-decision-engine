import json

import pytest
from fastapi.testclient import TestClient

from credit_risk.api import dependencies as deps
from credit_risk.api.main import app


@pytest.fixture()
def client(trained, tmp_path, monkeypatch):
    monkeypatch.setenv("CREDIT_RISK_MODEL_PATH", str(trained["art"] / "model_bundle.joblib"))
    monkeypatch.setenv("CREDIT_RISK_LOG_PREDICTIONS", "true")
    monkeypatch.setenv("CREDIT_RISK_PREDICTION_LOG", str(tmp_path / "preds.jsonl"))
    deps.load_bundle(force=True)
    with TestClient(app) as c:
        c.log_path = tmp_path / "preds.jsonl"
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    j = client.get("/model-info").json()
    assert j["model_version"] and "disclaimer" in j and "CODE_GENDER" in j["excluded_sensitive_features"]
    assert set(j["risk_bands"]) == {"low_max", "medium_max"}


def test_predict_valid(client, valid_payload):
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 200
    j = r.json()
    assert 0 <= j["probability_of_default"] <= 1 and j["risk_category"] in {"Low Risk", "Medium Risk", "High Risk"}
    assert j["model_version"] and j["top_factors"] is None
    assert json.loads(client.log_path.read_text().splitlines()[-1])["model_version"] == j["model_version"]


def test_predict_with_explanation(client, valid_payload):
    j = client.post("/predict?explain=true", json=valid_payload).json()
    assert len(j["top_factors"]) == 5 and all(f["direction"] in ("increases_risk", "decreases_risk") for f in j["top_factors"])


def test_predict_partial_input_is_accepted(client):
    assert client.post("/predict", json={"AMT_CREDIT": 100000}).status_code == 200


@pytest.mark.parametrize("bad", [
    {"AMT_INCOME_TOTAL": -1}, {"AMT_CREDIT": "abc"}, {"EXT_SOURCE_1": 1.5}, {"NAME_CONTRACT_TYPE": "Mortgage"},
    {"CNT_CHILDREN": 3.5}, {"AMT_CREDIT": {"a": 1}},
])
def test_predict_invalid_input_is_rejected_safely(client, bad):
    r = client.post("/predict", json=bad)
    assert r.status_code == 422 and r.json()["detail"] == "Invalid input"
    body = json.dumps(r.json())
    assert '"input"' not in body and "abc" not in body and "Traceback" not in r.text   # submitted values are never echoed


def test_predict_malformed_json(client):
    assert client.post("/predict", content="not json", headers={"content-type": "application/json"}).status_code == 422


def test_batch(client, valid_payload):
    r = client.post("/predict/batch", json={"applicants": [valid_payload, {"AMT_CREDIT": 20000}, {}]})
    j = r.json()
    assert r.status_code == 200 and j["count"] == 3 and [p["index"] for p in j["predictions"]] == [0, 1, 2]
    assert all(0 <= p["probability_of_default"] <= 1 for p in j["predictions"])


def test_batch_validation_and_limits(client, valid_payload, monkeypatch):
    assert client.post("/predict/batch", json={"applicants": []}).status_code == 422
    assert client.post("/predict/batch", json={"applicants": [valid_payload, {"AMT_CREDIT": -5}]}).status_code == 422
    monkeypatch.setattr(deps, "settings", lambda: {"max_batch_size": 2, "log_predictions": False, "prediction_log": None})
    assert client.post("/predict/batch", json={"applicants": [valid_payload] * 3}).status_code == 413


def test_internal_errors_do_not_leak(client, valid_payload, monkeypatch):
    from credit_risk.models.predict import ModelBundle

    def boom(self, df):
        raise RuntimeError("secret internal path /srv/model")

    monkeypatch.setattr(ModelBundle, "predict_proba", boom)
    c = TestClient(app, raise_server_exceptions=False)
    r = c.post("/predict", json=valid_payload)
    assert r.status_code == 500 and "secret" not in r.text and r.json()["detail"] == "Internal server error"


def test_unavailable_model_returns_503_and_degraded_health(tmp_path, monkeypatch):
    monkeypatch.setenv("CREDIT_RISK_MODEL_PATH", str(tmp_path / "missing.joblib"))
    deps.load_bundle(force=True)
    with TestClient(app) as c:
        assert c.get("/health").json() == {"status": "degraded", "model_loaded": False}
        assert c.post("/predict", json={"AMT_CREDIT": 1000}).status_code == 503


def test_openapi_docs_available(client):
    assert client.get("/openapi.json").status_code == 200 and client.get("/docs").status_code == 200
