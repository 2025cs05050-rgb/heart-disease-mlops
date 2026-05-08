"""Integration tests for the FastAPI service."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.pkl"

requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model artifact not found — run `python -m src.train` first.",
)


@pytest.fixture(scope="module")
def client():
    from api.app import app
    with TestClient(app) as c:
        yield c


@requires_model
def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "heart-disease-api"
    assert "/predict" in body["endpoints"]


@requires_model
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


@requires_model
def test_predict_happy_path(client, sample_patient):
    resp = client.post("/predict", json=sample_patient)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("Disease", "No disease")
    assert 0.0 <= body["probability"] <= 1.0
    assert 0.5 <= body["confidence"] <= 1.0
    # Model version header is informational
    assert body["model_version"]


@requires_model
def test_predict_disease_case(client, sample_patient):
    """A high-risk profile should get a higher P(disease) than a healthy one."""
    high_risk = {**sample_patient, "age": 70, "cp": 4, "thalach": 110,
                 "exang": 1, "oldpeak": 3.5, "ca": 3, "thal": 7}
    low_risk = {**sample_patient, "age": 40, "cp": 2, "thalach": 180,
                "exang": 0, "oldpeak": 0.5, "ca": 0, "thal": 3}
    p_high = client.post("/predict", json=high_risk).json()["probability"]
    p_low = client.post("/predict", json=low_risk).json()["probability"]
    assert p_high > p_low


@requires_model
def test_predict_validation_error(client, sample_patient):
    bad = {**sample_patient, "age": -5}  # invalid age
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


@requires_model
def test_predict_missing_field(client, sample_patient):
    bad = {k: v for k, v in sample_patient.items() if k != "thal"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


@requires_model
def test_request_id_header(client, sample_patient):
    resp = client.post("/predict", json=sample_patient)
    assert "X-Request-ID" in resp.headers
    assert len(resp.headers["X-Request-ID"]) >= 6


@requires_model
def test_metrics_endpoint(client):
    """If the prometheus instrumentator is installed it should expose /metrics."""
    resp = client.get("/metrics")
    if resp.status_code == 404:
        pytest.skip("prometheus_fastapi_instrumentator not installed")
    assert resp.status_code == 200
    assert "http_requests_total" in resp.text or "process_" in resp.text


@requires_model
def test_custom_prediction_metrics_exposed(client, sample_patient):
    """After a /predict call the custom prediction metrics show up in /metrics."""
    client.post("/predict", json=sample_patient)
    resp = client.get("/metrics")
    if resp.status_code == 404:
        pytest.skip("prometheus_fastapi_instrumentator not installed")
    body = resp.text
    assert "heart_predictions_total" in body
    assert "heart_prediction_latency_seconds" in body
    assert "heart_prediction_probability" in body


@requires_model
def test_request_id_is_propagated(client, sample_patient):
    """Caller-supplied X-Request-ID should be echoed back unchanged."""
    rid = "test-rid-12345"
    resp = client.post("/predict", json=sample_patient,
                       headers={"X-Request-ID": rid})
    assert resp.headers["X-Request-ID"] == rid


def test_json_log_formatter_emits_valid_json():
    """The JsonFormatter produces parseable JSON with mandatory fields."""
    import json
    import logging

    from api.logging_config import JsonFormatter

    formatter = JsonFormatter(service="heart-disease-api", version="v1.0.0")
    record = logging.LogRecord(
        name="heart-api", level=logging.INFO, pathname=__file__, lineno=1,
        msg="prediction", args=(), exc_info=None,
    )
    record.event = "prediction"
    record.request_id = "abc123"
    record.probability = 0.42

    payload = json.loads(formatter.format(record))
    assert payload["level"] == "INFO"
    assert payload["service"] == "heart-disease-api"
    assert payload["version"] == "v1.0.0"
    assert payload["message"] == "prediction"
    assert payload["event"] == "prediction"
    assert payload["request_id"] == "abc123"
    assert payload["probability"] == 0.42
    assert "ts" in payload
