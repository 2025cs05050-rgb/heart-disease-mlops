"""Tests for the trained model artifact and the prediction wrapper."""
from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from src.predict import FEATURE_ORDER, load_model, predict

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.pkl"

requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model artifact not found — run `python -m src.train` first.",
)


@requires_model
def test_model_loads():
    model = load_model()
    # Sklearn pipelines expose predict / predict_proba
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


@requires_model
def test_predict_single_record(sample_patient):
    out = predict(sample_patient)
    assert isinstance(out, list) and len(out) == 1
    rec = out[0]
    assert set(rec.keys()) == {"prediction", "label", "probability",
                                "confidence"}
    assert rec["prediction"] in (0, 1)
    assert 0.0 <= rec["probability"] <= 1.0
    assert 0.5 <= rec["confidence"] <= 1.0
    assert rec["label"] in ("Disease", "No disease")


@requires_model
def test_predict_batch(sample_patient):
    batch = [sample_patient, {**sample_patient, "age": 70, "ca": 3}]
    out = predict(batch)
    assert len(out) == 2


@requires_model
def test_predict_missing_feature_raises(sample_patient):
    bad = {k: v for k, v in sample_patient.items() if k != "thal"}
    with pytest.raises(ValueError, match="thal"):
        predict(bad)


@requires_model
def test_feature_order_is_complete(sample_patient):
    """Sample patient must cover every feature the model expects."""
    assert set(FEATURE_ORDER) == set(sample_patient.keys())


@requires_model
def test_load_model_caches():
    """``lru_cache`` should return the same object on repeated calls."""
    assert load_model() is load_model()


@requires_model
def test_model_artifact_has_pipeline():
    """Persisted artifact must contain both preprocessing and estimator."""
    model = joblib.load(MODEL_PATH)
    step_names = [name for name, _ in model.steps]
    assert "preprocessor" in step_names
    assert "model" in step_names
