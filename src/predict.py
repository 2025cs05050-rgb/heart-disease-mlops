"""Inference helper: load the trained model and predict heart-disease risk.

The persisted ``model.pkl`` is a complete sklearn ``Pipeline`` containing
the preprocessing transformer (imputers + scaler + one-hot) and the
classifier, so inference takes raw feature dicts and applies the exact
same transformations as training.
"""
from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .data_loader import CATEGORICAL_FEATURES, NUMERIC_FEATURES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "model.pkl"

FEATURE_ORDER: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@lru_cache(maxsize=2)
def load_model(path: str | Path = DEFAULT_MODEL_PATH):
    """Load and cache the persisted model pipeline."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python -m src.train` first."
        )
    return joblib.load(path)


def _to_dataframe(records: dict | Iterable[dict]) -> pd.DataFrame:
    if isinstance(records, dict):
        records = [records]
    df = pd.DataFrame(list(records))
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return df[FEATURE_ORDER]


def predict(records: dict | Iterable[dict],
            model: Any | None = None) -> list[dict]:
    """Predict heart-disease risk for one or many patients.

    Returns
    -------
    list of dict
        Each dict has keys ``prediction`` (0/1), ``label`` (str),
        ``probability`` (float, P(disease)), and ``confidence``
        (max class probability).
    """
    model = model or load_model()
    X = _to_dataframe(records)

    proba = model.predict_proba(X)
    preds = proba.argmax(axis=1)

    out: list[dict] = []
    for p, row in zip(preds, proba, strict=True):
        prob_disease = float(row[1])
        out.append({
            "prediction": int(p),
            "label": "Disease" if p == 1 else "No disease",
            "probability": round(prob_disease, 4),
            "confidence": round(float(row.max()), 4),
        })
    return out


SAMPLE_PATIENT: dict = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
}


if __name__ == "__main__":
    print("Sample patient:", SAMPLE_PATIENT)
    print("Result        :", predict(SAMPLE_PATIENT))
