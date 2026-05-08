"""Pytest fixtures shared across test modules.

Ensures the project root is on ``sys.path`` so ``import src.*`` works
regardless of how pytest is invoked (locally, in CI, from a sub-folder).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def sample_patient() -> dict:
    """A valid feature dict matching row 1 of the Cleveland dataset."""
    return {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
    }


@pytest.fixture
def tiny_dataframe(sample_patient) -> pd.DataFrame:
    """A small DataFrame with both classes for fast pipeline tests."""
    rows = [
        {**sample_patient, "target": 0},
        {**sample_patient, "age": 67, "cp": 4, "thalach": 108,
         "exang": 1, "ca": 3, "target": 1},
        {**sample_patient, "age": 41, "sex": 0, "cp": 2,
         "trestbps": 130, "thalach": 172, "target": 0},
        {**sample_patient, "age": 56, "cp": 4, "exang": 1,
         "oldpeak": 1.8, "ca": 2, "thal": 7, "target": 1},
    ]
    return pd.DataFrame(rows)
