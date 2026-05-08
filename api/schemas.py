"""Pydantic request/response schemas for the heart-disease API.

Validation ranges follow the UCI dataset documentation so that
clearly-out-of-range inputs are rejected at the edge.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    """Raw patient features expected by the model."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
            }
        }
    )

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="0 = female, 1 = male")
    cp: Literal[1, 2, 3, 4] = Field(..., description="Chest pain type")
    trestbps: float = Field(..., ge=50, le=260,
                            description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=80, le=700,
                        description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG result")
    thalach: float = Field(..., ge=50, le=250,
                           description="Maximum heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0.0, le=10.0,
                           description="ST depression vs rest")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak ST")
    ca: Literal[0, 1, 2, 3] = Field(..., description="# major vessels (0-3)")
    thal: Literal[3, 6, 7] = Field(...,
                                   description="3=normal, 6=fixed, 7=reversible")


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    label: Literal["No disease", "Disease"]
    probability: float = Field(..., ge=0.0, le=1.0,
                               description="P(disease)")
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
