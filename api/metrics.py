"""Custom Prometheus metrics for the heart-disease API.

These complement the HTTP-level metrics produced by
``prometheus_fastapi_instrumentator`` with prediction-specific counters
and histograms useful for ML monitoring (drift detection, class-balance
sanity checks, latency SLOs).
"""
from __future__ import annotations

from prometheus_client import Counter, Histogram

PREDICTIONS_TOTAL = Counter(
    "heart_predictions_total",
    "Total predictions served, labelled by predicted class.",
    labelnames=("label", "model_version"),
)

PREDICTION_PROBABILITY = Histogram(
    "heart_prediction_probability",
    "Distribution of predicted P(disease) values.",
    labelnames=("model_version",),
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

PREDICTION_LATENCY = Histogram(
    "heart_prediction_latency_seconds",
    "End-to-end latency of /predict (model.predict + serialization).",
    labelnames=("model_version",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

PREDICTION_ERRORS = Counter(
    "heart_prediction_errors_total",
    "Failed predictions (model errors, not validation 4xx).",
    labelnames=("model_version", "error_type"),
)
