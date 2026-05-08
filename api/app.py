"""FastAPI service exposing the heart-disease classifier.

Endpoints
---------
GET  /          -> service banner
GET  /health    -> liveness/readiness probe (model loaded?)
POST /predict   -> single-patient prediction (JSON in/out)
GET  /metrics   -> Prometheus exposition (added by instrumentator)
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from src.predict import load_model, predict

from .logging_config import configure_logging
from .metrics import (
    PREDICTION_ERRORS,
    PREDICTION_LATENCY,
    PREDICTION_PROBABILITY,
    PREDICTIONS_TOTAL,
)
from .schemas import HealthResponse, PatientFeatures, PredictionResponse

MODEL_PATH = Path(os.getenv(
    "MODEL_PATH",
    str(Path(__file__).resolve().parents[1] / "models" / "model.pkl"),
))
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")

configure_logging(service="heart-disease-api", version=MODEL_VERSION)
log = logging.getLogger("heart-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model(MODEL_PATH)
        app.state.model_loaded = True
        log.info("Model loaded", extra={"event": "model_loaded",
                                        "model_path": str(MODEL_PATH)})
    except Exception as exc:  # noqa: BLE001
        app.state.model_loaded = False
        log.error("Failed to load model",
                  extra={"event": "model_load_failed", "error": str(exc)})
    yield


app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts heart-disease risk from patient health features.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


# Optional Prometheus instrumentation (Task 8). Stays a no-op if the
# package is missing so the API still works in minimal environments.
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    log.info("Prometheus /metrics endpoint enabled")
except ImportError:
    log.warning("prometheus_fastapi_instrumentator not installed — "
                "skipping /metrics")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    log.info(
        "request",
        extra={
            "event": "http_request",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "client": request.client.host if request.client else None,
        },
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/")
def root() -> dict:
    return {
        "service": "heart-disease-api",
        "version": MODEL_VERSION,
        "docs": "/docs",
        "endpoints": ["/health", "/predict", "/metrics"],
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = bool(getattr(app.state, "model_loaded", False))
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(features: PatientFeatures,
                     request: Request) -> PredictionResponse:
    if not getattr(app.state, "model_loaded", False):
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = request.headers.get("X-Request-ID", "-")
    start = time.perf_counter()
    try:
        result = predict(features.model_dump())[0]
    except Exception as exc:  # noqa: BLE001
        PREDICTION_ERRORS.labels(MODEL_VERSION, type(exc).__name__).inc()
        log.exception("Prediction failed",
                      extra={"event": "prediction_failed",
                             "request_id": request_id,
                             "error_type": type(exc).__name__})
        raise HTTPException(status_code=500,
                            detail=f"Prediction error: {exc}") from exc

    elapsed = time.perf_counter() - start
    PREDICTION_LATENCY.labels(MODEL_VERSION).observe(elapsed)
    PREDICTIONS_TOTAL.labels(result["label"], MODEL_VERSION).inc()
    PREDICTION_PROBABILITY.labels(MODEL_VERSION).observe(result["probability"])

    log.info(
        "prediction",
        extra={
            "event": "prediction",
            "request_id": request_id,
            "prediction": result["prediction"],
            "label": result["label"],
            "probability": round(result["probability"], 4),
            "confidence": round(result["confidence"], 4),
            "latency_ms": round(elapsed * 1000, 2),
            "model_version": MODEL_VERSION,
        },
    )

    return PredictionResponse(
        prediction=result["prediction"],
        label=result["label"],
        probability=result["probability"],
        confidence=result["confidence"],
        model_version=MODEL_VERSION,
    )
