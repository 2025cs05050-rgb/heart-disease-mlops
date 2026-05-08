# syntax=docker/dockerfile:1.7
# ----------------------------------------------------------------------
# Multi-stage build for the Heart Disease FastAPI service.
# Stage 1 (builder) installs deps and trains the model from scratch so
# the final image is fully reproducible from source.
# Stage 2 (runtime) carries only the slim wheels + trained artifact.
# ----------------------------------------------------------------------

ARG PYTHON_VERSION=3.11

# ====== builder ======
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build/runtime deps first (better layer cache)
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Copy source needed to train + serve
COPY src ./src
COPY scripts ./scripts
COPY api ./api

# Fetch dataset and train so the image ships with a baked model.
RUN python scripts/download_data.py && \
    python -m src.train

# ====== runtime ======
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:${PATH}" \
    MODEL_PATH=/app/models/model.pkl \
    MODEL_VERSION=v1.0.0 \
    PORT=8000

# Run as non-root for safety
RUN groupadd --system app && \
    useradd --system --create-home --gid app appuser

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy source + trained artifact
COPY --from=builder /build/src ./src
COPY --from=builder /build/api ./api
COPY --from=builder /build/models ./models

RUN chown -R appuser:app /app /home/appuser/.local

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)" \
    || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
