# MLOps Assignment I — Heart Disease Risk Prediction

End-to-end MLOps pipeline on the **UCI Heart Disease (Cleveland)** dataset:
EDA → preprocessing pipeline → model training (LogReg + RandomForest) →
MLflow experiment tracking → packaging → CI/CD → FastAPI Docker container →
Kubernetes deployment → Prometheus/Grafana monitoring.

**Course:** MLOps (S2-25_AMLCSZG523) | **Total marks:** 50

---

## Project layout

```
Assignment1/
├── data/
│   ├── raw/            # downloaded UCI files
│   └── processed/      # cleaned CSV
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── notebooks/
│   └── 01_eda.ipynb
├── api/
│   ├── app.py
│   └── schemas.py
├── tests/
├── docker/
│   └── Dockerfile
├── k8s/
├── monitoring/
├── .github/workflows/
├── scripts/
│   └── download_data.py
├── requirements.txt
└── README.md
```

## Quick start

```bash
# 1. Create environment
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Download dataset
python scripts/download_data.py

# 3. Run EDA notebook
jupyter notebook notebooks/01_eda.ipynb

# 4. Train models (with MLflow tracking)
python -m src.train

# 5. View MLflow UI
mlflow ui --backend-store-uri ./mlruns

# 6. Run API locally
uvicorn api.app:app --reload --port 8000

# 7. Run tests
pytest -v

# 8. Build Docker image (multi-stage; trains the model inside the build)
docker build -t heart-disease-api:latest -f docker/Dockerfile .
docker run --rm -p 8000:8000 heart-disease-api:latest

# 9. Smoke-test the running container
curl http://localhost:8000/health
```

## API endpoints

| Method | Path        | Description                                    |
|--------|-------------|------------------------------------------------|
| GET    | `/`         | Service banner                                 |
| GET    | `/health`   | Liveness/readiness (model loaded?)             |
| POST   | `/predict`  | Single-patient prediction (JSON in/out)        |
| GET    | `/metrics`  | Prometheus metrics (Task 8)                    |
| GET    | `/docs`     | Interactive Swagger UI                         |

## Sample API request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,
       "restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,
       "slope":3,"ca":0,"thal":6}'
```

## Monitoring & logging

The API ships with structured **JSON logs** on stdout and a Prometheus
`/metrics` endpoint exposing both standard HTTP metrics and four custom
prediction metrics (`heart_predictions_total`,
`heart_prediction_probability`, `heart_prediction_latency_seconds`,
`heart_prediction_errors_total`).

A ready-to-run observability stack lives under `monitoring/`:

```bash
docker compose -f monitoring/docker-compose.yml up --build
# API:        http://localhost:8000/docs
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin / admin)
```

The pre-provisioned Grafana dashboard (10 panels: throughput, latency
percentiles, class balance, score distribution heatmap, error rates) is
auto-loaded under **Dashboards → ML → Heart Disease API**. See
[`monitoring/README.md`](monitoring/README.md) for details.

## Kubernetes deployment

Manifests for Minikube / Docker-Desktop K8s / managed clusters are in
[`k8s/`](k8s/README.md). Apply with `kubectl apply -k k8s/`.

## Report

A consolidated PDF report (~11 pages, cover + 12 sections + appendices)
is built from real metrics + figures by:

```bash
python scripts/generate_report.py
# -> reports/MLOps_Assignment1_Report.pdf
```

## Dataset

UCI Heart Disease — Cleveland processed file (303 rows × 14 columns).
Features: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
oldpeak, slope, ca, thal`. Target `num` is binarised: 0 = no disease,
1 = disease present (originally 1–4).

Source: https://archive.ics.uci.edu/dataset/45/heart+disease
