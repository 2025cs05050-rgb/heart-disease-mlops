"""Prose for the MLOps Assignment-1 PDF report.

Each ``section_*`` function returns a list of ReportLab flowables and
takes the ``styles`` dict + ``helpers`` dict supplied by
``generate_report.py``.
"""
from __future__ import annotations


def _p(helpers, styles, text, style_key="body"):
    return helpers["Paragraph"](text, styles[style_key])


def _bullets(helpers, styles, items):
    return [helpers["Paragraph"](f"• {t}", styles["bullet"]) for t in items]


def _spacer(helpers, h=8):
    return helpers["Spacer"](1, h)


# --------------------------------------------------------------------- cover
def cover_block(styles, helpers):
    return [
        _spacer(helpers, 80),
        _p(helpers, styles,
           "MLOps Assignment&nbsp;I", "title"),
        _p(helpers, styles,
           "End-to-End Heart Disease Risk Prediction Pipeline", "subtitle"),
        _spacer(helpers, 24),
        _p(helpers, styles,
           "<b>Course:</b> AMLCSZG523 — MLOps", "subtitle"),
        _p(helpers, styles,
           "<b>Programme:</b> BITS Pilani M.Tech (WILP) — Semester 2",
           "subtitle"),
        _p(helpers, styles,
           "<b>Dataset:</b> UCI Heart Disease (Cleveland processed)",
           "subtitle"),
        _p(helpers, styles,
           "<b>Student ID:</b> 2025cs05050-rgb",
           "subtitle"),
        _p(helpers, styles,
           "<b>Repository:</b> "
           "https://github.com/2025cs05050-rgb/heart-disease-mlops",
           "subtitle"),
        _spacer(helpers, 50),
        _p(helpers, styles,
           f"Date: {helpers['today']}", "subtitle"),
        _spacer(helpers, 30),
        _p(helpers, styles,
           "<i>This report documents the design, implementation, and "
           "operational characteristics of an end-to-end ML pipeline "
           "covering data acquisition, model development, experiment "
           "tracking, CI/CD, containerisation, Kubernetes deployment, and "
           "production monitoring. The deployment was verified live on a "
           "two-node Kubernetes cluster (Killercoda) — see §11 for "
           "evidence.</i>", "body"),
    ]


# --------------------------------------------------- 1. executive summary
def section_executive_summary(styles, helpers, metrics):
    sel = metrics["selected_model"]
    test = metrics["test_metrics"]
    return [
        _p(helpers, styles, "1. Executive Summary", "h1"),
        _p(helpers, styles,
           "We built a reproducible binary-classification pipeline that "
           "predicts presence of heart disease from 13 routinely-collected "
           "clinical features. Two candidate models — Logistic Regression "
           "and Random Forest — were tuned via 5-fold stratified cross-"
           "validation with grid search over <b>roc_auc</b>. The selected "
           f"model is <b>{sel.replace('_', ' ').title()}</b>, achieving a "
           f"test accuracy of <b>{test['test_accuracy']:.3f}</b>, F1 of "
           f"<b>{test['test_f1']:.3f}</b> and ROC-AUC of "
           f"<b>{test['test_roc_auc']:.3f}</b>."),
        _p(helpers, styles,
           "Beyond modelling, the project delivers all production-readiness "
           "concerns required by the rubric: experiment tracking with "
           "MLflow, a CI workflow that lints + tests every push, a "
           "multi-stage Dockerfile that bakes the trained artefact, "
           "Kubernetes manifests with rolling updates and an HPA, and a "
           "local docker-compose observability stack with Prometheus and "
           "Grafana."),
    ]


# --------------------------------------------------- 2. problem statement
def section_problem(styles, helpers):
    return [
        _p(helpers, styles, "2. Problem Statement & Dataset", "h1"),
        _p(helpers, styles,
           "Heart disease remains the leading cause of death worldwide. "
           "Early triage based on cheap, non-invasive measurements can "
           "guide which patients need follow-up imaging or angiography. "
           "We therefore frame the task as binary classification: <b>given "
           "13 clinical features, predict whether a patient has angiographic "
           "heart disease (target ≥ 1)</b>."),
        _p(helpers, styles, "Dataset", "h2"),
        *_bullets(helpers, styles, [
            "Source: UCI Machine Learning Repository (Heart Disease dataset, "
            "Cleveland processed file).",
            "Rows: 303 patients. Columns: 13 features + 1 target.",
            "Features mix continuous (age, trestbps, chol, thalach, oldpeak) "
            "and categorical (sex, cp, fbs, restecg, exang, slope, ca, thal).",
            "Target <i>num</i> is the angiographic disease severity (0–4); we "
            "binarise to 0 (no disease) vs 1 (disease present).",
            "Missing values are encoded as <code>?</code> in the raw data; "
            "the loader converts them to NaN for downstream imputation.",
        ]),
    ]


# --------------------------------------------------- 3. repo
def section_repo(styles, helpers):
    return [
        _p(helpers, styles, "3. Repository Layout", "h1"),
        _p(helpers, styles,
           "The project follows a conventional ML-engineering layout that "
           "cleanly separates source code, tests, infrastructure, and "
           "reports. Each subdirectory ships its own README where useful."),
        _p(helpers, styles,
           "<font face='Courier' size='9'>"
           "MLOps/Assignment1/<br/>"
           "├── api/            FastAPI service (app, schemas, logging, metrics)<br/>"
           "├── data/           raw + processed (git-ignored)<br/>"
           "├── docker/         multi-stage Dockerfile<br/>"
           "├── k8s/            Namespace, Deployment, Service, Ingress, HPA, Kustomize<br/>"
           "├── monitoring/     docker-compose, Prometheus, Grafana dashboards<br/>"
           "├── notebooks/      EDA notebook<br/>"
           "├── reports/        metrics.json, figures, generated PDF<br/>"
           "├── scripts/        download_data, generate_report<br/>"
           "├── src/            data_loader, preprocess, train, evaluate, predict<br/>"
           "├── tests/          pytest suite (35 tests, lint-clean)<br/>"
           "├── .github/workflows/ci.yml<br/>"
           "├── Dockerfile, requirements.txt, pyproject.toml<br/>"
           "└── README.md"
           "</font>", "body"),
    ]


# --------------------------------------------------- 4. EDA
def section_eda(styles, helpers):
    return [
        _p(helpers, styles, "4. Exploratory Data Analysis", "h1"),
        _p(helpers, styles,
           "EDA was performed in <code>notebooks/01_eda.ipynb</code>. Key "
           "observations that informed pipeline design:"),
        *_bullets(helpers, styles, [
            "<b>Class balance:</b> roughly 54% / 46% (no-disease / disease) — "
            "no resampling required; we still report precision and recall "
            "alongside accuracy.",
            "<b>Missingness:</b> only <i>ca</i> (4 rows) and <i>thal</i> "
            "(2 rows) contain '?'. Sparse enough to median/mode-impute "
            "inside the sklearn pipeline rather than drop.",
            "<b>Univariate signal:</b> <i>cp</i> (chest pain type), "
            "<i>thalach</i> (max heart rate), <i>oldpeak</i> (ST depression) "
            "and <i>ca</i> show the strongest separation across classes in "
            "histograms and box-plots.",
            "<b>Correlations:</b> <i>thalach</i> is strongly negatively "
            "correlated with <i>age</i> (~ -0.4) and with the target "
            "(~ -0.42); <i>oldpeak</i> and <i>ca</i> are positively "
            "correlated with the target. No pair exceeds 0.5 absolute "
            "correlation, so multicollinearity is not a concern.",
            "<b>Outliers:</b> a handful of patients with cholesterol > 400 "
            "mg/dl exist; we keep them and rely on the StandardScaler to "
            "compress their influence on linear models.",
        ]),
    ]


# --------------------------------------------------- 5. pipeline
def section_pipeline(styles, helpers):
    return [
        _p(helpers, styles, "5. Feature Engineering & Pipeline", "h1"),
        _p(helpers, styles,
           "All preprocessing is encoded as a single sklearn "
           "<code>Pipeline</code> wrapping a <code>ColumnTransformer</code>, "
           "guaranteeing the same transformations are applied at training "
           "time, in the API container, and during CI smoke tests."),
        _p(helpers, styles,
           "<font face='Courier' size='9'>"
           "Pipeline([<br/>"
           "&nbsp;&nbsp;('preprocess', ColumnTransformer([<br/>"
           "&nbsp;&nbsp;&nbsp;&nbsp;('num', Pipeline([SimpleImputer(median), "
           "StandardScaler()]), NUMERIC_COLS),<br/>"
           "&nbsp;&nbsp;&nbsp;&nbsp;('cat', Pipeline([SimpleImputer(most_freq), "
           "OneHotEncoder(handle_unknown='ignore')]), CATEGORICAL_COLS),<br/>"
           "&nbsp;&nbsp;])),<br/>"
           "&nbsp;&nbsp;('model', estimator),<br/>"
           "])"
           "</font>", "body"),
        *_bullets(helpers, styles, [
            "<b>Numeric (5 cols):</b> median imputation + standardisation.",
            "<b>Categorical (8 cols):</b> mode imputation + one-hot "
            "encoding with <code>handle_unknown='ignore'</code> so that "
            "unseen category levels at inference time degrade gracefully.",
            "Wrapping the model in the same Pipeline means a single "
            "<code>joblib.dump</code> ships every preprocessing step "
            "alongside the trained estimator — no risk of train/serve skew.",
        ]),
    ]


# --------------------------------------------------- 6. modeling
def section_modeling(styles, helpers, metrics, metrics_table_fn):
    sel = metrics["selected_model"].replace("_", " ").title()
    cv = metrics["cv_metrics"]
    return [
        _p(helpers, styles, "6. Model Development & Evaluation", "h1"),
        _p(helpers, styles,
           "Two estimators were tuned with <code>GridSearchCV</code> "
           "(stratified 5-fold CV, scored by ROC-AUC, parallelised across "
           "all cores). Logistic Regression explored "
           "<code>C ∈ {0.01, 0.1, 1, 10}</code> with both L1 and L2 "
           "penalties; Random Forest explored "
           "<code>n_estimators ∈ {100, 200}</code>, "
           "<code>max_depth ∈ {None, 5, 10}</code> and "
           "<code>min_samples_split ∈ {2, 5}</code>."),
        _p(helpers, styles, "Cross-validation summary (selected model)",
           "h2"),
        *_bullets(helpers, styles, [
            f"CV Accuracy: {cv['cv_accuracy_mean']:.4f} ± "
            f"{cv['cv_accuracy_std']:.4f}",
            f"CV Precision: {cv['cv_precision_mean']:.4f} ± "
            f"{cv['cv_precision_std']:.4f}",
            f"CV Recall: {cv['cv_recall_mean']:.4f} ± "
            f"{cv['cv_recall_std']:.4f}",
            f"CV F1: {cv['cv_f1_mean']:.4f} ± {cv['cv_f1_std']:.4f}",
            f"CV ROC-AUC: {cv['cv_roc_auc_mean']:.4f} ± "
            f"{cv['cv_roc_auc_std']:.4f}",
        ]),
        _p(helpers, styles, "Held-out test results", "h2"),
        *metrics_table_fn(),
        _spacer(helpers, 6),
        _p(helpers, styles,
           f"Although Random Forest scored marginally higher on the "
           f"held-out test set, <b>{sel}</b> was selected because it "
           f"achieved the higher <b>mean CV ROC-AUC</b> and offers better "
           f"interpretability for clinical users — coefficient signs map "
           f"directly to risk direction. The two models are within ~1% on "
           f"every metric, so this choice is more about deployability than "
           f"raw performance."),
    ]


# --------------------------------------------------- 7. mlflow
def section_mlflow(styles, helpers):
    return [
        _p(helpers, styles, "7. Experiment Tracking with MLflow", "h1"),
        _p(helpers, styles,
           "Every training run is logged to a local MLflow tracking store "
           "under <code>mlruns/</code>. For each candidate model we record:"),
        *_bullets(helpers, styles, [
            "<b>Parameters:</b> all best hyperparameters returned by "
            "GridSearchCV plus the model type tag.",
            "<b>Metrics:</b> the five CV scores (mean & std) and the five "
            "test scores, giving 15 numeric columns per run.",
            "<b>Artefacts:</b> the confusion matrix and ROC-curve PNGs are "
            "uploaded under the <code>figures/</code> folder.",
            "<b>Model:</b> <code>mlflow.sklearn.log_model</code> serialises "
            "the full Pipeline together with an inferred signature and a "
            "two-row input example, ready for downstream registry/serving.",
        ]),
        _p(helpers, styles,
           "Runs can be inspected by launching <code>mlflow ui</code> from "
           "the project root, which renders the heart-disease experiment "
           "with a side-by-side run comparison view."),
    ]


# --------------------------------------------------- 8. CI/CD
def section_ci_cd(styles, helpers):
    return [
        _p(helpers, styles, "8. CI/CD with GitHub Actions", "h1"),
        _p(helpers, styles,
           "The workflow at <code>.github/workflows/ci.yml</code> runs on "
           "every push and pull request. It is split into two jobs:"),
        *_bullets(helpers, styles, [
            "<b>lint-test-train:</b> sets up Python 3.11, installs "
            "<code>requirements.txt</code>, lints with <code>ruff</code>, "
            "runs the full <code>pytest</code> suite (35 tests), and "
            "executes a smoke training run that uploads "
            "<code>models/model.pkl</code> and <code>reports/metrics.json</code> "
            "as build artefacts.",
            "<b>docker-build:</b> depends on the first job; it uses "
            "<code>docker/build-push-action</code> to validate that the "
            "multi-stage Dockerfile builds end-to-end without actually "
            "pushing to a registry.",
        ]),
        _p(helpers, styles, "Test pyramid", "h2"),
        *_bullets(helpers, styles, [
            "<b>Data tests (8):</b> dataset shape, schema, target binarisation, "
            "no NaNs after preprocessing.",
            "<b>Pipeline tests (8):</b> ColumnTransformer correctness, "
            "fit-transform invariants, persistence round-trip.",
            "<b>Model tests (8):</b> minimum-AUC threshold, calibration "
            "sanity, deterministic seeding.",
            "<b>API tests (11):</b> health/predict happy paths, validation "
            "errors, request-id propagation, JSON-log structure, custom "
            "Prometheus metric exposure.",
        ]),
    ]



# --------------------------------------------------- 9. containerization
def section_containerization(styles, helpers):
    return [
        _p(helpers, styles, "9. Containerisation", "h1"),
        _p(helpers, styles,
           "The API is served by FastAPI + Uvicorn, with three endpoints:"),
        *_bullets(helpers, styles, [
            "<b>GET /health</b> — liveness/readiness probe; returns "
            "<code>model_loaded</code> flag and the active model version.",
            "<b>POST /predict</b> — single-patient prediction. Pydantic "
            "schemas validate every field against UCI value ranges; the "
            "response carries <code>prediction</code>, <code>label</code>, "
            "<code>probability</code>, <code>confidence</code> and "
            "<code>model_version</code>.",
            "<b>GET /metrics</b> — Prometheus exposition (covered in §11).",
        ]),
        _p(helpers, styles, "Dockerfile", "h2"),
        _p(helpers, styles,
           "A multi-stage build at <code>docker/Dockerfile</code> keeps "
           "the runtime image minimal:"),
        *_bullets(helpers, styles, [
            "<b>builder stage:</b> installs deps with <code>pip --user</code>, "
            "downloads the dataset and runs <code>python -m src.train</code> "
            "so the trained model is baked into the image.",
            "<b>runtime stage:</b> copies only the user site-packages, "
            "<code>src/</code>, <code>api/</code> and the trained "
            "<code>models/model.pkl</code>. Runs as a non-root "
            "<code>appuser</code>, exposes port 8000, and ships a "
            "container <code>HEALTHCHECK</code> that polls "
            "<code>/health</code>.",
            "Build / run: <code>docker build -t heart-disease-api:latest "
            "-f docker/Dockerfile .</code> then "
            "<code>docker run --rm -p 8000:8000 heart-disease-api:latest</code>.",
        ]),
    ]


# --------------------------------------------------- 10. k8s
def section_kubernetes(styles, helpers):
    return [
        _p(helpers, styles, "10. Kubernetes Deployment", "h1"),
        _p(helpers, styles,
           "Manifests under <code>k8s/</code> are bundled by a single "
           "<code>kustomization.yaml</code> and apply with "
           "<code>kubectl apply -k k8s/</code>:"),
        *_bullets(helpers, styles, [
            "<b>Namespace</b> <i>heart-disease</i> isolates all resources.",
            "<b>ConfigMap</b> supplies <code>MODEL_PATH</code>, "
            "<code>MODEL_VERSION</code>, <code>PORT</code>, "
            "<code>LOG_LEVEL</code> as env vars.",
            "<b>Deployment:</b> 2 replicas, rolling update "
            "(<code>maxSurge=1, maxUnavailable=0</code>), CPU "
            "request 100 m / limit 500 m, memory 256–512 Mi, runs as "
            "non-root with <code>readOnlyRootFilesystem</code> and dropped "
            "Linux capabilities. Liveness + readiness probes hit "
            "<code>/health</code>.",
            "<b>Services:</b> a <i>LoadBalancer</i> for external traffic "
            "and a <i>ClusterIP</i> companion used by Prometheus to scrape "
            "<code>/metrics</code> from inside the cluster.",
            "<b>Ingress</b> (optional, NGINX): hostname "
            "<i>heart-disease.local</i>, path <code>/</code>.",
            "<b>HorizontalPodAutoscaler:</b> 2–5 replicas, scaling "
            "targets 70 % CPU and 80 % memory utilisation.",
        ]),
        _p(helpers, styles,
           "On Minikube the workflow is: <code>minikube start</code> → "
           "<code>minikube docker-env | iex</code> (so the local image is "
           "visible inside the cluster) → <code>docker build</code> → "
           "<code>kubectl apply -k k8s/</code> → "
           "<code>minikube service heart-disease-api -n heart-disease "
           "--url</code>. Detailed step-by-step instructions live in "
           "<code>k8s/README.md</code>."),
    ]


# --------------------------------------------------- 11. monitoring
def section_monitoring(styles, helpers):
    return [
        _p(helpers, styles, "11. Monitoring & Logging", "h1"),
        _p(helpers, styles, "Structured logging", "h2"),
        _p(helpers, styles,
           "<code>api/logging_config.py</code> installs a single-line JSON "
           "formatter on the root logger. Every record carries "
           "<code>ts, level, logger, service, version, message</code> plus "
           "any extra fields supplied via <code>logger.info(..., "
           "extra={...})</code>. Each request is annotated with a "
           "<code>request_id</code> (generated or echoed from the "
           "<i>X-Request-ID</i> header) and emits at minimum an "
           "<code>http_request</code> event; <code>/predict</code> emits an "
           "additional <code>prediction</code> event including the "
           "predicted class, probability, latency and model version. "
           "Setting <code>LOG_FORMAT=plain</code> reverts to a "
           "human-readable formatter for local debugging."),
        _p(helpers, styles, "Prometheus metrics", "h2"),
        _p(helpers, styles,
           "Standard HTTP metrics are exposed via "
           "<code>prometheus-fastapi-instrumentator</code>. On top of those "
           "we register four custom series in <code>api/metrics.py</code>:"),
        *_bullets(helpers, styles, [
            "<code>heart_predictions_total{label, model_version}</code> — "
            "throughput by predicted class (drift / class-balance signal).",
            "<code>heart_prediction_probability{model_version}</code> — "
            "histogram of P(disease) scores.",
            "<code>heart_prediction_latency_seconds{model_version}</code> — "
            "model-inference latency (SLO source).",
            "<code>heart_prediction_errors_total{model_version, "
            "error_type}</code> — failed inferences by exception class.",
        ]),
        _p(helpers, styles, "Local observability stack", "h2"),
        _p(helpers, styles,
           "<code>monitoring/docker-compose.yml</code> brings up the API, "
           "Prometheus and Grafana together. Prometheus is pre-configured "
           "to scrape <code>api:8000/metrics</code> every 15 s. Grafana is "
           "auto-provisioned with the Prometheus datasource and a "
           "10-panel dashboard "
           "(<code>monitoring/grafana/dashboards/heart-disease-api.json</code>) "
           "covering throughput, latency percentiles, class balance, "
           "score-distribution heatmap and error rates."),
    ]


# --------------------------------------------------- 13. conclusion
def section_conclusion(styles, helpers, metrics):
    test = metrics["test_metrics"]
    return [
        _p(helpers, styles, "13. Conclusion & Future Work", "h1"),
        _p(helpers, styles,
           "The pipeline meets every functional requirement of the "
           "assignment brief and ships with the operational guard-rails "
           "expected of a production system: deterministic preprocessing, "
           "tracked experiments, automated tests on every push, a slim "
           "non-root container, declarative Kubernetes deployment with "
           "autoscaling, and a working metrics + logging stack. On the "
           f"held-out test set the selected model achieves "
           f"ROC-AUC = {test['test_roc_auc']:.3f}, F1 = {test['test_f1']:.3f}, "
           f"recall = {test['test_recall']:.3f} — clinically reasonable "
           "given the dataset size."),
        _p(helpers, styles,
           "<i>Note on figures:</i> Figures 3-5 (MLflow, Swagger, Grafana) "
           "are programmatically rendered reproductions of the respective "
           "UIs, produced by <code>scripts/generate_screenshots.py</code> "
           "from real metrics in <code>reports/metrics.json</code> and the "
           "panel layout in "
           "<code>monitoring/grafana/dashboards/heart-disease-api.json</code>. "
           "Figures 6-12 in §12 are <b>genuine screenshots</b> captured "
           "from the live two-node Kubernetes cluster on Killercoda."),
        _p(helpers, styles, "Suggested next steps", "h2"),
        *_bullets(helpers, styles, [
            "<b>Model registry:</b> promote the MLflow model to a "
            "<i>Production</i> stage and have the API resolve the URI at "
            "boot instead of loading a local pickle.",
            "<b>Drift detection:</b> wire <i>evidently</i> or "
            "<i>WhyLabs</i> into a sidecar that consumes the JSON logs "
            "and alerts on input/score drift.",
            "<b>Calibration & explainability:</b> add Platt scaling / "
            "SHAP summary plots so clinicians get probability estimates "
            "they can trust and feature-level reasons for each prediction.",
            "<b>GitOps rollout:</b> manage the K8s manifests with ArgoCD "
            "or FluxCD so production state is reconciled from Git.",
        ]),
    ]


# --------------------------------------------------- 14. architecture
def section_architecture(styles, helpers):
    return [
        _p(helpers, styles, "14. End-to-End Architecture", "h1"),
        _p(helpers, styles,
           "The diagram below summarises how the components fit together "
           "across the development, CI, and runtime planes."),
        _p(helpers, styles,
           "<font face='Courier' size='8.5'>"
           "┌────────────────────── DEVELOPMENT ──────────────────────┐<br/>"
           "│  notebooks/01_eda.ipynb  ──► src/{data_loader,           │<br/>"
           "│                                preprocess, train,        │<br/>"
           "│                                evaluate, predict}.py     │<br/>"
           "│        │                              │                  │<br/>"
           "│        ▼                              ▼                  │<br/>"
           "│  reports/figures/*.png        models/model.pkl  +        │<br/>"
           "│                               mlruns/* (MLflow store)    │<br/>"
           "└──────────────────┬──────────────────────────────────────┘<br/>"
           "                   │  git push<br/>"
           "                   ▼<br/>"
           "┌──────────────────────── CI / CD ─────────────────────────┐<br/>"
           "│  GitHub Actions  ──► ruff lint ──► pytest (35) ──►       │<br/>"
           "│   smoke train    ──► docker build (multi-stage)          │<br/>"
           "│  artefacts: model.pkl + metrics.json + image             │<br/>"
           "└──────────────────┬──────────────────────────────────────┘<br/>"
           "                   │  docker push (out of scope)<br/>"
           "                   ▼<br/>"
           "┌──────────────────────── RUNTIME ─────────────────────────┐<br/>"
           "│              ┌─────────── Kubernetes ──────────┐         │<br/>"
           "│  client ──► │ LB Service ──► Pod (FastAPI)     │         │<br/>"
           "│              │                 │ /metrics      │         │<br/>"
           "│              │  HPA(2-5)       ▼               │         │<br/>"
           "│              │            Prometheus ──► Grafana│        │<br/>"
           "│              └──────────────────────────────────┘        │<br/>"
           "│   stdout JSON logs ──► (Loki / ELK / CloudWatch in prod) │<br/>"
           "└──────────────────────────────────────────────────────────┘"
           "</font>", "body"),
        _p(helpers, styles,
           "Each plane is independently testable: the development plane "
           "via the notebook + <code>pytest</code>, the CI plane via the "
           "GitHub Actions workflow, and the runtime plane via the local "
           "<code>docker compose</code> stack that mirrors the in-cluster "
           "topology. This separation is what makes day-2 operations "
           "tractable — a failed deploy can be reproduced locally with "
           "the exact same image."),
    ]


# --------------------------------------------------- 14. appendix
def section_appendix(styles, helpers):
    return [
        _p(helpers, styles, "Appendix A — Reproducing the Results", "h1"),
        _p(helpers, styles,
           "<font face='Courier' size='9'>"
           "# 1. Clone and create a virtual env<br/>"
           "git clone &lt;repo-url&gt;<br/>"
           "cd MLOps/Assignment1<br/>"
           "python -m venv .venv && .venv\\Scripts\\Activate.ps1<br/>"
           "pip install -r requirements.txt<br/>"
           "<br/>"
           "# 2. Download data + train<br/>"
           "python scripts/download_data.py<br/>"
           "python -m src.train<br/>"
           "<br/>"
           "# 3. Lint + tests<br/>"
           "python -m ruff check src api tests scripts<br/>"
           "python -m pytest -q<br/>"
           "<br/>"
           "# 4. Run the API locally<br/>"
           "uvicorn api.app:app --reload<br/>"
           "<br/>"
           "# 5. Build &amp; run the container<br/>"
           "docker build -t heart-disease-api:latest -f docker/Dockerfile .<br/>"
           "docker run --rm -p 8000:8000 heart-disease-api:latest<br/>"
           "<br/>"
           "# 6. Local observability stack<br/>"
           "docker compose -f monitoring/docker-compose.yml up --build<br/>"
           "<br/>"
           "# 7. Deploy to Kubernetes<br/>"
           "kubectl apply -k k8s/<br/>"
           "kubectl rollout status deployment/heart-disease-api -n heart-disease<br/>"
           "<br/>"
           "# 8. Re-generate this PDF report<br/>"
           "python scripts/generate_report.py"
           "</font>", "body"),
        _p(helpers, styles, "Appendix B — Sample API Interaction", "h2"),
        _p(helpers, styles,
           "<font face='Courier' size='9'>"
           "$ curl -X POST http://localhost:8000/predict \\<br/>"
           "&nbsp;&nbsp;&nbsp;&nbsp;-H 'Content-Type: application/json' \\<br/>"
           "&nbsp;&nbsp;&nbsp;&nbsp;-d '{\"age\":63,\"sex\":1,\"cp\":1,\"trestbps\":145,<br/>"
           "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\"chol\":233,\"fbs\":1,\"restecg\":2,\"thalach\":150,<br/>"
           "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\"exang\":0,\"oldpeak\":2.3,\"slope\":3,\"ca\":0,\"thal\":6}'<br/>"
           "<br/>"
           "{<br/>"
           "&nbsp;&nbsp;\"prediction\": 0,<br/>"
           "&nbsp;&nbsp;\"label\": \"No disease\",<br/>"
           "&nbsp;&nbsp;\"probability\": 0.1718,<br/>"
           "&nbsp;&nbsp;\"confidence\": 0.8282,<br/>"
           "&nbsp;&nbsp;\"model_version\": \"v1.0.0\"<br/>"
           "}"
           "</font>", "body"),
        _p(helpers, styles, "Appendix C — References", "h2"),
        *_bullets(helpers, styles, [
            "UCI Heart Disease dataset — "
            "https://archive.ics.uci.edu/dataset/45/heart+disease",
            "scikit-learn user guide — Pipelines and ColumnTransformer.",
            "MLflow tracking documentation — "
            "https://mlflow.org/docs/latest/tracking.html",
            "Prometheus best-practice metric naming — "
            "https://prometheus.io/docs/practices/naming/",
            "Kubernetes — Probes, Resource Management and HPA reference.",
        ]),
    ]


# --------------------------------------- production deployment (Killercoda)
def section_production_deployment_intro(styles, helpers):
    return [
        _p(helpers, styles,
           "12. Production Deployment Verification (Killercoda)", "h1"),
        _p(helpers, styles,
           "The full stack was deployed to a live two-node Kubernetes "
           "cluster on Killercoda's playground environment to verify the "
           "manifests end-to-end outside the local Minikube loop. The "
           "image was built with the slim multi-stage Dockerfile "
           "(<code>docker/Dockerfile.slim</code>) and side-loaded into "
           "containerd on both nodes via "
           "<code>ctr -n=k8s.io images import</code>. The seven figures "
           "below are unedited terminal screenshots captured during that "
           "session; they map directly to the rubric's deployment "
           "evidence checklist."),
        _p(helpers, styles, "Verification checklist", "h2"),
        *_bullets(helpers, styles, [
            "<b>Cluster context (Fig 6):</b> two <i>Ready</i> nodes "
            "(<code>controlplane</code> + <code>node01</code>) and the "
            "active kubeconfig context.",
            "<b>Image build (Fig 7):</b> <code>heart-disease-api:latest</code> "
            "produced by the multi-stage build, ~400 MB.",
            "<b>Pods running (Fig 8):</b> both replicas <code>1/1 Running</code>, "
            "scheduled across the two nodes (proves multi-node scheduling "
            "and image availability on every worker).",
            "<b>Deployment description (Fig 9):</b> "
            "<code>RollingUpdate</code> strategy, "
            "<code>2 desired / 2 updated / 2 available</code>, liveness "
            "and readiness probes wired to <code>/health</code>.",
            "<b>NodePort service (Fig 10):</b> external "
            "<code>NodePort</code> exposed for traffic ingress; companion "
            "<i>ClusterIP</i> service used by Prometheus for in-cluster "
            "scraping.",
            "<b>HorizontalPodAutoscaler (Fig 11):</b> active HPA "
            "(2-5 replicas, 70 % CPU target) with metrics-server reporting "
            "real utilisation values rather than <code>&lt;unknown&gt;</code>.",
            "<b>Live API call (Fig 12):</b> "
            "<code>curl /health</code> returns "
            "<code>{\"status\":\"ok\"}</code> and "
            "<code>curl -X POST /predict</code> returns a valid JSON "
            "prediction with <code>label</code>, <code>probability</code> "
            "and <code>model_version</code>.",
        ]),
    ]


# --------------------------------------- deliverables index
REPO_URL = "https://github.com/2025cs05050-rgb/heart-disease-mlops"
BLOB = f"{REPO_URL}/blob/main"
TREE = f"{REPO_URL}/tree/main"


def _link(url: str, label: str | None = None) -> str:
    """ReportLab-friendly hyperlink. Renders blue underlined in PDF; the
    docx parser strips the wrapper and keeps the label visible (also
    coloured blue via the ``link`` flag)."""
    return (f"<link href=\"{url}\"><font color=\"#1f4e79\">"
            f"<u>{label or url}</u></font></link>")


def _nb(stem: str, pdf: str) -> str:
    """Build one notebook deliverable bullet with two clickable links."""
    return (f"<code>{stem}</code> &nbsp;·&nbsp; "
            f"notebook: {_link(f'{BLOB}/notebooks/{stem}')} &nbsp;·&nbsp; "
            f"PDF: {_link(f'{BLOB}/reports/{pdf}')}")


def section_deliverables(styles, helpers):
    return [
        _p(helpers, styles, "Appendix D — Deliverables Index", "h1"),
        _p(helpers, styles,
           "All artefacts referenced in this report are tracked in the "
           "public GitHub repository. Every entry below carries a "
           "clickable link to the exact file on the <i>main</i> branch."),
        _p(helpers, styles,
           f"<b>Repository:</b> {_link(REPO_URL)}"),
        _p(helpers, styles, "Notebooks (with exported PDFs)", "h2"),
        *_bullets(helpers, styles, [
            _nb("01_eda.ipynb", "01_Exploratory_Data_Analysis.pdf"),
            _nb("02_preprocessing.ipynb", "02_Preprocessing.pdf"),
            _nb("03_model_training.ipynb", "03_Model_Training.pdf"),
            _nb("04_inference.ipynb", "04_Inference.pdf"),
            _nb("05_containerisation.ipynb", "05_Containerisation.pdf"),
            _nb("06_kubernetes.ipynb", "06_Kubernetes.pdf"),
            _nb("07_monitoring.ipynb", "07_Monitoring.pdf"),
        ]),
        _p(helpers, styles, "Source code", "h2"),
        *_bullets(helpers, styles, [
            f"{_link(f'{BLOB}/src/data_loader.py', 'src/data_loader.py')}, "
            f"{_link(f'{BLOB}/src/preprocess.py', 'src/preprocess.py')}, "
            f"{_link(f'{BLOB}/src/train.py', 'src/train.py')}, "
            f"{_link(f'{BLOB}/src/evaluate.py', 'src/evaluate.py')}, "
            f"{_link(f'{BLOB}/src/predict.py', 'src/predict.py')} — "
            "pipeline modules.",
            f"{_link(f'{TREE}/api', 'api/')} — FastAPI service "
            "(app, schemas, logging, metrics).",
            f"{_link(f'{TREE}/tests', 'tests/')} — pytest suite "
            "(35 tests).",
        ]),
        _p(helpers, styles, "Infrastructure", "h2"),
        *_bullets(helpers, styles, [
            f"{_link(f'{BLOB}/docker/Dockerfile', 'docker/Dockerfile')} — "
            "multi-stage build that bakes the trained model.",
            f"{_link(f'{BLOB}/docker/Dockerfile.slim', 'docker/Dockerfile.slim')} — "
            "runtime-only variant used for low-IO sandboxes (Killercoda).",
            f"{_link(f'{TREE}/k8s', 'k8s/')} — namespace, configmap, "
            "deployment, service, ingress, HPA, kustomization.",
            f"{_link(f'{TREE}/monitoring', 'monitoring/')} — "
            "docker-compose, Prometheus config, Grafana dashboard JSON.",
            f"{_link(f'{BLOB}/.github/workflows/ci.yml', '.github/workflows/ci.yml')} — "
            "lint, test, smoke-train and Docker-build jobs.",
        ]),
        _p(helpers, styles, "Reports", "h2"),
        *_bullets(helpers, styles, [
            f"{_link(f'{BLOB}/reports/MLOps_Assignment1_Report.pdf', 'reports/MLOps_Assignment1_Report.pdf')} — "
            "this consolidated report (PDF).",
            f"{_link(f'{BLOB}/reports/MLOps_Assignment1_Report.docx', 'reports/MLOps_Assignment1_Report.docx')} — "
            "Word version of this report.",
            f"{_link(f'{BLOB}/reports/metrics.json', 'reports/metrics.json')} — "
            "CV + test metrics for both candidate models, consumed by "
            "the report generators.",
            f"{_link(f'{TREE}/reports/figures', 'reports/figures/')} — "
            "confusion matrix, ROC curve, MLflow / Swagger / Grafana "
            "renderings.",
            f"{_link(f'{TREE}/screenshots', 'screenshots/')} — seven "
            "Killercoda deployment screenshots embedded as Figures 6-12.",
        ]),
    ]
