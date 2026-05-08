"""MLflow + Swagger UI mockup renderers (split out to keep file sizes small)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


def mlflow_runs(out: Path, metrics: dict) -> None:
    """MLflow-style side-by-side run comparison using real metrics."""
    bg = "#ffffff"
    header = "#0194e2"
    row_alt = "#f5f7fa"
    text = "#202124"

    fig = plt.figure(figsize=(12, 6.5), facecolor=bg)
    fig.suptitle("MLflow · Experiments · heart-disease",
                 color=text, fontsize=12, fontweight="bold", y=0.97, x=0.05,
                 ha="left")

    ax = fig.add_axes([0.04, 0.06, 0.92, 0.84])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Toolbar mock
    ax.add_patch(Rectangle((0, 9.2), 10, 0.5, facecolor="#eef1f5",
                           edgecolor="none"))
    ax.text(0.1, 9.45, "Compare runs · 2 selected", color=text, fontsize=8.5,
            fontweight="bold")
    ax.text(9.95, 9.45, "Sort by: cv_roc_auc_mean ▼", color="#666",
            fontsize=8, ha="right")

    # Header
    cols = ["Run name", "Status", "C / n_estimators", "penalty / depth",
            "cv_roc_auc", "test_accuracy", "test_f1", "test_roc_auc"]
    col_x = [0.15, 1.85, 2.85, 4.30, 5.55, 6.65, 7.70, 8.75]

    ax.add_patch(Rectangle((0, 8.5), 10, 0.55, facecolor=header,
                           edgecolor="none"))
    for x, c in zip(col_x, cols, strict=True):
        ax.text(x, 8.78, c, color="white", fontsize=8, fontweight="bold")

    # Two real rows from metrics.json
    by_name = {r["name"]: r for r in metrics["all_results"]}
    lr = by_name["logistic_regression"]
    rf = by_name["random_forest"]
    rows = [
        ("logistic_regression", "FINISHED", "1.0", "l1",
         lr["cv_metrics"]["cv_roc_auc_mean"],
         lr["test_metrics"]["test_accuracy"],
         lr["test_metrics"]["test_f1"],
         lr["test_metrics"]["test_roc_auc"]),
        ("random_forest", "FINISHED", "100",
         f"depth={rf['best_params']['model__max_depth']}",
         rf["cv_metrics"]["cv_roc_auc_mean"],
         rf["test_metrics"]["test_accuracy"],
         rf["test_metrics"]["test_f1"],
         rf["test_metrics"]["test_roc_auc"]),
    ]
    y = 7.85
    for i, row in enumerate(rows):
        if i % 2 == 0:
            ax.add_patch(Rectangle((0, y - 0.25), 10, 0.55,
                                   facecolor=row_alt, edgecolor="none"))
        for x, val in zip(col_x, row, strict=True):
            if isinstance(val, float):
                ax.text(x, y, f"{val:.4f}", color=text, fontsize=8.5)
            elif val == "FINISHED":
                ax.add_patch(FancyBboxPatch((x, y - 0.13), 0.85, 0.32,
                                            boxstyle="round,pad=0.02",
                                            facecolor="#dff5d5",
                                            edgecolor="#56a64b", linewidth=0.8))
                ax.text(x + 0.42, y, val, color="#2e7d32", fontsize=7,
                        ha="center", va="center", fontweight="bold")
            else:
                ax.text(x, y, str(val), color=text, fontsize=8.5)
        y -= 0.7

    # Parallel coordinate-ish bars for the two ROC-AUC values
    ax.text(0.15, 6.0, "Test ROC-AUC", color=text, fontsize=10,
            fontweight="bold")
    bar_y = [5.2, 4.3]
    bar_vals = [lr["test_metrics"]["test_roc_auc"],
                rf["test_metrics"]["test_roc_auc"]]
    for y, name, v, col in zip(bar_y,
                               ["logistic_regression", "random_forest"],
                               bar_vals,
                               ["#0194e2", "#ff9800"], strict=True):
        ax.text(0.15, y + 0.12, name, color=text, fontsize=8.5)
        ax.add_patch(Rectangle((2.5, y - 0.05), 6 * v, 0.35,
                               facecolor=col, edgecolor="none"))
        ax.text(2.5 + 6 * v + 0.1, y + 0.12, f"{v:.4f}",
                color=text, fontsize=8.5, va="center")

    # Caption
    ax.text(0.15, 3.2, "Metrics, hyperparameters, the two PNG artefacts "
            "(confusion matrix + ROC) and the\nserialised sklearn pipeline "
            "are logged on every run by src/train.py.", color="#555",
            fontsize=8.5)

    fig.savefig(out, dpi=140, facecolor=bg, bbox_inches="tight")
    plt.close(fig)


def swagger_ui(out: Path) -> None:
    """FastAPI / Swagger-UI style screenshot of the API endpoints."""
    bg = "#fafafa"
    text = "#3b4151"

    fig = plt.figure(figsize=(11, 7.5), facecolor=bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Top bar
    ax.add_patch(Rectangle((0, 9.4), 10, 0.6, facecolor="#1b1b1b",
                           edgecolor="none"))
    ax.text(0.2, 9.7, "Swagger UI", color="#89bf04", fontsize=11,
            fontweight="bold", va="center")
    ax.text(9.8, 9.7, "/openapi.json", color="#bbb", fontsize=8,
            ha="right", va="center")

    # Title
    ax.text(0.2, 8.85, "Heart Disease Risk API", color=text, fontsize=18,
            fontweight="bold")
    ax.text(0.2, 8.45, "v1.0.0   OAS 3.1", color="#888", fontsize=8.5)
    ax.text(0.2, 8.10, "Predicts heart-disease risk from patient health "
            "features.", color=text, fontsize=9.5)

    # Endpoints
    endpoints = [
        ("GET", "#61affe", "/", "Service banner"),
        ("GET", "#61affe", "/health", "Liveness/readiness (model loaded?)"),
        ("POST", "#49cc90", "/predict", "Single-patient prediction"),
        ("GET", "#61affe", "/metrics", "Prometheus exposition"),
        ("GET", "#61affe", "/docs", "Interactive Swagger UI"),
    ]
    y = 7.30
    for method, col, path, desc in endpoints:
        ax.add_patch(Rectangle((0.2, y - 0.32), 9.6, 0.55,
                               facecolor="#ffffff",
                               edgecolor=col, linewidth=1.2))
        ax.add_patch(Rectangle((0.3, y - 0.23), 0.85, 0.38,
                               facecolor=col, edgecolor="none"))
        ax.text(0.72, y - 0.04, method, color="white", fontsize=9,
                fontweight="bold", ha="center", va="center")
        ax.text(1.30, y - 0.04, path, color=text, fontsize=10,
                fontweight="bold", va="center")
        ax.text(4.00, y - 0.04, desc, color="#555", fontsize=9, va="center")
        y -= 0.75

    fig.savefig(out, dpi=140, facecolor=bg, bbox_inches="tight")
    plt.close(fig)
