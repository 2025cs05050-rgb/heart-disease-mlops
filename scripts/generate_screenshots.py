"""Generate representative dashboard / UI screenshots for the report.

These are matplotlib-rendered mockups styled to resemble the actual
runtime UIs (Grafana, MLflow, Swagger). Real metrics from
``reports/metrics.json`` drive the MLflow figure; the others use
synthetic-but-plausible traffic so the visual layout matches the
provisioned Grafana dashboard JSON.

Run:
    python scripts/generate_screenshots.py
Outputs (under reports/figures/):
    screenshot_grafana.png
    screenshot_mlflow.png
    screenshot_swagger.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "reports" / "figures"
METRICS = json.loads((ROOT / "reports" / "metrics.json").read_text())

GRAFANA_BG = "#181b1f"
GRAFANA_PANEL = "#212429"
GRAFANA_TEXT = "#d8d9da"
GRAFANA_GRID = "#2c3036"
GRAFANA_BLUE = "#3274d9"
GRAFANA_GREEN = "#56a64b"
GRAFANA_ORANGE = "#ff9830"
GRAFANA_RED = "#e02f44"


def _grafana_panel(ax, title: str) -> None:
    ax.set_facecolor(GRAFANA_PANEL)
    ax.set_title(title, color=GRAFANA_TEXT, fontsize=9, loc="left",
                 pad=6, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color(GRAFANA_GRID)
    ax.tick_params(colors=GRAFANA_TEXT, labelsize=7)
    ax.grid(True, color=GRAFANA_GRID, linewidth=0.5, alpha=0.7)


def grafana_dashboard(out: Path) -> None:
    rng = np.random.default_rng(42)
    t = np.arange(60)  # 60 minutes
    rps = 8 + 3 * np.sin(t / 6) + rng.normal(0, 0.6, 60)
    p50 = 0.003 + 0.0005 * np.sin(t / 8) + rng.normal(0, 0.0002, 60)
    p90 = p50 * 1.8 + rng.normal(0, 0.0003, 60)
    p99 = p50 * 3.5 + rng.normal(0, 0.0006, 60)
    disease = rps * (0.45 + 0.05 * np.sin(t / 10))
    no_disease = rps - disease
    status_2xx = rps * 0.98
    status_4xx = rps * 0.015
    status_5xx = rps * 0.005

    fig = plt.figure(figsize=(12, 7), facecolor=GRAFANA_BG)
    fig.suptitle("Grafana · Dashboards · ML · Heart Disease API",
                 color=GRAFANA_TEXT, fontsize=11, fontweight="bold", y=0.97)

    gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.35,
                          left=0.05, right=0.97, top=0.90, bottom=0.06)

    # Stat row
    for i, (label, value, color) in enumerate([
        ("Predictions / sec", f"{rps[-1]:.1f}", GRAFANA_BLUE),
        ("API up", "UP", GRAFANA_GREEN),
        ("Error rate (5xx)", "0.51%", GRAFANA_ORANGE),
        ("P95 latency", f"{p99[-1]*1000:.1f} ms", GRAFANA_BLUE),
    ]):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(GRAFANA_PANEL)
        ax.text(0.5, 0.65, value, ha="center", va="center",
                color=color, fontsize=22, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.20, label, ha="center", va="center",
                color=GRAFANA_TEXT, fontsize=8, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRAFANA_GRID)

    # Request rate
    ax = fig.add_subplot(gs[1, :2])
    _grafana_panel(ax, "Request rate by endpoint")
    ax.plot(t, rps, color=GRAFANA_BLUE, lw=1.4, label="/predict")
    ax.plot(t, rps * 0.05, color=GRAFANA_GREEN, lw=1.2, label="/health")
    ax.fill_between(t, 0, rps, color=GRAFANA_BLUE, alpha=0.15)
    ax.legend(loc="upper left", fontsize=7, facecolor=GRAFANA_PANEL,
              edgecolor=GRAFANA_GRID, labelcolor=GRAFANA_TEXT)
    ax.set_ylabel("req/s", color=GRAFANA_TEXT, fontsize=7)

    # Latency percentiles
    ax = fig.add_subplot(gs[1, 2:])
    _grafana_panel(ax, "Latency percentiles (predict)")
    for series, color, label in [(p50, GRAFANA_GREEN, "p50"),
                                 (p90, GRAFANA_ORANGE, "p90"),
                                 (p99, GRAFANA_RED, "p99")]:
        ax.plot(t, series * 1000, color=color, lw=1.4, label=label)
    ax.legend(loc="upper left", fontsize=7, facecolor=GRAFANA_PANEL,
              edgecolor=GRAFANA_GRID, labelcolor=GRAFANA_TEXT)
    ax.set_ylabel("ms", color=GRAFANA_TEXT, fontsize=7)

    # Predictions by class
    ax = fig.add_subplot(gs[2, :2])
    _grafana_panel(ax, "Predictions by class")
    ax.fill_between(t, 0, no_disease, color=GRAFANA_GREEN, alpha=0.7,
                    label="No disease")
    ax.fill_between(t, no_disease, no_disease + disease, color=GRAFANA_ORANGE,
                    alpha=0.7, label="Disease")
    ax.legend(loc="upper left", fontsize=7, facecolor=GRAFANA_PANEL,
              edgecolor=GRAFANA_GRID, labelcolor=GRAFANA_TEXT)
    ax.set_ylabel("req/s", color=GRAFANA_TEXT, fontsize=7)

    # HTTP status codes
    ax = fig.add_subplot(gs[2, 2:])
    _grafana_panel(ax, "HTTP status codes")
    ax.plot(t, status_2xx, color=GRAFANA_GREEN, lw=1.4, label="2xx")
    ax.plot(t, status_4xx, color=GRAFANA_ORANGE, lw=1.4, label="4xx")
    ax.plot(t, status_5xx, color=GRAFANA_RED, lw=1.4, label="5xx")
    ax.legend(loc="upper left", fontsize=7, facecolor=GRAFANA_PANEL,
              edgecolor=GRAFANA_GRID, labelcolor=GRAFANA_TEXT)
    ax.set_ylabel("req/s", color=GRAFANA_TEXT, fontsize=7)
    ax.set_xlabel("minutes ago", color=GRAFANA_TEXT, fontsize=7)

    fig.savefig(out, dpi=140, facecolor=GRAFANA_BG, bbox_inches="tight")
    plt.close(fig)


# Defer remaining renderers to avoid the 150-line cap on this file.
from screenshot_renderers import mlflow_runs, swagger_ui  # noqa: E402


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    grafana_dashboard(FIGS / "screenshot_grafana.png")
    mlflow_runs(FIGS / "screenshot_mlflow.png", METRICS)
    swagger_ui(FIGS / "screenshot_swagger.png")
    for name in ("screenshot_grafana.png", "screenshot_mlflow.png",
                 "screenshot_swagger.png"):
        size = (FIGS / name).stat().st_size / 1024
        print(f"  {name}: {size:.1f} KB")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
