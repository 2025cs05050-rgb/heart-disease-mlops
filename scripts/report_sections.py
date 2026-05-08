"""Section builders for the MLOps Assignment-1 PDF report.

Kept in a separate module so ``generate_report.py`` stays focused on
typography/layout concerns.
"""
from __future__ import annotations

from pathlib import Path

from report_content import (
    cover_block,
    section_appendix,
    section_architecture,
    section_ci_cd,
    section_conclusion,
    section_containerization,
    section_deliverables,
    section_eda,
    section_executive_summary,
    section_kubernetes,
    section_mlflow,
    section_modeling,
    section_monitoring,
    section_pipeline,
    section_problem,
    section_production_deployment_intro,
    section_repo,
)


def _metrics_table(helpers: dict, metrics: dict, styles: dict) -> list:
    """Side-by-side test metrics for both candidate models."""
    from reportlab.lib.colors import HexColor, white

    Table, TableStyle = helpers["Table"], helpers["TableStyle"]
    PRIMARY = helpers["PRIMARY"]

    rows = [["Metric", "Logistic Regression", "Random Forest"]]
    metric_keys = ("test_accuracy", "test_precision", "test_recall",
                   "test_f1", "test_roc_auc")
    by_name = {r["name"]: r for r in metrics["all_results"]}
    for key in metric_keys:
        rows.append([
            key.replace("test_", "").upper(),
            f"{by_name['logistic_regression']['test_metrics'][key]:.4f}",
            f"{by_name['random_forest']['test_metrics'][key]:.4f}",
        ])
    table = Table(rows, hAlign="LEFT", colWidths=[4.5 * helpers["cm"]] * 3)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [white, HexColor("#f4f7fb")]),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cccccc")),
    ]))
    return [table]


def _image(helpers: dict, path: Path, width_cm: float, caption: str,
           styles: dict) -> list:
    """Embed an image preserving its native aspect ratio."""
    from PIL import Image as PILImage

    Image, Spacer, Paragraph = (helpers["Image"], helpers["Spacer"],
                                helpers["Paragraph"])
    cm = helpers["cm"]
    if not path.exists():
        return [Paragraph(f"<i>[Figure missing: {path.name}]</i>",
                          styles["caption"])]
    with PILImage.open(path) as im:
        w_px, h_px = im.size
    aspect = h_px / w_px
    img = Image(str(path), width=width_cm * cm,
                height=width_cm * cm * aspect)
    return [img, Paragraph(caption, styles["caption"]), Spacer(1, 4)]


def build_sections(*, styles: dict, metrics: dict, figures_dir: Path,
                   helpers: dict,
                   screenshots_dir: Path | None = None) -> list:
    """Compose the full Platypus story (list of flowables)."""
    story: list = []

    def metrics_table_fn():
        return _metrics_table(helpers, metrics, styles)

    def image_fn(path, width_cm, caption):
        return _image(helpers, path, width_cm, caption, styles)

    # ---- Cover page ----
    story += cover_block(styles, helpers)
    story.append(helpers["PageBreak"]())

    # ---- 1. Executive summary + Problem statement ----
    story += section_executive_summary(styles, helpers, metrics)
    story += section_problem(styles, helpers)
    story.append(helpers["PageBreak"]())

    # ---- 2. Repo & EDA ----
    story += section_repo(styles, helpers)
    story += section_eda(styles, helpers)
    story.append(helpers["PageBreak"]())

    # ---- 3. Pipeline + modeling (with metrics table & figures) ----
    story += section_pipeline(styles, helpers)
    story += section_modeling(styles, helpers, metrics, metrics_table_fn)
    story += image_fn(figures_dir / "cm_random_forest.png", 9,
                      "Figure 1 — Confusion matrix on the held-out test set "
                      "(Random Forest).")
    story += image_fn(figures_dir / "roc_random_forest.png", 9,
                      "Figure 2 — ROC curve, test AUC = "
                      f"{metrics['all_results'][1]['test_metrics']['test_roc_auc']:.4f}.")
    story.append(helpers["PageBreak"]())

    # ---- 4. MLflow + CI/CD ----
    story += section_mlflow(styles, helpers)
    story += image_fn(figures_dir / "screenshot_mlflow.png", 15,
                      "Figure 3 — MLflow run-comparison view for the "
                      "<i>heart-disease</i> experiment "
                      "(rendered from logged metrics).")
    story += section_ci_cd(styles, helpers)
    story.append(helpers["PageBreak"]())

    # ---- 5. Container + K8s ----
    story += section_containerization(styles, helpers)
    story += image_fn(figures_dir / "screenshot_swagger.png", 15,
                      "Figure 4 — Swagger UI at <code>/docs</code> "
                      "showing the five service endpoints.")
    story += section_kubernetes(styles, helpers)
    story.append(helpers["PageBreak"]())

    # ---- 6. Monitoring ----
    story += section_monitoring(styles, helpers)
    story += image_fn(figures_dir / "screenshot_grafana.png", 16,
                      "Figure 5 — Pre-provisioned Grafana dashboard "
                      "<i>ML › Heart Disease API</i>: stat row, request "
                      "rate, latency percentiles, class balance and HTTP "
                      "status codes.")
    story.append(helpers["PageBreak"]())

    # ---- 7. Production deployment evidence (Killercoda screenshots) ----
    story += section_production_deployment_intro(styles, helpers)
    if screenshots_dir is not None:
        shots = [
            ("01_cluster_context.png",
             "Figure 6 — <code>kubectl get nodes</code> showing the "
             "two-node cluster (controlplane + node01) both <i>Ready</i>."),
            ("02_docker_image.png",
             "Figure 7 — <code>docker images heart-disease-api</code>: "
             "image produced by the multi-stage build."),
            ("03_pods_running.png",
             "Figure 8 — <code>kubectl -n heart-disease get pods -o wide</code>: "
             "both replicas <code>1/1 Running</code> across the two nodes."),
            ("04_deployment_describe.png",
             "Figure 9 — <code>kubectl describe deployment</code>: "
             "RollingUpdate strategy, 2/2 available, probes wired."),
            ("05_service_nodeport.png",
             "Figure 10 — <code>kubectl get svc</code>: NodePort service "
             "exposed for external traffic."),
            ("06_hpa.png",
             "Figure 11 — <code>kubectl get hpa</code>: active HPA "
             "(2-5 replicas, 70 % CPU target) with live metrics."),
            ("07_curl_health_predict.png",
             "Figure 12 — Live <code>curl</code> calls: <code>/health</code> "
             "and <code>/predict</code> returning valid JSON."),
        ]
        for filename, caption in shots:
            story += image_fn(screenshots_dir / filename, 15, caption)
    story.append(helpers["PageBreak"]())

    # ---- 8. Architecture + Conclusion + Appendix ----
    story += section_architecture(styles, helpers)
    story.append(helpers["PageBreak"]())

    story += section_conclusion(styles, helpers, metrics)
    story += section_appendix(styles, helpers)
    story += section_deliverables(styles, helpers)

    return story
