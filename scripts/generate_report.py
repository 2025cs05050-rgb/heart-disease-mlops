"""Generate the MLOps Assignment-1 PDF report.

Pulls real metrics from ``reports/metrics.json`` and embeds the
confusion-matrix / ROC plots produced by ``src.train``.

Run:
    python scripts/generate_report.py
Output:
    reports/MLOps_Assignment1_Report.pdf
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Local helpers live alongside this script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_sections import build_sections  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METRICS = json.loads((ROOT / "reports" / "metrics.json").read_text())
OUT = ROOT / "reports" / "MLOps_Assignment1_Report.pdf"

PRIMARY = HexColor("#1f4e79")
ACCENT = HexColor("#c0392b")
GREY = HexColor("#555555")


def make_styles() -> dict:
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontSize=24, textColor=PRIMARY,
            spaceAfter=18, alignment=TA_CENTER, leading=28,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontSize=14,
            textColor=GREY, alignment=TA_CENTER, spaceAfter=6,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontSize=16, textColor=PRIMARY,
            spaceBefore=14, spaceAfter=8, leading=20,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontSize=12, textColor=ACCENT,
            spaceBefore=10, spaceAfter=4, leading=16,
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontSize=10.5,
            alignment=TA_JUSTIFY, leading=14, spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["BodyText"], fontSize=10.5,
            leftIndent=14, bulletIndent=2, leading=14, spaceAfter=2,
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["Italic"], fontSize=9, textColor=GREY,
            alignment=TA_CENTER, spaceAfter=10,
        ),
        "code": ParagraphStyle(
            "code", parent=base["Code"], fontSize=8.5, textColor=HexColor("#222"),
            backColor=HexColor("#f4f4f4"), borderPadding=4, leading=11,
            spaceAfter=8, alignment=TA_LEFT,
        ),
    }
    return styles


def _on_page(canvas, doc):
    """Footer + header on every page except the cover."""
    canvas.saveState()
    page_num = canvas.getPageNumber()
    if page_num > 1:
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(GREY)
        canvas.drawString(2 * cm, 1.2 * cm,
                          "MLOps Assignment-1 — Heart Disease Risk Pipeline")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {page_num}")
        canvas.setStrokeColor(HexColor("#cccccc"))
        canvas.line(2 * cm, 1.5 * cm, A4[0] - 2 * cm, 1.5 * cm)
    canvas.restoreState()


def build() -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="MLOps Assignment-1 Report",
        author="BITS MTech — AMLCSZG523",
    )
    styles = make_styles()
    story = build_sections(
        styles=styles,
        metrics=METRICS,
        figures_dir=ROOT / "reports" / "figures",
        screenshots_dir=ROOT / "screenshots",
        helpers={
            "Paragraph": Paragraph,
            "Spacer": Spacer,
            "PageBreak": PageBreak,
            "Table": Table,
            "TableStyle": TableStyle,
            "Image": Image,
            "cm": cm,
            "PRIMARY": PRIMARY,
            "ACCENT": ACCENT,
            "GREY": GREY,
            "today": date.today().isoformat(),
        },
    )
    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return OUT


if __name__ == "__main__":
    out = build()
    print(f"PDF written -> {out}  ({out.stat().st_size / 1024:.1f} KB)")
