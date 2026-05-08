"""Download the UCI Heart Disease (Cleveland processed) dataset.

Tries the UCI mirror first, then falls back to a local copy under
``MLOps/heart+disease/processed.cleveland.data`` if the network is
unavailable. The result is written to ``data/raw/processed.cleveland.data``.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import requests

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TARGET = RAW_DIR / "processed.cleveland.data"

# Local fallback (sibling folder in this repository)
LOCAL_FALLBACK = (
    ROOT.parent / "heart+disease" / "processed.cleveland.data"
)


def download() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[download] GET {UCI_URL}")
        resp = requests.get(UCI_URL, timeout=15)
        resp.raise_for_status()
        TARGET.write_bytes(resp.content)
        print(f"[download] saved -> {TARGET} ({TARGET.stat().st_size} bytes)")
        return TARGET
    except Exception as exc:  # noqa: BLE001
        print(f"[download] UCI fetch failed: {exc}")

    if LOCAL_FALLBACK.exists():
        print(f"[download] using local fallback: {LOCAL_FALLBACK}")
        shutil.copy2(LOCAL_FALLBACK, TARGET)
        print(f"[download] saved -> {TARGET}")
        return TARGET

    print("[download] ERROR: no network and no local fallback found.",
          file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    download()
