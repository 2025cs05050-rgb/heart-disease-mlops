"""Load and clean the UCI Heart Disease (Cleveland) dataset.

The raw file ``processed.cleveland.data`` has 14 numeric attributes with
missing values encoded as ``"?"``. The target ``num`` originally ranges
0-4 (severity); we binarise it to 0 (no disease) / 1 (disease present),
which is the standard formulation used in the literature.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS: list[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

# Categorical vs numeric split (used by the preprocessing pipeline)
NUMERIC_FEATURES: list[str] = [
    "age", "trestbps", "chol", "thalach", "oldpeak",
]
CATEGORICAL_FEATURES: list[str] = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
]
TARGET: str = "target"

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "processed.cleveland.data"
PROCESSED_PATH = ROOT / "data" / "processed" / "heart_cleveland.csv"


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Read the raw UCI file into a DataFrame with proper column names."""
    path = Path(path) if path else RAW_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. "
            "Run `python scripts/download_data.py` first."
        )
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and binarise the dataset.

    - Drops rows where the target is missing.
    - Coerces all features to numeric (NaN for non-parseable values).
    - Binarises ``num`` -> ``target`` (0 vs >=1).
    """
    df = df.copy()
    df = df.dropna(subset=["num"])
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    return df.reset_index(drop=True)


def load_clean(path: Path | str | None = None) -> pd.DataFrame:
    """Convenience wrapper: load raw + clean in one call."""
    return clean(load_raw(path))


def save_processed(df: pd.DataFrame,
                   out_path: Path | str | None = None) -> Path:
    out_path = Path(out_path) if out_path else PROCESSED_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[TARGET]), df[TARGET]


def summary(df: pd.DataFrame) -> dict:
    """Return a small summary dict useful for EDA / logging."""
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "n_missing": int(df.isna().sum().sum()),
        "class_balance": df[TARGET].value_counts(normalize=True).to_dict()
        if TARGET in df.columns
        else None,
    }


if __name__ == "__main__":
    raw = load_raw()
    print(f"Raw shape: {raw.shape}, missing: {int(raw.isna().sum().sum())}")
    df = clean(raw)
    out = save_processed(df)
    print(f"Cleaned shape: {df.shape}, saved -> {out}")
    print("Summary:", summary(df))
