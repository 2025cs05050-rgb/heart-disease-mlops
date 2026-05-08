"""Unit tests for ``src.data_loader``."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    CATEGORICAL_FEATURES,
    COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
    clean,
    load_clean,
    load_raw,
    save_processed,
    split_features_target,
    summary,
)


def test_columns_definition():
    """Column lists must stay in sync with the dataset shape."""
    assert len(COLUMNS) == 14
    assert COLUMNS[-1] == "num"
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) == \
        set(COLUMNS) - {"num"}


def test_load_raw_shape():
    df = load_raw()
    assert df.shape == (303, 14)
    assert list(df.columns) == COLUMNS


def test_load_raw_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "does_not_exist.data")


def test_clean_binarises_target():
    df = clean(load_raw())
    assert TARGET in df.columns
    assert "num" not in df.columns
    assert set(df[TARGET].unique()) <= {0, 1}


def test_clean_no_missing_target():
    df = clean(load_raw())
    assert df[TARGET].notna().all()


def test_clean_preserves_row_count():
    raw = load_raw()
    df = clean(raw)
    # Cleveland file has no missing target -> rows preserved
    assert df.shape[0] == raw.shape[0] == 303


def test_clean_handles_non_numeric(tiny_dataframe):
    """Cleaning should coerce any '?' strings into NaN floats."""
    raw_like = tiny_dataframe.drop(columns=["target"]).copy()
    raw_like["num"] = [0, 2, 0, 3]
    raw_like.loc[0, "ca"] = "?"
    cleaned = clean(raw_like)
    assert pd.isna(cleaned.loc[0, "ca"])
    assert cleaned.loc[1, TARGET] == 1
    assert cleaned.loc[2, TARGET] == 0


def test_split_features_target():
    df = load_clean()
    X, y = split_features_target(df)
    assert TARGET not in X.columns
    assert y.name == TARGET
    assert len(X) == len(y) == 303


def test_summary_keys():
    s = summary(load_clean())
    assert {"n_rows", "n_cols", "n_missing", "class_balance"} <= s.keys()
    assert s["n_rows"] == 303
    # Cleveland data is roughly balanced
    assert 0.3 < s["class_balance"][1] < 0.7


def test_save_processed(tmp_path: Path):
    df = load_clean()
    out = save_processed(df, tmp_path / "out.csv")
    assert out.exists()
    reread = pd.read_csv(out)
    assert reread.shape == df.shape
