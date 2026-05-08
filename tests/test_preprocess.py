"""Unit tests for ``src.preprocess``."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_loader import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_clean,
    split_features_target,
)
from src.preprocess import build_pipeline, build_preprocessor


def test_build_preprocessor_type():
    pre = build_preprocessor()
    assert isinstance(pre, ColumnTransformer)


def test_preprocessor_outputs_finite_array():
    X, _ = split_features_target(load_clean())
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] >= len(NUMERIC_FEATURES)
    assert np.all(np.isfinite(Xt))


def test_preprocessor_imputes_missing_values():
    """Median/most-frequent imputation removes NaN before scaling/OHE."""
    X, _ = split_features_target(load_clean())
    # Inject some missing values into a numeric and a categorical column
    X = X.copy()
    X.loc[X.index[:5], "trestbps"] = np.nan
    X.loc[X.index[:5], "thal"] = np.nan
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    assert not np.isnan(Xt).any()


def test_pipeline_with_estimator(tiny_dataframe):
    X = tiny_dataframe.drop(columns=["target"])
    y = tiny_dataframe["target"]
    pipe = build_pipeline(LogisticRegression(max_iter=200))
    assert isinstance(pipe, Pipeline)
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)) <= {0, 1}


def test_pipeline_handles_unknown_category(tiny_dataframe):
    """OneHotEncoder configured with handle_unknown='ignore' must
    accept new categories at inference time without raising."""
    X = tiny_dataframe.drop(columns=["target"])
    y = tiny_dataframe["target"]
    pipe = build_pipeline(LogisticRegression(max_iter=200))
    pipe.fit(X, y)

    new_row = X.iloc[[0]].copy()
    new_row.loc[:, "thal"] = 99  # unseen category
    preds = pipe.predict(new_row)
    assert preds.shape == (1,)


def test_categorical_columns_ohe_expands_features():
    X, _ = split_features_target(load_clean())
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    # OHE on 8 categorical columns adds at least one column per category
    assert Xt.shape[1] > len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)


def test_columns_preserved_after_dataframe_input():
    """The transformer should accept a DataFrame and ignore extra cols."""
    X, _ = split_features_target(load_clean())
    X_extra = X.copy()
    X_extra["extra_col"] = 1.0
    pre = build_preprocessor()
    pre.fit(X_extra)
    Xt = pre.transform(X_extra)
    assert Xt.shape[0] == X_extra.shape[0]
