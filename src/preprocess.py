"""Reusable preprocessing pipeline for the Heart Disease dataset.

Numeric features  -> median imputation + standard scaling
Categorical feats -> most-frequent imputation + one-hot encoding

Returning a single ``ColumnTransformer`` keeps train and inference
identical, which is critical for reproducibility.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _ohe() -> OneHotEncoder:
    """OneHotEncoder compatible with sklearn >=1.2 (sparse_output kw)."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    """Build the column-wise preprocessing pipeline.

    Returns
    -------
    ColumnTransformer
        A fitted/un-fitted transformer that can be plugged into any
        sklearn ``Pipeline`` or saved/loaded with ``joblib``.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _ohe()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator) -> Pipeline:
    """Wrap any sklearn estimator with the standard preprocessor."""
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", estimator),
    ])
