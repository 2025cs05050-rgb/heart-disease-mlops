"""Train Logistic Regression and Random Forest classifiers.

- Performs stratified 5-fold cross-validation
- Hyperparameter tuning with ``GridSearchCV``
- Logs everything (params, metrics, plots, model artifacts) to MLflow
  if the package is available; otherwise still saves a local pickle.
- Selects the best model by mean CV ROC-AUC and persists it to
  ``models/model.pkl`` together with the metric report in JSON.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from .data_loader import load_clean, split_features_target
from .evaluate import (
    compute_metrics,
    cv_summary,
    plot_confusion_matrix,
    plot_roc_curve,
)
from .preprocess import build_pipeline

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGS_DIR = REPORTS_DIR / "figures"

CV_SCORERS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def get_model_specs() -> dict[str, dict]:
    """Define candidate models + their hyperparameter grids."""
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=1000, solver="liblinear"),
            "param_grid": {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__penalty": ["l1", "l2"],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
            "param_grid": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_split": [2, 5],
            },
        },
    }


def _try_mlflow():
    try:
        import mlflow
        return mlflow
    except ImportError:
        log.warning("mlflow not installed — skipping experiment logging.")
        return None


def train_one(name: str, spec: dict, X_train, y_train, X_test, y_test,
              cv: StratifiedKFold, mlflow=None) -> dict:
    log.info("=== Training %s ===", name)
    pipe = build_pipeline(spec["estimator"])

    grid = GridSearchCV(
        pipe, spec["param_grid"], cv=cv, scoring="roc_auc",
        n_jobs=-1, refit=True,
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    cv_res = cross_validate(best, X_train, y_train, cv=cv,
                            scoring=CV_SCORERS, n_jobs=-1)
    cv_metrics = cv_summary(cv_res)

    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
    test_metrics = {f"test_{k}": v for k, v in
                    compute_metrics(y_test, y_pred, y_proba).items()}

    cm_path = plot_confusion_matrix(y_test, y_pred,
                                    FIGS_DIR / f"cm_{name}.png")
    roc_path = plot_roc_curve(y_test, y_proba,
                              FIGS_DIR / f"roc_{name}.png", label=name)

    log.info("  best params: %s", grid.best_params_)
    log.info("  CV ROC-AUC : %.4f ± %.4f",
             cv_metrics["cv_roc_auc_mean"], cv_metrics["cv_roc_auc_std"])
    log.info("  test metrics: %s",
             {k: round(v, 4) for k, v in test_metrics.items()})

    if mlflow is not None:
        with mlflow.start_run(run_name=name):
            mlflow.log_params({k.replace("model__", ""): v
                               for k, v in grid.best_params_.items()})
            mlflow.log_param("model_type", name)
            mlflow.log_metrics({**cv_metrics, **test_metrics})
            mlflow.log_artifact(str(cm_path), artifact_path="figures")
            mlflow.log_artifact(str(roc_path), artifact_path="figures")
            try:
                from mlflow.models.signature import infer_signature
                signature = infer_signature(X_train, best.predict(X_train))
                input_example = X_train.head(2)
                try:
                    mlflow.sklearn.log_model(
                        best, artifact_path="model",
                        signature=signature, input_example=input_example,
                    )
                except TypeError:
                    mlflow.sklearn.log_model(
                        best, name="model",
                        signature=signature, input_example=input_example,
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning("mlflow.sklearn.log_model failed: %s", exc)

    return {
        "name": name,
        "best_params": grid.best_params_,
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "estimator": best,
    }


def main(test_size: float = 0.2, random_state: int = 42,
         experiment: str = "heart-disease") -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_clean()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    log.info("Train: %s, Test: %s", X_train.shape, X_test.shape)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    mlflow = _try_mlflow()
    if mlflow is not None:
        mlflow.set_tracking_uri(f"file:{(ROOT / 'mlruns').as_posix()}")
        mlflow.set_experiment(experiment)

    results = []
    for name, spec in get_model_specs().items():
        results.append(
            train_one(name, spec, X_train, y_train, X_test, y_test, cv, mlflow)
        )

    best = max(results, key=lambda r: r["cv_metrics"]["cv_roc_auc_mean"])
    log.info("Selected best model: %s (CV ROC-AUC=%.4f)",
             best["name"], best["cv_metrics"]["cv_roc_auc_mean"])

    model_path = MODELS_DIR / "model.pkl"
    joblib.dump(best["estimator"], model_path)
    log.info("Saved model -> %s", model_path)

    report = {
        "selected_model": best["name"],
        "best_params": best["best_params"],
        "cv_metrics": best["cv_metrics"],
        "test_metrics": best["test_metrics"],
        "all_results": [
            {k: v for k, v in r.items() if k != "estimator"} for r in results
        ],
    }
    report_path = REPORTS_DIR / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    log.info("Saved report -> %s", report_path)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--experiment", default="heart-disease")
    args = parser.parse_args()
    main(args.test_size, args.random_state, args.experiment)
