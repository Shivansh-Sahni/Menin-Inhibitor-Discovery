"""Baseline activity and hERG models."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from .features import SmilesFeatureTransformer


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def train_menin_activity_model(
    compounds: pd.DataFrame,
    models_dir: Path,
    reports_dir: Path,
    *,
    random_state: int = 13,
) -> dict:
    """Train a dependency-light pActivity regression baseline."""

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    data = compounds.dropna(subset=["smiles", "p_activity_median"]).copy()
    data = data[data["smiles"].astype(str).str.len() > 0]

    if len(data) < 40:
        metrics = {"status": "insufficient_data", "n_compounds": int(len(data))}
        _write_json(reports_dir / "menin_activity_model_metrics.json", metrics)
        return metrics

    X_train, X_test, y_train, y_test = train_test_split(
        data["smiles"],
        data["p_activity_median"].astype(float),
        test_size=0.2,
        random_state=random_state,
    )

    pipeline = Pipeline(
        [
            ("features", SmilesFeatureTransformer()),
            ("model", Ridge()),
        ]
    )
    grid = GridSearchCV(
        pipeline,
        {"model__alpha": [0.1, 1.0, 3.0, 10.0, 30.0]},
        scoring="neg_mean_absolute_error",
        cv=5,
        n_jobs=1,
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    preds = best.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    metrics = {
        "status": "trained",
        "model": "hashed-SMILES Ridge regression",
        "n_compounds": int(len(data)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "best_alpha": float(grid.best_params_["model__alpha"]),
        "test_mae_pchembl": float(mean_absolute_error(y_test, preds)),
        "test_rmse_pchembl": rmse,
        "test_r2": float(r2_score(y_test, preds)),
        "interpretation": "A fast baseline for triage only; RDKit fingerprints and scaffold split are next-step upgrades.",
    }

    with (models_dir / "menin_activity_ridge.pkl").open("wb") as fh:
        pickle.dump(best, fh)
    _write_json(reports_dir / "menin_activity_model_metrics.json", metrics)

    pred_df = pd.DataFrame(
        {
            "smiles": X_test.values,
            "observed_p_activity_median": y_test.values,
            "predicted_p_activity_median": preds,
            "absolute_error": np.abs(y_test.values - preds),
        }
    ).sort_values("absolute_error", ascending=False)
    pred_df.to_csv(reports_dir / "menin_activity_model_test_predictions.csv", index=False)
    return metrics


def train_herg_classifier_and_predict(
    herg_compounds: pd.DataFrame,
    menin_compounds: pd.DataFrame,
    models_dir: Path,
    reports_dir: Path,
    *,
    random_state: int = 13,
) -> dict:
    """Train hERG blocker classifier and score menin compounds."""

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    data = herg_compounds.dropna(subset=["smiles", "herg_blocker_label"]).copy()
    data = data[data["smiles"].astype(str).str.len() > 0]
    data["herg_blocker_label"] = data["herg_blocker_label"].astype(int)

    if len(data) < 80 or data["herg_blocker_label"].nunique() < 2:
        metrics = {
            "status": "insufficient_data",
            "n_compounds": int(len(data)),
            "n_classes": int(data["herg_blocker_label"].nunique()) if len(data) else 0,
        }
        _write_json(reports_dir / "herg_classifier_metrics.json", metrics)
        return metrics

    X_train, X_test, y_train, y_test = train_test_split(
        data["smiles"],
        data["herg_blocker_label"],
        test_size=0.2,
        random_state=random_state,
        stratify=data["herg_blocker_label"],
    )

    pipeline = Pipeline(
        [
            ("features", SmilesFeatureTransformer()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )
    grid = GridSearchCV(
        pipeline,
        {"model__C": [0.05, 0.1, 0.3, 1.0, 3.0]},
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        n_jobs=1,
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "status": "trained",
        "model": "hashed-SMILES logistic regression",
        "n_compounds": int(len(data)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_label": "hERG activity <=10 uM",
        "negative_label": "hERG activity >=30 uM",
        "best_C": float(grid.best_params_["model__C"]),
        "test_roc_auc": float(roc_auc_score(y_test, proba)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "interpretation": "Coarse liability screen; experimental hERG assay labels and applicability-domain checks are needed before decision use.",
    }

    with (models_dir / "herg_liability_logistic.pkl").open("wb") as fh:
        pickle.dump(best, fh)
    _write_json(reports_dir / "herg_classifier_metrics.json", metrics)

    menin = menin_compounds.copy()
    if not menin.empty:
        menin["predicted_herg_blocker_probability"] = best.predict_proba(menin["smiles"])[:, 1]
        menin["predicted_herg_risk"] = np.select(
            [
                menin["predicted_herg_blocker_probability"] >= 0.70,
                menin["predicted_herg_blocker_probability"] <= 0.30,
            ],
            ["high", "low"],
            default="medium",
        )
        menin.to_csv(reports_dir / "menin_with_predicted_herg_risk.csv", index=False)

    test_df = pd.DataFrame(
        {
            "smiles": X_test.values,
            "observed_herg_blocker_label": y_test.values,
            "predicted_herg_blocker_probability": proba,
            "predicted_label_0p5": pred,
        }
    )
    test_df.to_csv(reports_dir / "herg_classifier_test_predictions.csv", index=False)
    return metrics


def run_models(processed_dir: Path, models_dir: Path, reports_dir: Path) -> dict[str, dict]:
    menin_path = processed_dir / "menin_compounds_curated.csv"
    herg_path = processed_dir / "herg_compounds_curated.csv"
    menin = pd.read_csv(menin_path) if menin_path.exists() else pd.DataFrame()
    herg = pd.read_csv(herg_path) if herg_path.exists() else pd.DataFrame()

    return {
        "menin_activity": train_menin_activity_model(menin, models_dir, reports_dir),
        "herg_liability": train_herg_classifier_and_predict(herg, menin, models_dir, reports_dir),
    }
