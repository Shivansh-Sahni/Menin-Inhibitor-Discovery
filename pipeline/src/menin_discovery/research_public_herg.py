"""Reproducible public Sun hERG comparators with an untouched source-held-out set."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from .features import RDKIT_DESCRIPTOR_COLUMNS
from .research_common import atomic_write_csv, atomic_write_json, atomic_write_parquet, atomic_write_text
from .research_modeling import (
    continuous_herg_metrics,
    herg_classification_metrics,
    structure_feature_frame,
)


def prepare_sun_source_holdout(public_root: str | Path) -> dict[str, pd.DataFrame | int]:
    """Curate train/validation tables without resolving conflicts by averaging."""

    root = Path(public_root)
    records = pd.read_parquet(root / "public_herg_normalized.parquet")
    quarantine = pd.read_parquet(root / "public_herg_quarantine.parquet")
    conflict_ids = set(quarantine["structure_id"].dropna().astype(str))
    valid_records = records[records["structure_valid"].fillna(False) & records["structure_id"].notna()].copy()
    raw_train_ids = set(
        valid_records.loc[valid_records["dataset_role"] != "validation", "structure_id"].astype(str)
    )
    raw_validation_ids = set(
        valid_records.loc[valid_records["dataset_role"] == "validation", "structure_id"].astype(str)
    )
    # Count the source overlap before conflict quarantine.  The one known
    # overlap is also a measurement disagreement; excluding conflicts first
    # would make the leakage audit incorrectly report zero.
    overlap = raw_train_ids & raw_validation_ids
    records = valid_records[~valid_records["structure_id"].astype(str).isin(conflict_ids | overlap)].copy()
    validation = records[records["dataset_role"] == "validation"].copy()
    validation_pic50 = pd.to_numeric(validation["pic50_value"], errors="coerce")
    validation["canonical_blocker_class"] = np.select(
        [
            validation_pic50.ge(5.0).fillna(False).to_numpy(dtype=bool),
            validation_pic50.le(6.0 - np.log10(30.0)).fillna(False).to_numpy(dtype=bool),
        ],
        [1.0, 0.0],
        default=np.nan,
    )

    structures = records[["structure_id", "raw_smiles", "computed_mw_g_mol"]].drop_duplicates("structure_id")
    descriptors = structure_feature_frame(
        structures.rename(columns={"structure_id": "compound_id", "raw_smiles": "standardized_smiles"})
    )
    descriptors = descriptors.rename(columns={"compound_id": "structure_id"})

    classification_train = records[
        (records["dataset_role"] == "classification") & records["canonical_blocker_class"].notna()
    ].copy()
    classification_train = classification_train.drop_duplicates(
        ["structure_id", "canonical_blocker_class"]
    ).drop_duplicates("structure_id", keep=False)
    classification_validation = validation[validation["canonical_blocker_class"].notna()].copy()
    classification_validation = classification_validation.drop_duplicates(
        ["structure_id", "canonical_blocker_class"]
    ).drop_duplicates("structure_id", keep=False)

    regression_train = records[
        (records["dataset_role"] == "regression") & records["pic50_value"].notna()
    ].copy()
    # Conflicting structures were removed above; replicate-identical rows can
    # be collapsed without suppressing disagreements.
    regression_train = regression_train.groupby("structure_id", as_index=False).agg(
        pic50_value=("pic50_value", "median"),
        raw_smiles=("raw_smiles", "first"),
        computed_mw_g_mol=("computed_mw_g_mol", "first"),
    )
    regression_validation = (
        validation[validation["pic50_value"].notna()]
        .groupby("structure_id", as_index=False)
        .agg(
            pic50_value=("pic50_value", "median"),
            raw_smiles=("raw_smiles", "first"),
            computed_mw_g_mol=("computed_mw_g_mol", "first"),
        )
    )

    def attach(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.merge(descriptors, on="structure_id", how="inner", validate="one_to_one")

    return {
        "classification_train": attach(classification_train),
        "classification_validation": attach(classification_validation),
        "regression_train": attach(regression_train),
        "regression_validation": attach(regression_validation),
        "quarantined_structure_count": len(conflict_ids),
        "train_validation_overlap_removed": len(overlap),
    }


def run_sun_public_reproduction(
    public_root: str | Path,
    output_dir: str | Path,
    *,
    random_state: int = 20260721,
) -> dict[str, Any]:
    """Fit SVC/SVR and tree comparators on the paper's source-held-out roles."""

    prepared = prepare_sun_source_holdout(public_root)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    feature_columns = [column for column in RDKIT_DESCRIPTOR_COLUMNS if column != "invalid_structure"]
    classification_train = prepared["classification_train"]
    classification_validation = prepared["classification_validation"]
    regression_train = prepared["regression_train"]
    regression_validation = prepared["regression_validation"]
    assert isinstance(classification_train, pd.DataFrame)
    assert isinstance(classification_validation, pd.DataFrame)
    assert isinstance(regression_train, pd.DataFrame)
    assert isinstance(regression_validation, pd.DataFrame)

    classifiers = {
        "svc_rbf_rdkit": Pipeline(
            [
                ("impute", SimpleImputer()),
                ("scale", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=3.0,
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "extra_trees_rdkit": Pipeline(
            [
                ("impute", SimpleImputer()),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
    regressors = {
        "svr_rbf_rdkit": Pipeline(
            [("impute", SimpleImputer()), ("scale", StandardScaler()), ("model", SVR(C=3.0, epsilon=0.15))]
        ),
        "extra_trees_rdkit": Pipeline(
            [
                ("impute", SimpleImputer()),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=500,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    classification_rows: list[dict[str, Any]] = []
    classification_predictions: list[pd.DataFrame] = []
    X_train = classification_train[feature_columns].replace([np.inf, -np.inf], np.nan)
    y_train = classification_train["canonical_blocker_class"].astype(int).to_numpy()
    X_validation = classification_validation[feature_columns].replace([np.inf, -np.inf], np.nan)
    y_validation = classification_validation["canonical_blocker_class"].astype(int).to_numpy()
    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_validation)[:, 1]
        classification_rows.append(
            {
                "model": name,
                "train_role": "classification",
                "validation_role": "validation_source_held_out",
                "n_train": len(y_train),
                **herg_classification_metrics(y_validation, probability),
            }
        )
        classification_predictions.append(
            pd.DataFrame(
                {
                    "structure_id": classification_validation["structure_id"].astype(str),
                    "model": name,
                    "observed_blocker": y_validation,
                    "blocker_probability": probability,
                    "computed_mw_g_mol": classification_validation["computed_mw_g_mol"],
                }
            )
        )

    regression_rows: list[dict[str, Any]] = []
    regression_predictions: list[pd.DataFrame] = []
    X_train = regression_train[feature_columns].replace([np.inf, -np.inf], np.nan)
    y_train = regression_train["pic50_value"].astype(float).to_numpy()
    X_validation = regression_validation[feature_columns].replace([np.inf, -np.inf], np.nan)
    y_validation = regression_validation["pic50_value"].astype(float).to_numpy()
    for name, model in regressors.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_validation)
        regression_rows.append(
            {
                "model": name,
                "train_role": "regression",
                "validation_role": "validation_source_held_out",
                "n_train": len(y_train),
                **continuous_herg_metrics(y_validation, prediction),
            }
        )
        regression_predictions.append(
            pd.DataFrame(
                {
                    "structure_id": regression_validation["structure_id"].astype(str),
                    "model": name,
                    "observed_pic50": y_validation,
                    "predicted_pic50": prediction,
                    "computed_mw_g_mol": regression_validation["computed_mw_g_mol"],
                }
            )
        )

    classification_metrics = pd.DataFrame(classification_rows)
    regression_metrics = pd.DataFrame(regression_rows)
    atomic_write_csv(destination / "classification_source_holdout_metrics.csv", classification_metrics)
    atomic_write_csv(destination / "regression_source_holdout_metrics.csv", regression_metrics)
    atomic_write_parquet(
        destination / "classification_source_holdout_predictions.parquet",
        pd.concat(classification_predictions, ignore_index=True),
    )
    atomic_write_parquet(
        destination / "regression_source_holdout_predictions.parquet",
        pd.concat(regression_predictions, ignore_index=True),
    )
    audit = {
        "quarantined_structure_count": int(prepared["quarantined_structure_count"]),
        "train_validation_overlap_removed": int(prepared["train_validation_overlap_removed"]),
        "classification_train_rows": len(classification_train),
        "classification_validation_rows": len(classification_validation),
        "regression_train_rows": len(regression_train),
        "regression_validation_rows": len(regression_validation),
        "source_class_polarity": "source 0 normalized to canonical blocker 1",
        "regression_conversion": "pIC50 = 9 - stored log10(IC50 nM)",
        "reproduction_status": "partial_missing_paper_specific_atom_typing_and_correction_mapping",
    }
    atomic_write_json(destination / "reproduction_audit.json", audit)
    atomic_write_text(
        destination / "reproduction_limitations.md",
        "# Sun/Wang/Shen reproduction boundary\n\n"
        "The supplied structures, corrected class polarity, and corrected pIC50 values are reproduced with "
        "independent RDKit SVC/SVR and tree comparators. Opposite-class/conflicting structures and the published "
        "train-validation overlap are excluded. The paper-specific atom typing and the complete correction-factor "
        "mapping were not supplied, so this is not represented as an exact implementation of the published model.\n",
    )
    return audit


__all__ = ["prepare_sun_source_holdout", "run_sun_public_reproduction"]
