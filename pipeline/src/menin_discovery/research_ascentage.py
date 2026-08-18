"""Validated loading and transparent recovery for the Ascentage hERG source table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "source_row",
    "internal_id",
    "standardized_smiles",
    "structure_id",
    "herg_ic50_relation",
    "herg_ic50_censoring",
    "herg_pic50_relation",
    "herg_pic50_lower_bound",
    "herg_pic50_upper_bound",
    "synthesis_status",
    "series_status",
}
FIRST_DERIVED_PREDICTION_COLUMN = "predicted_pic50"


def validate_ascentage_source(frame: pd.DataFrame) -> None:
    """Validate the fixed 2026-07-28 source boundary and Angelo clarifications."""

    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Ascentage source table is missing columns: {missing}")
    if FIRST_DERIVED_PREDICTION_COLUMN in frame:
        raise ValueError("Ascentage source boundary contains derived model predictions")
    if len(frame) != 76 or frame["structure_id"].nunique() != 76:
        raise ValueError("Expected 76 unique Ascentage source structures")
    if frame["internal_id"].duplicated().any():
        raise ValueError("Ascentage source contains duplicate internal IDs")
    exact = frame["herg_ic50_relation"].eq("=")
    censored = frame["herg_ic50_relation"].eq(">")
    missing_measurement = frame["herg_ic50_censoring"].eq("missing")
    if (int(exact.sum()), int(censored.sum()), int(missing_measurement.sum())) != (61, 7, 8):
        raise ValueError("Expected 61 exact, 7 right-censored, and 8 missing source outcomes")
    if not frame.loc[censored, "herg_pic50_relation"].eq("<").all():
        raise ValueError("IC50 >30 uM records must transform to left-censored pIC50")
    expected_bound = 6.0 - np.log10(30.0)
    if not np.allclose(
        frame.loc[censored, "herg_pic50_upper_bound"].astype(float),
        expected_bound,
        atol=1e-10,
    ):
        raise ValueError("The seven >30 uM records do not share the correct pIC50 upper bound")
    if not frame.loc[missing_measurement, "synthesis_status"].eq("not_synthesized").all():
        raise ValueError("Blank source outcomes must remain explicitly not synthesized")
    measured = ~missing_measurement
    if not frame.loc[measured, "synthesis_status"].eq("synthesized_by_cro").all():
        raise ValueError("Measured source compounds must retain CRO synthesis provenance")
    if not frame["series_status"].eq("same_internal_medicinal_chemistry_series_confirmed").all():
        raise ValueError("All source compounds must retain the confirmed same-series status")


def load_ascentage_source(
    canonical_path: Path,
    *,
    recovery_artifact: Path,
) -> pd.DataFrame:
    """Load canonical source data or recover its source-only columns transparently.

    The retained evaluation artifact starts with the unchanged normalized source
    columns and appends model outputs beginning at ``predicted_pic50``.  The
    fallback removes every appended field and validates the original outcome
    counts.  It does not recreate the missing DOCX/CDX binaries.
    """

    if canonical_path.exists():
        frame = pd.read_parquet(canonical_path)
        frame.attrs["source_boundary"] = "canonical_normalized_records"
    else:
        if not recovery_artifact.exists():
            raise FileNotFoundError(
                f"Neither canonical source {canonical_path} nor recovery artifact {recovery_artifact} exists"
            )
        retained = pd.read_parquet(recovery_artifact)
        if FIRST_DERIVED_PREDICTION_COLUMN not in retained:
            raise ValueError("Recovery artifact lacks the derived-boundary marker")
        boundary = retained.columns.get_loc(FIRST_DERIVED_PREDICTION_COLUMN)
        frame = retained.iloc[:, :boundary].copy()
        frame.attrs["source_boundary"] = "recovered_source_only_columns_from_retained_evaluation_artifact"
    validate_ascentage_source(frame)
    return frame
