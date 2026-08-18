#!/usr/bin/env python3
"""Recover the source-only Ascentage normalized table from its retained report artifact.

This is a transparent repair for a cleaned staging directory. It restores the
normalized tabular source boundary, but it does not claim to recreate the
missing original DOCX or embedded CDX binaries.
"""

from __future__ import annotations

import json
from pathlib import Path

from menin_discovery.research_ascentage import load_ascentage_source
from menin_discovery.research_common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "research/data/pk_herg/canonical/ascentage_herg_2026_07_28"
CANONICAL = OUTPUT / "normalized_records.parquet"
RECOVERY = ROOT / "research/reports/pk_herg/ascentage_herg_extension/predictions.parquet"


def recover() -> dict[str, object]:
    # Loading with an intentionally absent target exercises the validated
    # source-only recovery path even when this command is rerun.
    source = load_ascentage_source(
        OUTPUT / "__source_recovery_input_absent__.parquet",
        recovery_artifact=RECOVERY,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(CANONICAL, source)
    review_columns = [
        "source_row",
        "internal_id",
        "ascentage_id",
        "submitted_herg_ic50_um",
        "herg_ic50_value_um",
        "herg_ic50_relation",
        "herg_pic50_value",
        "herg_pic50_relation",
        "synthesis_status",
        "herg_measurement_status",
        "herg_missing_reason",
        "menin_potency_status",
        "series_status",
        "synthesis_provider",
        "standardized_smiles",
        "standard_inchi_key",
        "structure_id",
        "computed_mw_g_mol",
        "scaffold",
        "assay_protocol_status",
        "decision_track_eligible",
    ]
    atomic_write_csv(OUTPUT / "normalized_records.csv", source[review_columns])
    summary = {
        "status": "normalized_table_recovered_from_retained_evaluation_artifact",
        "records": int(len(source)),
        "unique_structures": int(source["structure_id"].nunique()),
        "exact_measurements": int(source["herg_ic50_relation"].eq("=").sum()),
        "right_censored_measurements": int(source["herg_ic50_relation"].eq(">").sum()),
        "not_synthesized_structures": int(source["synthesis_status"].eq("not_synthesized").sum()),
        "derived_prediction_columns_removed": True,
        "original_docx_available": False,
        "embedded_cdx_files_available": False,
        "source_limitation": (
            "The original source DOCX and embedded CDX binaries are no longer present. "
            "The normalized fields were recovered exactly from the source-prefix columns "
            "of the retained evaluation artifact."
        ),
    }
    atomic_write_json(OUTPUT / "recovery_qc.json", summary)
    notice = """# Ascentage normalized-table recovery notice

The original staging directory was removed after the extension analysis. The normalized
source fields in this directory were recovered from the unchanged source-prefix columns of
`research/reports/pk_herg/ascentage_herg_extension/predictions.parquet`; all columns beginning
with the first model field (`predicted_pic50`) were excluded.

Validation requires exactly 76 unique structures, 61 exact IC50 values, seven `>30 uM`
right-censored IC50 values (pIC50 `<4.522879`), and eight blank/not-synthesized structures.
All compounds retain Dr. Aguilar's same-series clarification and CRO synthesis provenance.

This repair does **not** recreate the original DOCX or its embedded CDX files. Those raw binaries
must be reacquired from Dr. Aguilar for a publication archive. Until then, the recovered table is
sufficient for reproducible modeling but not a complete raw-evidence package.
"""
    atomic_write_text(OUTPUT / "RECOVERY_NOTICE.md", notice)
    return summary


def main() -> None:
    print(json.dumps(recover(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
