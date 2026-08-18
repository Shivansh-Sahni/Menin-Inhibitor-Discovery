from __future__ import annotations

import math

import pandas as pd
from menin_discovery.platform_data_schema import (
    canonical_relation,
    concentration_interval_to_nm,
    parse_interval,
    parse_numeric,
    relation_from_value,
    validate_table,
)


def test_interval_and_ambiguous_value_grammar_is_fail_closed() -> None:
    assert parse_interval("1-2 uM") == (1.0, 2.0)
    assert parse_interval("between -2e-3 and -1e-3 M") == (-0.002, -0.001)
    assert parse_interval("2-1 uM") is None
    assert relation_from_value("1-2 uM") == "interval"
    assert relation_from_value("2-1 uM") == "not_reported"
    assert relation_from_value("1 ± 0.2") == "~"
    assert relation_from_value("1+/-0.2") == "~"
    assert relation_from_value("ca. 1") == "~"
    assert canonical_relation("range") == "interval"
    assert math.isnan(parse_numeric("2-1 uM"))
    low, high, unit, status = concentration_interval_to_nm("IC50", "1-2", "uM")
    assert (low, high, unit, status) == (1000.0, 2000.0, "nM", "converted")


def test_source_and_task_composite_keys_and_strict_task_boolean() -> None:
    sources = pd.DataFrame(
        {
            "source_id": ["SRC-X", "SRC-X"],
            "snapshot_id": ["SNP-A", "SNP-B"],
            "source_name": ["x", "x"],
            "source_version": ["1", "1"],
            "retrieval_date_utc": ["2026-01-01", "2026-01-02"],
            "source_url": ["https://x", "https://x"],
            "license_name": ["CC", "CC"],
            "license_status": ["verified", "verified"],
            "access_class": ["public_redistributable", "public_redistributable"],
        }
    )
    assert validate_table("sources", sources) == []

    base = {
        "task_id": ["TASK-X", "TASK-X"],
        "task_type": ["binding_kd", "binding_kd"],
        "observation_id": ["OBS-1", "OBS-2"],
        "molecule_id": ["MOL-1", "MOL-2"],
        "protein_id": ["PROT-1", "PROT-1"],
        "assay_id": ["ASSAY-1", "ASSAY-1"],
        "source_id": ["SRC-X", "SRC-X"],
        "snapshot_id": ["SNP-A", "SNP-A"],
        "source_record_id": ["R1", "R2"],
        "label_kind": ["continuous_exact", "continuous_exact"],
        "label_value": [1.0, 2.0],
        "label_relation": ["=", "="],
        "label_lower_bound": [1.0, 2.0],
        "label_upper_bound": [1.0, 2.0],
        "label_unit": ["nM", "nM"],
        "observation_kind": ["experimental_summary", "experimental_summary"],
        "access_class": ["public_redistributable", "public_redistributable"],
        "default_task_eligible": [True, True],
        "required_modalities": [
            "small_molecule_structure;protein_sequence",
            "small_molecule_structure;protein_sequence",
        ],
    }
    assert validate_table("tasks", pd.DataFrame(base)) == []
    invalid = pd.DataFrame(base)
    invalid["default_task_eligible"] = ["True", "False"]
    assert any(issue["code"] == "invalid_boolean" for issue in validate_table("tasks", invalid))


def test_blank_observation_kind_is_rejected() -> None:
    from menin_discovery.platform_data_schema import TABLE_REQUIRED_COLUMNS

    row = {column: "x" for column in TABLE_REQUIRED_COLUMNS["observations"]}
    row.update(
        {
            "relation": "=",
            "evidence_domain": "binding",
            "evidence_stage": pd.NA,
            "development_stage": "unknown",
            "result_status": "reported",
            "quality_grade": "identity_resolved",
            "access_class": "public_redistributable",
            "inclusion_status": "included",
            "observation_kind": "",
        }
    )
    issues = validate_table("observations", pd.DataFrame([row]))
    assert any(
        issue["code"] == "missing_required_value" and issue["column"] == "observation_kind"
        for issue in issues
    )
