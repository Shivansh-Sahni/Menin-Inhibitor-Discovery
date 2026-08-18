from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from menin_discovery.platform_external_admission import (
    RIGHTS_CLEARED,
    RIGHTS_UNRESOLVED,
    SCHEMA_VERSION,
    ExternalAdmissionError,
    _atomic_json,
    _bindingdb_measurement_analysis,
    _validate_analysis_bindings,
    classify_admission_candidate,
    validate_exact_schema,
    verify_external_admission_analysis,
)
from menin_discovery.platform_external_normalization import (
    canonical_json_bytes,
    document_with_sha256,
)


def _write_identified(path: Path, body: dict[str, object]) -> dict[str, object]:
    document = document_with_sha256(body)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def test_prediction_is_quarantined_not_promoted_to_observed_label() -> None:
    result = classify_admission_candidate(
        {"predicted_affinity_nm": 10.0},
        rights_status=RIGHTS_CLEARED,
        molecule_match_count=1,
        target_match_count=1,
    )
    assert result["admission_state"] == "quarantine_prediction_is_not_observed_evidence"
    assert result["prediction_treated_as_observation"] is False
    assert result["model_label_admitted"] is False


def test_pre_admitted_label_attempt_fails_closed() -> None:
    with pytest.raises(ExternalAdmissionError, match="pre-admit"):
        classify_admission_candidate(
            {"model_label_admitted": True},
            rights_status=RIGHTS_CLEARED,
            molecule_match_count=1,
            target_match_count=1,
        )


@pytest.mark.parametrize(
    "field",
    ["has_posted_outcome_measures_module", "has_posted_adverse_events_module"],
)
def test_absence_of_clinical_module_is_unknown_not_negative(field: str) -> None:
    result = classify_admission_candidate(
        {field: False},
        rights_status=RIGHTS_CLEARED,
        source_kind="clinical_inventory",
    )
    assert result["admission_state"] == "review_absence_of_posted_module_is_not_a_negative_outcome"
    assert result["negative_label_inferred_from_absence"] is False


def test_ambiguous_identity_is_quarantined_when_rights_are_cleared() -> None:
    result = classify_admission_candidate(
        {},
        rights_status=RIGHTS_CLEARED,
        molecule_match_count=2,
        target_match_count=1,
    )
    assert result["admission_state"] == "quarantine_ambiguous_identity_link"
    assert result["canonical_observation_admitted"] is False


def test_rights_block_precedes_otherwise_complete_link() -> None:
    result = classify_admission_candidate(
        {},
        rights_status=RIGHTS_UNRESOLVED,
        molecule_match_count=1,
        target_match_count=1,
    )
    assert result["admission_state"] == "blocked_rights_or_access_not_cleared"
    assert result["model_label_admitted"] is False


@pytest.mark.parametrize(
    "actual",
    [
        ("source_id",),
        ("source_id", "value", "unexpected"),
        ("value", "source_id"),
    ],
)
def test_schema_drift_fails_closed(actual: tuple[str, ...]) -> None:
    with pytest.raises(ExternalAdmissionError, match="Schema drift"):
        validate_exact_schema(actual, ("source_id", "value"), context="fixture")


def test_identified_json_is_byte_deterministic(tmp_path: Path) -> None:
    body = {
        "schema_version": SCHEMA_VERSION,
        "states": {"review": 2, "blocked": 1},
        "external_model_labels_admitted": 0,
    }
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left_document = _atomic_json(left, body)
    right_document = _atomic_json(right, body)
    assert left.read_bytes() == right.read_bytes()
    assert left_document == right_document


def test_measurement_accounting_uses_measurement_key_not_only_source_row(tmp_path: Path) -> None:
    root = tmp_path / "normalized"
    (root / "bindingdb").mkdir(parents=True)
    table = pa.table(
        {
            "source_row_number_one_based": [1, 1],
            "measurement_key": ["Kd:primary", "Kd:secondary"],
            "endpoint_type": ["Kd", "Kd"],
            "relation": ["=", "="],
            "value_nm": [10.0, 10.0],
            "parse_status": ["parsed_candidate", "parsed_candidate"],
            "candidate_evidence_admitted": [True, True],
            "model_label_admitted": [False, False],
        }
    )
    pq.write_table(table, root / "bindingdb/affinity_observations.parquet")
    report, signatures = _bindingdb_measurement_analysis(root, {1: ("molecule", "protein")})
    assert report["exact_unique_dual_link_measurements"] == 2
    assert len(signatures[("molecule", "protein", "Kd")]) == 2
    assert {item[3] for item in signatures[("molecule", "protein", "Kd")]} == {
        "Kd:primary",
        "Kd:secondary",
    }


def test_analysis_binding_rejects_code_or_component_set_drift() -> None:
    from menin_discovery import platform_external_admission as admission

    components = [{"path": "observations/part-00000.parquet", "sha256": "a" * 64}]
    code_sha = admission.sha256_file(Path(admission.__file__).resolve())
    report = {
        "analyzer_version": admission.ANALYZER_VERSION,
        "analyzer_code_sha256": code_sha,
        "normalized_input": {"manifest_physical_sha256": "b" * 64},
        "canonical_input": {
            "manifest_physical_sha256": "c" * 64,
            "verified_used_components": components,
        },
    }
    manifest = {
        "analyzer_version": admission.ANALYZER_VERSION,
        "input_bindings": {
            "analyzer_code_sha256": code_sha,
            "normalized_manifest_physical_sha256": "b" * 64,
            "canonical_manifest_physical_sha256": "c" * 64,
            "canonical_used_component_set_sha256": hashlib.sha256(
                canonical_json_bytes(components)
            ).hexdigest(),
        },
    }
    _validate_analysis_bindings(manifest, report)
    manifest["input_bindings"]["canonical_used_component_set_sha256"] = "0" * 64
    with pytest.raises(ExternalAdmissionError, match="code or input binding changed"):
        _validate_analysis_bindings(manifest, report)


def test_output_inventory_path_traversal_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    root.mkdir()
    entries = [
        {
            "path": "../escape.json",
            "artifact_role": "machine_readable_report",
            "bytes": 0,
            "sha256": "0" * 64,
        }
    ]
    _write_identified(
        root / "external_admission_analysis_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "decision": "not_ready_for_external_canonical_admission_or_model_training",
            "report_internal_sha256": "0" * 64,
            "output_inventory": {
                "entries": entries,
                "entries_sha256": hashlib.sha256(canonical_json_bytes(entries)).hexdigest(),
                "entry_count": 1,
                "total_bytes": 0,
                "excluded_paths": ["external_admission_analysis_manifest.json"],
            },
            "external_canonical_observations_admitted": 0,
            "external_model_labels_admitted": 0,
            "substantive_model_training_performed": False,
            "substantive_model_training_authorized": False,
        },
    )
    with pytest.raises(ExternalAdmissionError, match="Unsafe analysis artifact path"):
        verify_external_admission_analysis(root)


def test_output_verifier_rejects_nonzero_label_boundary(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    root.mkdir()
    _write_identified(
        root / "external_admission_analysis_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "decision": "invalid",
            "report_internal_sha256": "0" * 64,
            "output_inventory": {
                "entries": [],
                "entries_sha256": hashlib.sha256(canonical_json_bytes([])).hexdigest(),
                "entry_count": 0,
                "total_bytes": 0,
                "excluded_paths": ["external_admission_analysis_manifest.json"],
            },
            "external_canonical_observations_admitted": 0,
            "external_model_labels_admitted": 1,
            "substantive_model_training_performed": False,
            "substantive_model_training_authorized": False,
        },
    )
    with pytest.raises(ExternalAdmissionError, match="zero-label"):
        verify_external_admission_analysis(root)
