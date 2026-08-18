from __future__ import annotations

from pathlib import Path

import pytest
from menin_discovery.platform_herg_training_surfaces import (
    SURFACE_APPROXIMATE,
    SURFACE_CLEAN,
    SURFACE_CONFIRMED_WT,
    SURFACE_FUNCTIONAL_NUMERIC,
    SURFACE_PIC50,
    HergTrainingSurfaceError,
    _implementation_binding,
    _is_clinical_qt,
    _label_decision,
    _lineage_annotations,
    _surface_membership,
    _validate_implementation_binding,
)


def _row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source_family": "chembl_herg_specialized_view",
        "structure_id": "S1",
        "structure_model_eligible": True,
        "target_variant": "wild_type_or_unspecified",
        "measurement_modality": "patch_clamp_electrophysiology",
        "endpoint_class": "potency_ic50",
        "native_relation": "=",
        "native_value": 10.0,
        "native_unit": "nM",
        "derived_binary_label": None,
    }
    row.update(updates)
    return row


def test_primary_native_numeric_preserves_relation_value_and_unit() -> None:
    decision = _label_decision(_row(native_relation=">", native_value=100.0), {})
    assert decision == {
        "kind": "native_numeric_relation_preserved",
        "binary": None,
        "numeric": 100.0,
        "relation": ">",
        "unit": "nM",
        "primary": True,
        "sensitivity": True,
        "reason": "native_numeric_endpoint_relation_and_unit_complete",
        "q0_consensus": False,
        "q0_label": None,
    }


def test_approximate_numeric_is_sensitivity_only() -> None:
    decision = _label_decision(_row(native_relation="~"), {})
    assert decision["kind"] == "native_numeric_approximate_sensitivity"
    assert decision["primary"] is False
    assert decision["sensitivity"] is True
    assert decision["relation"] == "~"


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"native_unit": None}, "native_numeric_unit_missing_or_unresolved"),
        ({"native_relation": None}, "native_numeric_relation_missing_or_unresolved"),
        ({"native_value": None}, "native_numeric_value_missing_or_nonfinite"),
        ({"structure_model_eligible": False}, "invalid_or_missing_standardized_structure"),
    ],
)
def test_incomplete_numeric_rows_are_not_primary(updates: dict[str, object], reason: str) -> None:
    decision = _label_decision(_row(**updates), {})
    assert decision["primary"] is False
    assert decision["sensitivity"] is False
    assert decision["reason"] == reason


def test_fixed_dose_requires_structure_consensus_and_never_becomes_ic50() -> None:
    row = _row(
        source_family="pubchem_aid720551",
        target_variant="wild_type",
        derived_binary_label=1,
        native_value=None,
        native_unit=None,
        native_relation=None,
        endpoint_class="categorical_activity_call",
        measurement_modality="high_throughput_thallium_flux",
    )
    eligible = _label_decision(row, {"S1": 1})
    assert eligible["kind"] == "fixed_dose_binary"
    assert eligible["binary"] == 1
    assert eligible["numeric"] is None
    assert eligible["primary"] is True

    conflicting = _label_decision(row, {"S1": 0})
    assert conflicting["kind"] == "none"
    assert conflicting["primary"] is False
    assert conflicting["reason"] == "fixed_dose_structure_lacks_unique_consensus_label"


def test_clinical_qt_is_context_only_even_when_numeric() -> None:
    row = _row(measurement_modality="clinical_qt_in_vivo")
    assert _is_clinical_qt(row) is True
    decision = _label_decision(row, {})
    assert decision["kind"] == "clinical_context_not_label"
    assert decision["primary"] is False
    assert decision["sensitivity"] is False


def test_explicit_mutant_is_fail_closed() -> None:
    with pytest.raises(HergTrainingSurfaceError, match="explicit mutant"):
        _label_decision(_row(target_variant="mutant_or_variant"), {})


def test_lineage_annotations_are_sorted_and_take_maximum_strength() -> None:
    rows = [
        {
            "lineage_group_id": "LG2",
            "automated_evidence_strength": "weak",
            "observation_ids_json": '["O1","O2"]',
        },
        {
            "lineage_group_id": "LG1",
            "automated_evidence_strength": "strong",
            "observation_ids_json": '["O1"]',
        },
    ]
    annotations = _lineage_annotations(rows)
    assert annotations["O1"] == {"ids": ["LG1", "LG2"], "maximum": "strong"}
    assert annotations["O2"] == {"ids": ["LG2"], "maximum": "weak"}


def test_nested_surface_membership_uses_explicit_flags() -> None:
    row = {
        "primary_training_eligible": True,
        "confirmed_wt_fixed_dose_primary": False,
        "preclinical_native_numeric_primary": True,
        "standardized_pic50_primary": True,
        "functional_how_measured_numeric_primary": True,
        "functional_how_measured_pic50_primary": True,
        "approximate_relation_sensitivity_only": False,
        "curated_functional_t1_review_candidate": False,
    }
    assert _surface_membership(row, SURFACE_CLEAN)
    assert _surface_membership(row, SURFACE_PIC50)
    assert _surface_membership(row, SURFACE_FUNCTIONAL_NUMERIC)
    assert not _surface_membership(row, SURFACE_CONFIRMED_WT)
    assert not _surface_membership(row, SURFACE_APPROXIMATE)


def test_implementation_binding_rejects_source_tampering(tmp_path: Path) -> None:
    source = tmp_path / "release_builder.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    manifest = {"implementation": _implementation_binding(source)}
    assert _validate_implementation_binding(manifest) == source.resolve()

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(HergTrainingSurfaceError, match="implementation binding mismatch"):
        _validate_implementation_binding(manifest)
