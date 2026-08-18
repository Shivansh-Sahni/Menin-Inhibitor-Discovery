from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import menin_discovery.platform_herg_operational_tiers as operational
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from menin_discovery.platform_herg_operational_tiers import (
    MANIFEST_NAME,
    O0,
    O1,
    O2,
    O3,
    O3_QT,
    QT_RECORD_OUTPUT,
    RECORD_OUTPUT,
    SUMMARY_OUTPUT,
    HergOperationalTierError,
    build_herg_operational_tiers,
    main,
    validate_herg_operational_tiers,
)


def _parquet(path: Path, columns: Mapping[str, Sequence[object]]) -> None:
    pq.write_table(pa.table(columns), path)


def _inputs(tmp_path: Path, *, promoted_qt: bool = False) -> tuple[Path, Path]:
    hierarchy = tmp_path / "hierarchy"
    clinical = tmp_path / "clinical"
    hierarchy.mkdir()
    clinical.mkdir()
    (hierarchy / "manifest.json").write_text("{}\n", encoding="utf-8")
    (clinical / "herg_clinical_links_manifest.json").write_text("{}\n", encoding="utf-8")

    size = 1_000
    structures = [f"HSTR-{index:04d}" for index in range(size)]
    observations = [f"HOBS-{index:04d}" for index in range(size)]
    _parquet(
        hierarchy / "observation_ledger.parquet",
        {
            "observation_id": observations,
            "source_family": ["quantitative_pic50_release"] * size,
            "source_record_id": [f"SRC-{index:04d}" for index in range(size)],
            "structure_id": structures,
            "structure_valid": [True] * size,
            "assay_id": [None] * size,
            "native_value": [5.0] * size,
            "pic50_value": [5.0] * size,
            "derived_binary_label": [1.0] * size,
        },
    )
    _parquet(
        clinical / "structure_development_annotations.parquet",
        {
            "molecule_id": structures,
            "clinical_development_annotation": [True] * size,
            "clinical_cardiac_label_admitted": [False] * size,
            "model_label_admitted": [False] * size,
            "chembl_max_phase": [2.0] * size,
            "chembl_first_approval": [2000] * size,
            "drugsfda_application_numbers_json": [json.dumps([f"NDA{index:06d}"]) for index in range(size)],
        },
    )
    _parquet(
        clinical / "exact_name_structure_link_audit.parquet",
        {
            "source_kind": ["clinicaltrials_intervention"] * size + ["drugsfda_ingredient"] * size,
            "source_record_id": [f"CTI-{index:04d}" for index in range(size)]
            + [f"FDA-{index:04d}" for index in range(size)],
            "nct_id": [f"NCT{index:08d}" for index in range(size)] + [None] * size,
            "application_number": [None] * size + [f"NDA{index:06d}" for index in range(size)],
            "product_number": [None] * size + ["001"] * size,
            "linked_molecule_id": structures + structures,
            "source_record_dummy": [""] * (size * 2),
            "link_is_exact_and_unique": [True] * (size * 2),
            "model_label_admitted": [False] * (size * 2),
        },
    )
    _parquet(
        clinical / "t3_posted_qt_trial_result_candidates.parquet",
        {
            "candidate_id": [f"QT-{index:04d}" for index in range(size)],
            "molecule_id": structures,
            "nct_id": [f"NCT{index:08d}" for index in range(size)],
            "endpoint_candidate_id": [f"END-{index:04d}" for index in range(size)],
            "candidate_rule_passed": [True] * size,
            "clinical_herg_label_admitted": [False] * size,
            "model_label_admitted": [promoted_qt] + [False] * (size - 1),
            "reported_numeric_value_count": [1] * size,
            "value_records_json": [json.dumps([{"value": "4.2"}])] * size,
            "denominator_records_json": [json.dumps([{"value": "20"}])] * size,
            "source_page_path": ["page.json"] * size,
            "raw_json_pointer": ["/results/0"] * size,
        },
    )
    return hierarchy, clinical


def _disable_fixture_upstream_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(operational, "validate_herg_hierarchy", lambda _root: {})
    monkeypatch.setattr(operational, "verify_herg_clinical_links", lambda _root: {})


def test_builds_grain_explicit_indexes_and_preserves_disclosures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hierarchy, clinical = _inputs(tmp_path)
    _disable_fixture_upstream_verification(monkeypatch)
    output = tmp_path / "operational"
    manifest = build_herg_operational_tiers(hierarchy, clinical, output)

    summaries = {row["operational_stage"]: row for row in pq.read_table(output / SUMMARY_OUTPUT).to_pylist()}
    assert summaries[O0]["headline_record_grain"] == "observation"
    assert summaries[O1]["headline_record_grain"] == "observation"
    assert summaries[O2]["headline_record_grain"] == "structure"
    assert summaries[O3]["headline_record_grain"] == "intervention_link"
    assert summaries[O3_QT]["headline_record_grain"] == "result_value"
    assert {row["headline_record_count"] for row in summaries.values()} == {1_000}
    assert summaries[O3]["unique_trials"] == 1_000
    assert summaries[O3_QT]["indexed_record_count"] == 2_000

    records = pq.read_table(output / RECORD_OUTPUT).to_pylist()
    clinical_records = [row for row in records if row["operational_stage"] in {O2, O3}]
    assert len(records) == 4_000
    assert not any(row["clinical_context_used_as_herg_label"] for row in clinical_records)
    assert not any(row["model_label_admitted_from_context"] for row in clinical_records)

    qt = pq.read_table(output / QT_RECORD_OUTPUT).to_pylist()
    assert sum(row["record_grain"] == "result_value" for row in qt) == 1_000
    assert sum(row["record_grain"] == "denominator" for row in qt) == 1_000
    assert manifest["disclosed_counts"][O3_QT]["numeric_result_values"] == 1_000
    assert validate_herg_operational_tiers(output)["manifest_sha256"] == manifest["manifest_sha256"]
    assert main(["--output-root", str(output), "--validate-only"]) == 0


def test_determinism_hash_tamper_and_clinical_label_promotion_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hierarchy, clinical = _inputs(tmp_path)
    _disable_fixture_upstream_verification(monkeypatch)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = build_herg_operational_tiers(hierarchy, clinical, first)
    second_manifest = build_herg_operational_tiers(hierarchy, clinical, second)
    assert first_manifest == second_manifest
    for name in (MANIFEST_NAME, RECORD_OUTPUT, QT_RECORD_OUTPUT, SUMMARY_OUTPUT):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    manifest_path = first / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["minimum_headline_record_count"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(HergOperationalTierError, match="manifest digest mismatch"):
        validate_herg_operational_tiers(first)

    promoted_root = tmp_path / "promoted_inputs"
    promoted_root.mkdir()
    promoted_hierarchy, promoted_clinical = _inputs(promoted_root, promoted_qt=True)
    with pytest.raises(HergOperationalTierError, match="posted QT context was promoted"):
        build_herg_operational_tiers(promoted_hierarchy, promoted_clinical, tmp_path / "promoted_output")
