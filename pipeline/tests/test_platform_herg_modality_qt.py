from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import menin_discovery.platform_herg_modality_qt as modality_qt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from menin_discovery.platform_herg_modality_qt import (
    EXCLUSION_OUTPUT,
    MANIFEST_NAME,
    MODALITY_OUTPUT,
    QC_OUTPUT,
    QT_OUTPUT,
    HergModalityQtError,
    _classify_modality,
    build_herg_modality_qt,
    main,
    validate_herg_modality_qt,
)


def _parquet(path: Path, columns: Mapping[str, Sequence[object]]) -> None:
    pq.write_table(pa.table(columns), path)


def _inputs(tmp_path: Path, *, promoted_qt: bool = False) -> tuple[Path, Path]:
    hierarchy = tmp_path / "hierarchy"
    clinical = tmp_path / "clinical"
    hierarchy.mkdir(parents=True)
    clinical.mkdir(parents=True)
    (hierarchy / "manifest.json").write_text("{}\n", encoding="utf-8")
    (clinical / "herg_clinical_links_manifest.json").write_text("{}\n", encoding="utf-8")

    _parquet(
        hierarchy / "observation_ledger.parquet",
        {
            "observation_id": ["H1", "H2", "H3", "H4", "H5", "H6"],
            "structure_id": ["S1", "S2", "S3", "S4", "S5", "S6"],
            "source_family": [
                "pubchem_aid720551",
                "quantitative_pic50_release",
                "chembl_herg_specialized_view",
                "chembl_herg_specialized_view",
                "chembl_herg_specialized_view",
                "chembl_herg_specialized_view",
            ],
            "source_record_id": ["R1", "R2", "R3", "R4", "R5", "R6"],
            "assay_id": ["AID720551", None, "C3", "C4", "C5", "C6"],
            "target_variant": [
                "wild_type",
                "wild_type_or_unspecified",
                "mutant_or_variant",
                "wild_type_or_unspecified",
                "wild_type_or_unspecified",
                "wild_type_or_unspecified",
            ],
            "assay_family": [
                "source_reported_qhts",
                "mixed_unresolved_compilation",
                "binding",
                "functional",
                "binding",
                "other",
            ],
            "native_endpoint": ["activity_outcome", "pIC50", "IC50", "IC50", "Ki", "Inhibition"],
            "native_label": ["Active", None, None, None, None, None],
            "native_aux_json": [
                json.dumps({"assay_name": "qHTS for wild-type KCNH2"}),
                json.dumps({"reported_source": "PubChem"}),
                json.dumps({"assay_description": "T623S mutant by patch clamp"}),
                json.dumps({"assay_description": "hERG inhibition by manual patch clamp assay"}),
                json.dumps({"assay_description": "Displacement of [3H]dofetilide from hERG"}),
                json.dumps({"assay_description": "Inhibition of human hERG"}),
            ],
        },
    )
    values = [
        json.dumps([{"value": "12.4", "group_id": "G1"}]),
        json.dumps([{"value": "3", "class_title": "QTcF >500 msec"}]),
    ]
    denominators = [
        json.dumps([{"value": "20", "group_id": "G1"}]),
        json.dumps([{"value": "40", "group_id": "G1"}]),
    ]
    _parquet(
        clinical / "t3_posted_qt_trial_result_candidates.parquet",
        {
            "candidate_id": ["Q1", "Q2"],
            "molecule_id": ["S1", "S2"],
            "nct_id": ["NCT1", "NCT2"],
            "endpoint_candidate_id": ["E1", "E2"],
            "candidate_classification": [
                "qt_qtc_interval_measure_candidate",
                "qt_qtc_event_or_threshold_candidate",
            ],
            "title_or_term": ["Change From Baseline in QTcF", "Participants With QTcB >500 msec"],
            "description_or_organ_system": ["Fridericia corrected interval", "Bazett threshold"],
            "unit_of_measure": ["msec", "Participants"],
            "time_frame": ["Day 1", "Week 4"],
            "value_records_json": values,
            "denominator_records_json": denominators,
            "reported_numeric_value_count": [1, 1],
            "candidate_rule_passed": [True, True],
            "clinical_herg_label_admitted": [False, False],
            "model_label_admitted": [promoted_qt, False],
            "source_page_path": ["page1.json", "page2.json"],
            "raw_json_pointer": ["/studies/0", "/studies/1"],
        },
    )
    return hierarchy, clinical


def _disable_upstream_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(modality_qt, "validate_herg_hierarchy", lambda _root: {})
    monkeypatch.setattr(modality_qt, "verify_herg_clinical_links", lambda _root: {})


def test_builds_wild_type_method_ontology_and_separate_qt_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hierarchy, clinical = _inputs(tmp_path)
    _disable_upstream_validation(monkeypatch)
    output = tmp_path / "output"
    manifest = build_herg_modality_qt(hierarchy, clinical, output)

    methods = {row["observation_id"]: row for row in pq.read_table(output / MODALITY_OUTPUT).to_pylist()}
    assert set(methods) == {"H1", "H2", "H4", "H5", "H6"}
    assert methods["H1"]["measurement_modality"] == "high_throughput_thallium_flux"
    assert methods["H1"]["automation_class"] == "automated"
    assert methods["H1"]["dose_design"] == "fixed_dose_categorical"
    assert methods["H2"]["wild_type_evidence_scope"] == "wild_type_or_unspecified"
    assert methods["H2"]["measurement_modality"] == "unresolved"
    assert methods["H4"]["measurement_modality"] == "patch_clamp_electrophysiology"
    assert methods["H4"]["automation_class"] == "manual"
    assert methods["H5"]["measurement_modality"] == "radioligand_binding"
    assert methods["H6"]["measurement_modality"] == "unresolved"
    assert methods["H6"]["automation_class"] == "unresolved"

    exclusions = pq.read_table(output / EXCLUSION_OUTPUT).to_pylist()
    assert [row["observation_id"] for row in exclusions] == ["H3"]
    assert exclusions[0]["exclusion_reason"] == "explicit_mutant_or_variant_outside_wild_type_scope"

    qt = {row["candidate_id"]: row for row in pq.read_table(output / QT_OUTPUT).to_pylist()}
    assert qt["Q1"]["qt_phenotype_class"] == "interval_measurement"
    assert json.loads(qt["Q1"]["correction_methods_json"]) == ["QTcF"]
    assert qt["Q1"]["qt_metric_semantics"] == "qt_or_qtc_change_from_baseline"
    assert qt["Q2"]["qt_phenotype_class"] == "event_or_threshold"
    assert json.loads(qt["Q2"]["correction_methods_json"]) == ["QTcB"]
    assert not any(row["herg_potency_derived"] or row["qt_used_as_herg_label"] for row in qt.values())

    assert pq.read_table(output / QC_OUTPUT).num_rows >= 8
    assert manifest["counts"]["explicit_mutant_exclusions"] == 1
    assert manifest["counts"]["wild_type_evidence_scope"] == {
        "confirmed_wild_type": 1,
        "wild_type_or_unspecified": 4,
    }
    assert validate_herg_modality_qt(output)["manifest_sha256"] == manifest["manifest_sha256"]
    assert main(["--output-root", str(output), "--validate-only"]) == 0
    # Exact reruns are deterministic no-ops and do not overwrite the build.
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    rerun = build_herg_modality_qt(hierarchy, clinical, output)
    assert rerun == manifest
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_determinism_tamper_and_qt_label_promotion_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hierarchy, clinical = _inputs(tmp_path)
    _disable_upstream_validation(monkeypatch)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = build_herg_modality_qt(hierarchy, clinical, first)
    second_manifest = build_herg_modality_qt(hierarchy, clinical, second)
    assert first_manifest == second_manifest
    for name in (MANIFEST_NAME, MODALITY_OUTPUT, EXCLUSION_OUTPUT, QT_OUTPUT, QC_OUTPUT):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    manifest_path = first / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scientific_contract"]["qt_is_herg_potency"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(HergModalityQtError, match="manifest digest mismatch"):
        validate_herg_modality_qt(first)

    promoted_root = tmp_path / "promoted"
    promoted_root.mkdir()
    promoted_hierarchy, promoted_clinical = _inputs(promoted_root, promoted_qt=True)
    with pytest.raises(HergModalityQtError, match="promoted into a hERG/model label"):
        build_herg_modality_qt(promoted_hierarchy, promoted_clinical, tmp_path / "bad")


def test_qt_modality_requires_curated_assay_or_native_endpoint() -> None:
    legitimate = _classify_modality(
        "chembl_herg_specialized_view",
        "binding",
        "hERG inhibition relevant to QT prolongation",
        "CHEMBL3706049",
        "IC50",
    )
    assert legitimate[0] == "binding_unspecified"
    phenotype = _classify_modality(
        "chembl_herg_specialized_view",
        "functional",
        "Effect on prolongation of heart rate-corrected QT interval",
        "CHEMBL820994",
        "EC10",
    )
    assert phenotype[0] == "clinical_qt_in_vivo"


def test_unknown_target_scope_and_inconsistent_numeric_qt_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hierarchy, clinical = _inputs(tmp_path)
    _disable_upstream_validation(monkeypatch)
    table = pq.read_table(hierarchy / "observation_ledger.parquet").to_pydict()
    table["target_variant"][0] = "probably_wild_type"
    _parquet(hierarchy / "observation_ledger.parquet", table)
    with pytest.raises(HergModalityQtError, match="unknown target_variant"):
        build_herg_modality_qt(hierarchy, clinical, tmp_path / "unknown")

    hierarchy2, clinical2 = _inputs(tmp_path / "second_case")
    qt = pq.read_table(clinical2 / "t3_posted_qt_trial_result_candidates.parquet").to_pydict()
    qt["reported_numeric_value_count"][0] = 2
    _parquet(clinical2 / "t3_posted_qt_trial_result_candidates.parquet", qt)
    with pytest.raises(HergModalityQtError, match="numeric-result count"):
        build_herg_modality_qt(hierarchy2, clinical2, tmp_path / "bad_numeric")
