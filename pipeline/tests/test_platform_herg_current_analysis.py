from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from menin_discovery.platform_herg_current_analysis import (
    MANIFEST_NAME,
    OUTPUTS,
    REPORT_NAME,
    HergCurrentAnalysisError,
    build_herg_current_analysis,
    main,
    validate_herg_current_analysis,
)


def _write(path: Path, data: dict[str, list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(data), path)


def _task_base(structures: list[str], smiles: list[str], task_id: str) -> dict[str, list[object]]:
    n = len(structures)
    return {
        "task_id": list[object]([task_id] * n),
        "structure_id": list[object](structures),
        "standardized_smiles": list[object](smiles),
        "target_class": [None] * n,
        "target_pic50": [None] * n,
        "target_relation": [None] * n,
        "source_family": ["test_source"] * n,
        "assay_id": [None] * n,
        "eligible": [True] * n,
        "model_split": list[object](["train", "train", "validation", "test", "test", "validation"][:n]),
        "scaffold_group_id": list[object]([f"G{i}" for i in range(n)]),
    }


def _inputs(root: Path) -> tuple[Path, Path, Path]:
    hierarchy = root / "hierarchy"
    quality = root / "quality"
    modality = root / "modality"
    structures = ["S1", "S2", "S3", "S4", "S5", "S6"]
    smiles = ["CCO", "c1ccccc1", "CCN", "CC(=O)O", "CCCC", "c1ccncc1"]

    _write(
        hierarchy / "observation_ledger.parquet",
        {
            "observation_id": ["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8"],
            "structure_id": ["S1", "S1", "S2", "S3", "S4", "S5", "S6", "S6"],
            "source_family": [
                "screen",
                "functional",
                "screen",
                "screen",
                "screen",
                "screen",
                "screen",
                "functional",
            ],
            "derived_binary_label": [0, 1, 0, 1, 0, 0, 1, 1],
            "pic50_value": [4.0, 5.0, None, 6.0, None, None, 7.0, 7.4],
            "native_endpoint": [
                "activity_outcome",
                "IC50",
                "activity_outcome",
                "pIC50",
                "activity_outcome",
                "activity_outcome",
                "pIC50",
                "IC50",
            ],
        },
    )
    _write(
        modality / "herg_measurement_modality_index.parquet",
        {
            "observation_id": ["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8"],
            "structure_id": ["S1", "S1", "S2", "S3", "S4", "S5", "S6", "S6"],
            "source_family": [
                "screen",
                "functional",
                "screen",
                "screen",
                "screen",
                "screen",
                "screen",
                "functional",
            ],
            "native_endpoint": [
                "activity_outcome",
                "IC50",
                "activity_outcome",
                "pIC50",
                "activity_outcome",
                "activity_outcome",
                "pIC50",
                "IC50",
            ],
            "measurement_modality": ["flux", "patch", "flux", "patch", "flux", "flux", "patch", "patch"],
            "automation_class": [
                "automated",
                "manual",
                "automated",
                "manual",
                "automated",
                "automated",
                "manual",
                "manual",
            ],
            "dose_design": [
                "fixed",
                "response",
                "fixed",
                "response",
                "fixed",
                "fixed",
                "response",
                "response",
            ],
            "wild_type_evidence_scope": ["confirmed_wild_type"] * 8,
            "modality_confidence": ["high"] * 8,
            "automation_confidence": ["high"] * 8,
            "dose_design_confidence": ["high"] * 8,
        },
    )

    q0 = _task_base(structures, smiles, "Q0")
    q0["target_class"] = [0, 0, 1, 0, 0, 1]
    _write(quality / "q0_weak_fixed_dose_binary.parquet", q0)

    q1 = _task_base(["S1", "S1", "S6"], ["CCO", "CCO", "c1ccncc1"], "Q1")
    q1["target_pic50"] = [4.0, 5.0, 7.4]
    q1["target_relation"] = ["=", "=", "="]
    q1["source_family"] = ["screen", "functional", "functional"]
    q1["assay_id"] = ["A1", "A2", "A3"]
    _write(quality / "q1_quantitative_pic50.parquet", q1)

    q2 = _task_base(["S1", "S6"], ["CCO", "c1ccncc1"], "Q2")
    _write(quality / "q2_functional_assay_aware.parquet", q2)
    c0 = _task_base(["S1", "S5"], ["CCO", "CCCC"], "C0")
    _write(quality / "c0_clinical_development_context.parquet", c0)
    _write(
        quality / "c1_qt_context_endpoints.parquet",
        {
            "structure_id": ["S1", "S5"],
            "context_eligible": [True, True],
            "candidate_classification": ["interval", "event"],
            "nct_id": ["NCT1", "NCT2"],
            "model_split": ["train", "test"],
            "reported_numeric_value_count": [3, 2],
        },
    )
    return hierarchy, quality, modality


def test_build_validate_and_scientific_contract(tmp_path: Path) -> None:
    hierarchy, quality, modality = _inputs(tmp_path)
    output = tmp_path / "analysis"
    report = tmp_path / "report"
    manifest = build_herg_current_analysis(hierarchy, quality, modality, output, report)

    assert {path.name for path in output.iterdir()} == {*OUTPUTS, MANIFEST_NAME}
    assert manifest["counts"]["q0_eligible_structures"] == 6
    assert manifest["counts"]["exact_pic50_replicate_structures"] == 1
    assert not manifest["scientific_contract"]["causal_claims_established"]
    assert not manifest["scientific_contract"]["qt_used_as_herg_label"]
    assert pq.read_table(output / "q0_structure_descriptors.parquet").num_rows == 6
    assert pq.read_table(output / "descriptor_associations.parquet").num_rows == 11
    assert pq.read_table(output / "descriptor_interactions.parquet").num_rows >= 25
    assert pq.read_table(output / "measurement_disagreements.parquet").num_rows >= 2
    disagreement_axes = set(
        pq.read_table(output / "measurement_disagreements.parquet")["comparison_axis"].to_pylist()
    )
    assert "automation_class" in disagreement_axes
    text = (report / REPORT_NAME).read_text(encoding="utf-8")
    assert "does **not** establish" in text
    assert "design superiorities" in text
    assert validate_herg_current_analysis(output) == manifest
    assert main(["--output-root", str(output), "--validate-only"]) == 0


def test_determinism_noop_and_tamper_detection(tmp_path: Path) -> None:
    hierarchy, quality, modality = _inputs(tmp_path)
    output = tmp_path / "analysis"
    report = tmp_path / "report"
    first = build_herg_current_analysis(hierarchy, quality, modality, output, report)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    second = build_herg_current_analysis(hierarchy, quality, modality, output, report)
    assert first == second
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}

    descriptor = output / "q0_structure_descriptors.parquet"
    frame = pq.read_table(descriptor).to_pandas()
    frame.loc[0, "logp"] = 999.0
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), descriptor)
    with pytest.raises(HergCurrentAnalysisError, match="artifact digest mismatch"):
        validate_herg_current_analysis(output)


def test_manifest_contract_promotion_fails_closed(tmp_path: Path) -> None:
    hierarchy, quality, modality = _inputs(tmp_path)
    output = tmp_path / "analysis"
    build_herg_current_analysis(hierarchy, quality, modality, output, tmp_path / "report")
    path = output / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["scientific_contract"]["qt_used_as_herg_label"] = True
    body = dict(manifest)
    body.pop("manifest_sha256")
    manifest["manifest_sha256"] = (
        __import__("hashlib")
        .sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
        .hexdigest()
    )
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(HergCurrentAnalysisError, match="keep QT separate"):
        validate_herg_current_analysis(output)


def test_requires_complete_inputs(tmp_path: Path) -> None:
    hierarchy, quality, modality = _inputs(tmp_path)
    frame = pq.read_table(quality / "q0_weak_fixed_dose_binary.parquet").to_pandas()
    frame = frame.drop(columns=["scaffold_group_id"])
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), quality / "q0_weak_fixed_dose_binary.parquet"
    )
    with pytest.raises(HergCurrentAnalysisError, match="missing required columns"):
        build_herg_current_analysis(hierarchy, quality, modality, tmp_path / "analysis", tmp_path / "report")
