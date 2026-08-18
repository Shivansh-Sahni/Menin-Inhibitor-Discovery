from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from menin_discovery.platform_structure_metadata import (
    SIFTS_COLUMNS,
    StructureMetadataError,
    _coverage_rows,
    _safe_relative,
    _validate_runtime_bindings,
    build_acquisition_manifest,
    parse_entry_type_line,
    parse_sifts_header,
    verify_acquisition,
)


def _raw_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    root.mkdir(parents=True)
    sifts = root / "pdb_chain_uniprot.tsv.gz"
    with gzip.open(sifts, "wt", encoding="utf-8", newline="") as handle:
        handle.write("# 2026/08/03 - 16:01 | PDB: 31.26 | UniProt: 2026.03\n")
        handle.write("\t".join(SIFTS_COLUMNS) + "\n")
        handle.write("101m\tA\tP02185\t1\t154\t0\t153\t1\t154\n")
    entry_type = root / "pdb_entry_type.txt"
    entry_type.write_text("101m\tprot\tdiffraction\n", encoding="utf-8")
    (root / "pdb_chain_uniprot.http_headers.txt").write_text(
        f"HTTP/1.1 200 OK\nDate: fixture\nContent-Length: {sifts.stat().st_size}\nETag: sifts\n",
        encoding="utf-8",
    )
    (root / "pdb_entry_type.http_headers.txt").write_text(
        f"HTTP/2 200\nDate: fixture\nContent-Length: {entry_type.stat().st_size}\nETag: entry\n",
        encoding="utf-8",
    )
    return root


def test_release_header_is_exact_and_versioned() -> None:
    assert parse_sifts_header("# 2026/08/03 - 16:01 | PDB: 31.26 | UniProt: 2026.03\n") == {
        "date": "2026/08/03",
        "time": "16:01",
        "pdb_version": "31.26",
        "uniprot_version": "2026.03",
    }


@pytest.mark.parametrize(
    "line",
    [
        "2026/08/03 PDB 31.26",
        "# 2026/08/03 - 16:01 | PDB: 31.26",
        "# current | PDB: 31.26 | UniProt: 2026.03",
    ],
)
def test_release_header_drift_fails_closed(line: str) -> None:
    with pytest.raises(StructureMetadataError, match="release header"):
        parse_sifts_header(line)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("101m\tprot\tdiffraction\n", ("101m", "prot", "diffraction")),
        ("1abc\tprot-nuc\tNMR\n", ("1abc", "prot-nuc", "NMR")),
        ("2xyz\tprot\tEM\n", ("2xyz", "prot", "EM")),
    ],
)
def test_entry_type_parser(line: str, expected: tuple[str, str, str]) -> None:
    assert parse_entry_type_line(line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "bad\tprot\tdiffraction\n",
        "101m\tprot\tpredicted\n",
        "101m\tprot\n",
        "101m\tprot\tdiffraction\textra\n",
    ],
)
def test_entry_type_schema_and_prediction_drift_fail_closed(line: str) -> None:
    with pytest.raises(StructureMetadataError, match="entry-type"):
        parse_entry_type_line(line)


def test_acquisition_manifest_binds_exact_raw_bytes(tmp_path: Path) -> None:
    root = _raw_fixture(tmp_path)
    built = build_acquisition_manifest(root)
    assert built["coordinate_files_downloaded"] == 0
    assert built["predicted_structure_files_downloaded"] == 0
    assert built["model_labels_admitted"] == 0
    assert verify_acquisition(root)["manifest_sha256"] == built["manifest_sha256"]
    with (root / "pdb_entry_type.txt").open("a", encoding="utf-8") as handle:
        handle.write("102m\tprot\tdiffraction\n")
    with pytest.raises(StructureMetadataError, match="inventory identity"):
        verify_acquisition(root)


def test_acquisition_manifest_is_deterministic(tmp_path: Path) -> None:
    left = _raw_fixture(tmp_path / "left")
    right = _raw_fixture(tmp_path / "right")
    build_acquisition_manifest(left)
    build_acquisition_manifest(right)
    assert (left / "acquisition_manifest.json").read_bytes() == (
        right / "acquisition_manifest.json"
    ).read_bytes()


def test_runtime_binding_rejects_analyzer_or_acquisition_drift(tmp_path: Path) -> None:
    root = _raw_fixture(tmp_path)
    acquisition = build_acquisition_manifest(root)
    from menin_discovery import platform_structure_metadata as structure

    bindings = {
        "input_bindings": {
            "acquisition_manifest_physical_sha256": structure.sha256_file(root / "acquisition_manifest.json"),
            "acquisition_manifest_internal_sha256": acquisition["manifest_sha256"],
            "analyzer_code_sha256": structure.sha256_file(Path(structure.__file__).resolve()),
        }
    }
    _validate_runtime_bindings(bindings, acquisition, root)
    bindings["input_bindings"]["analyzer_code_sha256"] = "0" * 64
    with pytest.raises(StructureMetadataError, match="runtime input/code binding changed"):
        _validate_runtime_bindings(bindings, acquisition, root)


@pytest.mark.parametrize("value", ["../escape", "/absolute", "a/../../b", ""])
def test_output_path_traversal_is_rejected(value: str) -> None:
    with pytest.raises(StructureMetadataError, match="Unsafe"):
        _safe_relative(value, context="fixture")


def test_coverage_is_candidate_only_not_construct_or_sequence_version_claim() -> None:
    universe = [
        {
            "universe_kind": "canonical_chembl37_protein",
            "protein_id": "PROT-1",
            "uniprot_accession": "P12345",
            "sequence_sha256": "a" * 64,
            "sequence_length": 100,
            "accession_resolution_status": "resolved",
        }
    ]
    summaries = {
        "P12345": {
            "pdb": {"1abc"},
            "chains": {("1abc", "A")},
            "segments": 1,
            "methods": {"diffraction": {"1abc"}},
            "min": 10,
            "max": 59,
        }
    }
    row = list(_coverage_rows(universe, summaries))[0]
    assert row["pdb_entry_count"] == 1
    assert row["span_fraction_of_frozen_sequence"] == 0.5
    assert row["coverage_interpretation"] == "outer_span_proxy_not_observed_residue_coverage"
    assert row["construct_identity_verified"] is False
    assert row["sequence_version_verified"] is False
    assert row["predicted_structure_count"] == 0
    assert row["model_label_admitted"] is False


def test_outer_span_fraction_above_one_is_retained_as_warning_evidence() -> None:
    universe = [
        {
            "universe_kind": "canonical_chembl37_protein",
            "protein_id": "PROT-1",
            "uniprot_accession": "P12345",
            "sequence_sha256": "a" * 64,
            "sequence_length": 10,
            "accession_resolution_status": "resolved",
        }
    ]
    summaries = {
        "P12345": {
            "pdb": {"1abc"},
            "chains": {("1abc", "A")},
            "segments": 1,
            "methods": {"diffraction": {"1abc"}},
            "min": 1,
            "max": 15,
        }
    }
    row = list(_coverage_rows(universe, summaries))[0]
    assert row["span_fraction_of_frozen_sequence"] == 1.5
    assert row["sequence_version_verified"] is False


def test_manifest_is_valid_json_and_has_no_training_action(tmp_path: Path) -> None:
    root = _raw_fixture(tmp_path)
    build_acquisition_manifest(root)
    value = json.loads((root / "acquisition_manifest.json").read_text(encoding="utf-8"))
    assert value["substantive_model_training_performed"] is False
    assert "training_actions" not in value
