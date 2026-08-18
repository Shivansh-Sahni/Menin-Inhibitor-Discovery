from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from menin_discovery import platform_affinity_training_surfaces as affinity_module
from menin_discovery.platform_affinity_training_surfaces import (
    _BDB_REQUIRED_COLUMNS,
    _CHEMBL_COLUMNS,
    ENDPOINTS,
    AffinityTrainingSurfaceError,
    _initialize_work_database,
    _normalize_accessions,
    _parse_bindingdb_value,
    _register_target,
    _resolve_structure,
    _resolve_target_groups,
    _sha256,
    _split,
    _UnionFind,
    build_affinity_training_surfaces,
    validate_affinity_training_surfaces,
)

_STRING_COLUMNS = set(_CHEMBL_COLUMNS) - {
    "label_value",
    "label_lower_bound",
    "label_upper_bound",
    "document_year",
    "default_task_eligible",
}
_CHEMBL_FIXTURE_SCHEMA = pa.schema(
    [
        pa.field(
            column,
            pa.large_string()
            if column in _STRING_COLUMNS
            else pa.bool_()
            if column == "default_task_eligible"
            else pa.int64()
            if column == "document_year"
            else pa.float64(),
        )
        for column in _CHEMBL_COLUMNS
    ]
)


def _chembl_row(
    *,
    observation_id: str,
    endpoint: str,
    relation: str,
    value: float | None,
    lower: float | None,
    upper: float | None,
    smiles: str,
    accession: str,
    sequence: str,
    doi: str,
) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in _CHEMBL_COLUMNS}
    row.update(
        {
            "observation_id": observation_id,
            "source_record_id": f"ChEMBL:activity:{observation_id}",
            "source_id": "SRC-CHEMBL",
            "molecule_id": f"MOL-{observation_id}",
            "structure_id": f"OLD-{observation_id}",
            "standardized_smiles": smiles,
            "canonical_target_id": accession,
            "protein_id": f"PROT-{accession}",
            "sequence": sequence,
            "target_name": f"target {accession}",
            "species": "Homo sapiens",
            "endpoint": endpoint,
            "label_relation": relation,
            "label_value": value,
            "label_lower_bound": lower,
            "label_upper_bound": upper,
            "label_unit": "nM",
            "label_text": "",
            "label_kind": "continuous_exact" if relation == "=" else "continuous_censored",
            "assay_id": f"ASSAY-{observation_id}",
            "assay_family": "binding",
            "description": "fixture assay",
            "matrix": "",
            "route": "",
            "document_doi": doi,
            "document_pubmed_id": "",
            "document_patent_id": "",
            "document_year": 2025,
            "default_task_eligible": True,
            "inclusion_status": "included",
        }
    )
    return row


def _write_chembl_fixture(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "build_manifest.json").write_text('{"fixture":true}\n')
    rows_by_task: dict[tuple[str, str], list[dict[str, object]]] = {
        ("Kd", "exact"): [
            _chembl_row(
                observation_id="1",
                endpoint="Kd",
                relation="=",
                value=10.0,
                lower=10.0,
                upper=10.0,
                smiles="CCO",
                accession="P00001",
                sequence="AAA",
                doi="10.1000/shared",
            )
        ],
        ("IC50", "censored"): [
            _chembl_row(
                observation_id="2",
                endpoint="IC50",
                relation="interval",
                value=None,
                lower=7.0,
                upper=9.0,
                smiles="CCCl",
                accession="P00002",
                sequence="BBB",
                doi="10.1000/interval",
            )
        ],
    }
    task_datasets: dict[str, object] = {}
    for endpoint in ENDPOINTS:
        for kind in ("exact", "censored"):
            key = f"default::default__binding__{endpoint.casefold()}__binding__nm__continuous_{kind}"
            directory = root / "tasks" / key.replace("default::", "").replace("__", "_")
            directory.mkdir(parents=True)
            path = directory / "part-00000.parquet"
            rows = rows_by_task.get((endpoint, kind), [])
            pq.write_table(pa.Table.from_pylist(rows, schema=_CHEMBL_FIXTURE_SCHEMA), path)
            task_datasets[key] = {
                "row_count": len(rows),
                "part_count": 1,
                "parts": [
                    {
                        "path": path.relative_to(root).as_posix(),
                        "rows": len(rows),
                        "sha256": _sha256(path),
                        "arrow_schema_sha256": "fixture-source-schema",
                    }
                ],
            }
    (root / "task_datasets.json").write_text(json.dumps(task_datasets, indent=2, sort_keys=True) + "\n")


def _bindingdb_row(**values: str) -> dict[str, str]:
    row = {name: "" for name in _BDB_REQUIRED_COLUMNS.values()}
    row.update(
        {
            _BDB_REQUIRED_COLUMNS["chain_count"]: "1",
            _BDB_REQUIRED_COLUMNS["organism"]: "Homo sapiens",
            _BDB_REQUIRED_COLUMNS["publication_date"]: "2025",
            _BDB_REQUIRED_COLUMNS["source"]: "Curated from the literature by BindingDB",
        }
    )
    row.update(values)
    return row


def _write_bindingdb_fixture(root: Path) -> tuple[Path, Path]:
    header = list(_BDB_REQUIRED_COLUMNS.values())
    c = _BDB_REQUIRED_COLUMNS
    rows = [
        _bindingdb_row(
            **{
                c["reactant_set"]: "1",
                c["smiles"]: "CCO",
                c["target_name"]: "target P00001",
                c["sequence"]: "AAA",
                c["swissprot"]: "P00001",
                c["Kd"]: "10",
                c["doi"]: "10.1000/shared",
                c["source"]: "ChEMBL",
            }
        ),
        _bindingdb_row(
            **{
                c["reactant_set"]: "2",
                c["smiles"]: "OCC",
                c["target_name"]: "target P00001",
                c["sequence"]: "AAA",
                c["swissprot"]: "P00001",
                c["Kd"]: "10",
                c["doi"]: "https://doi.org/10.1000/shared",
            }
        ),
        _bindingdb_row(
            **{
                c["reactant_set"]: "3",
                c["smiles"]: "CCO",
                c["target_name"]: "target P00001",
                c["sequence"]: "AAA",
                c["swissprot"]: "P00001",
                c["Ki"]: ">20",
                c["doi"]: "10.1000/ki",
            }
        ),
        _bindingdb_row(
            **{
                c["reactant_set"]: "4",
                c["smiles"]: "CCN",
                c["target_name"]: "target P00002",
                c["sequence"]: "BBB",
                c["swissprot"]: "P00002",
                c["IC50"]: "<100",
                c["doi"]: "10.1000/ic50",
            }
        ),
        _bindingdb_row(
            **{
                c["reactant_set"]: "5",
                c["smiles"]: "CCC",
                c["target_name"]: "target P00003",
                c["sequence"]: "CCC",
                c["swissprot"]: "P00003",
                c["EC50"]: "50",
                c["doi"]: "10.1000/ec50",
            }
        ),
        _bindingdb_row(
            **{
                c["reactant_set"]: "6",
                c["smiles"]: "CCN",
                c["target_name"]: "target P00002",
                c["sequence"]: "BBB",
                c["swissprot"]: "P00002",
                c["IC50"]: "<100",
                c["doi"]: "10.1000/ic50",
            }
        ),
        _bindingdb_row(
            **{
                c["reactant_set"]: "7",
                c["smiles"]: "CCCC",
                c["target_name"]: "target P00004",
                c["sequence"]: "DDD",
                c["swissprot"]: "P00004",
                c["Kd"]: "5",
                c["doi"]: "10.1000/rights",
                c["source"]: "Taylor Research Group, UCSD",
            }
        ),
        _bindingdb_row(
            **{
                c["reactant_set"]: "8",
                c["smiles"]: "not-a-smiles",
                c["target_name"]: "target P00003",
                c["sequence"]: "CCC",
                c["swissprot"]: "P00003",
                c["IC50"]: "4",
                c["doi"]: "10.1000/invalid",
            }
        ),
        _bindingdb_row(
            **{
                c["reactant_set"]: "9",
                c["smiles"]: "CCF",
                c["target_name"]: "complex",
                c["sequence"]: "EEE",
                c["swissprot"]: "P00005",
                c["IC50"]: "4",
                c["doi"]: "10.1000/complex",
                c["chain_count"]: "2",
            }
        ),
        _bindingdb_row(
            **{
                c["reactant_set"]: "10",
                c["smiles"]: "CCBr",
                c["target_name"]: "target P00006",
                c["sequence"]: "FFF",
                c["swissprot"]: "P00006",
                c["Ki"]: "~5",
                c["doi"]: "10.1000/approximate",
            }
        ),
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=header, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    archive_path = root / "BindingDB_All_202608_tsv.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BindingDB_All_202608.tsv", buffer.getvalue())
    acquisition_path = root / "acquisition_manifest.json"
    acquisition_path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "source": "BindingDB",
                        "path": archive_path.name,
                        "bytes": archive_path.stat().st_size,
                        "sha256": _sha256(archive_path),
                        "checksum_verified": True,
                    }
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return archive_path, acquisition_path


@pytest.fixture
def affinity_release(tmp_path: Path) -> tuple[Path, Path]:
    chembl_root = tmp_path / "chembl"
    _write_chembl_fixture(chembl_root)
    archive_path, acquisition_path = _write_bindingdb_fixture(tmp_path)
    output_root = tmp_path / "release"
    build_affinity_training_surfaces(
        canonical_chembl_root=chembl_root,
        bindingdb_archive=archive_path,
        acquisition_manifest=acquisition_path,
        output_root=output_root,
    )
    return output_root, acquisition_path


def test_bindingdb_value_parser_preserves_censoring() -> None:
    assert _parse_bindingdb_value("12.5") == ("=", 12.5, 12.5, 12.5)
    assert _parse_bindingdb_value("<= 12.5") == ("<=", 12.5, None, 12.5)
    assert _parse_bindingdb_value(">1e3") == (">", 1000.0, 1000.0, None)
    assert _parse_bindingdb_value("~5") is None
    assert _parse_bindingdb_value("0") is None


def test_target_components_connect_accession_and_sequence_without_label_access() -> None:
    records = {}
    union = _UnionFind()
    first = _register_target(
        records,
        union,
        accessions=_normalize_accessions("P00001"),
        sequence="AAA",
        target_name="one",
        organism="human",
        source_dataset="A",
    )
    second = _register_target(
        records,
        union,
        accessions=_normalize_accessions("P00001"),
        sequence="",
        target_name="one alias",
        organism="human",
        source_dataset="B",
    )
    assert first is not None and second is not None
    lookup, targets, groups = _resolve_target_groups(records, union, {first.target_id, second.target_id})
    assert len(groups) == 1
    assert {
        lookup[first.target_id]["target_leakage_group_id"],
        lookup[second.target_id]["target_leakage_group_id"],
    } == {groups[0]["target_leakage_group_id"]}
    resolved = next(row for row in targets if row["target_id"] == second.target_id)
    assert resolved["sequence"] == "AAA"
    assert resolved["sequence_source"] == "resolved_from_unambiguous_accession_sequence_component"


def test_split_is_identifier_only_and_deterministic() -> None:
    assert _split("SCF-X", dimension="ligand_scaffold") == _split("SCF-X", dimension="ligand_scaffold")
    assert _split("TGRP-X", dimension="target_component") in {"train", "validation", "test"}


def test_scaffold_failure_uses_exact_structure_no_leak_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection = _initialize_work_database(tmp_path / "work.sqlite")

    def fail_scaffold(*args: object, **kwargs: object) -> str:
        raise RuntimeError("fixture scaffold failure")

    monkeypatch.setattr(affinity_module.MurckoScaffold, "MurckoScaffoldSmiles", fail_scaffold)
    ligand = _resolve_structure(connection, "CCO")
    connection.close()
    assert ligand is not None
    assert ligand["scaffold_derivation_status"] == "murcko_failure_exact_structure_fallback"
    assert ligand["scaffold_group_id"]


def test_fixture_release_preserves_endpoints_bounds_and_removes_mirrors(
    affinity_release: tuple[Path, Path],
) -> None:
    output_root, _ = affinity_release
    manifest = validate_affinity_training_surfaces(output_root)
    assert manifest["counts"]["total"]["observations"] == 5
    assert manifest["counts"]["endpoints"]["Kd"]["observations"] == 1
    assert manifest["counts"]["endpoints"]["Ki"]["observations"] == 1
    assert manifest["counts"]["endpoints"]["IC50"]["observations"] == 2
    assert manifest["counts"]["endpoints"]["EC50"]["observations"] == 1
    assert manifest["counts"]["exclusions"]["explicit_chembl_source_mirror"] == 1
    assert manifest["counts"]["exclusions"]["same_document_chembl_mirror"] == 1
    assert manifest["counts"]["exclusions"]["same_document_internal_exact_mirror"] == 1
    ic50 = pq.read_table(output_root / "observations" / "ic50").to_pylist()
    interval = next(row for row in ic50 if row["label_relation"] == "interval")
    assert interval["label_value_nM"] is None
    assert interval["label_lower_bound_nM"] == 7.0
    assert interval["label_upper_bound_nM"] == 9.0
    left = next(row for row in ic50 if row["label_relation"] == "<")
    assert left["label_lower_bound_nM"] is None
    assert left["label_upper_bound_nM"] == 100.0
    assert left["source_dataset"] == "BindingDB_202608"


def test_validator_rejects_artifact_tampering_and_extra_membership(
    affinity_release: tuple[Path, Path], tmp_path: Path
) -> None:
    output_root, _ = affinity_release
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    for source in output_root.rglob("*"):
        if source.is_file():
            target = tampered / source.relative_to(output_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    report = tampered / "AFFINITY_TRAINING_SURFACES.md"
    report.write_text(report.read_text() + "tampered\n")
    with pytest.raises(AffinityTrainingSurfaceError, match="artifact binding"):
        validate_affinity_training_surfaces(tampered)

    extra = tmp_path / "extra"
    extra.mkdir()
    for source in output_root.rglob("*"):
        if source.is_file():
            target = extra / source.relative_to(output_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    (extra / "unexpected.txt").write_text("no")
    with pytest.raises(AffinityTrainingSurfaceError, match="unexpected output membership"):
        validate_affinity_training_surfaces(extra)


def test_validator_rejects_bound_input_changes(
    affinity_release: tuple[Path, Path],
) -> None:
    output_root, acquisition_path = affinity_release
    acquisition_path.write_text(acquisition_path.read_text() + " ")
    with pytest.raises(AffinityTrainingSurfaceError, match="input binding"):
        validate_affinity_training_surfaces(output_root)


def test_production_release_validates_when_present() -> None:
    root = Path("research/data/platform/processed/affinity_training/v1_0_chembl37_bindingdb202608")
    if not root.exists():
        pytest.skip("production affinity release is not built")
    manifest = validate_affinity_training_surfaces(root)
    assert manifest["counts"]["total"]["primary_observations"] >= 3_000_000
    assert manifest["counts"]["endpoints"]["IC50"]["observations"] >= 2_000_000
