from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from menin_discovery.platform_external_normalization import (
    BINDINGDB_ALLOWED_ORIGIN,
    BINDINGDB_ARTICLES_ARCHIVE,
    BINDINGDB_ARTICLES_MEMBER,
    BINDINGDB_EXCLUDED_ORIGIN,
    BINDINGDB_QUARANTINED_ORIGIN,
    BINDINGDB_SOURCE_ID,
    CLINICALTRIALS_SOURCE_ID,
    DAILYMED_SOURCE_ID,
    DRUGSFDA_SOURCE_ID,
    SCHEMA_VERSION,
    SOURCE_SCHEMA_VERSION,
    UNIPROT_SOURCE_ID,
    InputBinding,
    NormalizationError,
    _portable_report_path,
    arrow_schema_sha256,
    canonical_json_bytes,
    document_with_sha256,
    load_and_verify_input,
    normalize_bindingdb,
    normalize_clinicaltrials,
    normalize_regulatory_inventories,
    normalize_uniprot,
    parse_affinity_value,
    sha256_file,
    verify_document_sha256,
    verify_external_normalized_output,
)


def _write_identified(path: Path, body: dict[str, object]) -> dict[str, object]:
    document = document_with_sha256(body)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def _binding(source_id: str, root: Path, manifest: Mapping[str, object]) -> InputBinding:
    manifest_path = root / f"{source_id}_manifest.json"
    return InputBinding(
        source_id=source_id,
        root=root,
        manifest_path=manifest_path,
        manifest=dict(manifest),
        declared_manifest_sha256=str(manifest.get("manifest_sha256", "fixture")),
        physical_manifest_sha256="0" * 64,
        physical_manifest_bytes=0,
        bundle_entries_sha256="1" * 64,
        bundle_entry_count=0,
        bundle_total_bytes=0,
    )


def _verifier_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "normalized"
    root.mkdir(parents=True)
    parquet_path = root / "fixture.parquet"
    schema = pa.schema(
        [
            pa.field("value", pa.int64(), nullable=False),
            pa.field("model_label_admitted", pa.bool_(), nullable=False),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist([{"value": 1, "model_label_admitted": False}], schema=schema), parquet_path
    )
    entry = {
        "path": parquet_path.name,
        "artifact_role": "normalized_parquet",
        "rows": 1,
        "bytes": parquet_path.stat().st_size,
        "sha256": sha256_file(parquet_path),
        "arrow_schema_sha256": arrow_schema_sha256(pq.ParquetFile(parquet_path).schema_arrow),
    }
    inventory = {
        "entries": [entry],
        "entries_sha256": hashlib.sha256(canonical_json_bytes([entry])).hexdigest(),
        "entry_count": 1,
        "total_bytes": entry["bytes"],
        "excluded_paths": ["external_public_normalized_manifest.json"],
    }
    manifest_path = root / "external_public_normalized_manifest.json"
    _write_identified(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": "external_public_normalized",
            "inputs": [],
            "input_manifest_set_sha256": hashlib.sha256(canonical_json_bytes([])).hexdigest(),
            "source_to_output_reconciliation": {},
            "output_inventory": inventory,
            "output_row_count": 1,
            "canonical_observations_admitted": 0,
            "model_labels_admitted": 0,
            "substantive_model_training_performed": False,
            "zero_training_flag": True,
            "endpoint_pooling_performed": False,
            "silent_cross_source_identity_replacement_performed": False,
        },
    )
    return root, manifest_path


def test_manifest_identity_and_exact_recursive_bundle(tmp_path: Path) -> None:
    root = tmp_path / "fixture_source"
    root.mkdir()
    artifact = root / "artifact.txt"
    artifact.write_text("frozen\n", encoding="utf-8")
    entry = {
        "path": artifact.name,
        "bytes": artifact.stat().st_size,
        "sha256": sha256_file(artifact),
    }
    manifest_name = "fixture_source_manifest.json"
    manifest = _write_identified(
        root / manifest_name,
        {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "source_id": root.name,
            "release_id": "fixture",
            "snapshot_status": "complete",
            "files": [
                {
                    "local_path": artifact.name,
                    "acquired_bytes": artifact.stat().st_size,
                    "acquired_sha256": sha256_file(artifact),
                }
            ],
            "exact_physical_file_count": 1,
            "exact_physical_bytes": artifact.stat().st_size,
            "bundle_inventory": {
                "entries": [entry],
                "entries_sha256": hashlib.sha256(canonical_json_bytes([entry])).hexdigest(),
                "entry_count": 1,
                "total_bytes": artifact.stat().st_size,
                "excluded_paths": [manifest_name],
            },
        },
    )
    assert verify_document_sha256(manifest)
    verified = load_and_verify_input(root, manifest_name)
    assert verified.bundle_entry_count == 1

    extra = root / "unexpected.txt"
    extra.write_text("not declared", encoding="utf-8")
    with pytest.raises(NormalizationError, match="membership changed"):
        load_and_verify_input(root, manifest_name)
    extra.unlink()
    artifact.write_text("drifted\n", encoding="utf-8")
    with pytest.raises(NormalizationError, match="byte count changed|SHA-256 changed"):
        load_and_verify_input(root, manifest_name)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" 8.700 ", ("=", "8.7", 8.7, "parsed_candidate")),
        (">10000", (">", "10000", 10000.0, "parsed_candidate")),
        ("<= 1e-3", ("<=", "0.001", 0.001, "parsed_candidate")),
        ("~5", ("~", "5", 5.0, "parsed_candidate")),
        ("10-20", (None, None, None, "unparsed_raw_value_quarantine")),
        ("inactive", (None, None, None, "unparsed_raw_value_quarantine")),
        ("", (None, None, None, "blank")),
    ],
)
def test_affinity_parser_is_conservative(raw: str, expected: tuple[object, ...]) -> None:
    assert parse_affinity_value(raw) == expected


def test_bindingdb_row_disposition_preservation_and_no_endpoint_pooling(tmp_path: Path) -> None:
    root = tmp_path / BINDINGDB_SOURCE_ID
    root.mkdir()
    columns = [
        "BindingDB Reactant_set_id",
        "Ligand SMILES",
        "Ligand InChI",
        "Ligand InChI Key",
        "BindingDB MonomerID",
        "BindingDB Ligand Name",
        "Target Name",
        "Target Source Organism According to Curator or DataSource",
        "Ki (nM)",
        "IC50 (nM)",
        "Kd (nM)",
        "EC50 (nM)",
        "Curation/DataSource",
        "Article DOI",
        "BindingDB Entry DOI",
        "PMID",
        "Date of publication",
        "Number of Protein Chains in Target (>1 implies a multichain complex)",
        "BindingDB Target Chain Sequence 1",
        "UniProt (SwissProt) Primary ID of Target Chain 1",
        "UniProt (TrEMBL) Primary ID of Target Chain 1",
    ]
    values = [
        [
            "1",
            "CC",
            "InChI=1S/C2H6",
            "OTMSDBZUPAUEDD-UHFFFAOYSA-N",
            "7",
            "ethane",
            "target A",
            "human",
            "<10",
            "",
            "5",
            "",
            BINDINGDB_ALLOWED_ORIGIN,
            "10/example",
            "10/bdb",
            "1",
            "2020",
            "1",
            "ACDE",
            "P12345",
            "",
        ],
        [
            "1",
            "C",
            "",
            "VNWKTOKETHGBQD-UHFFFAOYSA-N",
            "8",
            "methane",
            "target B",
            "human",
            "",
            "inactive",
            "",
            "",
            BINDINGDB_EXCLUDED_ORIGIN,
            "",
            "",
            "",
            "2021",
            "1",
            "ACD",
            "Q99999",
            "",
        ],
        [
            "2",
            "",
            "",
            "",
            "9",
            "none",
            "target C",
            "mouse",
            "",
            "",
            "",
            ">1000",
            BINDINGDB_QUARANTINED_ORIGIN,
            "",
            "",
            "",
            "2022",
            "0",
            "",
            "",
            "",
        ],
    ]
    archive_path = root / BINDINGDB_ARTICLES_ARCHIVE
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(values)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(BINDINGDB_ARTICLES_MEMBER, buffer.getvalue())
    manifest = {
        "parse_inventory": [
            {
                "file": BINDINGDB_ARTICLES_ARCHIVE,
                "members": [{"columns": columns, "data_row_count": 3}],
            }
        ],
        "articles_origin_and_endpoint_audit": {
            "curation_data_source_counts": {
                BINDINGDB_ALLOWED_ORIGIN: 1,
                BINDINGDB_EXCLUDED_ORIGIN: 1,
                BINDINGDB_QUARANTINED_ORIGIN: 1,
            },
            "endpoint_nonblank_row_counts": {"Ki (nM)": 1, "IC50 (nM)": 1, "Kd (nM)": 1, "EC50 (nM)": 1},
            "duplicate_reactant_set_id_rows": 1,
            "physical_measurement_rows": 3,
        },
    }
    output = tmp_path / "normalized"
    artifacts, summary = normalize_bindingdb(_binding(BINDINGDB_SOURCE_ID, root, manifest), output)
    assert [item.rows for item in artifacts] == [3, 2]
    assert summary["candidate_rows"] == 1
    assert summary["excluded_chembl_mirror_rows"] == 1
    assert summary["quarantined_taylor_rows"] == 1
    assert summary["duplicate_reactant_set_id_excess_rows"] == 1
    article = pq.read_table(output / "bindingdb/article_rows.parquet").to_pylist()
    assert all(not row["model_label_admitted"] for row in article)
    assert json.loads(article[0]["source_record_json"])["Ligand SMILES"] == "CC"
    observations = pq.read_table(output / "bindingdb/affinity_observations.parquet").to_pylist()
    assert {row["endpoint_type"] for row in observations} == {"Ki", "Kd"}
    assert {row["endpoint_pooling_key"] for row in observations} == {"Ki", "Kd"}


def test_uniprot_sequence_inventory_and_quarantines(tmp_path: Path) -> None:
    root = tmp_path / UNIPROT_SOURCE_ID
    (root / "pages").mkdir(parents=True)
    sequence = "ACDE"
    page = {
        "results": [
            {
                "primaryAccession": "P12345",
                "entryType": "UniProtKB reviewed (Swiss-Prot)",
                "uniProtkbId": "TEST_HUMAN",
                "proteinDescription": {"recommendedName": {"fullName": {"value": "Test"}}},
                "organism": {"taxonId": 9606, "scientificName": "Homo sapiens"},
                "entryAudit": {"entryVersion": 2},
                "sequence": {
                    "value": sequence,
                    "length": len(sequence),
                    "md5": hashlib.md5(sequence.encode(), usedforsecurity=False).hexdigest(),
                    "version": 1,
                },
            },
            {"primaryAccession": "A0A0000001", "entryType": "Inactive"},
        ]
    }
    (root / "pages/page_000000.json").write_text(json.dumps(page), encoding="utf-8")
    resolution_rows = [
        {
            "requested_accession": "P12345",
            "resolution_state": "resolved_primary",
            "returned_primary_accession": "P12345",
            "returned_primary_accessions": ["P12345"],
            "entry_type": "UniProtKB reviewed (Swiss-Prot)",
            "replacement_state": "not_reported_by_search_endpoint",
        },
        {
            "requested_accession": "A0A0000001",
            "resolution_state": "resolved_primary",
            "returned_primary_accession": "A0A0000001",
            "returned_primary_accessions": ["A0A0000001"],
            "entry_type": "Inactive",
            "replacement_state": "not_reported_by_search_endpoint",
        },
        {
            "requested_accession": "BAD",
            "resolution_state": "non_uniprot_identifier_syntax_quarantine",
            "returned_primary_accessions": [],
            "replacement_state": "not_applicable_non_uniprot_identifier",
        },
        {
            "requested_accession": "Q00001",
            "resolution_state": "ambiguous_multi_mapped_quarantine",
            "returned_primary_accessions": ["Q00001", "Q00002"],
            "replacement_state": "not_reported_by_search_endpoint",
        },
    ]
    resolution_path = root / "accession_resolution.jsonl"
    resolution_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in resolution_rows),
        encoding="utf-8",
    )
    membership_rows = [
        {
            "source_id": "chembl_37_target_components",
            "source_file": "target_components.parquet",
            "source_row_index_zero_based": 0,
            "source_target_id": "CHEMBL1",
            "source_component_id": 1,
            "source_accession_value": "P12345",
            "normalized_identifier": "P12345",
            "admission_state": "request_candidate",
        },
        {
            "source_id": "chembl_37_target_components",
            "source_file": "target_components.parquet",
            "source_row_index_zero_based": 1,
            "source_target_id": "CHEMBL2",
            "source_component_id": 2,
            "source_accession_value": "BAD",
            "normalized_identifier": "BAD",
            "admission_state": "identifier_syntax_quarantine",
        },
    ]
    membership_path = root / "accession_source_membership.jsonl"
    membership_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in membership_rows),
        encoding="utf-8",
    )
    manifest = {
        "pages": [{"page_index": 0, "path": "pages/page_000000.json", "returned_primary_count": 2}],
        "protein_entry_inventory": {
            "unique_returned_primary_entries": 2,
            "sequence_ready_entries": 1,
            "entry_type_counts": {"UniProtKB reviewed (Swiss-Prot)": 1, "Inactive": 1},
        },
        "accession_resolution_inventory": {"path": resolution_path.name, "rows": 4},
        "resolution_counts": {
            "resolved": 2,
            "ambiguous_multi_mapped": 1,
            "non_uniprot_identifier_syntax_quarantine": 1,
        },
        "accession_source_membership": {
            "path": membership_path.name,
            "identifier_reference_rows": 2,
            "valid_uniprot_accession_reference_rows": 1,
            "identifier_syntax_quarantine_reference_rows": 1,
        },
    }
    output = tmp_path / "normalized"
    artifacts, summary = normalize_uniprot(_binding(UNIPROT_SOURCE_ID, root, manifest), output)
    assert [item.rows for item in artifacts] == [2, 4, 2]
    assert summary["sequence_ready_entries"] == 1
    assert summary["inactive_returned_entries"] == 1
    assert summary["silent_chembl_identity_replacements"] == 0
    returned = pq.read_table(output / "uniprot/returned_entries.parquet").to_pylist()
    assert returned[0]["sequence"] == sequence
    memberships = pq.read_table(output / "uniprot/source_membership.parquet").to_pylist()
    assert memberships[0]["normalized_identifier"] == "P12345"
    assert memberships[0]["returned_primary_accession"] == "P12345"
    assert not memberships[0]["silent_identity_replacement_performed"]


def test_registry_and_regulatory_outputs_are_inventory_only(tmp_path: Path) -> None:
    clinical_root = tmp_path / CLINICALTRIALS_SOURCE_ID
    clinical_root.mkdir()
    broad_path = clinical_root / "broad.jsonl"
    broad_path.write_text(
        json.dumps(
            {"nct_id": "NCT1", "page_index": 0, "study_index_within_page": 0, "study_sha256": "a" * 64}
        )
        + "\n",
        encoding="utf-8",
    )
    heuristic_path = clinical_root / "heuristic.jsonl"
    heuristic_path.write_text(
        json.dumps(
            {
                "nct_id": "NCT1",
                "page_index": 0,
                "study_index_within_page": 0,
                "study_sha256": "a" * 64,
                "projected_heuristic_term_matches": ["QT_or_QTc"],
                "has_posted_outcome_measures_module": True,
                "has_posted_adverse_events_module": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    clinical_manifest = {
        "alias_independent_all_drug_cohort": {
            "cohort_snapshot_key": "v1",
            "nct_membership_inventory": {"path": broad_path.name, "rows": 1},
            "unique_nct_count": 1,
        },
        "cardiac_safety_heuristic_cohort": {
            "cohort_snapshot_key": "v1",
            "nct_membership_inventory": {"path": heuristic_path.name, "rows": 1},
            "unique_nct_count": 1,
        },
    }
    output = tmp_path / "normalized"
    artifacts, summary = normalize_clinicaltrials(
        _binding(CLINICALTRIALS_SOURCE_ID, clinical_root, clinical_manifest), output
    )
    assert artifacts[0].rows == 2
    assert summary["broad_vs_heuristic_membership_overlap"] == 1
    clinical_rows = pq.read_table(output / "clinicaltrials/cohort_membership.parquet").to_pylist()
    assert all(not row["model_label_admitted"] for row in clinical_rows)
    assert all(not row["canonical_observation_admitted"] for row in clinical_rows)

    drugs_root = tmp_path / DRUGSFDA_SOURCE_ID
    daily_root = tmp_path / DAILYMED_SOURCE_ID
    drugs_root.mkdir()
    daily_root.mkdir()
    drugs_manifest = {
        "archive_member_table": {
            "members": [
                {
                    "archive_member_path": "Products.txt",
                    "data_row_count": 2,
                    "malformed_width_rows": 1,
                    "parse_integrity": "failed",
                }
            ],
            "txt_table_count": 1,
            "total_data_rows": 2,
            "source_width_anomaly_rows": 1,
        },
        "relational_key_and_join_audit": {
            "total_blank_primary_key_rows": 3,
            "total_missing_foreign_key_rows_across_relations": 4,
        },
    }
    daily_manifest = {
        "files": [
            {
                "artifact_role": "human_prescription_release_part",
                "local_path": "part.zip",
                "acquired_bytes": 10,
                "acquired_sha256": "b" * 64,
                "archive_integrity": {
                    "file_member_count": 2,
                    "total_member_uncompressed_bytes": 20,
                },
            }
        ],
        "release_part_count": 1,
        "expected_and_verified_file_member_count": 2,
    }
    artifacts, summary = normalize_regulatory_inventories(
        _binding(DRUGSFDA_SOURCE_ID, drugs_root, drugs_manifest),
        _binding(DAILYMED_SOURCE_ID, daily_root, daily_manifest),
        output,
    )
    assert artifacts[0].rows == 2
    assert summary["dailymed_section_extraction_attempted"] is False
    regulatory = pq.read_table(output / "regulatory/archive_inventory.parquet").to_pylist()
    assert all(not row["model_label_admitted"] for row in regulatory)
    assert SCHEMA_VERSION.startswith("platform-external-normalization/")


def test_existing_output_verifier_and_portable_report_paths(tmp_path: Path) -> None:
    root, _manifest_path = _verifier_fixture(tmp_path)
    result = verify_external_normalized_output(root)
    assert result["status"] == "passed"
    assert result["aggregate_parquet_rows"] == 1
    assert result["zero_label_training_and_identity_replacement_contract"] == "passed"
    portable = _portable_report_path(Path.cwd() / "research/data/example")
    assert portable == "research/data/example"
    assert not portable.startswith("/")
    assert "/Users/" not in portable and "/home/" not in portable


def test_existing_output_verifier_rejects_byte_tamper(tmp_path: Path) -> None:
    root, _manifest_path = _verifier_fixture(tmp_path)
    with (root / "fixture.parquet").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(NormalizationError, match="byte count changed"):
        verify_external_normalized_output(root)


def test_existing_output_verifier_rejects_extra_and_symlink(tmp_path: Path) -> None:
    extra_root, _manifest_path = _verifier_fixture(tmp_path / "extra")
    (extra_root / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(NormalizationError, match="membership changed"):
        verify_external_normalized_output(extra_root)

    symlink_root, _manifest_path = _verifier_fixture(tmp_path / "symlink")
    link = symlink_root / "forbidden-link"
    try:
        link.symlink_to(symlink_root / "fixture.parquet")
    except OSError:
        pytest.skip("Filesystem does not permit symlink fixture")
    with pytest.raises(NormalizationError, match="Symlink prohibited"):
        verify_external_normalized_output(symlink_root)


def test_existing_output_verifier_rejects_declared_schema_mismatch(tmp_path: Path) -> None:
    root, manifest_path = _verifier_fixture(tmp_path)
    parquet_path = root / "fixture.parquet"
    changed_schema = pa.schema(
        [
            pa.field("value", pa.int64(), nullable=False),
            pa.field("model_label_admitted", pa.bool_(), nullable=False),
            pa.field("unexpected", pa.string()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(
            [{"value": 1, "model_label_admitted": False, "unexpected": "drift"}],
            schema=changed_schema,
        ),
        parquet_path,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    entry = body["output_inventory"]["entries"][0]
    entry["bytes"] = parquet_path.stat().st_size
    entry["sha256"] = sha256_file(parquet_path)
    body["output_inventory"]["total_bytes"] = entry["bytes"]
    body["output_inventory"]["entries_sha256"] = hashlib.sha256(
        canonical_json_bytes(body["output_inventory"]["entries"])
    ).hexdigest()
    _write_identified(manifest_path, body)
    with pytest.raises(NormalizationError, match="Arrow schema changed"):
        verify_external_normalized_output(root)
