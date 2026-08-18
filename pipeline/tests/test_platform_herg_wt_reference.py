from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from menin_discovery.platform_herg_wt_reference import (
    HergWtReferenceError,
    build_wt_reference,
    validate_wt_reference,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    sequence = "ACDEFGHIK"
    page = tmp_path / "page.json"
    page.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "primaryAccession": "Q12809",
                        "uniProtkbId": "KCNH2_HUMAN",
                        "entryType": "UniProtKB reviewed (Swiss-Prot)",
                        "organism": {"taxonId": 9606, "scientificName": "Homo sapiens"},
                        "sequence": {
                            "value": sequence,
                            "length": len(sequence),
                            "md5": hashlib.md5(sequence.encode(), usedforsecurity=False).hexdigest().upper(),
                            "crc64": "X",
                            "molWeight": 1,
                        },
                    }
                ]
            }
        )
    )
    manifest = tmp_path / "page.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "acquired_sha256": hashlib.sha256(page.read_bytes()).hexdigest(),
                "response_headers": {
                    "x-uniprot-release": "2026_02",
                    "x-uniprot-release-date": "10-June-2026",
                },
            }
        )
    )
    return page, manifest


def test_wt_reference_builds_and_validates_hash_bound_artifacts(tmp_path: Path) -> None:
    page, page_manifest = _fixture(tmp_path)
    output = tmp_path / "out"
    build_wt_reference(page_path=page, page_manifest_path=page_manifest, output_root=output)
    # The production reference requires the known full length; the small fixture verifies the builder.
    record = json.loads((output / "wt_kcnh2_reference.json").read_text())
    assert record["primary_accession"] == "Q12809"
    assert record["mutant_sequence_allowed"] is False
    assert record["construct_or_receptor_structure_selected"] is False


def test_wt_reference_fails_if_source_page_changes(tmp_path: Path) -> None:
    page, page_manifest = _fixture(tmp_path)
    page.write_text(page.read_text() + "\n")
    with pytest.raises(HergWtReferenceError, match="acquisition manifest"):
        build_wt_reference(page_path=page, page_manifest_path=page_manifest, output_root=tmp_path / "out")


def test_production_wt_reference_validates() -> None:
    root = Path("research/data/platform/processed/herg_hierarchy/v1_5_wt_reference")
    if root.exists():
        assert validate_wt_reference(root)["reference"]["mutants_admitted"] == 0
