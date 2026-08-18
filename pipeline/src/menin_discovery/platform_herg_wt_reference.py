"""Freeze the reviewed human wild-type KCNH2 sequence for future hERG work."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "platform-herg-wt-reference/1.0"
ACCESSION = "Q12809"
ENTRY_NAME = "KCNH2_HUMAN"
TAXONOMY_ID = 9606


class HergWtReferenceError(RuntimeError):
    """Raised when the WT sequence reference cannot be proven or validated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_hash(payload: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _find_entry(page: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    matches = [
        (index, row)
        for index, row in enumerate(page.get("results", []))
        if row.get("primaryAccession") == ACCESSION
    ]
    if len(matches) != 1:
        raise HergWtReferenceError(f"expected one {ACCESSION} entry, found {len(matches)}")
    return matches[0]


def build_wt_reference(*, page_path: Path, page_manifest_path: Path, output_root: Path) -> dict[str, Any]:
    """Extract one reviewed, human, canonical UniProt sequence without network access."""

    page = json.loads(page_path.read_text())
    page_manifest = json.loads(page_manifest_path.read_text())
    if page_manifest.get("acquired_sha256") != _sha256(page_path):
        raise HergWtReferenceError("UniProt page does not match its acquisition manifest")
    if page_manifest.get("response_headers", {}).get("x-uniprot-release") != "2026_02":
        raise HergWtReferenceError("unexpected UniProt release")
    index, entry = _find_entry(page)
    sequence = entry.get("sequence", {}).get("value", "")
    sequence_sha256 = hashlib.sha256(sequence.encode()).hexdigest()
    sequence_md5 = hashlib.md5(sequence.encode(), usedforsecurity=False).hexdigest().upper()
    organism = entry.get("organism", {})
    if (
        entry.get("uniProtkbId") != ENTRY_NAME
        or entry.get("entryType") != "UniProtKB reviewed (Swiss-Prot)"
        or organism.get("taxonId") != TAXONOMY_ID
        or entry.get("sequence", {}).get("length") != len(sequence)
        or entry.get("sequence", {}).get("md5") != sequence_md5
        or not sequence
    ):
        raise HergWtReferenceError("Q12809 identity, review status, organism, or sequence check failed")

    record = {
        "schema_version": SCHEMA_VERSION,
        "reference_scope": "human_reviewed_canonical_full_length_sequence",
        "target": "wild_type_KCNH2_hERG_Kv11.1",
        "primary_accession": ACCESSION,
        "entry_name": ENTRY_NAME,
        "entry_type": entry["entryType"],
        "taxonomy_id": TAXONOMY_ID,
        "organism_scientific_name": organism.get("scientificName"),
        "uniprot_release": page_manifest["response_headers"]["x-uniprot-release"],
        "uniprot_release_date": page_manifest["response_headers"].get("x-uniprot-release-date"),
        "source_result_index_zero_based": index,
        "sequence": sequence,
        "sequence_length": len(sequence),
        "sequence_md5": sequence_md5,
        "sequence_sha256": sequence_sha256,
        "sequence_crc64": entry["sequence"].get("crc64"),
        "molecular_weight_da": entry["sequence"].get("molWeight"),
        "allowed_for_wt_protein_features": True,
        "mutant_sequence_allowed": False,
        "construct_or_receptor_structure_selected": False,
        "scientific_limitations": [
            "This freezes the reviewed canonical full-length sequence, not an experimental construct.",
            "It does not select a receptor conformation, membrane state, or docking structure.",
            "Any truncated construct must receive a new construct ID and explicit residue mapping.",
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    record_path = output_root / "wt_kcnh2_reference.json"
    fasta_path = output_root / "Q12809_KCNH2_HUMAN.fasta"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    fasta_path.write_text(
        f">sp|{ACCESSION}|{ENTRY_NAME} canonical reviewed UniProt release 2026_02\n"
        + "\n".join(sequence[offset : offset + 60] for offset in range(0, len(sequence), 60))
        + "\n"
    )
    artifacts = {
        path.name: {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in [record_path, fasta_path]
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "inputs": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in [page_path, page_manifest_path]
        ],
        "artifacts": artifacts,
        "counts": {"reference_sequences": 1, "amino_acids": len(sequence)},
        "reference": {
            "primary_accession": ACCESSION,
            "entry_name": ENTRY_NAME,
            "taxonomy_id": TAXONOMY_ID,
            "sequence_sha256": sequence_sha256,
            "mutants_admitted": 0,
        },
        "scientific_contract": {
            "protein_features_computed": False,
            "receptor_selected": False,
            "docking_enabled": False,
            "canonical_sequence_frozen": True,
        },
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    (output_root / "wt_reference_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def validate_wt_reference(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "wt_reference_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise HergWtReferenceError("manifest self-hash mismatch")
    for binding in manifest["inputs"]:
        path = Path(binding["path"])
        if (
            not path.is_file()
            or path.stat().st_size != binding["bytes"]
            or _sha256(path) != binding["sha256"]
        ):
            raise HergWtReferenceError(f"input binding failed: {path}")
    for name, binding in manifest["artifacts"].items():
        path = output_root / name
        if (
            not path.is_file()
            or path.stat().st_size != binding["bytes"]
            or _sha256(path) != binding["sha256"]
        ):
            raise HergWtReferenceError(f"artifact binding failed: {name}")
    record = json.loads((output_root / "wt_kcnh2_reference.json").read_text())
    sequence = record["sequence"]
    if (
        record["primary_accession"] != ACCESSION
        or record["entry_name"] != ENTRY_NAME
        or record["taxonomy_id"] != TAXONOMY_ID
        or record["sequence_length"] != len(sequence)
        or len(sequence) != 1159
        or hashlib.sha256(sequence.encode()).hexdigest() != record["sequence_sha256"]
        or record["mutant_sequence_allowed"]
        or record["construct_or_receptor_structure_selected"]
    ):
        raise HergWtReferenceError("WT reference semantic validation failed")
    fasta_sequence = "".join((output_root / "Q12809_KCNH2_HUMAN.fasta").read_text().splitlines()[1:])
    if fasta_sequence != sequence:
        raise HergWtReferenceError("FASTA/JSON sequence mismatch")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=Path)
    parser.add_argument("--page-manifest", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_wt_reference(args.output_root)
    else:
        if args.page is None or args.page_manifest is None:
            parser.error("--page and --page-manifest are required when building")
        build_wt_reference(
            page_path=args.page,
            page_manifest_path=args.page_manifest,
            output_root=args.output_root,
        )
        validate_wt_reference(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
