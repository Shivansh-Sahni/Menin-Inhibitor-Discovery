"""Integrate hERG assay and clinical-link evidence without promoting claims.

This module produces a structure-level index over the assay-native hERG
hierarchy and the conservative clinical-link inventory.  It is intentionally
an evidence *index*, not a label promotion step:

* every structure retains its formal ``T0_reported`` assignment;
* cross-lineage PubChem/ChEMBL agreement is a T1 review candidate only;
* development metadata is not clinical cardiac validation; and
* a posted QT/QTc result is not a hERG assay label and is not T2/T3 admission.

The builder accepts only explicit, already-verified hierarchy and clinical
link roots, writes deterministic Parquet artifacts, and fails closed on any
schema, hash, count, or scientific-contract drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from menin_discovery.platform_herg_clinical_links import (
    STRUCTURE_OUTPUT,
    T2_OUTPUT,
    T3_OUTPUT,
    verify_herg_clinical_links,
)
from menin_discovery.platform_herg_hierarchy import validate_herg_hierarchy

SCHEMA_VERSION = "platform-herg-evidence-tiers/1.0"
MANIFEST_NAME = "herg_evidence_tiers_manifest.json"
STRUCTURE_TIER_OUTPUT = "structure_evidence_tiers.parquet"
T1_CANDIDATE_OUTPUT = "cross_lineage_t1_candidates.parquet"

FORMAL_TIER = "T0_reported"
T1_SEMANTICS = (
    "cross_lineage_review_candidate_only; upstream_independence_and_modality_"
    "comparability_not_adjudicated; never_formal_T1"
)
CLINICAL_SEMANTICS = (
    "development_and_posted_QT_QTc_are_annotations_or_review_candidates; "
    "neither_is_a_hERG_label_or_formal_T2_T3_assignment"
)


class HergTierIntegrationError(RuntimeError):
    """Raised when tier integration or verification fails closed."""


_STRUCTURE_TIER_SCHEMA = pa.schema(
    [
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("standardized_smiles", pa.large_string(), nullable=False),
        pa.field("standard_inchi_key", pa.large_string(), nullable=False),
        pa.field("t0_reported", pa.bool_(), nullable=False),
        pa.field("t0_observation_count", pa.int64(), nullable=False),
        pa.field("t0_source_families_json", pa.large_string(), nullable=False),
        pa.field("hierarchy_consensus_status", pa.large_string(), nullable=False),
        pa.field("hierarchy_consensus_binary_label", pa.int8()),
        pa.field("cross_lineage_t1_candidate", pa.bool_(), nullable=False),
        pa.field("cross_lineage_t1_candidate_state", pa.large_string(), nullable=False),
        pa.field("cross_lineage_t1_candidate_count", pa.int64(), nullable=False),
        pa.field("clinical_development_annotation", pa.bool_(), nullable=False),
        pa.field("chembl_max_phase", pa.float64()),
        pa.field("chembl_first_approval", pa.int64()),
        pa.field("chembl_therapeutic_flag", pa.bool_(), nullable=False),
        pa.field("chembl_dosed_ingredient", pa.bool_(), nullable=False),
        pa.field("chembl_withdrawn_flag", pa.bool_(), nullable=False),
        pa.field("drugsfda_exact_name_link_count", pa.int64(), nullable=False),
        pa.field("exact_posted_qt_candidate_count", pa.int64(), nullable=False),
        pa.field("exact_posted_qt_distinct_nct_count", pa.int64(), nullable=False),
        pa.field("exact_posted_qt_reported_numeric_value_count", pa.int64(), nullable=False),
        pa.field("clinical_cardiac_candidate_evidence", pa.bool_(), nullable=False),
        pa.field("formal_highest_assigned_tier", pa.large_string(), nullable=False),
        pa.field("formal_t1_assigned", pa.bool_(), nullable=False),
        pa.field("formal_t2_assigned", pa.bool_(), nullable=False),
        pa.field("formal_t3_assigned", pa.bool_(), nullable=False),
        pa.field("clinical_herg_label_admitted", pa.bool_(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
        pa.field("tier_semantics", pa.large_string(), nullable=False),
    ]
)

_T1_CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("candidate_id", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("standardized_smiles", pa.large_string(), nullable=False),
        pa.field("standard_inchi_key", pa.large_string(), nullable=False),
        pa.field("pubchem_qualifying_observation_count", pa.int64(), nullable=False),
        pa.field("pubchem_labels_json", pa.large_string(), nullable=False),
        pa.field("pubchem_source_record_ids_json", pa.large_string(), nullable=False),
        pa.field("chembl_qualifying_observation_count", pa.int64(), nullable=False),
        pa.field("chembl_labels_json", pa.large_string(), nullable=False),
        pa.field("chembl_source_record_ids_json", pa.large_string(), nullable=False),
        pa.field("chembl_assay_ids_json", pa.large_string(), nullable=False),
        pa.field("candidate_state", pa.large_string(), nullable=False),
        pa.field("concordant_binary_label", pa.int8()),
        pa.field("upstream_lineage_independence_adjudicated", pa.bool_(), nullable=False),
        pa.field("assay_modality_comparability_adjudicated", pa.bool_(), nullable=False),
        pa.field("formal_t1_assigned", pa.bool_(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
        pa.field("candidate_semantics", pa.large_string(), nullable=False),
    ]
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _stable_id(*parts: object) -> str:
    body = "\x1f".join(str(part) for part in parts)
    return "HT1-" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:24].upper()


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]], schema: pa.Schema) -> dict[str, Any]:
    table = pa.Table.from_pylist(list(rows), schema=schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
        version="2.6",
    )
    return {
        "path": path.name,
        "rows": table.num_rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "arrow_schema_sha256": _schema_sha256(schema),
    }


def _checked_root(root: str | os.PathLike[str], *, role: str) -> Path:
    path = Path(root).resolve()
    if path.is_symlink() or not path.is_dir():
        raise HergTierIntegrationError(f"missing or unsafe {role} root: {path}")
    return path


def _input_binding(role: str, path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HergTierIntegrationError(f"missing or unsafe {role} input: {path}")
    return {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _candidate_state(pubchem_labels: set[int], chembl_labels: set[int]) -> tuple[str, int | None]:
    if len(pubchem_labels) != 1 or len(chembl_labels) != 1:
        return "internally_ambiguous_review_candidate", None
    pubchem_label = next(iter(pubchem_labels))
    chembl_label = next(iter(chembl_labels))
    if pubchem_label == chembl_label:
        return "concordant_review_candidate", pubchem_label
    return "discordant_review_candidate", None


def _build_t1_candidates(
    observations: Sequence[dict[str, Any]], structures: Mapping[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    qualifying: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    allowed_families = {"pubchem_aid720551", "chembl_herg_specialized_view"}
    for row in observations:
        structure_id = row.get("structure_id")
        family = str(row.get("source_family"))
        label = row.get("derived_binary_label")
        if (
            structure_id is not None
            and family in allowed_families
            and row.get("t1_candidate") is True
            and label in {0, 1}
        ):
            qualifying[str(structure_id)][family].append(row)

    candidates: list[dict[str, Any]] = []
    states: dict[str, str] = {}
    for structure_id in sorted(qualifying):
        evidence = qualifying[structure_id]
        pubchem = evidence.get("pubchem_aid720551", [])
        chembl = evidence.get("chembl_herg_specialized_view", [])
        if not pubchem or not chembl:
            continue
        if structure_id not in structures:
            raise HergTierIntegrationError("qualifying observation refers to an unknown hierarchy structure")
        pubchem_labels = {int(row["derived_binary_label"]) for row in pubchem}
        chembl_labels = {int(row["derived_binary_label"]) for row in chembl}
        state, concordant_label = _candidate_state(pubchem_labels, chembl_labels)
        structure = structures[structure_id]
        candidates.append(
            {
                "candidate_id": _stable_id(structure_id, state),
                "structure_id": structure_id,
                "standardized_smiles": structure["standardized_smiles"],
                "standard_inchi_key": structure["standard_inchi_key"],
                "pubchem_qualifying_observation_count": len(pubchem),
                "pubchem_labels_json": _canonical_json(sorted(pubchem_labels)),
                "pubchem_source_record_ids_json": _canonical_json(
                    sorted(str(row["source_record_id"]) for row in pubchem)
                ),
                "chembl_qualifying_observation_count": len(chembl),
                "chembl_labels_json": _canonical_json(sorted(chembl_labels)),
                "chembl_source_record_ids_json": _canonical_json(
                    sorted(str(row["source_record_id"]) for row in chembl)
                ),
                "chembl_assay_ids_json": _canonical_json(
                    sorted({str(row["assay_id"]) for row in chembl if row.get("assay_id") is not None})
                ),
                "candidate_state": state,
                "concordant_binary_label": concordant_label,
                "upstream_lineage_independence_adjudicated": False,
                "assay_modality_comparability_adjudicated": False,
                "formal_t1_assigned": False,
                "model_label_admitted": False,
                "candidate_semantics": T1_SEMANTICS,
            }
        )
        states[structure_id] = state
    return candidates, states


def build_herg_evidence_tiers(
    hierarchy_root: str | os.PathLike[str],
    clinical_links_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Create the unified candidate inventory while assigning only T0."""

    hierarchy = _checked_root(hierarchy_root, role="hERG hierarchy")
    clinical = _checked_root(clinical_links_root, role="clinical-link")
    try:
        validate_herg_hierarchy(hierarchy)
        verify_herg_clinical_links(clinical)
    except Exception as error:
        raise HergTierIntegrationError("upstream hERG artifact verification failed") from error

    output = Path(output_root).resolve()
    if output.exists() and (output.is_symlink() or not output.is_dir() or any(output.iterdir())):
        raise HergTierIntegrationError("output directory must be absent or empty and may not be a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)

    hierarchy_rows = pq.read_table(hierarchy / "hierarchy_annotations.parquet").to_pylist()
    observations = pq.read_table(hierarchy / "observation_ledger.parquet").to_pylist()
    structures = {str(row["structure_id"]): row for row in hierarchy_rows}
    if len(structures) != len(hierarchy_rows):
        raise HergTierIntegrationError("hierarchy annotations are not one row per structure")

    observed_counts: dict[str, int] = defaultdict(int)
    observed_families: dict[str, set[str]] = defaultdict(set)
    for row in observations:
        structure_id = row.get("structure_id")
        if row.get("structure_valid") is True and structure_id is not None:
            key = str(structure_id)
            if key not in structures:
                raise HergTierIntegrationError("valid observation refers to an unknown hierarchy structure")
            observed_counts[key] += 1
            observed_families[key].add(str(row["source_family"]))
    for structure_id, row in structures.items():
        if observed_counts[structure_id] != int(row["reported_observation_count"]):
            raise HergTierIntegrationError("hierarchy/observation reported count mismatch")
        if _canonical_json(sorted(observed_families[structure_id])) != row["source_families_json"]:
            raise HergTierIntegrationError("hierarchy/observation source-family mismatch")

    t1_candidates, t1_states = _build_t1_candidates(observations, structures)

    development_rows = pq.read_table(clinical / STRUCTURE_OUTPUT).to_pylist()
    development: dict[str, dict[str, Any]] = {}
    for row in development_rows:
        structure_id = str(row["molecule_id"])
        if structure_id in development:
            raise HergTierIntegrationError("clinical development annotations are not unique by structure")
        if structure_id not in structures:
            raise HergTierIntegrationError("clinical development annotation refers to an unknown structure")
        development[structure_id] = row

    t2_rows = pq.read_table(clinical / T2_OUTPUT).to_pylist()
    t3_rows = pq.read_table(clinical / T3_OUTPUT).to_pylist()
    t2_keys = {(row["candidate_id"], row["molecule_id"], row["nct_id"]) for row in t2_rows}
    t3_keys = {(row["candidate_id"], row["molecule_id"], row["nct_id"]) for row in t3_rows}
    if t2_keys != t3_keys:
        raise HergTierIntegrationError("T2/T3 posted-QT candidate inventories diverge")

    posted_qt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in t3_rows:
        structure_id = str(row["molecule_id"])
        if structure_id not in structures:
            raise HergTierIntegrationError("posted QT candidate refers to an unknown structure")
        if row.get("candidate_rule_passed") is not True or row.get("model_label_admitted") is not False:
            raise HergTierIntegrationError("posted QT evidence violated candidate-only semantics")
        posted_qt[structure_id].append(row)

    structure_rows: list[dict[str, Any]] = []
    for structure_id in sorted(structures):
        hierarchy_row = structures[structure_id]
        development_row = development.get(structure_id)
        qt_rows = posted_qt.get(structure_id, [])
        candidate_state = t1_states.get(structure_id, "no_cross_lineage_candidate")
        structure_rows.append(
            {
                "structure_id": structure_id,
                "standardized_smiles": hierarchy_row["standardized_smiles"],
                "standard_inchi_key": hierarchy_row["standard_inchi_key"],
                "t0_reported": True,
                "t0_observation_count": observed_counts[structure_id],
                "t0_source_families_json": _canonical_json(sorted(observed_families[structure_id])),
                "hierarchy_consensus_status": hierarchy_row["consensus_status"],
                "hierarchy_consensus_binary_label": hierarchy_row["consensus_binary_label"],
                "cross_lineage_t1_candidate": structure_id in t1_states,
                "cross_lineage_t1_candidate_state": candidate_state,
                "cross_lineage_t1_candidate_count": int(structure_id in t1_states),
                "clinical_development_annotation": bool(
                    development_row and development_row["clinical_development_annotation"]
                ),
                "chembl_max_phase": development_row["chembl_max_phase"] if development_row else None,
                "chembl_first_approval": development_row["chembl_first_approval"]
                if development_row
                else None,
                "chembl_therapeutic_flag": bool(
                    development_row and development_row["chembl_therapeutic_flag"]
                ),
                "chembl_dosed_ingredient": bool(
                    development_row and development_row["chembl_dosed_ingredient"]
                ),
                "chembl_withdrawn_flag": bool(development_row and development_row["chembl_withdrawn_flag"]),
                "drugsfda_exact_name_link_count": int(
                    development_row["drugsfda_exact_name_link_count"] if development_row else 0
                ),
                "exact_posted_qt_candidate_count": len(qt_rows),
                "exact_posted_qt_distinct_nct_count": len({str(row["nct_id"]) for row in qt_rows}),
                "exact_posted_qt_reported_numeric_value_count": sum(
                    int(row["reported_numeric_value_count"]) for row in qt_rows
                ),
                "clinical_cardiac_candidate_evidence": bool(qt_rows),
                "formal_highest_assigned_tier": FORMAL_TIER,
                "formal_t1_assigned": False,
                "formal_t2_assigned": False,
                "formal_t3_assigned": False,
                "clinical_herg_label_admitted": False,
                "model_label_admitted": False,
                "tier_semantics": CLINICAL_SEMANTICS,
            }
        )

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        artifacts = [
            _write_parquet(staging / STRUCTURE_TIER_OUTPUT, structure_rows, _STRUCTURE_TIER_SCHEMA),
            _write_parquet(staging / T1_CANDIDATE_OUTPUT, t1_candidates, _T1_CANDIDATE_SCHEMA),
        ]
        input_bindings = [
            _input_binding("hierarchy_manifest", hierarchy / "manifest.json"),
            _input_binding("hierarchy_observation_ledger", hierarchy / "observation_ledger.parquet"),
            _input_binding("hierarchy_annotations", hierarchy / "hierarchy_annotations.parquet"),
            _input_binding("clinical_links_manifest", clinical / "herg_clinical_links_manifest.json"),
            _input_binding("clinical_development_annotations", clinical / STRUCTURE_OUTPUT),
            _input_binding("clinical_t2_candidates", clinical / T2_OUTPUT),
            _input_binding("clinical_t3_posted_qt_candidates", clinical / T3_OUTPUT),
        ]
        state_counts: dict[str, int] = defaultdict(int)
        for row in t1_candidates:
            state_counts[str(row["candidate_state"])] += 1
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": "herg_unified_candidate_evidence_tiers",
            "input_bindings": input_bindings,
            "input_set_sha256": hashlib.sha256(_canonical_json(input_bindings).encode("utf-8")).hexdigest(),
            "scientific_contract": {
                "formal_highest_assigned_tier": FORMAL_TIER,
                "formal_t1_assignments": 0,
                "formal_t2_assignments": 0,
                "formal_t3_assignments": 0,
                "cross_lineage_state": T1_SEMANTICS,
                "clinical_state": CLINICAL_SEMANTICS,
                "clinical_development_is_t2": False,
                "posted_qt_is_herg_label": False,
                "absence_is_negative_evidence": False,
            },
            "counts": {
                "structures": len(structure_rows),
                "T0_reported_structures": len(structure_rows),
                "cross_lineage_T1_candidates": len(t1_candidates),
                "cross_lineage_candidate_states": dict(sorted(state_counts.items())),
                "clinical_development_annotations": sum(
                    bool(row["clinical_development_annotation"]) for row in structure_rows
                ),
                "structures_with_posted_QT_candidates": sum(
                    bool(row["clinical_cardiac_candidate_evidence"]) for row in structure_rows
                ),
                "posted_QT_candidate_records": sum(
                    int(row["exact_posted_qt_candidate_count"]) for row in structure_rows
                ),
                "formal_T1_assignments": 0,
                "formal_T2_assignments": 0,
                "formal_T3_assignments": 0,
                "model_labels_admitted": 0,
            },
            "artifacts": artifacts,
            "artifact_set_sha256": hashlib.sha256(_canonical_json(artifacts).encode("utf-8")).hexdigest(),
        }
        manifest["manifest_sha256"] = hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        validate_herg_evidence_tiers(staging)
        if output.exists():
            output.rmdir()
        os.replace(staging, output)
        return validate_herg_evidence_tiers(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_herg_evidence_tiers(output_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Verify deterministic artifacts and enforce zero-promotion semantics."""

    root = Path(output_root)
    manifest_path = root / MANIFEST_NAME
    if root.is_symlink() or not root.is_dir() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise HergTierIntegrationError(f"missing or unsafe evidence-tier output: {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HergTierIntegrationError("unreadable evidence-tier manifest") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise HergTierIntegrationError("unexpected evidence-tier manifest schema")
    declared_digest = manifest.get("manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest() != declared_digest:
        raise HergTierIntegrationError("evidence-tier manifest digest mismatch")
    contract = manifest.get("scientific_contract", {})
    counts = manifest.get("counts", {})
    if (
        contract.get("formal_highest_assigned_tier") != FORMAL_TIER
        or contract.get("clinical_development_is_t2") is not False
        or contract.get("posted_qt_is_herg_label") is not False
        or any(
            int(counts.get(key, -1)) != 0
            for key in (
                "formal_T1_assignments",
                "formal_T2_assignments",
                "formal_T3_assignments",
                "model_labels_admitted",
            )
        )
    ):
        raise HergTierIntegrationError("evidence-tier scientific contract was promoted or weakened")

    schemas = {
        STRUCTURE_TIER_OUTPUT: _STRUCTURE_TIER_SCHEMA,
        T1_CANDIDATE_OUTPUT: _T1_CANDIDATE_SCHEMA,
    }
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or {item.get("path") for item in artifacts} != set(schemas):
        raise HergTierIntegrationError("evidence-tier artifact membership mismatch")
    if {path.name for path in root.iterdir()} != {MANIFEST_NAME, *schemas} or any(
        path.is_symlink() or not path.is_file() for path in root.iterdir()
    ):
        raise HergTierIntegrationError("evidence-tier output contains unexpected or unsafe members")
    bindings = manifest.get("input_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise HergTierIntegrationError("evidence-tier input bindings are missing")
    for binding in bindings:
        if not isinstance(binding, Mapping):
            raise HergTierIntegrationError("evidence-tier input binding is malformed")
        path = Path(str(binding.get("path", "")))
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(binding.get("bytes", -1))
            or _sha256_file(path) != binding.get("sha256")
        ):
            raise HergTierIntegrationError(f"evidence-tier input binding mismatch: {path}")
    for artifact in artifacts:
        path = root / str(artifact["path"])
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(artifact.get("bytes", -1))
            or _sha256_file(path) != artifact.get("sha256")
        ):
            raise HergTierIntegrationError(f"evidence-tier artifact hash mismatch: {path.name}")
        parquet = pq.ParquetFile(path)
        if parquet.schema_arrow != schemas[path.name]:
            raise HergTierIntegrationError(f"evidence-tier artifact schema mismatch: {path.name}")
        if parquet.metadata is None or parquet.metadata.num_rows != int(artifact.get("rows", -1)):
            raise HergTierIntegrationError(f"evidence-tier artifact row-count mismatch: {path.name}")

    structures = pq.read_table(root / STRUCTURE_TIER_OUTPUT).to_pylist()
    structure_ids = [row["structure_id"] for row in structures]
    if len(structure_ids) != len(set(structure_ids)) or structure_ids != sorted(structure_ids):
        raise HergTierIntegrationError("structure-tier table is not unique and sorted")
    for row in structures:
        if (
            row["t0_reported"] is not True
            or row["formal_highest_assigned_tier"] != FORMAL_TIER
            or any(
                row[field] is not False
                for field in (
                    "formal_t1_assigned",
                    "formal_t2_assigned",
                    "formal_t3_assigned",
                    "clinical_herg_label_admitted",
                    "model_label_admitted",
                )
            )
        ):
            raise HergTierIntegrationError("a structure was promoted beyond T0 or admitted as a label")
        has_t1 = bool(row["cross_lineage_t1_candidate"])
        if int(row["cross_lineage_t1_candidate_count"]) != int(has_t1):
            raise HergTierIntegrationError("structure T1 candidate count/flag mismatch")
        if bool(row["clinical_cardiac_candidate_evidence"]) != (row["exact_posted_qt_candidate_count"] > 0):
            raise HergTierIntegrationError("structure clinical-candidate count/flag mismatch")

    candidates = pq.read_table(root / T1_CANDIDATE_OUTPUT).to_pylist()
    candidate_ids = [row["candidate_id"] for row in candidates]
    candidate_structure_ids = [row["structure_id"] for row in candidates]
    if (
        len(candidate_ids) != len(set(candidate_ids))
        or len(candidate_structure_ids) != len(set(candidate_structure_ids))
        or candidate_structure_ids != sorted(candidate_structure_ids)
        or not set(candidate_structure_ids).issubset(set(structure_ids))
    ):
        raise HergTierIntegrationError("T1 candidate table identity contract failed")
    valid_states = {
        "concordant_review_candidate",
        "discordant_review_candidate",
        "internally_ambiguous_review_candidate",
    }
    for row in candidates:
        if (
            row["candidate_state"] not in valid_states
            or row["pubchem_qualifying_observation_count"] < 1
            or row["chembl_qualifying_observation_count"] < 1
            or row["upstream_lineage_independence_adjudicated"] is not False
            or row["assay_modality_comparability_adjudicated"] is not False
            or row["formal_t1_assigned"] is not False
            or row["model_label_admitted"] is not False
        ):
            raise HergTierIntegrationError("T1 candidate scientific contract failed")
        labels = json.loads(row["pubchem_labels_json"]), json.loads(row["chembl_labels_json"])
        expected_state, expected_label = _candidate_state(set(labels[0]), set(labels[1]))
        if row["candidate_state"] != expected_state or row["concordant_binary_label"] != expected_label:
            raise HergTierIntegrationError("T1 candidate concordance state mismatch")

    if counts.get("structures") != len(structures) or counts.get("cross_lineage_T1_candidates") != len(
        candidates
    ):
        raise HergTierIntegrationError("evidence-tier manifest count mismatch")
    if sum(int(row["exact_posted_qt_candidate_count"]) for row in structures) != counts.get(
        "posted_QT_candidate_records"
    ):
        raise HergTierIntegrationError("posted-QT manifest count mismatch")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hierarchy-root", type=Path)
    parser.add_argument("--clinical-links-root", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.validate_only:
        validate_herg_evidence_tiers(args.output_root)
        return 0
    if args.hierarchy_root is None or args.clinical_links_root is None:
        raise SystemExit("build mode requires --hierarchy-root and --clinical-links-root")
    build_herg_evidence_tiers(args.hierarchy_root, args.clinical_links_root, args.output_root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
