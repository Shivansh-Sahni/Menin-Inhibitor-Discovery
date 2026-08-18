"""Build deterministic, review-oriented hERG assets before HPC modeling.

These tables prioritize evidence for human review.  In particular, the
``gold_standard_evaluation_candidates`` table is a candidate inventory, not an
adjudicated gold standard and not an authorization to use every row as a test
label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .platform_herg_master_dataset import validate_herg_master_dataset

SCHEMA_VERSION = "platform-herg-pre-hpc-assets/1.0"
MANIFEST_NAME = "herg_pre_hpc_assets_manifest.json"
REPORT_NAME = "HERG_PRE_HPC_ASSETS.md"
GOLD_OUTPUT = "gold_standard_evaluation_candidates.parquet"
CONFLICT_OUTPUT = "replicated_pic50_conflict_review_queue.parquet"
PROTOCOL_OUTPUT = "assay_protocol_enrichment_priority_queue.parquet"
PIC50_CONFLICT_TOLERANCE = 1e-6


class HergPreHpcAssetError(RuntimeError):
    """Raised when pre-HPC asset compilation or validation fails closed."""


_GOLD_SCHEMA = pa.schema(
    [
        pa.field("candidate_rank", pa.int64(), nullable=False),
        pa.field("candidate_id", pa.large_string(), nullable=False),
        pa.field("candidate_status", pa.large_string(), nullable=False),
        pa.field("priority_score", pa.float64(), nullable=False),
        pa.field("observation_id", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("standardized_smiles", pa.large_string(), nullable=False),
        pa.field("standard_inchi_key", pa.large_string(), nullable=False),
        pa.field("wild_type_evidence_scope", pa.large_string(), nullable=False),
        pa.field("source_family", pa.large_string(), nullable=False),
        pa.field("source_record_id", pa.large_string(), nullable=False),
        pa.field("assay_id", pa.large_string()),
        pa.field("assay_family", pa.large_string(), nullable=False),
        pa.field("measurement_modality", pa.large_string(), nullable=False),
        pa.field("automation_class", pa.large_string(), nullable=False),
        pa.field("native_endpoint", pa.large_string(), nullable=False),
        pa.field("native_relation", pa.large_string()),
        pa.field("native_value", pa.float64()),
        pa.field("native_unit", pa.large_string()),
        pa.field("potency_relation_pic50", pa.large_string(), nullable=False),
        pa.field("potency_pic50_point", pa.float64()),
        pa.field("potency_pic50_lower_bound", pa.float64()),
        pa.field("potency_pic50_upper_bound", pa.float64()),
        pa.field("potency_censoring", pa.large_string(), nullable=False),
        pa.field("protocol_completeness_score", pa.int8(), nullable=False),
        pa.field("protocol_unresolved_fields_json", pa.large_string(), nullable=False),
        pa.field("replicated_exact_pic50_count", pa.int64(), nullable=False),
        pa.field("model_split", pa.large_string(), nullable=False),
        pa.field("scaffold_group_id", pa.large_string(), nullable=False),
        pa.field("original_native_aux_json", pa.large_string(), nullable=False),
        pa.field("candidate_limitations_json", pa.large_string(), nullable=False),
    ]
)

_CONFLICT_SCHEMA = pa.schema(
    [
        pa.field("review_rank", pa.int64(), nullable=False),
        pa.field("review_id", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("standardized_smiles", pa.large_string(), nullable=False),
        pa.field("standard_inchi_key", pa.large_string(), nullable=False),
        pa.field("review_priority", pa.large_string(), nullable=False),
        pa.field("priority_score", pa.float64(), nullable=False),
        pa.field("exact_replicate_count", pa.int64(), nullable=False),
        pa.field("pic50_minimum", pa.float64(), nullable=False),
        pa.field("pic50_median", pa.float64(), nullable=False),
        pa.field("pic50_mean", pa.float64(), nullable=False),
        pa.field("pic50_maximum", pa.float64(), nullable=False),
        pa.field("pic50_range", pa.float64(), nullable=False),
        pa.field("pic50_sample_standard_deviation", pa.float64()),
        pa.field("source_count", pa.int64(), nullable=False),
        pa.field("assay_count", pa.int64(), nullable=False),
        pa.field("modality_count", pa.int64(), nullable=False),
        pa.field("automation_class_count", pa.int64(), nullable=False),
        pa.field("maximum_protocol_completeness", pa.int8(), nullable=False),
        pa.field("wild_type_scopes_json", pa.large_string(), nullable=False),
        pa.field("source_families_json", pa.large_string(), nullable=False),
        pa.field("assay_ids_json", pa.large_string(), nullable=False),
        pa.field("measurement_modalities_json", pa.large_string(), nullable=False),
        pa.field("automation_classes_json", pa.large_string(), nullable=False),
        pa.field("observation_ids_json", pa.large_string(), nullable=False),
        pa.field("source_record_ids_json", pa.large_string(), nullable=False),
        pa.field("replicate_evidence_json", pa.large_string(), nullable=False),
        pa.field("review_reason", pa.large_string(), nullable=False),
    ]
)

_PROTOCOL_SCHEMA = pa.schema(
    [
        pa.field("priority_rank", pa.int64(), nullable=False),
        pa.field("priority_id", pa.large_string(), nullable=False),
        pa.field("priority_score", pa.float64(), nullable=False),
        pa.field("assay_catalog_id", pa.large_string(), nullable=False),
        pa.field("source_family", pa.large_string(), nullable=False),
        pa.field("assay_id", pa.large_string()),
        pa.field("assay_family", pa.large_string(), nullable=False),
        pa.field("observation_count", pa.int64(), nullable=False),
        pa.field("unresolved_field_count", pa.int64(), nullable=False),
        pa.field("unresolved_fields_json", pa.large_string(), nullable=False),
        pa.field("protocol_completeness_score", pa.int8(), nullable=False),
        pa.field("host_systems_json", pa.large_string(), nullable=False),
        pa.field("named_platforms_json", pa.large_string(), nullable=False),
        pa.field("recording_configurations_json", pa.large_string(), nullable=False),
        pa.field("source_automation_classes_json", pa.large_string(), nullable=False),
        pa.field("raw_protocol_text_json", pa.large_string(), nullable=False),
        pa.field("source_contract_evidence_json", pa.large_string(), nullable=False),
        pa.field("enrichment_action", pa.large_string(), nullable=False),
    ]
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join("" if part is None else str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode()).hexdigest()[:24].upper()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked(path: Path, columns: Sequence[str]) -> Path:
    if path.is_symlink() or not path.is_file():
        raise HergPreHpcAssetError(f"missing or unsafe input: {path}")
    missing = sorted(set(columns) - set(pq.ParquetFile(path).schema_arrow.names))
    if missing:
        raise HergPreHpcAssetError(f"{path.name} is missing columns: {missing}")
    return path.resolve()


def _artifact(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    return {
        "path": path.name,
        "rows": parquet.metadata.num_rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "arrow_schema_sha256": hashlib.sha256(parquet.schema_arrow.serialize().to_pybytes()).hexdigest(),
    }


def _write(path: Path, rows: Sequence[Mapping[str, Any]], schema: pa.Schema) -> dict[str, Any]:
    table = pa.Table.from_pylist(list(rows), schema=schema)
    pq.write_table(
        table, path, compression="zstd", use_dictionary=False, row_group_size=65_536, version="2.6"
    )
    return _artifact(path)


def _protocol_map(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in pq.read_table(path).to_pylist():
        key = (str(row["source_family"]), str(row["assay_id"] or ""), str(row["assay_family"]))
        output[key] = row
    return output


def _protocol_for(
    row: Mapping[str, Any], protocols: Mapping[tuple[str, str, str], dict[str, Any]]
) -> dict[str, Any]:
    key = (str(row["source_family"]), str(row.get("assay_id") or ""), str(row["assay_family"]))
    return protocols.get(key, {"protocol_completeness_score": 0, "unresolved_fields_json": "[]"})


def _eligible_q2_observations(task_path: Path) -> set[str]:
    return {
        str(row["observation_id"])
        for row in pq.read_table(
            task_path, columns=["task_id", "observation_id", "eligible", "use_as_training_label"]
        ).to_pylist()
        if row["task_id"] == "Q2_FUNCTIONAL_ASSAY_AWARE"
        and row["observation_id"]
        and row["eligible"]
        and row["use_as_training_label"]
    }


def _exact_replicate_counts(observations: Sequence[Mapping[str, Any]]) -> Counter[str]:
    return Counter(
        str(row["structure_id"])
        for row in observations
        if row["structure_id"] and row["endpoint_standardization_status"] == "exact_standardized"
    )


def _gold_candidates(
    observations: Sequence[Mapping[str, Any]],
    q2_eligible: set[str],
    protocols: Mapping[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    replicates = _exact_replicate_counts(observations)
    output: list[dict[str, Any]] = []
    modality_score = {
        "patch_clamp_electrophysiology": 35.0,
        "functional_electrophysiology": 30.0,
        "functional_ion_flux": 18.0,
        "functional_unspecified": 8.0,
    }
    for row in observations:
        if str(row["observation_id"]) not in q2_eligible:
            continue
        if row["endpoint_standardization_status"] not in {"exact_standardized", "censored_standardized"}:
            continue
        if row["endpoint_class"] != "potency_ic50":
            continue
        protocol = _protocol_for(row, protocols)
        try:
            native_aux = json.loads(str(row["native_aux_json"]))
        except json.JSONDecodeError:
            native_aux = {}
        completeness = int(protocol["protocol_completeness_score"])
        score = (
            (30.0 if row["wild_type_evidence_scope"] == "confirmed_wild_type" else 10.0)
            + modality_score.get(str(row["measurement_modality"]), 4.0)
            + (24.0 if row["endpoint_standardization_status"] == "exact_standardized" else 14.0)
            + completeness * 4.0
            + (
                10.0
                if row["automation_class"] == "manual"
                else (5.0 if row["automation_class"] == "automated" else 0.0)
            )
            + min(10, max(0, replicates[str(row["structure_id"])] - 1))
        )
        limitations = ["candidate_not_adjudicated", "single_reported_observation_not_independent_retest"]
        if row["wild_type_evidence_scope"] != "confirmed_wild_type":
            limitations.append("wild_type_not_explicitly_confirmed_in_source_record")
        if row["endpoint_standardization_status"] == "censored_standardized":
            limitations.append("one_sided_censored_potency_boundary")
        if completeness < 4:
            limitations.append("incomplete_protocol_metadata")
        if native_aux.get("data_validity_comment"):
            limitations.append("source_data_validity_comment_requires_review")
            score -= 25.0
        if native_aux.get("potential_duplicate"):
            limitations.append("source_marks_potential_duplicate")
            score -= 15.0
        if native_aux.get("standard_flag") not in {None, 1}:
            limitations.append("source_standard_flag_not_asserted")
            score -= 10.0
        confidence = native_aux.get("confidence_score")
        if isinstance(confidence, (int, float)):
            score += min(10.0, max(0.0, float(confidence))) * 0.5
        output.append(
            {
                "candidate_rank": 0,
                "candidate_id": _stable_id("HGOLD", row["observation_id"]),
                "candidate_status": "evaluation_candidate_not_adjudicated_gold_standard",
                "priority_score": score,
                "observation_id": str(row["observation_id"]),
                "structure_id": str(row["structure_id"]),
                "standardized_smiles": str(row["standardized_smiles"]),
                "standard_inchi_key": str(row["standard_inchi_key"]),
                "wild_type_evidence_scope": str(row["wild_type_evidence_scope"]),
                "source_family": str(row["source_family"]),
                "source_record_id": str(row["source_record_id"]),
                "assay_id": row["assay_id"],
                "assay_family": str(row["assay_family"]),
                "measurement_modality": str(row["measurement_modality"]),
                "automation_class": str(row["automation_class"]),
                "native_endpoint": str(row["native_endpoint"]),
                "native_relation": row["native_relation"],
                "native_value": row["native_value"],
                "native_unit": row["native_unit"],
                "potency_relation_pic50": str(row["potency_relation_pic50"]),
                "potency_pic50_point": row["potency_pic50_point"],
                "potency_pic50_lower_bound": row["potency_pic50_lower_bound"],
                "potency_pic50_upper_bound": row["potency_pic50_upper_bound"],
                "potency_censoring": str(row["potency_censoring"]),
                "protocol_completeness_score": completeness,
                "protocol_unresolved_fields_json": str(protocol["unresolved_fields_json"]),
                "replicated_exact_pic50_count": replicates[str(row["structure_id"])],
                "model_split": str(row["model_split"]),
                "scaffold_group_id": str(row["scaffold_group_id"]),
                "original_native_aux_json": str(row["native_aux_json"]),
                "candidate_limitations_json": _canonical_json(sorted(limitations)),
            }
        )
    output.sort(key=lambda row: (-float(row["priority_score"]), str(row["candidate_id"])))
    for rank, row in enumerate(output, 1):
        row["candidate_rank"] = rank
    return output


def _conflict_queue(
    observations: Sequence[Mapping[str, Any]], protocols: Mapping[tuple[str, str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        if (
            row["structure_id"]
            and row["endpoint_standardization_status"] == "exact_standardized"
            and row["potency_pic50_point"] is not None
        ):
            groups[str(row["structure_id"])].append(row)
    output: list[dict[str, Any]] = []
    for structure_id, rows in groups.items():
        if len(rows) < 2:
            continue
        values = [float(row["potency_pic50_point"]) for row in rows]
        value_range = max(values) - min(values)
        if value_range <= PIC50_CONFLICT_TOLERANCE:
            continue
        modalities = sorted({str(row["measurement_modality"]) for row in rows})
        sources = sorted({str(row["source_family"]) for row in rows})
        assays = sorted({str(row["assay_id"]) for row in rows if row["assay_id"]})
        automation = sorted({str(row["automation_class"]) for row in rows})
        max_protocol = max(int(_protocol_for(row, protocols)["protocol_completeness_score"]) for row in rows)
        priority = (
            "critical"
            if value_range >= 2
            else ("high" if value_range >= 1 else ("moderate" if value_range >= 0.5 else "low"))
        )
        score = (
            value_range * 25.0
            + min(20.0, len(rows) * 2.0)
            + len(modalities) * 5.0
            + len(sources) * 3.0
            + max_protocol
        )
        first = rows[0]
        evidence = [
            {
                "observation_id": row["observation_id"],
                "source_record_id": row["source_record_id"],
                "pic50": row["potency_pic50_point"],
                "source_family": row["source_family"],
                "assay_id": row["assay_id"],
                "measurement_modality": row["measurement_modality"],
                "automation_class": row["automation_class"],
                "wild_type_scope": row["wild_type_evidence_scope"],
            }
            for row in sorted(rows, key=lambda item: str(item["observation_id"]))
        ]
        output.append(
            {
                "review_rank": 0,
                "review_id": _stable_id("HCONFLICT", structure_id),
                "structure_id": structure_id,
                "standardized_smiles": str(first["standardized_smiles"]),
                "standard_inchi_key": str(first["standard_inchi_key"]),
                "review_priority": priority,
                "priority_score": score,
                "exact_replicate_count": len(rows),
                "pic50_minimum": min(values),
                "pic50_median": statistics.median(values),
                "pic50_mean": statistics.fmean(values),
                "pic50_maximum": max(values),
                "pic50_range": value_range,
                "pic50_sample_standard_deviation": statistics.stdev(values),
                "source_count": len(sources),
                "assay_count": len(assays),
                "modality_count": len(modalities),
                "automation_class_count": len(automation),
                "maximum_protocol_completeness": max_protocol,
                "wild_type_scopes_json": _canonical_json(
                    sorted({str(row["wild_type_evidence_scope"]) for row in rows})
                ),
                "source_families_json": _canonical_json(sources),
                "assay_ids_json": _canonical_json(assays),
                "measurement_modalities_json": _canonical_json(modalities),
                "automation_classes_json": _canonical_json(automation),
                "observation_ids_json": _canonical_json(sorted(str(row["observation_id"]) for row in rows)),
                "source_record_ids_json": _canonical_json(
                    sorted(str(row["source_record_id"]) for row in rows)
                ),
                "replicate_evidence_json": _canonical_json(evidence),
                "review_reason": "same standardized structure has exact pIC50 range greater than 1e-6; adjudicate source identity, assay method, protocol, and duplicate lineage",
            }
        )
    output.sort(key=lambda row: (-float(row["priority_score"]), str(row["review_id"])))
    for rank, row in enumerate(output, 1):
        row["review_rank"] = rank
    return output


def _protocol_queue(protocol_path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in pq.read_table(protocol_path).to_pylist():
        unresolved = json.loads(str(row["unresolved_fields_json"]))
        if not unresolved:
            continue
        observations = int(row["observation_count"])
        score = math.log10(observations + 1) * 20.0 + len(unresolved) * 8.0 + min(20.0, observations / 1000.0)
        output.append(
            {
                "priority_rank": 0,
                "priority_id": _stable_id("HPROTOCOL", row["assay_catalog_id"]),
                "priority_score": score,
                "assay_catalog_id": str(row["assay_catalog_id"]),
                "source_family": str(row["source_family"]),
                "assay_id": row["assay_id"],
                "assay_family": str(row["assay_family"]),
                "observation_count": observations,
                "unresolved_field_count": len(unresolved),
                "unresolved_fields_json": str(row["unresolved_fields_json"]),
                "protocol_completeness_score": int(row["protocol_completeness_score"]),
                "host_systems_json": str(row["host_systems_json"]),
                "named_platforms_json": str(row["named_platforms_json"]),
                "recording_configurations_json": str(row["recording_configurations_json"]),
                "source_automation_classes_json": str(row["source_automation_classes_json"]),
                "raw_protocol_text_json": str(row["raw_protocol_text_json"]),
                "source_contract_evidence_json": str(row["source_contract_evidence_json"]),
                "enrichment_action": "retrieve primary assay publication or repository protocol and record only explicitly reported unresolved fields",
            }
        )
    output.sort(key=lambda row: (-float(row["priority_score"]), str(row["priority_id"])))
    for rank, row in enumerate(output, 1):
        row["priority_rank"] = rank
    return output


def build_herg_pre_hpc_assets(*, master_root: Path, output_root: Path, report_root: Path) -> dict[str, Any]:
    """Build review queues from the frozen master release."""

    master = master_root.resolve()
    validate_herg_master_dataset(master)
    observation_path = _checked(
        master / "observation_master.parquet", ["observation_id", "endpoint_standardization_status"]
    )
    task_path = _checked(master / "task_membership.parquet", ["task_id", "observation_id", "eligible"])
    protocol_path = _checked(
        master / "assay_protocol_index.parquet", ["assay_catalog_id", "unresolved_fields_json"]
    )
    output = output_root.resolve()
    report = report_root.resolve()
    if output.exists() or report.exists():
        raise HergPreHpcAssetError("output_root and report_root must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    report_staging = Path(tempfile.mkdtemp(prefix=f".{report.name}.", dir=report.parent))
    try:
        observations = pq.read_table(observation_path).to_pylist()
        protocols = _protocol_map(protocol_path)
        gold = _gold_candidates(observations, _eligible_q2_observations(task_path), protocols)
        conflicts = _conflict_queue(observations, protocols)
        protocol_queue = _protocol_queue(protocol_path)
        artifacts = {
            GOLD_OUTPUT: _write(staging / GOLD_OUTPUT, gold, _GOLD_SCHEMA),
            CONFLICT_OUTPUT: _write(staging / CONFLICT_OUTPUT, conflicts, _CONFLICT_SCHEMA),
            PROTOCOL_OUTPUT: _write(staging / PROTOCOL_OUTPUT, protocol_queue, _PROTOCOL_SCHEMA),
        }
        counts = {
            "gold_candidates": len(gold),
            "gold_confirmed_wild_type": sum(
                row["wild_type_evidence_scope"] == "confirmed_wild_type" for row in gold
            ),
            "gold_exact": sum(row["potency_censoring"] == "exact" for row in gold),
            "gold_censored": sum(row["potency_censoring"] != "exact" for row in gold),
            "replicated_pic50_conflicts": len(conflicts),
            "critical_conflicts": sum(row["review_priority"] == "critical" for row in conflicts),
            "high_conflicts": sum(row["review_priority"] == "high" for row in conflicts),
            "protocol_enrichment_priorities": len(protocol_queue),
            "protocol_impacted_observations_sum_exclusive": sum(
                int(row["observation_count"]) for row in protocol_queue
            ),
        }
        body = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": "wild_type_herg_v1_4_pre_hpc_review_assets",
            "inputs": [
                {
                    "path": str(path),
                    "rows": pq.ParquetFile(path).metadata.num_rows,
                    "sha256": _sha256_file(path),
                }
                for path in (observation_path, task_path, protocol_path)
            ]
            + [
                {
                    "path": str(master / "herg_master_manifest.json"),
                    "sha256": _sha256_file(master / "herg_master_manifest.json"),
                }
            ],
            "policies": {
                "gold_status": "candidate_only_not_adjudicated_gold_standard",
                "gold_admission": "Q2-eligible exact or correctly one-sided-censored functional IC50 with standardized structure",
                "conflict_queue": "two_or_more exact standardized pIC50 values for one structure with range greater than 1e-6",
                "conflict_numerical_equivalence_tolerance_pic50": PIC50_CONFLICT_TOLERANCE,
                "protocol_queue": "one_or_more unresolved protocol fields; ranking favors observation impact and missing dimensions",
                "no_labels_invented": True,
            },
            "counts": counts,
            "artifacts": artifacts,
        }
        manifest = dict(body)
        manifest["manifest_sha256"] = hashlib.sha256(_canonical_json(body).encode()).hexdigest()
        (staging / MANIFEST_NAME).write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
        report_lines = [
            "# hERG pre-HPC review assets",
            "",
            f"- Evaluation candidates: {counts['gold_candidates']:,} ({counts['gold_exact']:,} exact; {counts['gold_censored']:,} censored).",
            f"- Confirmed-WT evaluation candidates: {counts['gold_confirmed_wild_type']:,}; unspecified target status is retained rather than upgraded.",
            f"- Broader standardized-potency conflict structures: {counts['replicated_pic50_conflicts']:,} using pIC50 range > {PIC50_CONFLICT_TOLERANCE:g} ({counts['critical_conflicts']:,} critical; {counts['high_conflicts']:,} high).",
            f"- Protocol enrichment priorities: {counts['protocol_enrichment_priorities']:,} assays.",
            "",
            "The first table is explicitly a prioritization queue, not an adjudicated gold standard. Rows require duplicate-lineage review, protocol verification, and ideally independent retesting before becoming a locked evaluation panel.",
            "",
            "The broader conflict queue uses exact standardized IC50/pIC50 interpretations and excludes numerical conversion noise at or below 1e-6 pIC50. It preserves every observation and source record used in each range calculation. The protocol queue retrieves missing metadata; it must not infer absent voltage, temperature, host, timing, or platform details.",
            "",
            "No model was trained and no upstream artifact was modified.",
        ]
        (report_staging / REPORT_NAME).write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        os.replace(staging, output)
        os.replace(report_staging, report)
        validate_herg_pre_hpc_assets(output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(report_staging, ignore_errors=True)
        raise


def validate_herg_pre_hpc_assets(output_root: Path) -> dict[str, Any]:
    """Validate hashes, schemas, ordering, and candidate-only semantics."""

    root = output_root.resolve()
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    supplied = manifest.pop("manifest_sha256", None)
    expected = hashlib.sha256(_canonical_json(manifest).encode()).hexdigest()
    manifest["manifest_sha256"] = supplied
    if supplied != expected:
        raise HergPreHpcAssetError("manifest digest mismatch")
    for binding in manifest["inputs"]:
        source = Path(str(binding["path"]))
        if source.is_symlink() or not source.is_file() or _sha256_file(source) != binding["sha256"]:
            raise HergPreHpcAssetError(f"input binding mismatch: {source}")
        if "rows" in binding and pq.ParquetFile(source).metadata.num_rows != binding["rows"]:
            raise HergPreHpcAssetError(f"input row count mismatch: {source}")
    schemas = {
        GOLD_OUTPUT: _GOLD_SCHEMA,
        CONFLICT_OUTPUT: _CONFLICT_SCHEMA,
        PROTOCOL_OUTPUT: _PROTOCOL_SCHEMA,
    }
    for name, schema in schemas.items():
        path = root / name
        meta = manifest["artifacts"][name]
        if _sha256_file(path) != meta["sha256"] or pq.ParquetFile(path).schema_arrow != schema:
            raise HergPreHpcAssetError(f"artifact hash or schema mismatch: {name}")
        if pq.ParquetFile(path).metadata.num_rows != meta["rows"]:
            raise HergPreHpcAssetError(f"artifact row count mismatch: {name}")
    gold = pq.read_table(root / GOLD_OUTPUT).to_pylist()
    if any(row["candidate_status"] != "evaluation_candidate_not_adjudicated_gold_standard" for row in gold):
        raise HergPreHpcAssetError("candidate promoted to adjudicated gold standard")
    if any(
        row["potency_relation_pic50"] in {"<", "<="}
        and (row["potency_pic50_point"] is not None or row["potency_pic50_upper_bound"] is None)
        for row in gold
    ):
        raise HergPreHpcAssetError("invalid upper-bounded gold candidate")
    if [row["candidate_rank"] for row in gold] != list(range(1, len(gold) + 1)):
        raise HergPreHpcAssetError("candidate ranks are not deterministic")
    conflicts = pq.read_table(root / CONFLICT_OUTPUT).to_pylist()
    if any(
        row["exact_replicate_count"] < 2 or row["pic50_range"] <= PIC50_CONFLICT_TOLERANCE
        for row in conflicts
    ):
        raise HergPreHpcAssetError("invalid replicate conflict row")
    protocol = pq.ParquetFile(root / PROTOCOL_OUTPUT).metadata.num_rows
    if len(gold) != manifest["counts"]["gold_candidates"]:
        raise HergPreHpcAssetError("gold candidate count mismatch")
    if len(conflicts) != manifest["counts"]["replicated_pic50_conflicts"]:
        raise HergPreHpcAssetError("conflict count mismatch")
    if protocol != manifest["counts"]["protocol_enrichment_priorities"]:
        raise HergPreHpcAssetError("protocol priority count mismatch")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        validate_herg_pre_hpc_assets(args.output_root)
    else:
        if args.master_root is None or args.report_root is None:
            raise HergPreHpcAssetError("--master-root and --report-root are required when building")
        build_herg_pre_hpc_assets(
            master_root=args.master_root, output_root=args.output_root, report_root=args.report_root
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
