"""Build a grain-explicit operational hERG evidence hierarchy.

The operational hierarchy is a compact index over verified hERG hierarchy and
clinical-link artifacts.  It deliberately separates an ``operational_stage``
from the physical ``record_grain`` used for each headline:

* O0 indexes all reported hERG observations;
* O1 indexes curated or quantitative preclinical observations;
* O2 indexes structures with development or regulatory annotations;
* O3 indexes exact, unique clinical-trial intervention links; and
* O3-QT indexes result-value and denominator records from posted QT/QTc
  endpoints without copying the native payload.

O2 and O3 are context, not hERG labels.  The builder and validator fail closed
if clinical context is promoted into a hERG or model label, if headline grain
is conflated with a stage, or if a declared headline has fewer than 1,000
records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from menin_discovery.platform_herg_clinical_links import (
    LINK_AUDIT_OUTPUT,
    STRUCTURE_OUTPUT,
    T3_OUTPUT,
    verify_herg_clinical_links,
)
from menin_discovery.platform_herg_clinical_links import (
    MANIFEST_NAME as CLINICAL_MANIFEST_NAME,
)
from menin_discovery.platform_herg_hierarchy import validate_herg_hierarchy

SCHEMA_VERSION = "platform-herg-operational-tiers/1.1"
MANIFEST_NAME = "herg_operational_tiers_manifest.json"
RECORD_OUTPUT = "operational_stage_records.parquet"
QT_RECORD_OUTPUT = "operational_qt_record_index.parquet"
SUMMARY_OUTPUT = "operational_stage_summary.parquet"
MIN_HEADLINE_COUNT = 1_000

O0 = "O0_PUBLIC_REPORTED_HERG"
O1 = "O1_CURATED_QUANTITATIVE_PRECLINICAL"
O2 = "O2_CLINICAL_DEVELOPMENT_REGULATORY"
O3 = "O3_CLINICAL_TRIAL_INTERVENTION_REPORTED"
O3_QT = "O3_QT_POSTED_RESULT"
STAGE_ORDER = (O0, O1, O2, O3, O3_QT)
HEADLINE_GRAIN = {
    O0: "observation",
    O1: "observation",
    O2: "structure",
    O3: "intervention_link",
    O3_QT: "result_value",
}
PRECLINICAL_SOURCE_FAMILIES = frozenset({"chembl_herg_specialized_view", "quantitative_pic50_release"})
CLINICAL_STAGES = frozenset({O2, O3, O3_QT})


class HergOperationalTierError(RuntimeError):
    """Raised when operational-tier construction or validation fails closed."""


_RECORD_SCHEMA = pa.schema(
    [
        pa.field("operational_stage", pa.large_string(), nullable=False),
        pa.field("record_grain", pa.large_string(), nullable=False),
        pa.field("record_id", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string()),
        pa.field("observation_id", pa.large_string()),
        pa.field("source_family", pa.large_string()),
        pa.field("source_record_id", pa.large_string(), nullable=False),
        pa.field("nct_id", pa.large_string()),
        pa.field("endpoint_candidate_id", pa.large_string()),
        pa.field("upstream_artifact", pa.large_string(), nullable=False),
        pa.field("upstream_row_pointer", pa.large_string(), nullable=False),
        pa.field("direct_herg_assay_evidence", pa.bool_(), nullable=False),
        pa.field("clinical_context_only", pa.bool_(), nullable=False),
        pa.field("clinical_context_used_as_herg_label", pa.bool_(), nullable=False),
        pa.field("model_label_admitted_from_context", pa.bool_(), nullable=False),
    ]
)

_QT_RECORD_SCHEMA = pa.schema(
    [
        pa.field("operational_stage", pa.large_string(), nullable=False),
        pa.field("record_grain", pa.large_string(), nullable=False),
        pa.field("record_id", pa.large_string(), nullable=False),
        pa.field("source_candidate_id", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("nct_id", pa.large_string(), nullable=False),
        pa.field("endpoint_candidate_id", pa.large_string(), nullable=False),
        pa.field("record_ordinal", pa.int64(), nullable=False),
        pa.field("reported_value_is_numeric", pa.bool_(), nullable=False),
        pa.field("source_page_path", pa.large_string(), nullable=False),
        pa.field("raw_json_pointer", pa.large_string(), nullable=False),
        pa.field("clinical_context_used_as_herg_label", pa.bool_(), nullable=False),
        pa.field("model_label_admitted_from_context", pa.bool_(), nullable=False),
    ]
)

_SUMMARY_SCHEMA = pa.schema(
    [
        pa.field("operational_stage", pa.large_string(), nullable=False),
        pa.field("stage_name", pa.large_string(), nullable=False),
        pa.field("headline_record_grain", pa.large_string(), nullable=False),
        pa.field("headline_record_count", pa.int64(), nullable=False),
        pa.field("indexed_record_count", pa.int64(), nullable=False),
        pa.field("unique_structures", pa.int64(), nullable=False),
        pa.field("unique_trials", pa.int64(), nullable=False),
        pa.field("unique_endpoints", pa.int64(), nullable=False),
        pa.field("direct_herg_assay_evidence", pa.bool_(), nullable=False),
        pa.field("clinical_context_used_as_herg_label", pa.bool_(), nullable=False),
        pa.field("model_label_admitted_from_context", pa.bool_(), nullable=False),
        pa.field("disclosed_counts_json", pa.large_string(), nullable=False),
        pa.field("stage_semantics", pa.large_string(), nullable=False),
    ]
)

_STAGE_NAME = {
    O0: "All clean reported hERG evidence",
    O1: "Curated or quantitative preclinical hERG evidence",
    O2: "Clinical-development or regulatory context",
    O3: "Exact clinical-trial intervention links",
    O3_QT: "Posted QT/QTc result-record index",
}

_STAGE_SEMANTICS = {
    O0: "native_or_derived_hERG_assay_observation_reference",
    O1: "curated_ChEMBL_or_quantitative_pIC50_observation_reference",
    O2: "structure_level_development_or_regulatory_context_not_a_hERG_label",
    O3: "exact_unique_intervention_structure_link_not_a_hERG_label",
    O3_QT: "posted_QT_QTc_result_or_denominator_reference_not_a_hERG_label",
}


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


def _checked_root(root: str | os.PathLike[str], *, role: str) -> Path:
    path = Path(root).resolve()
    if path.is_symlink() or not path.is_dir():
        raise HergOperationalTierError(f"missing or unsafe {role} root: {path}")
    return path


def _input_binding(role: str, path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HergOperationalTierError(f"missing or unsafe {role} input: {path}")
    return {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _artifact(path: Path, schema: pa.Schema, rows: int) -> dict[str, Any]:
    return {
        "path": path.name,
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "arrow_schema_sha256": _schema_sha256(schema),
    }


def _write_table(path: Path, rows: Sequence[Mapping[str, Any]], schema: pa.Schema) -> dict[str, Any]:
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
    return _artifact(path, schema, table.num_rows)


def _write_batches(
    path: Path, schema: pa.Schema, batches: Iterable[Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    row_count = 0
    writer = pq.ParquetWriter(
        path,
        schema,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
        version="2.6",
    )
    try:
        for rows in batches:
            if not rows:
                continue
            table = pa.Table.from_pylist(list(rows), schema=schema)
            writer.write_table(table)
            row_count += table.num_rows
    finally:
        writer.close()
    return _artifact(path, schema, row_count)


def _parquet_rows(path: Path, columns: Sequence[str] | None = None) -> Iterable[list[dict[str, Any]]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=25_000, columns=columns):
        yield batch.to_pylist()


def _numeric(value: Any) -> bool:
    if value is None:
        return False
    try:
        return Decimal(str(value).strip()).is_finite()
    except (InvalidOperation, ValueError):
        return False


def _load_json_list(value: Any, *, field: str) -> list[Any]:
    try:
        decoded = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as error:
        raise HergOperationalTierError(f"invalid JSON in {field}") from error
    if not isinstance(decoded, list):
        raise HergOperationalTierError(f"{field} is not a JSON list")
    return decoded


def _record_row(
    *,
    stage: str,
    grain: str,
    record_id: str,
    structure_id: Any,
    observation_id: Any,
    source_family: Any,
    source_record_id: str,
    nct_id: Any,
    endpoint_candidate_id: Any,
    upstream_artifact: str,
    direct_assay: bool,
) -> dict[str, Any]:
    clinical = stage in CLINICAL_STAGES
    return {
        "operational_stage": stage,
        "record_grain": grain,
        "record_id": record_id,
        "structure_id": None if structure_id is None else str(structure_id),
        "observation_id": None if observation_id is None else str(observation_id),
        "source_family": None if source_family is None else str(source_family),
        "source_record_id": source_record_id,
        "nct_id": None if nct_id is None else str(nct_id),
        "endpoint_candidate_id": (None if endpoint_candidate_id is None else str(endpoint_candidate_id)),
        "upstream_artifact": upstream_artifact,
        "upstream_row_pointer": source_record_id,
        "direct_herg_assay_evidence": direct_assay,
        "clinical_context_only": clinical,
        "clinical_context_used_as_herg_label": False,
        "model_label_admitted_from_context": False,
    }


def build_herg_operational_tiers(
    hierarchy_root: str | os.PathLike[str],
    clinical_links_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Build a deterministic operational-stage index from explicit roots."""

    hierarchy = _checked_root(hierarchy_root, role="hERG hierarchy")
    clinical = _checked_root(clinical_links_root, role="clinical-link")
    try:
        validate_herg_hierarchy(hierarchy)
        verify_herg_clinical_links(clinical)
    except Exception as error:
        raise HergOperationalTierError("upstream hERG artifact verification failed") from error

    observation_path = hierarchy / "observation_ledger.parquet"
    development_path = clinical / STRUCTURE_OUTPUT
    audit_path = clinical / LINK_AUDIT_OUTPUT
    qt_path = clinical / T3_OUTPUT
    required = (observation_path, development_path, audit_path, qt_path)
    if any(path.is_symlink() or not path.is_file() for path in required):
        raise HergOperationalTierError("required operational-tier input is missing or unsafe")

    output = Path(output_root).resolve()
    if output.exists() and (output.is_symlink() or not output.is_dir() or any(output.iterdir())):
        raise HergOperationalTierError("output directory must be absent or empty and may not be a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)

    counters: dict[str, dict[str, Any]] = {
        stage: {
            "headline": 0,
            "indexed": 0,
            "structures": set(),
            "trials": set(),
            "endpoints": set(),
        }
        for stage in STAGE_ORDER
    }
    disclosed: dict[str, dict[str, int]] = {stage: {} for stage in STAGE_ORDER}
    o0_valid_structure_observations = 0
    o1_valid_structure_observations = 0
    o1_native_numeric = 0
    o1_native_numeric_structures: set[str] = set()
    o1_pic50 = 0
    o1_pic50_structures: set[str] = set()
    o1_decisive = 0
    o1_decisive_structures: set[str] = set()
    o1_assays: set[str] = set()
    o1_source_records: set[str] = set()

    observation_columns = [
        "observation_id",
        "source_family",
        "source_record_id",
        "structure_id",
        "structure_valid",
        "assay_id",
        "native_value",
        "pic50_value",
        "derived_binary_label",
    ]

    def record_batches() -> Iterable[Sequence[Mapping[str, Any]]]:
        nonlocal o0_valid_structure_observations
        nonlocal o1_valid_structure_observations, o1_native_numeric, o1_pic50, o1_decisive

        for input_rows in _parquet_rows(observation_path, observation_columns):
            output_rows: list[dict[str, Any]] = []
            for row in input_rows:
                observation_id = str(row["observation_id"])
                source_record_id = str(row["source_record_id"])
                structure_id = row.get("structure_id")
                output_rows.append(
                    _record_row(
                        stage=O0,
                        grain="observation",
                        record_id=f"O0:{observation_id}",
                        structure_id=structure_id,
                        observation_id=observation_id,
                        source_family=row["source_family"],
                        source_record_id=source_record_id,
                        nct_id=None,
                        endpoint_candidate_id=None,
                        upstream_artifact=observation_path.name,
                        direct_assay=True,
                    )
                )
                counters[O0]["headline"] += 1
                counters[O0]["indexed"] += 1
                if row.get("structure_valid") is True and structure_id is not None:
                    value = str(structure_id)
                    counters[O0]["structures"].add(value)
                    o0_valid_structure_observations += 1
            yield output_rows

        for input_rows in _parquet_rows(observation_path, observation_columns):
            output_rows = []
            for row in input_rows:
                if row["source_family"] not in PRECLINICAL_SOURCE_FAMILIES:
                    continue
                observation_id = str(row["observation_id"])
                source_record_id = str(row["source_record_id"])
                structure_id = row.get("structure_id")
                output_rows.append(
                    _record_row(
                        stage=O1,
                        grain="observation",
                        record_id=f"O1:{observation_id}",
                        structure_id=structure_id,
                        observation_id=observation_id,
                        source_family=row["source_family"],
                        source_record_id=source_record_id,
                        nct_id=None,
                        endpoint_candidate_id=None,
                        upstream_artifact=observation_path.name,
                        direct_assay=True,
                    )
                )
                counters[O1]["headline"] += 1
                counters[O1]["indexed"] += 1
                o1_source_records.add(source_record_id)
                if row.get("assay_id") is not None:
                    o1_assays.add(str(row["assay_id"]))
                if row.get("structure_valid") is True and structure_id is not None:
                    value = str(structure_id)
                    counters[O1]["structures"].add(value)
                    o1_valid_structure_observations += 1
                    if row.get("native_value") is not None:
                        o1_native_numeric_structures.add(value)
                    if row.get("pic50_value") is not None:
                        o1_pic50_structures.add(value)
                    if row.get("derived_binary_label") is not None:
                        o1_decisive_structures.add(value)
                if row.get("native_value") is not None:
                    o1_native_numeric += 1
                if row.get("pic50_value") is not None:
                    o1_pic50 += 1
                if row.get("derived_binary_label") is not None:
                    o1_decisive += 1
            yield output_rows

        for input_rows in _parquet_rows(development_path):
            output_rows = []
            for row in input_rows:
                if (
                    row.get("clinical_cardiac_label_admitted") is not False
                    or row.get("model_label_admitted") is not False
                ):
                    raise HergOperationalTierError("clinical development context was promoted to a label")
                if row.get("clinical_development_annotation") is not True:
                    continue
                structure_id = str(row["molecule_id"])
                output_rows.append(
                    _record_row(
                        stage=O2,
                        grain="structure",
                        record_id=f"O2:{structure_id}",
                        structure_id=structure_id,
                        observation_id=None,
                        source_family="clinical_development_regulatory_annotation",
                        source_record_id=structure_id,
                        nct_id=None,
                        endpoint_candidate_id=None,
                        upstream_artifact=development_path.name,
                        direct_assay=False,
                    )
                )
                counters[O2]["headline"] += 1
                counters[O2]["indexed"] += 1
                counters[O2]["structures"].add(structure_id)
            yield output_rows

        for input_rows in _parquet_rows(audit_path):
            output_rows = []
            for row in input_rows:
                if row.get("model_label_admitted") is not False:
                    raise HergOperationalTierError("clinical link context was promoted to a model label")
                if (
                    row.get("source_kind") != "clinicaltrials_intervention"
                    or row.get("link_is_exact_and_unique") is not True
                ):
                    continue
                if row.get("linked_molecule_id") is None or row.get("nct_id") is None:
                    raise HergOperationalTierError("exact trial link lacks structure or NCT identity")
                source_record_id = str(row["source_record_id"])
                structure_id = str(row["linked_molecule_id"])
                nct_id = str(row["nct_id"])
                output_rows.append(
                    _record_row(
                        stage=O3,
                        grain="intervention_link",
                        record_id=f"O3:{source_record_id}",
                        structure_id=structure_id,
                        observation_id=None,
                        source_family="clinicaltrials_intervention",
                        source_record_id=source_record_id,
                        nct_id=nct_id,
                        endpoint_candidate_id=None,
                        upstream_artifact=audit_path.name,
                        direct_assay=False,
                    )
                )
                counters[O3]["headline"] += 1
                counters[O3]["indexed"] += 1
                counters[O3]["structures"].add(structure_id)
                counters[O3]["trials"].add(nct_id)
            yield output_rows

    qt_numeric_declared = 0
    qt_numeric_indexed = 0
    qt_source_candidates: set[str] = set()
    qt_trial_structure: set[tuple[str, str]] = set()

    def qt_batches() -> Iterable[Sequence[Mapping[str, Any]]]:
        nonlocal qt_numeric_declared, qt_numeric_indexed
        for input_rows in _parquet_rows(qt_path):
            output_rows: list[dict[str, Any]] = []
            for row in input_rows:
                if (
                    row.get("candidate_rule_passed") is not True
                    or row.get("clinical_herg_label_admitted") is not False
                    or row.get("model_label_admitted") is not False
                ):
                    raise HergOperationalTierError("posted QT context was promoted or is unqualified")
                candidate_id = str(row["candidate_id"])
                structure_id = str(row["molecule_id"])
                nct_id = str(row["nct_id"])
                endpoint_id = str(row["endpoint_candidate_id"])
                page_path = str(row["source_page_path"])
                pointer = str(row["raw_json_pointer"])
                values = _load_json_list(row["value_records_json"], field="value_records_json")
                denominators = _load_json_list(
                    row["denominator_records_json"], field="denominator_records_json"
                )
                qt_source_candidates.add(candidate_id)
                qt_trial_structure.add((nct_id, structure_id))
                counters[O3_QT]["structures"].add(structure_id)
                counters[O3_QT]["trials"].add(nct_id)
                counters[O3_QT]["endpoints"].add(endpoint_id)
                qt_numeric_declared += int(row["reported_numeric_value_count"])
                for ordinal, value_record in enumerate(values):
                    if not isinstance(value_record, Mapping):
                        raise HergOperationalTierError("QT result-value record is not an object")
                    is_numeric = _numeric(value_record.get("value"))
                    qt_numeric_indexed += int(is_numeric)
                    output_rows.append(
                        {
                            "operational_stage": O3_QT,
                            "record_grain": "result_value",
                            "record_id": f"O3QT:{candidate_id}:result:{ordinal}",
                            "source_candidate_id": candidate_id,
                            "structure_id": structure_id,
                            "nct_id": nct_id,
                            "endpoint_candidate_id": endpoint_id,
                            "record_ordinal": ordinal,
                            "reported_value_is_numeric": is_numeric,
                            "source_page_path": page_path,
                            "raw_json_pointer": pointer,
                            "clinical_context_used_as_herg_label": False,
                            "model_label_admitted_from_context": False,
                        }
                    )
                    counters[O3_QT]["headline"] += 1
                    counters[O3_QT]["indexed"] += 1
                for ordinal, denominator in enumerate(denominators):
                    if not isinstance(denominator, Mapping):
                        raise HergOperationalTierError("QT denominator record is not an object")
                    output_rows.append(
                        {
                            "operational_stage": O3_QT,
                            "record_grain": "denominator",
                            "record_id": f"O3QT:{candidate_id}:denominator:{ordinal}",
                            "source_candidate_id": candidate_id,
                            "structure_id": structure_id,
                            "nct_id": nct_id,
                            "endpoint_candidate_id": endpoint_id,
                            "record_ordinal": ordinal,
                            "reported_value_is_numeric": False,
                            "source_page_path": page_path,
                            "raw_json_pointer": pointer,
                            "clinical_context_used_as_herg_label": False,
                            "model_label_admitted_from_context": False,
                        }
                    )
                    counters[O3_QT]["indexed"] += 1
            yield output_rows

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        record_artifact = _write_batches(staging / RECORD_OUTPUT, _RECORD_SCHEMA, record_batches())
        qt_artifact = _write_batches(staging / QT_RECORD_OUTPUT, _QT_RECORD_SCHEMA, qt_batches())
        if qt_numeric_declared != qt_numeric_indexed:
            raise HergOperationalTierError("posted QT numeric-result count differs from compact index")

        disclosed[O0] = {
            "valid_structured_observations": o0_valid_structure_observations,
        }
        disclosed[O1] = {
            "valid_structured_observations": o1_valid_structure_observations,
            "unique_source_records": len(o1_source_records),
            "unique_assay_ids": len(o1_assays),
            "native_numeric_observations": o1_native_numeric,
            "native_numeric_structures": len(o1_native_numeric_structures),
            "normalized_pic50_observations": o1_pic50,
            "normalized_pic50_structures": len(o1_pic50_structures),
            "decisive_binary_observations": o1_decisive,
            "decisive_binary_structures": len(o1_decisive_structures),
        }

        development_rows = pq.read_table(development_path).to_pylist()
        development = [row for row in development_rows if row["clinical_development_annotation"]]
        fda_applications: set[str] = set()
        for row in development:
            fda_applications.update(
                str(value)
                for value in _load_json_list(
                    row["drugsfda_application_numbers_json"],
                    field="drugsfda_application_numbers_json",
                )
            )
        fda_exact_links = 0
        fda_structures: set[str] = set()
        for batch in _parquet_rows(audit_path):
            for row in batch:
                if row["source_kind"] == "drugsfda_ingredient" and row["link_is_exact_and_unique"]:
                    fda_exact_links += 1
                    fda_structures.add(str(row["linked_molecule_id"]))
        disclosed[O2] = {
            "phase_at_least_1_structures": sum(
                row.get("chembl_max_phase") is not None and float(row["chembl_max_phase"]) >= 1.0
                for row in development
            ),
            "first_approved_structures": sum(
                row.get("chembl_first_approval") is not None for row in development
            ),
            "fda_exact_ingredient_product_links": fda_exact_links,
            "fda_application_numbers": len(fda_applications),
            "fda_linked_structures": len(fda_structures),
        }

        o3_pairs: set[tuple[str, str]] = set()
        for batch in _parquet_rows(staging / RECORD_OUTPUT, ["operational_stage", "nct_id", "structure_id"]):
            for row in batch:
                if row["operational_stage"] == O3:
                    o3_pairs.add((str(row["nct_id"]), str(row["structure_id"])))
        disclosed[O3] = {"distinct_trial_structure_pairs": len(o3_pairs)}
        disclosed[O3_QT] = {
            "source_endpoint_records": len(qt_source_candidates),
            "result_value_records": counters[O3_QT]["headline"],
            "numeric_result_values": qt_numeric_indexed,
            "denominator_records": counters[O3_QT]["indexed"] - counters[O3_QT]["headline"],
            "distinct_trial_structure_pairs": len(qt_trial_structure),
        }

        summary_rows: list[dict[str, Any]] = []
        for stage in STAGE_ORDER:
            direct = stage in {O0, O1}
            summary_rows.append(
                {
                    "operational_stage": stage,
                    "stage_name": _STAGE_NAME[stage],
                    "headline_record_grain": HEADLINE_GRAIN[stage],
                    "headline_record_count": counters[stage]["headline"],
                    "indexed_record_count": counters[stage]["indexed"],
                    "unique_structures": len(counters[stage]["structures"]),
                    "unique_trials": len(counters[stage]["trials"]),
                    "unique_endpoints": len(counters[stage]["endpoints"]),
                    "direct_herg_assay_evidence": direct,
                    "clinical_context_used_as_herg_label": False,
                    "model_label_admitted_from_context": False,
                    "disclosed_counts_json": _canonical_json(disclosed[stage]),
                    "stage_semantics": _STAGE_SEMANTICS[stage],
                }
            )
        too_small = {
            row["operational_stage"]: row["headline_record_count"]
            for row in summary_rows
            if row["headline_record_count"] < MIN_HEADLINE_COUNT
        }
        if too_small:
            raise HergOperationalTierError(f"operational headline count is below 1000: {too_small}")
        summary_artifact = _write_table(staging / SUMMARY_OUTPUT, summary_rows, _SUMMARY_SCHEMA)

        artifacts = [record_artifact, qt_artifact, summary_artifact]
        input_bindings = [
            _input_binding("hierarchy_manifest", hierarchy / "manifest.json"),
            _input_binding("hierarchy_observation_ledger", observation_path),
            _input_binding("clinical_links_manifest", clinical / CLINICAL_MANIFEST_NAME),
            _input_binding("clinical_structure_development", development_path),
            _input_binding("clinical_exact_name_link_audit", audit_path),
            _input_binding("clinical_posted_qt_candidates", qt_path),
        ]
        headline_counts = {
            row["operational_stage"]: {
                "record_grain": row["headline_record_grain"],
                "record_count": row["headline_record_count"],
                "unique_structures": row["unique_structures"],
                "unique_trials": row["unique_trials"],
                "unique_endpoints": row["unique_endpoints"],
            }
            for row in summary_rows
        }
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": "herg_operational_evidence_hierarchy_v1_1",
            "minimum_headline_record_count": MIN_HEADLINE_COUNT,
            "preclinical_source_families": sorted(PRECLINICAL_SOURCE_FAMILIES),
            "stage_order": list(STAGE_ORDER),
            "scientific_contract": {
                "stage_and_record_grain_are_distinct": True,
                "O2_is_a_herg_label": False,
                "O3_is_a_herg_label": False,
                "posted_QT_is_a_herg_label": False,
                "clinical_context_model_labels_admitted": 0,
                "absence_is_negative_evidence": False,
            },
            "headline_counts": headline_counts,
            "disclosed_counts": disclosed,
            "input_bindings": input_bindings,
            "input_set_sha256": hashlib.sha256(_canonical_json(input_bindings).encode("utf-8")).hexdigest(),
            "artifacts": artifacts,
            "artifact_set_sha256": hashlib.sha256(_canonical_json(artifacts).encode("utf-8")).hexdigest(),
        }
        manifest["manifest_sha256"] = hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        validate_herg_operational_tiers(staging)
        if output.exists():
            output.rmdir()
        os.replace(staging, output)
        return validate_herg_operational_tiers(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_herg_operational_tiers(
    output_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Validate hashes, grains, counts, and the no-clinical-label contract."""

    root = Path(output_root)
    manifest_path = root / MANIFEST_NAME
    if root.is_symlink() or not root.is_dir() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise HergOperationalTierError(f"missing or unsafe operational-tier output: {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HergOperationalTierError("unreadable operational-tier manifest") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise HergOperationalTierError("unexpected operational-tier manifest schema")
    declared_digest = manifest.get("manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest() != declared_digest:
        raise HergOperationalTierError("operational-tier manifest digest mismatch")
    if manifest.get("minimum_headline_record_count") != MIN_HEADLINE_COUNT:
        raise HergOperationalTierError("operational-tier minimum headline contract changed")
    contract = manifest.get("scientific_contract", {})
    if (
        contract.get("stage_and_record_grain_are_distinct") is not True
        or contract.get("O2_is_a_herg_label") is not False
        or contract.get("O3_is_a_herg_label") is not False
        or contract.get("posted_QT_is_a_herg_label") is not False
        or contract.get("clinical_context_model_labels_admitted") != 0
    ):
        raise HergOperationalTierError("operational-tier scientific contract was weakened")

    schemas = {
        RECORD_OUTPUT: _RECORD_SCHEMA,
        QT_RECORD_OUTPUT: _QT_RECORD_SCHEMA,
        SUMMARY_OUTPUT: _SUMMARY_SCHEMA,
    }
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or {item.get("path") for item in artifacts} != set(schemas):
        raise HergOperationalTierError("operational-tier artifact membership mismatch")
    if {path.name for path in root.iterdir()} != {MANIFEST_NAME, *schemas} or any(
        path.is_symlink() or not path.is_file() for path in root.iterdir()
    ):
        raise HergOperationalTierError("operational-tier output contains unexpected members")
    bindings = manifest.get("input_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise HergOperationalTierError("operational-tier input bindings are missing")
    for binding in bindings:
        if not isinstance(binding, Mapping):
            raise HergOperationalTierError("operational-tier input binding is malformed")
        path = Path(str(binding.get("path", "")))
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(binding.get("bytes", -1))
            or _sha256_file(path) != binding.get("sha256")
        ):
            raise HergOperationalTierError(f"operational-tier input binding mismatch: {path}")
    for artifact in artifacts:
        path = root / str(artifact["path"])
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(artifact.get("bytes", -1))
            or _sha256_file(path) != artifact.get("sha256")
        ):
            raise HergOperationalTierError(f"operational-tier artifact hash mismatch: {path.name}")
        parquet = pq.ParquetFile(path)
        if parquet.schema_arrow != schemas[path.name]:
            raise HergOperationalTierError(f"operational-tier artifact schema mismatch: {path.name}")
        if parquet.metadata is None or parquet.metadata.num_rows != int(artifact.get("rows", -1)):
            raise HergOperationalTierError(f"operational-tier artifact row-count mismatch: {path.name}")

    records = pq.read_table(root / RECORD_OUTPUT).to_pylist()
    qt_records = pq.read_table(root / QT_RECORD_OUTPUT).to_pylist()
    summaries = pq.read_table(root / SUMMARY_OUTPUT).to_pylist()
    if [row["operational_stage"] for row in summaries] != list(STAGE_ORDER):
        raise HergOperationalTierError("operational-stage summary order or membership mismatch")
    record_ids = [row["record_id"] for row in records]
    qt_ids = [row["record_id"] for row in qt_records]
    if len(record_ids) != len(set(record_ids)) or len(qt_ids) != len(set(qt_ids)):
        raise HergOperationalTierError("operational record identity is not unique")

    expected: dict[str, dict[str, Any]] = {}
    for stage in (O0, O1, O2, O3):
        stage_rows = [row for row in records if row["operational_stage"] == stage]
        if any(row["record_grain"] != HEADLINE_GRAIN[stage] for row in stage_rows):
            raise HergOperationalTierError("operational stage and record grain were conflated")
        expected[stage] = {
            "headline": len(stage_rows),
            "indexed": len(stage_rows),
            "structures": len({row["structure_id"] for row in stage_rows if row["structure_id"]}),
            "trials": len({row["nct_id"] for row in stage_rows if row["nct_id"]}),
            "endpoints": len(
                {row["endpoint_candidate_id"] for row in stage_rows if row["endpoint_candidate_id"]}
            ),
        }
        for row in stage_rows:
            clinical = stage in CLINICAL_STAGES
            if (
                row["direct_herg_assay_evidence"] is clinical
                or row["clinical_context_only"] is not clinical
                or row["clinical_context_used_as_herg_label"] is not False
                or row["model_label_admitted_from_context"] is not False
            ):
                raise HergOperationalTierError("operational record violated label/context semantics")
    if {row["operational_stage"] for row in records} != {O0, O1, O2, O3}:
        raise HergOperationalTierError("operational record stage membership mismatch")

    if any(row["operational_stage"] != O3_QT for row in qt_records):
        raise HergOperationalTierError("QT index contains a non-QT operational stage")
    if any(
        row["record_grain"] not in {"result_value", "denominator"}
        or row["clinical_context_used_as_herg_label"] is not False
        or row["model_label_admitted_from_context"] is not False
        for row in qt_records
    ):
        raise HergOperationalTierError("QT index violated grain or label semantics")
    result_rows = [row for row in qt_records if row["record_grain"] == "result_value"]
    expected[O3_QT] = {
        "headline": len(result_rows),
        "indexed": len(qt_records),
        "structures": len({row["structure_id"] for row in qt_records}),
        "trials": len({row["nct_id"] for row in qt_records}),
        "endpoints": len({row["endpoint_candidate_id"] for row in qt_records}),
    }

    headline_manifest = manifest.get("headline_counts")
    if not isinstance(headline_manifest, Mapping) or set(headline_manifest) != set(STAGE_ORDER):
        raise HergOperationalTierError("operational headline manifest membership mismatch")
    disclosed_manifest = manifest.get("disclosed_counts")
    if not isinstance(disclosed_manifest, Mapping) or set(disclosed_manifest) != set(STAGE_ORDER):
        raise HergOperationalTierError("operational disclosed-count membership mismatch")
    for row in summaries:
        stage = str(row["operational_stage"])
        values = expected[stage]
        if (
            row["headline_record_grain"] != HEADLINE_GRAIN[stage]
            or row["headline_record_count"] != values["headline"]
            or row["indexed_record_count"] != values["indexed"]
            or row["unique_structures"] != values["structures"]
            or row["unique_trials"] != values["trials"]
            or row["unique_endpoints"] != values["endpoints"]
            or row["headline_record_count"] < MIN_HEADLINE_COUNT
            or row["clinical_context_used_as_herg_label"] is not False
            or row["model_label_admitted_from_context"] is not False
        ):
            raise HergOperationalTierError("operational summary count, grain, or label contract failed")
        declared = headline_manifest[stage]
        if declared != {
            "record_grain": row["headline_record_grain"],
            "record_count": row["headline_record_count"],
            "unique_structures": row["unique_structures"],
            "unique_trials": row["unique_trials"],
            "unique_endpoints": row["unique_endpoints"],
        }:
            raise HergOperationalTierError("operational manifest/summary headline mismatch")
        if _canonical_json(disclosed_manifest[stage]) != row["disclosed_counts_json"]:
            raise HergOperationalTierError("operational disclosed counts were not preserved")

    o3_rows = [row for row in records if row["operational_stage"] == O3]
    o3_pairs = {(row["nct_id"], row["structure_id"]) for row in o3_rows}
    qt_pairs = {(row["nct_id"], row["structure_id"]) for row in qt_records}
    recomputed_disclosures = {
        O3: {"distinct_trial_structure_pairs": len(o3_pairs)},
        O3_QT: {
            "source_endpoint_records": len({row["source_candidate_id"] for row in qt_records}),
            "result_value_records": len(result_rows),
            "numeric_result_values": sum(row["reported_value_is_numeric"] for row in result_rows),
            "denominator_records": len(qt_records) - len(result_rows),
            "distinct_trial_structure_pairs": len(qt_pairs),
        },
    }
    for stage, values in recomputed_disclosures.items():
        if disclosed_manifest[stage] != values:
            raise HergOperationalTierError("operational linked-record disclosure mismatch")
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
        validate_herg_operational_tiers(args.output_root)
        return 0
    if args.hierarchy_root is None or args.clinical_links_root is None:
        raise SystemExit("build mode requires --hierarchy-root and --clinical-links-root")
    build_herg_operational_tiers(args.hierarchy_root, args.clinical_links_root, args.output_root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
