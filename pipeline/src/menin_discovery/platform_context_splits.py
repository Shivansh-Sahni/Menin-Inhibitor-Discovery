"""Supplemental label-blind assay, document, and strict-temporal candidates.

These candidates answer context-generalization questions that the official
split suite does not.  They never replace official assignments, read labels,
open model-ready lockboxes, search seeds, authorize training, or claim model
performance.  Missing grouping context is routed to ``excluded_unknown``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pyarrow as pa
import pyarrow.parquet as pq

from .platform_corpus_readiness import _task_parts, _verify_canonical_corpus
from .platform_data_schema import clean_text
from .platform_data_sources import sha256_file
from .platform_features import stable_json_digest
from .platform_split_suite import verify_split_suite

SCHEMA_VERSION = "platform_context_split_candidates_v1"
REPORT_SCHEMA_VERSION = "platform_context_split_analysis_v1"
MANIFEST_SCHEMA_VERSION = "platform_context_split_analysis_manifest_v1"
DEFAULT_CANONICAL_ROOT = Path("research/data/platform/canonical/full_chembl37")
DEFAULT_QC_REPORT = Path("research/reports/platform/qc_report.json")
DEFAULT_SPLIT_ROOT = Path("research/data/platform/splits/full_chembl37")
DEFAULT_OUTPUT = Path("research/data/platform/splits/context_holdout_candidates")
DEFAULT_REPORT_OUTPUT = Path("research/reports/platform/context_split_analysis")
TASK_KEYS = (
    "default::default__herg__ic50__herg_functional__nm__continuous_exact",
    "default::default__binding__kd__binding__nm__continuous_exact",
)
STRATEGIES = ("assay_group_holdout", "document_group_holdout", "strict_temporal")
PARTITIONS = ("train", "validation", "test")
ROUTES = (*PARTITIONS, "excluded_unknown")
FORBIDDEN_LABEL_COLUMNS = frozenset(
    {
        "label_kind",
        "label_value",
        "label_text",
        "label_relation",
        "label_lower_bound",
        "label_upper_bound",
        "label_unit",
    }
)
CONTEXT_COLUMNS = ("observation_id", "assay_id", "document_id", "document_year")
FEATURE_COLUMNS = (
    "record_id",
    "molecule_id",
    "protein_id",
    "target_id",
    "source_id",
)
OUTPUT_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("split", pa.string()),
        ("group_id", pa.string()),
        ("strategy", pa.string()),
        ("molecule_id", pa.string()),
        ("protein_id", pa.string()),
        ("target_id", pa.string()),
        ("source_id", pa.string()),
        ("assay_id", pa.string()),
        ("document_id", pa.string()),
        ("document_year", pa.int64()),
    ]
)
Strategy = Literal["assay_group_holdout", "document_group_holdout", "strict_temporal"]


@dataclass(frozen=True)
class ContextSplitConfig:
    seed: int = 20260805
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    parquet_batch_size: int = 50_000
    temporal_rule: str = "whole_year_nearest_cumulative_70_85_with_one_year_minimum_per_partition_v1"

    def validate(self) -> None:
        if not all(0 < value < 1 for value in self.fractions.values()):
            raise ValueError("split fractions must be in (0, 1)")
        if abs(sum(self.fractions.values()) - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to one")
        if not 1 <= self.parquet_batch_size <= 250_000:
            raise ValueError("parquet_batch_size must be between 1 and 250000")
        if self.temporal_rule != (
            "whole_year_nearest_cumulative_70_85_with_one_year_minimum_per_partition_v1"
        ):
            raise ValueError("unsupported temporal rule")

    @property
    def fractions(self) -> dict[str, float]:
        return {
            "train": self.train_fraction,
            "validation": self.validation_fraction,
            "test": self.test_fraction,
        }


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_relative(value: object, field: str) -> str:
    text = clean_text(value)
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe {field}: {text!r}")
    return path.as_posix()


def _guard_columns(columns: Sequence[str], source: str) -> tuple[str, ...]:
    result = tuple(columns)
    lowered = {column.casefold() for column in result}
    if lowered & FORBIDDEN_LABEL_COLUMNS or any(column.startswith("label") for column in lowered):
        raise ValueError(f"label columns forbidden in {source}")
    return result


def _iter_batches(path: Path, columns: Sequence[str], batch_size: int):
    requested = _guard_columns(columns, path.as_posix())
    parquet = pq.ParquetFile(path)
    missing = set(requested) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"missing columns in {path}: {sorted(missing)}")
    yield from parquet.iter_batches(batch_size=batch_size, columns=list(requested))


def _task_slug(dataset_key: str) -> str:
    readable = re.sub(r"[^a-z0-9]+", "-", dataset_key.casefold()).strip("-")[:76]
    return f"{readable}-{hashlib.sha256(dataset_key.encode()).hexdigest()[:12]}"


def _group_digest(kind: str, value: str) -> str:
    return f"{kind}:{hashlib.sha256(value.encode()).hexdigest()[:24]}"


def _hash_partition(value: str, config: ContextSplitConfig) -> str:
    unit = int.from_bytes(hashlib.sha256(f"{config.seed}|{value}".encode()).digest()[:8], "big") / 2**64
    if unit < config.train_fraction:
        return "train"
    if unit < config.train_fraction + config.validation_fraction:
        return "validation"
    return "test"


def _temporal_boundaries(year_counts: Mapping[int, int]) -> tuple[int, int]:
    years = sorted(year_counts)
    if len(years) < 3:
        raise ValueError("strict temporal split requires at least three distinct years")
    total = sum(year_counts.values())
    cumulative = 0
    cumulative_by_index: list[int] = []
    for year in years:
        cumulative += year_counts[year]
        cumulative_by_index.append(cumulative)
    train_index = min(
        range(0, len(years) - 2),
        key=lambda index: (abs(cumulative_by_index[index] / total - 0.70), years[index]),
    )
    validation_index = min(
        range(train_index + 1, len(years) - 1),
        key=lambda index: (abs(cumulative_by_index[index] / total - 0.85), years[index]),
    )
    return years[train_index], years[validation_index]


def _partition_temporal(year: int | None, boundaries: tuple[int, int]) -> str:
    if year is None:
        return "excluded_unknown"
    train_end, validation_end = boundaries
    if year <= train_end:
        return "train"
    if year <= validation_end:
        return "validation"
    return "test"


def _load_task_rows(
    canonical: Any,
    canonical_entry: Mapping[str, Any],
    feature_path: Path,
    batch_size: int,
) -> list[dict[str, Any]]:
    context: dict[str, dict[str, Any]] = {}
    for path in _task_parts(canonical, canonical_entry):
        for batch in _iter_batches(path, CONTEXT_COLUMNS, batch_size):
            values = batch.to_pydict()
            for index, raw_id in enumerate(values["observation_id"]):
                record_id = clean_text(raw_id)
                if not record_id or record_id in context:
                    raise ValueError("canonical observation IDs must be nonblank and unique")
                raw_year = values["document_year"][index]
                context[record_id] = {
                    "assay_id": clean_text(values["assay_id"][index]),
                    "document_id": clean_text(values["document_id"][index]),
                    "document_year": None if raw_year is None else int(raw_year),
                }
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for batch in _iter_batches(feature_path, FEATURE_COLUMNS, batch_size):
        values = batch.to_pydict()
        for index, raw_id in enumerate(values["record_id"]):
            record_id = clean_text(raw_id)
            if record_id in seen or record_id not in context:
                raise ValueError("feature/canonical observation membership mismatch")
            seen.add(record_id)
            rows.append(
                {
                    "record_id": record_id,
                    **context[record_id],
                    **{
                        column: clean_text(values[column][index])
                        for column in FEATURE_COLUMNS
                        if column != "record_id"
                    },
                }
            )
    if seen != set(context):
        raise ValueError("feature/canonical observation membership mismatch")
    return sorted(rows, key=lambda row: row["record_id"])


def _route_rows(
    rows: Sequence[Mapping[str, Any]], strategy: Strategy, config: ContextSplitConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config.validate()
    boundaries: tuple[int, int] | None = None
    grouping_field = {
        "assay_group_holdout": "assay_id",
        "document_group_holdout": "document_id",
        "strict_temporal": "document_year",
    }[strategy]
    if strategy == "strict_temporal":
        counts = Counter(int(row["document_year"]) for row in rows if row.get("document_year") is not None)
        boundaries = _temporal_boundaries(counts)
    routed: list[dict[str, Any]] = []
    for row in rows:
        raw_group = row.get(grouping_field)
        if strategy == "strict_temporal":
            assert boundaries is not None
            year = None if raw_group is None else int(raw_group)
            partition = _partition_temporal(year, boundaries)
            group_id = "unknown:document_year" if year is None else f"year:{year}"
        else:
            group = clean_text(raw_group)
            partition = "excluded_unknown" if not group else _hash_partition(group, config)
            group_id = f"unknown:{grouping_field}" if not group else _group_digest(grouping_field, group)
        routed.append(
            {
                "record_id": row["record_id"],
                "split": partition,
                "group_id": group_id,
                "strategy": strategy,
                "molecule_id": row["molecule_id"],
                "protein_id": row["protein_id"],
                "target_id": row["target_id"],
                "source_id": row["source_id"],
                "assay_id": row["assay_id"],
                "document_id": row["document_id"],
                "document_year": row["document_year"],
            }
        )
    metadata = {
        "grouping_field": grouping_field,
        "temporal_boundaries": (
            None
            if boundaries is None
            else {"train_max_year": boundaries[0], "validation_max_year": boundaries[1]}
        ),
    }
    return routed, metadata


def _pairwise_overlap(values: Mapping[str, set[str]]) -> dict[str, Any]:
    counts = {
        f"{left}__{right}": len(values[left] & values[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    return {
        "evidence_scope": "exhaustive",
        "unique_counts": {partition: len(values[partition]) for partition in PARTITIONS},
        "pairwise_overlap_counts": counts,
        "any_overlap": any(counts.values()),
    }


def _audit_rows(rows: Sequence[Mapping[str, Any]], grouping_field: str) -> dict[str, Any]:
    fields = (
        "group_id",
        "molecule_id",
        "protein_id",
        "target_id",
        "source_id",
        "assay_id",
        "document_id",
    )
    sets: dict[str, dict[str, set[str]]] = {
        partition: {field: set() for field in fields} for partition in PARTITIONS
    }
    years: dict[str, list[int]] = {partition: [] for partition in PARTITIONS}
    counts: Counter[str] = Counter()
    for row in rows:
        partition = clean_text(row["split"])
        if partition not in ROUTES:
            raise ValueError(f"unknown route: {partition}")
        counts[partition] += 1
        if partition not in PARTITIONS:
            continue
        for field in fields:
            value = clean_text(row.get(field))
            if value:
                sets[partition][field].add(value)
        if row.get("document_year") is not None:
            years[partition].append(int(row["document_year"]))
    overlaps = {
        field: _pairwise_overlap({partition: sets[partition][field] for partition in PARTITIONS})
        for field in fields
    }
    year_ranges = {
        partition: {
            "minimum": min(years[partition], default=None),
            "maximum": max(years[partition], default=None),
            "rows": len(years[partition]),
        }
        for partition in PARTITIONS
    }
    chronological = bool(
        years["train"]
        and years["validation"]
        and years["test"]
        and max(years["train"]) < min(years["validation"])
        and max(years["validation"]) < min(years["test"])
    )
    return {
        "evidence_scope": "exhaustive_all_candidate_rows",
        "row_counts": {route: counts[route] for route in ROUTES},
        "exact_overlap": overlaps,
        "group_exclusion_passed": not overlaps["group_id"]["any_overlap"],
        "grouping_field": grouping_field,
        "year_ranges": year_ranges,
        "strict_chronology_passed": chronological,
        "label_columns_read": [],
        "test_labels_read": False,
    }


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]], batch_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(
        path,
        OUTPUT_SCHEMA,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    try:
        for start in range(0, len(rows), batch_size):
            table = pa.Table.from_pylist(list(rows[start : start + batch_size]), schema=OUTPUT_SCHEMA)
            writer.write_table(table, row_group_size=batch_size)
    finally:
        writer.close()


def _inventory(root: Path, *, exclude: frozenset[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"symlink forbidden: {path}")
        if stat.S_ISREG(mode):
            relative = path.relative_to(root).as_posix()
            if relative in exclude:
                continue
            entry: dict[str, Any] = {
                "relative_path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            if path.suffix == ".parquet":
                entry["rows"] = pq.ParquetFile(path).metadata.num_rows
            result[relative] = entry
        elif not stat.S_ISDIR(mode):
            raise ValueError(f"special entry forbidden: {path}")
    return result


def _closed_regular_files(root: Path, *, exclude: frozenset[str]) -> set[str]:
    result: set[str] = set()
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"symlink forbidden: {path}")
        if stat.S_ISREG(mode):
            relative = path.relative_to(root).as_posix()
            if relative not in exclude:
                result.add(relative)
        elif not stat.S_ISDIR(mode):
            raise ValueError(f"special entry forbidden: {path}")
    return result


class _Transaction:
    def __init__(self, target: Path) -> None:
        self.target = target

    def __enter__(self) -> Path:
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.staging = Path(tempfile.mkdtemp(prefix=f".{self.target.name}.building-", dir=self.target.parent))
        return self.staging

    def __exit__(self, kind: object, exc: object, traceback: object) -> None:
        if kind is not None:
            shutil.rmtree(self.staging, ignore_errors=True)
            return
        if self.target.exists():
            shutil.rmtree(self.staging, ignore_errors=True)
            raise FileExistsError(f"refusing to replace existing output: {self.target}")
        os.replace(self.staging, self.target)


def _write_summary(path: Path, report: Mapping[str, Any]) -> None:
    statuses = report["candidate_status_counts"]
    path.write_text(
        f"""# Supplemental context-split candidates

These are label-blind **candidate** assignments, not replacements for the
official split suite and not authorization to train. The exact hERG functional
IC50 pilot and exact binding Kd scale task were evaluated with assay-group,
document-group, and strict whole-year temporal rules.

- Materialized: {statuses.get("materialized", 0)}
- Skipped at the fixed rule: {sum(value for key, value in statuses.items() if key != "materialized")}
- Missing assay/document/year values are `excluded_unknown`, never silently assigned.
- Group exclusion and temporal chronology are recomputed exhaustively from every candidate row.
- Test features are used only for leakage auditing. No label column or model-ready lockbox was opened.

Context splits expose a different generalization question from molecular or
scaffold splits. They do not eliminate chemical similarity, protein-family
similarity, source dependence, publication bias, or prospective-validation
requirements. Near-similarity evidence remains in the separately bound deep
leakage report; this package reports exact identity/context overlap only.
""",
        encoding="utf-8",
    )


def materialize_context_split_candidates(
    canonical_root: str | os.PathLike[str] = DEFAULT_CANONICAL_ROOT,
    qc_report: str | os.PathLike[str] = DEFAULT_QC_REPORT,
    split_root: str | os.PathLike[str] = DEFAULT_SPLIT_ROOT,
    output_directory: str | os.PathLike[str] = DEFAULT_OUTPUT,
    report_directory: str | os.PathLike[str] = DEFAULT_REPORT_OUTPUT,
    config: ContextSplitConfig | None = None,
) -> dict[str, Any]:
    config = config or ContextSplitConfig()
    config.validate()
    output = Path(output_directory)
    report_output = Path(report_directory)
    if output.exists() or report_output.exists():
        raise FileExistsError("refusing to replace existing context-split outputs")
    canonical = _verify_canonical_corpus(Path(canonical_root), Path(qc_report))
    split_verification = verify_split_suite(split_root)
    split_root_path = Path(split_root).resolve()
    split_acceptance_path = split_root_path / "acceptance.json"
    split_acceptance = _strict_json(split_acceptance_path)
    source = split_acceptance.get("source_binding")
    if (
        not isinstance(source, Mapping)
        or source.get("canonical_build_manifest_sha256") != canonical.build_manifest_sha256
    ):
        raise ValueError("official split source binding differs from canonical input")
    split_tasks = {
        clean_text(task["dataset_key"]): task
        for task in split_acceptance.get("tasks", [])
        if isinstance(task, Mapping)
    }
    config_payload = asdict(config)
    task_statuses: list[dict[str, Any]] = []
    with _Transaction(output) as staging:
        for dataset_key in TASK_KEYS:
            canonical_entry = canonical.task_datasets.get(dataset_key)
            split_task = split_tasks.get(dataset_key)
            if not isinstance(canonical_entry, Mapping) or not isinstance(split_task, Mapping):
                raise ValueError(f"required task unavailable: {dataset_key}")
            feature_relative = _safe_relative(split_task["feature_projection_path"], "feature path")
            feature_path = split_root_path / feature_relative
            rows = _load_task_rows(canonical, canonical_entry, feature_path, config.parquet_batch_size)
            slug = _task_slug(dataset_key)
            candidates: list[dict[str, Any]] = []
            for raw_strategy in STRATEGIES:
                strategy: Strategy = raw_strategy  # type: ignore[assignment]
                strategy_root = staging / "tasks" / slug / strategy
                try:
                    routed, metadata = _route_rows(rows, strategy, config)
                except ValueError as exc:
                    status = {
                        "dataset_key": dataset_key,
                        "strategy": strategy,
                        "status": "skipped_infeasible",
                        "reason": str(exc),
                        "split_materialized": False,
                        "seed_search_performed": False,
                        "label_columns_read": [],
                        "test_labels_read": False,
                        "substantive_training_started": False,
                    }
                else:
                    audit = _audit_rows(routed, metadata["grouping_field"])
                    missing_partitions = [
                        partition for partition in PARTITIONS if audit["row_counts"][partition] == 0
                    ]
                    chronology_invalid = (
                        strategy == "strict_temporal" and not audit["strict_chronology_passed"]
                    )
                    if missing_partitions or not audit["group_exclusion_passed"] or chronology_invalid:
                        status = {
                            "dataset_key": dataset_key,
                            "strategy": strategy,
                            "status": "skipped_fixed_rule",
                            "reason": (
                                f"empty_partitions={missing_partitions}; "
                                f"group_exclusion={audit['group_exclusion_passed']}; "
                                f"strict_chronology={audit['strict_chronology_passed']}"
                            ),
                            "split_materialized": False,
                            "seed_search_performed": False,
                            "audit": audit,
                            "label_columns_read": [],
                            "test_labels_read": False,
                            "substantive_training_started": False,
                        }
                    else:
                        split_path = strategy_root / "split.parquet"
                        _write_parquet(split_path, routed, config.parquet_batch_size)
                        relative = split_path.relative_to(staging).as_posix()
                        status = {
                            "dataset_key": dataset_key,
                            "task_slug": slug,
                            "strategy": strategy,
                            "status": "materialized",
                            "split_materialized": True,
                            "split_path": relative,
                            "split_sha256": sha256_file(split_path),
                            "split_size_bytes": split_path.stat().st_size,
                            "split_rows": len(routed),
                            "configuration_sha256": stable_json_digest(config_payload),
                            "canonical_task_dataset_sha256": canonical_entry["dataset_sha256"],
                            "feature_projection_sha256": split_task["feature_projection"][
                                "feature_file_sha256"
                            ],
                            "metadata": metadata,
                            "audit": audit,
                            "seed_search_performed": False,
                            "official_split_replaced": False,
                            "label_columns_read": [],
                            "test_features_inspected": True,
                            "test_labels_read": False,
                            "substantive_training_started": False,
                        }
                _atomic_json(strategy_root / "status.json", status)
                candidates.append(status)
            task_status = {
                "dataset_key": dataset_key,
                "task_slug": slug,
                "canonical_rows": len(rows),
                "canonical_task_dataset_sha256": canonical_entry["dataset_sha256"],
                "feature_projection_path": feature_relative,
                "feature_projection_sha256": split_task["feature_projection"]["feature_file_sha256"],
                "candidates": candidates,
                "label_columns_read": [],
                "test_labels_read": False,
                "substantive_training_started": False,
            }
            _atomic_json(staging / "tasks" / slug / "task_status.json", task_status)
            task_statuses.append(task_status)
        inventory = _inventory(staging, exclude=frozenset())
        acceptance = {
            "schema_version": SCHEMA_VERSION,
            "configuration": config_payload,
            "configuration_sha256": stable_json_digest(config_payload),
            "source_binding": {
                **dict(source),
                "official_split_acceptance_sha256": sha256_file(split_acceptance_path),
                "official_split_component_inventory_sha256": split_acceptance["component_inventory_sha256"],
                "implementation_sha256": sha256_file(Path(__file__).resolve()),
            },
            "input_verification": split_verification,
            "task_order": list(TASK_KEYS),
            "strategy_order": list(STRATEGIES),
            "tasks": task_statuses,
            "component_inventory": inventory,
            "component_inventory_sha256": stable_json_digest(inventory),
            "label_access_contract": {
                "canonical_columns_requested": list(CONTEXT_COLUMNS),
                "feature_columns_requested": list(FEATURE_COLUMNS),
                "label_columns_requested": [],
                "test_features_inspected": True,
                "test_labels_read": False,
                "model_ready_test_lockbox_opened_or_hashed": False,
            },
            "candidate_only": True,
            "official_split_replaced": False,
            "substantive_training_ready": False,
            "substantive_training_authorized": False,
            "substantive_training_started": False,
        }
        _atomic_json(staging / "acceptance.json", acceptance)
    data_verification = verify_context_split_candidates(output, expected_config=config)
    acceptance = _strict_json(output / "acceptance.json")
    tasks = cast(list[dict[str, Any]], acceptance["tasks"])
    status_counts = Counter(candidate["status"] for task in tasks for candidate in task["candidates"])
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "configuration": config_payload,
        "configuration_sha256": stable_json_digest(config_payload),
        "context_split_acceptance_sha256": sha256_file(output / "acceptance.json"),
        "context_split_component_inventory_sha256": acceptance["component_inventory_sha256"],
        "source_binding": acceptance["source_binding"],
        "candidate_status_counts": dict(sorted(status_counts.items())),
        "tasks": tasks,
        "data_verification": data_verification,
        "evidence_scope": "exact identity/context overlap exhaustive; no near-similarity recomputation",
        "limitations": [
            "candidate assignments do not replace the official split suite",
            "temporal cutoffs are deterministic whole-year empirical-coverage boundaries, not prospective dates",
            "missing context is excluded_unknown",
            "all data remain from one canonical ChEMBL source",
            "chemical and protein near-similarity require the separately bound deep leakage audit",
            "no performance or clinical claim is supported",
        ],
        "claim_readiness": False,
        "substantive_training_ready": False,
        "substantive_training_authorized": False,
        "substantive_training_started": False,
    }
    with _Transaction(report_output) as staging:
        _atomic_json(staging / "report.json", report)
        _write_summary(staging / "summary.md", report)
        rows_for_csv = [
            {
                "dataset_key": task["dataset_key"],
                "strategy": candidate["strategy"],
                "status": candidate["status"],
                "train": candidate.get("audit", {}).get("row_counts", {}).get("train", 0),
                "validation": candidate.get("audit", {}).get("row_counts", {}).get("validation", 0),
                "test": candidate.get("audit", {}).get("row_counts", {}).get("test", 0),
                "excluded_unknown": candidate.get("audit", {})
                .get("row_counts", {})
                .get("excluded_unknown", 0),
                "group_exclusion_passed": candidate.get("audit", {}).get("group_exclusion_passed", False),
                "strict_chronology_passed": candidate.get("audit", {}).get("strict_chronology_passed", False),
            }
            for task in tasks
            for candidate in task["candidates"]
        ]
        with (staging / "decision_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows_for_csv[0]))
            writer.writeheader()
            writer.writerows(rows_for_csv)
        inventory = _inventory(staging, exclude=frozenset())
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "configuration_sha256": report["configuration_sha256"],
            "context_split_acceptance_sha256": report["context_split_acceptance_sha256"],
            "component_inventory": inventory,
            "component_inventory_sha256": stable_json_digest(inventory),
            "claim_readiness": False,
            "substantive_training_ready": False,
            "substantive_training_authorized": False,
            "substantive_training_started": False,
        }
        _atomic_json(staging / "manifest.json", manifest)
    report_verification = verify_context_split_report(report_output, output, expected_config=config)
    return {"data": data_verification, "report": report_verification}


def verify_context_split_candidates(
    output_directory: str | os.PathLike[str] = DEFAULT_OUTPUT,
    *,
    expected_config: ContextSplitConfig | None = None,
) -> dict[str, Any]:
    root = Path(output_directory).resolve()
    acceptance_path = root / "acceptance.json"
    acceptance = _strict_json(acceptance_path)
    if acceptance.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unrecognized context split schema")
    config = acceptance.get("configuration")
    if not isinstance(config, Mapping) or stable_json_digest(config) != acceptance.get(
        "configuration_sha256"
    ):
        raise ValueError("context split configuration digest mismatch")
    parsed = ContextSplitConfig(**config)
    parsed.validate()
    if expected_config is not None and asdict(expected_config) != dict(config):
        raise ValueError("context split configuration drift")
    source_binding = acceptance.get("source_binding")
    if not isinstance(source_binding, Mapping) or source_binding.get("implementation_sha256") != sha256_file(
        Path(__file__).resolve()
    ):
        raise ValueError("context split implementation binding changed")
    inventory = acceptance.get("component_inventory")
    if not isinstance(inventory, Mapping) or stable_json_digest(inventory) != acceptance.get(
        "component_inventory_sha256"
    ):
        raise ValueError("context split inventory digest mismatch")
    actual = _closed_regular_files(root, exclude=frozenset({"acceptance.json"}))
    if actual != set(inventory):
        raise ValueError("context split closed inventory membership mismatch")
    for relative, raw in inventory.items():
        if not isinstance(raw, Mapping):
            raise ValueError("malformed context inventory entry")
        safe = _safe_relative(relative, "component path")
        path = root / safe
        if raw.get("relative_path") != safe or path.stat().st_size != int(raw["size_bytes"]):
            raise ValueError(f"context component path/size mismatch: {relative}")
        if sha256_file(path) != raw["sha256"]:
            raise ValueError(f"context component hash mismatch: {relative}")
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows != int(raw["rows"]):
                raise ValueError("context Parquet row count mismatch")
            if any(name.casefold().startswith("label") for name in parquet.schema_arrow.names):
                raise ValueError("context split Parquet contains label columns")
    materialized = 0
    for task in acceptance.get("tasks", []):
        for candidate in task.get("candidates", []):
            if candidate.get("seed_search_performed") is not False:
                raise ValueError("context split used seed search")
            if candidate.get("test_labels_read") is not False:
                raise ValueError("context split read test labels")
            if candidate.get("status") != "materialized":
                continue
            materialized += 1
            split_path = root / _safe_relative(candidate["split_path"], "split path")
            if sha256_file(split_path) != candidate["split_sha256"]:
                raise ValueError("candidate split/status hash mismatch")
            rows = pq.read_table(split_path).to_pylist()
            audit = _audit_rows(rows, candidate["metadata"]["grouping_field"])
            if audit != candidate["audit"] or not audit["group_exclusion_passed"]:
                raise ValueError("context split audit regeneration mismatch")
            if candidate["strategy"] == "strict_temporal" and not audit["strict_chronology_passed"]:
                raise ValueError("temporal candidate is not strictly chronological")
    for key in (
        "candidate_only",
        "official_split_replaced",
        "substantive_training_ready",
        "substantive_training_authorized",
        "substantive_training_started",
    ):
        expected = key == "candidate_only"
        if acceptance.get(key) is not expected:
            raise ValueError(f"context split boundary changed: {key}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "verified",
        "acceptance_sha256": sha256_file(acceptance_path),
        "component_count": len(inventory),
        "materialized_candidates": materialized,
        "test_labels_read": False,
        "substantive_training_started": False,
    }


def verify_context_split_report(
    report_directory: str | os.PathLike[str] = DEFAULT_REPORT_OUTPUT,
    context_split_directory: str | os.PathLike[str] = DEFAULT_OUTPUT,
    *,
    expected_config: ContextSplitConfig | None = None,
) -> dict[str, Any]:
    root = Path(report_directory).resolve()
    manifest = _strict_json(root / "manifest.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unrecognized context report manifest")
    inventory = manifest.get("component_inventory")
    if not isinstance(inventory, Mapping) or stable_json_digest(inventory) != manifest.get(
        "component_inventory_sha256"
    ):
        raise ValueError("context report inventory digest mismatch")
    actual = _closed_regular_files(root, exclude=frozenset({"manifest.json"}))
    if actual != set(inventory):
        raise ValueError("context report inventory membership mismatch")
    for relative, raw in inventory.items():
        safe = _safe_relative(relative, "report component")
        path = root / safe
        if path.stat().st_size != int(raw["size_bytes"]) or sha256_file(path) != raw["sha256"]:
            raise ValueError(f"context report component changed: {relative}")
    report = _strict_json(root / "report.json")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unrecognized context report")
    config = report.get("configuration")
    if not isinstance(config, Mapping) or stable_json_digest(config) != report.get("configuration_sha256"):
        raise ValueError("context report configuration mismatch")
    if expected_config is not None and asdict(expected_config) != dict(config):
        raise ValueError("context report configuration drift")
    acceptance_path = Path(context_split_directory).resolve() / "acceptance.json"
    if sha256_file(acceptance_path) != report.get("context_split_acceptance_sha256"):
        raise ValueError("context report no longer binds candidate acceptance")
    if manifest.get("context_split_acceptance_sha256") != report.get("context_split_acceptance_sha256"):
        raise ValueError("context report manifest binding mismatch")
    for key in (
        "claim_readiness",
        "substantive_training_ready",
        "substantive_training_authorized",
        "substantive_training_started",
    ):
        if report.get(key) is not False:
            raise ValueError(f"context report boundary changed: {key}")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "verified",
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "report_sha256": sha256_file(root / "report.json"),
        "component_count": len(inventory),
        "substantive_training_started": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    build.add_argument("--qc-report", type=Path, default=DEFAULT_QC_REPORT)
    build.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    verify.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        result = materialize_context_split_candidates(
            args.canonical_root,
            args.qc_report,
            args.split_root,
            args.output,
            args.report_output,
        )
    else:
        result = {
            "data": verify_context_split_candidates(args.output),
            "report": verify_context_split_report(args.report_output, args.output),
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ContextSplitConfig",
    "materialize_context_split_candidates",
    "verify_context_split_candidates",
    "verify_context_split_report",
]
