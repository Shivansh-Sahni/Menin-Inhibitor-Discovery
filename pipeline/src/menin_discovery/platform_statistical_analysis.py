"""Manifest-bound, zero-training statistical census of a canonical platform build.

The analysis in this module is deliberately descriptive.  It never combines
incompatible endpoint/unit/relation/assay strata, never converts missing or
censored hERG measurements into classifier labels, and keeps Kd-derived free
energy in a separate sensitivity layer.  Large identity and numeric state is
kept in SQLite so peak Python memory is bounded by one Arrow batch plus small
summary counters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import html
import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import chi2_contingency

from .platform_data_schema import canonical_json, clean_text
from .platform_data_sources import sha256_file

ANALYSIS_VERSION = "platform-statistical-analysis-v1"
DEFAULT_BATCH_SIZE = 65_536
DEFAULT_SAMPLE_CAP = 100
DEFAULT_PLOT_CAP = 20

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SELF_HASH_POLICY = "analysis_manifest.json is necessarily excluded from its own artifact hash inventory"
_SCIENTIFIC_BOUNDARIES = [
    "No incompatible endpoint, unit, relation, assay-family, or exact/censor-bound stratum is pooled.",
    "hERG exact 10/30 micromolar support is not QT, torsades, cardiotoxicity, or clinical risk.",
    "Censored, intermediate, absent, incompatible, and excluded hERG rows are not classifier classes.",
    "Binding free energy is an opt-in exact-positive-Kd sensitivity derivation, never IC50-derived.",
    "Development annotations are metadata only and are neither outcomes nor labels.",
    "Association statistics describe the accepted source snapshot and support no causal or clinical claim.",
]
_INFERENCE_POLICY = {
    "analysis_kind": "exact descriptive census plus one explicitly screened association panel",
    "effect_size": "bias-corrected Cramer's V for tested document-decade/evidence-stage tables",
    "uncertainty_policy": (
        "no confidence intervals: rows exhaust the accepted snapshot and are not IID biological samples"
    ),
    "multiple_testing": "Benjamini-Hochberg over tested domains; untested panels retain explicit reasons",
    "prohibited_interpretations": [
        "causal effects",
        "clinical efficacy or clinical cardiotoxicity",
        "hERG as QT, torsades, or patient risk",
        "p-values as biological-population evidence",
    ],
}
_CSV_SCHEMAS: dict[str, tuple[str, ...]] = {
    "composition.csv": ("dataset", "dimension", "value", "rows"),
    "joint_composition.csv": (
        "dataset",
        "task_id",
        "evidence_domain",
        "endpoint",
        "unit",
        "relation",
        "assay_family",
        "source_id",
        "evidence_stage",
        "development_stage",
        "observation_kind",
        "label_kind",
        "inclusion_status",
        "rows",
    ),
    "missingness.csv": (
        "dataset",
        "field",
        "rows",
        "missing_rows",
        "present_rows",
        "missing_fraction",
    ),
    "attrition.csv": ("layer", "evidence_domain", "inclusion_status", "reason", "rows"),
    "model_input_exclusions.csv": (
        "task_scope",
        "stage_or_reason_kind",
        "value",
        "rows",
    ),
    "compatible_stratum_summaries.csv": (
        "dataset",
        "task_scope",
        "evidence_domain",
        "endpoint",
        "unit",
        "relation",
        "assay_family",
        "observation_kind",
        "measure_role",
        "rows",
        "minimum",
        "p01",
        "q1",
        "median",
        "q3",
        "p99",
        "maximum",
        "mean",
        "population_sd",
        "iqr",
        "quantile_method",
    ),
    "coverage_long_tail.csv": (
        "dataset",
        "dimension",
        "population_rows",
        "unique_values",
        "singleton_values",
        "values_with_2_to_5_rows",
        "values_with_6_to_10_rows",
        "values_with_over_10_rows",
        "top_1_share",
        "top_10_share",
        "top_100_share",
        "hhi",
        "effective_value_count_inverse_hhi",
        "gini_count_concentration",
    ),
    "coverage_top_entities.csv": (
        "dataset",
        "dimension",
        "rank",
        "value",
        "rows",
        "population_rows",
        "share",
        "cap",
    ),
    "duplicate_conflict_summary.csv": (
        "group_kind",
        "summary_kind",
        "group_size",
        "group_count",
        "rows",
        "repeated_group_count",
        "rows_in_repeated_groups",
        "maximum_group_size",
    ),
    "temporal_stage_composition.csv": (
        "layer",
        "evidence_domain",
        "endpoint",
        "document_decade",
        "evidence_stage",
        "development_stage",
        "rows",
    ),
    "herg_10_30uM_support.csv": (
        "population",
        "category",
        "rows",
        "threshold_low_nM",
        "threshold_high_nM",
        "class_policy",
    ),
    "kd_free_energy_sensitivity.csv": ("dimension", "value", "rows", "analysis_scope"),
    "development_metadata.csv": ("dimension", "value", "rows", "semantic_role"),
    "association_panel.csv": (
        "panel",
        "evidence_domain",
        "rows",
        "year_bins",
        "stage_levels",
        "status",
        "reason",
        "chi_square",
        "p_value",
        "q_value",
        "cramers_v_bias_corrected",
        "assumptions",
        "multiple_testing",
    ),
}
_NON_CSV_ARTIFACTS = {
    "deterministic_samples.json",
    "inference_policy.json",
    "plots/observation_domains.svg",
    "plots/top_task_types.svg",
}


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.part")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_scalar(row.get(column)) for column in columns})
    os.replace(temporary, path)
    return len(rows)


def _csv_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, ".17g")
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _missing_mask(series: pd.Series) -> pd.Series:
    missing = series.isna()
    if pd.api.types.is_string_dtype(series.dtype) or series.dtype == object:
        missing = missing | series.fillna("").astype(str).str.strip().eq("")
    if pd.api.types.is_float_dtype(series.dtype):
        numeric = pd.to_numeric(series, errors="coerce")
        missing = missing | ~pd.Series(np.isfinite(numeric), index=series.index)
    return missing


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _endpoint_key(value: Any) -> str:
    return "".join(character for character in clean_text(value).casefold() if character.isalnum())


def _year(value: Any) -> int | None:
    number = _finite(value)
    if number is None or not float(number).is_integer():
        return None
    integer = int(number)
    return integer if 1800 <= integer <= 2200 else None


def _year_bin(value: Any) -> str:
    integer = _year(value)
    return f"{integer // 10 * 10}s" if integer is not None else "<missing_or_invalid>"


class _HashSample:
    """Keep the lexicographically smallest SHA-256 identities in bounded memory."""

    def __init__(self, cap: int) -> None:
        if cap < 1:
            raise ValueError("sample cap must be positive")
        self.cap = cap
        self._heap: list[tuple[int, str, dict[str, Any]]] = []
        self.population_rows = 0

    def add(self, identity: str, payload: Mapping[str, Any]) -> None:
        self.population_rows += 1
        digest = int(hashlib.sha256(identity.encode("utf-8")).hexdigest(), 16)
        encoded = canonical_json(dict(payload))
        item = (-digest, encoded, dict(payload))
        if len(self._heap) < self.cap:
            heapq.heappush(self._heap, item)
        elif digest < -self._heap[0][0]:
            heapq.heapreplace(self._heap, item)

    def records(self) -> list[dict[str, Any]]:
        return [payload for _, _, payload in sorted(self._heap, key=lambda item: (-item[0], item[1]))]


@dataclass(frozen=True)
class _InputBinding:
    build_root: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    qc_report: dict[str, Any]
    qc_report_sha256: str
    component_records: tuple[dict[str, Any], ...]


def _load_and_verify_input(build_root: Path, qc_report_path: Path) -> _InputBinding:
    canonical = build_root.resolve()
    if canonical.name.startswith(".") or canonical.name.endswith(".building"):
        raise RuntimeError("Refusing a provisional canonical .building directory")
    manifest_path = canonical / "build_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Canonical input lacks build_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("Canonical build manifest must be a JSON object")
    inventory = manifest.get("component_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise RuntimeError("Canonical build manifest lacks a non-empty component inventory")
    records: list[dict[str, Any]] = []
    declared_paths: list[str] = []
    for candidate in inventory:
        if not isinstance(candidate, dict):
            raise RuntimeError("Canonical component inventory contains a non-object")
        relative = clean_text(candidate.get("path"))
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RuntimeError(f"Unsafe canonical component path: {relative!r}")
        declared_paths.append(relative)
        records.append(dict(candidate))
    if len(declared_paths) != len(set(declared_paths)):
        raise RuntimeError("Canonical component inventory contains duplicate paths")
    actual_paths: list[str] = []
    for path in sorted(canonical.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Canonical inputs may not contain symlinks: {path}")
        if path.is_file() and path.name != "build_manifest.json":
            actual_paths.append(path.relative_to(canonical).as_posix())
    if sorted(declared_paths) != actual_paths:
        raise RuntimeError("Canonical component inventory does not match exact recursive membership")
    for record in records:
        path = canonical / clean_text(record["path"])
        if path.stat().st_size != int(record.get("size_bytes", -1)):
            raise RuntimeError(f"Canonical component size mismatch: {record['path']}")
        if sha256_file(path) != clean_text(record.get("sha256")):
            raise RuntimeError(f"Canonical component SHA-256 mismatch: {record['path']}")
        if path.suffix.casefold() == ".parquet" and "rows" in record:
            metadata = pq.ParquetFile(path).metadata
            if metadata is None or int(metadata.num_rows) != int(record["rows"]):
                raise RuntimeError(f"Canonical component Parquet row mismatch: {record['path']}")
    qc_path = qc_report_path.resolve()
    if not qc_path.is_file():
        raise RuntimeError("A bound canonical QC report is required")
    qc_report = json.loads(qc_path.read_text(encoding="utf-8"))
    manifest_sha256 = sha256_file(manifest_path)
    if not isinstance(qc_report, dict) or not bool(qc_report.get("qc_passed")):
        raise RuntimeError("Canonical QC report is absent or not accepted")
    if clean_text(qc_report.get("build_manifest_sha256")) != manifest_sha256:
        raise RuntimeError("Canonical QC report is not bound to the current build manifest")
    return _InputBinding(
        build_root=canonical,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        qc_report=qc_report,
        qc_report_sha256=sha256_file(qc_path),
        component_records=tuple(sorted(records, key=lambda row: clean_text(row["path"]))),
    )


def _parquet_paths(binding: _InputBinding, prefix: str) -> list[Path]:
    return [
        binding.build_root / clean_text(record["path"])
        for record in binding.component_records
        if clean_text(record["path"]).startswith(prefix)
        and clean_text(record["path"]).casefold().endswith(".parquet")
    ]


def _batches(
    paths: Iterable[Path],
    *,
    required: Iterable[str] = (),
    batch_size: int,
) -> Iterator[pd.DataFrame]:
    required_set = set(required)
    for path in sorted(paths):
        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema_arrow.names)
        missing = sorted(required_set - columns)
        if missing:
            raise RuntimeError(f"{path} lacks analysis-required columns: {missing}")
        for batch in parquet.iter_batches(batch_size=batch_size):
            yield batch.to_pandas()


class _AnalysisState:
    """Bounded-memory counters plus disk-backed high-cardinality state."""

    def __init__(self, database_path: Path, sample_cap: int) -> None:
        self.connection = sqlite3.connect(database_path)
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.executescript(
            """
            CREATE TABLE assays (
                assay_id TEXT PRIMARY KEY,
                assay_family TEXT NOT NULL
            );
            CREATE TABLE protein_targets (
                protein_id TEXT PRIMARY KEY,
                canonical_target_id TEXT NOT NULL
            );
            CREATE TABLE coverage (
                dataset TEXT NOT NULL,
                dimension TEXT NOT NULL,
                value TEXT NOT NULL,
                rows INTEGER NOT NULL,
                PRIMARY KEY(dataset, dimension, value)
            );
            CREATE TABLE observation_assays (
                assay_id TEXT PRIMARY KEY,
                rows INTEGER NOT NULL
            );
            CREATE TABLE observation_joint (
                evidence_domain TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                unit TEXT NOT NULL,
                relation TEXT NOT NULL,
                assay_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                evidence_stage TEXT NOT NULL,
                development_stage TEXT NOT NULL,
                observation_kind TEXT NOT NULL,
                inclusion_status TEXT NOT NULL,
                rows INTEGER NOT NULL,
                PRIMARY KEY(
                    evidence_domain, endpoint, unit, relation, assay_id, source_id,
                    evidence_stage, development_stage, observation_kind, inclusion_status
                )
            );
            CREATE TABLE joint_composition (
                dataset TEXT NOT NULL,
                task_id TEXT NOT NULL,
                evidence_domain TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                unit TEXT NOT NULL,
                relation TEXT NOT NULL,
                assay_family TEXT NOT NULL,
                source_id TEXT NOT NULL,
                evidence_stage TEXT NOT NULL,
                development_stage TEXT NOT NULL,
                observation_kind TEXT NOT NULL,
                label_kind TEXT NOT NULL,
                inclusion_status TEXT NOT NULL,
                rows INTEGER NOT NULL,
                PRIMARY KEY(
                    dataset, task_id, evidence_domain, endpoint, unit, relation,
                    assay_family, source_id, evidence_stage, development_stage,
                    observation_kind, label_kind, inclusion_status
                )
            );
            CREATE TABLE numeric_values (
                dataset TEXT NOT NULL,
                task_scope TEXT NOT NULL,
                evidence_domain TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                unit TEXT NOT NULL,
                relation TEXT NOT NULL,
                assay_family TEXT NOT NULL,
                observation_kind TEXT NOT NULL,
                measure_role TEXT NOT NULL,
                document_year INTEGER,
                value REAL NOT NULL
            );
            CREATE TABLE observation_numeric (
                evidence_domain TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                unit TEXT NOT NULL,
                relation TEXT NOT NULL,
                assay_id TEXT NOT NULL,
                observation_kind TEXT NOT NULL,
                measure_role TEXT NOT NULL,
                document_year INTEGER,
                value REAL NOT NULL
            );
            CREATE TABLE duplicate_rows (
                group_kind TEXT NOT NULL,
                group_id TEXT NOT NULL,
                observation_id TEXT NOT NULL,
                molecule_id TEXT NOT NULL,
                protein_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                unit TEXT NOT NULL,
                relation TEXT NOT NULL,
                value REAL
            );
            CREATE TABLE global_conflict_rows (
                group_id TEXT NOT NULL,
                observation_id TEXT NOT NULL,
                molecule_id TEXT NOT NULL,
                protein_id TEXT NOT NULL,
                assay_id TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                unit TEXT NOT NULL,
                source_id TEXT NOT NULL,
                value REAL NOT NULL
            );
            CREATE TABLE herg_candidates (
                assay_id TEXT NOT NULL,
                observation_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                unit TEXT NOT NULL,
                canonical_value REAL,
                lower_bound REAL,
                upper_bound REAL,
                inclusion_status TEXT NOT NULL,
                observation_kind TEXT NOT NULL
            );
            """
        )
        self.composition: Counter[tuple[str, str, str]] = Counter()
        self.dataset_rows: Counter[str] = Counter()
        self.missing: Counter[tuple[str, str]] = Counter()
        self.denominators: Counter[tuple[str, str]] = Counter()
        self.attrition: Counter[tuple[str, str, str, str]] = Counter()
        self.readiness: Counter[tuple[str, str, str]] = Counter()
        self.temporal_stage: Counter[tuple[str, str, str, str, str, str]] = Counter()
        self.development: Counter[tuple[str, str]] = Counter()
        self.derivation: Counter[tuple[str, str]] = Counter()
        self.herg_emitted: Counter[str] = Counter()
        self.observation_sample = _HashSample(sample_cap)
        self.task_sample = _HashSample(sample_cap)

    def close(self) -> None:
        self.connection.close()

    def commit(self) -> None:
        self.connection.commit()

    def add_missingness(self, dataset: str, frame: pd.DataFrame) -> None:
        self.dataset_rows[dataset] += len(frame)
        for column in frame.columns:
            key = (dataset, str(column))
            self.denominators[key] += len(frame)
            self.missing[key] += int(_missing_mask(frame[column]).sum())

    def add_coverage(self, dataset: str, dimension: str, values: pd.Series) -> None:
        counts = values.fillna("").astype(str).str.strip().replace("", "<missing>").value_counts()
        self.connection.executemany(
            """
            INSERT INTO coverage(dataset, dimension, value, rows) VALUES (?, ?, ?, ?)
            ON CONFLICT(dataset, dimension, value) DO UPDATE SET rows=rows+excluded.rows
            """,
            [(dataset, dimension, str(value), int(count)) for value, count in counts.items()],
        )

    def add_composition(self, dataset: str, dimension: str, values: pd.Series) -> None:
        counts = values.fillna("").astype(str).str.strip().replace("", "<missing>").value_counts()
        self.composition.update(
            {(dataset, dimension, str(value)): int(count) for value, count in counts.items()}
        )


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], dataset: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{dataset} batch lacks required columns: {missing}")


def _load_entities(state: _AnalysisState, binding: _InputBinding, batch_size: int) -> None:
    assay_paths = _parquet_paths(binding, "assays/")
    protein_paths = _parquet_paths(binding, "proteins/")
    if not assay_paths or not protein_paths:
        raise RuntimeError("Canonical analysis requires assays and proteins datasets")
    for frame in _batches(assay_paths, required=("assay_id", "assay_family"), batch_size=batch_size):
        state.add_missingness("assays", frame)
        rows = [
            (clean_text(row.assay_id), clean_text(row.assay_family) or "<missing>")
            for row in frame[["assay_id", "assay_family"]].itertuples(index=False)
        ]
        state.connection.executemany("INSERT INTO assays(assay_id, assay_family) VALUES (?, ?)", rows)
    for frame in _batches(
        protein_paths,
        required=("protein_id", "canonical_target_id"),
        batch_size=batch_size,
    ):
        state.add_missingness("proteins", frame)
        rows = [
            (
                clean_text(row.protein_id),
                clean_text(row.canonical_target_id) or "<missing>",
            )
            for row in frame[["protein_id", "canonical_target_id"]].itertuples(index=False)
        ]
        state.connection.executemany(
            "INSERT INTO protein_targets(protein_id, canonical_target_id) VALUES (?, ?)", rows
        )
    molecule_paths = _parquet_paths(binding, "molecules/")
    for frame in _batches(molecule_paths, required=("molecule_id",), batch_size=batch_size):
        state.add_missingness("molecules", frame)
    state.commit()


def _numeric_records(
    frame: pd.DataFrame,
    *,
    dataset: str,
    task_scope: str,
    value_column: str,
    unit_column: str,
    relation_column: str,
    lower_column: str,
    upper_column: str,
    assay_column: str,
    assay_is_family: bool,
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    direct: list[tuple[Any, ...]] = []
    observations: list[tuple[Any, ...]] = []
    for row in frame.itertuples(index=False):
        values = row._asdict()
        relation = clean_text(values.get(relation_column))
        base = (
            clean_text(values.get("evidence_domain")) or "<missing>",
            clean_text(values.get("endpoint")) or "<missing>",
            clean_text(values.get(unit_column)) or "<missing>",
            relation or "<missing>",
        )
        observation_kind = clean_text(values.get("observation_kind")) or "<missing>"
        year = _year(values.get("document_year"))
        measures: list[tuple[str, float]] = []
        exact = _finite(values.get(value_column))
        lower = _finite(values.get(lower_column))
        upper = _finite(values.get(upper_column))
        if relation == "=" and exact is not None:
            measures.append(("exact_value", exact))
        elif relation in {">", ">=", "interval"} and lower is not None:
            measures.append(("lower_censor_bound", lower))
        if relation in {"<", "<=", "interval"} and upper is not None:
            measures.append(("upper_censor_bound", upper))
        for measure_role, number in measures:
            assay_value = clean_text(values.get(assay_column)) or "<missing>"
            if assay_is_family:
                direct.append(
                    (
                        dataset,
                        task_scope,
                        *base,
                        assay_value,
                        observation_kind,
                        measure_role,
                        year,
                        number,
                    )
                )
            else:
                observations.append((*base, assay_value, observation_kind, measure_role, year, number))
    return direct, observations


def _scan_observations(state: _AnalysisState, binding: _InputBinding, batch_size: int) -> None:
    paths = _parquet_paths(binding, "observations/")
    required = (
        "observation_id",
        "molecule_id",
        "protein_id",
        "assay_id",
        "source_id",
        "evidence_domain",
        "endpoint",
        "relation",
        "canonical_value",
        "canonical_unit",
        "lower_bound",
        "upper_bound",
        "observation_kind",
        "evidence_stage",
        "development_stage",
        "inclusion_status",
        "exclusion_reason",
        "dedup_group_id",
        "conflict_group_id",
    )
    if not paths:
        raise RuntimeError("Canonical analysis requires observation shards")
    for frame in _batches(paths, required=required, batch_size=batch_size):
        _require_columns(frame, required, "observations")
        state.add_missingness("observations", frame)
        for dimension, column in (
            ("evidence_domain", "evidence_domain"),
            ("endpoint", "endpoint"),
            ("unit", "canonical_unit"),
            ("relation", "relation"),
            ("source", "source_id"),
            ("evidence_stage", "evidence_stage"),
            ("development_stage", "development_stage"),
            ("result_status", "result_status"),
            ("observation_kind", "observation_kind"),
            ("inclusion_status", "inclusion_status"),
        ):
            if column in frame.columns:
                state.add_composition("observations", dimension, frame[column])
        for dimension, column in (
            ("molecule_id", "molecule_id"),
            ("protein_id", "protein_id"),
            ("assay_id", "assay_id"),
            ("source_id", "source_id"),
        ):
            state.add_coverage("observations", dimension, frame[column])
        assay_counts = (
            frame["assay_id"].fillna("").astype(str).str.strip().replace("", "<missing>").value_counts()
        )
        state.connection.executemany(
            """
            INSERT INTO observation_assays(assay_id, rows) VALUES (?, ?)
            ON CONFLICT(assay_id) DO UPDATE SET rows=rows+excluded.rows
            """,
            [(str(value), int(count)) for value, count in assay_counts.items()],
        )
        joint_counts: Counter[tuple[str, ...]] = Counter()
        for row in frame.itertuples(index=False):
            values = row._asdict()
            joint_counts[
                (
                    clean_text(values["evidence_domain"]) or "<missing>",
                    clean_text(values["endpoint"]) or "<missing>",
                    clean_text(values["canonical_unit"]) or "<missing>",
                    clean_text(values["relation"]) or "<missing>",
                    clean_text(values["assay_id"]) or "<missing>",
                    clean_text(values["source_id"]) or "<missing>",
                    clean_text(values["evidence_stage"]) or "<missing>",
                    clean_text(values["development_stage"]) or "<missing>",
                    clean_text(values["observation_kind"]) or "<missing>",
                    clean_text(values["inclusion_status"]) or "<missing>",
                )
            ] += 1
        state.connection.executemany(
            """
            INSERT INTO observation_joint VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO UPDATE SET rows=rows+excluded.rows
            """,
            [(*key, int(count)) for key, count in joint_counts.items()],
        )
        direct, observation_numeric = _numeric_records(
            frame,
            dataset="observations",
            task_scope="",
            value_column="canonical_value",
            unit_column="canonical_unit",
            relation_column="relation",
            lower_column="lower_bound",
            upper_column="upper_bound",
            assay_column="assay_id",
            assay_is_family=False,
        )
        if direct:
            raise AssertionError("Observation numeric records unexpectedly bypassed assay binding")
        state.connection.executemany(
            "INSERT INTO observation_numeric VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            observation_numeric,
        )
        duplicate_records: list[tuple[Any, ...]] = []
        global_conflict_records: list[tuple[str, str, str, str, str, str, str, str, float]] = []
        herg_records: list[tuple[Any, ...]] = []
        for row in frame.itertuples(index=False):
            values = row._asdict()
            identity = clean_text(values["observation_id"])
            payload = {
                "observation_id": identity,
                "evidence_domain": clean_text(values["evidence_domain"]),
                "endpoint": clean_text(values["endpoint"]),
                "relation": clean_text(values["relation"]),
                "canonical_unit": clean_text(values["canonical_unit"]),
                "inclusion_status": clean_text(values["inclusion_status"]),
            }
            state.observation_sample.add(identity, payload)
            domain = clean_text(values["evidence_domain"])
            status = clean_text(values["inclusion_status"]) or "<missing>"
            layer = "derived" if clean_text(values["observation_kind"]) == "derived" else "source"
            reasons = clean_text(values.get("exclusion_reason"))
            state.attrition[(layer, domain or "<missing>", status, "<all>")] += 1
            if reasons:
                state.attrition[(layer, domain or "<missing>", status, f"combination:{reasons}")] += 1
                for reason in reasons.split(";"):
                    if reason:
                        state.attrition[(layer, domain or "<missing>", status, f"reason:{reason}")] += 1
            temporal_key = (
                layer,
                domain or "<missing>",
                clean_text(values["endpoint"]) or "<missing>",
                _year_bin(values.get("document_year")),
                clean_text(values["evidence_stage"]) or "<missing>",
                clean_text(values["development_stage"]) or "<missing>",
            )
            state.temporal_stage[temporal_key] += 1
            for kind, column in (("dedup", "dedup_group_id"), ("conflict", "conflict_group_id")):
                group_id = clean_text(values.get(column))
                if group_id:
                    duplicate_records.append(
                        (
                            kind,
                            group_id,
                            identity,
                            clean_text(values["molecule_id"]),
                            clean_text(values["protein_id"]),
                            clean_text(values["source_id"]),
                            clean_text(values["endpoint"]),
                            clean_text(values["canonical_unit"]),
                            clean_text(values["relation"]),
                            _finite(values["canonical_value"]),
                        )
                    )
            exact_value = _finite(values["canonical_value"])
            if clean_text(values["relation"]) == "=" and exact_value is not None and exact_value > 0:
                context = [
                    clean_text(values["molecule_id"]),
                    clean_text(values["protein_id"]),
                    clean_text(values["assay_id"]),
                    clean_text(values["endpoint"]),
                    clean_text(values["canonical_unit"]),
                ]
                global_conflict_records.append(
                    (
                        hashlib.sha256(canonical_json(context).encode("utf-8")).hexdigest(),
                        identity,
                        context[0],
                        context[1],
                        context[2],
                        context[3],
                        context[4],
                        clean_text(values["source_id"]),
                        exact_value,
                    )
                )
            if domain.casefold() == "herg" and _endpoint_key(values["endpoint"]) == "ic50":
                herg_records.append(
                    (
                        clean_text(values["assay_id"]),
                        identity,
                        clean_text(values["relation"]),
                        clean_text(values["canonical_unit"]),
                        _finite(values["canonical_value"]),
                        _finite(values["lower_bound"]),
                        _finite(values["upper_bound"]),
                        status,
                        clean_text(values["observation_kind"]),
                    )
                )
        state.connection.executemany(
            "INSERT INTO duplicate_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", duplicate_records
        )
        state.connection.executemany(
            "INSERT INTO global_conflict_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", global_conflict_records
        )
        state.connection.executemany(
            "INSERT INTO herg_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", herg_records
        )
        state.commit()
    state.connection.execute(
        """
        INSERT INTO numeric_values
        SELECT 'observations', '', n.evidence_domain, n.endpoint, n.unit, n.relation,
               COALESCE(a.assay_family, '<unresolved_assay_family>'), n.observation_kind,
               n.measure_role, n.document_year, n.value
        FROM observation_numeric AS n
        LEFT JOIN assays AS a USING(assay_id)
        """
    )
    state.connection.execute(
        """
        INSERT INTO joint_composition
        SELECT 'observations', '<not_applicable>', j.evidence_domain, j.endpoint,
               j.unit, j.relation,
               COALESCE(a.assay_family, '<unresolved_assay_family>'), j.source_id,
               j.evidence_stage, j.development_stage, j.observation_kind,
               '<not_applicable>', j.inclusion_status, SUM(j.rows)
        FROM observation_joint AS j LEFT JOIN assays AS a USING(assay_id)
        GROUP BY j.evidence_domain, j.endpoint, j.unit, j.relation,
                 COALESCE(a.assay_family, '<unresolved_assay_family>'), j.source_id,
                 j.evidence_stage, j.development_stage, j.observation_kind,
                 j.inclusion_status
        """
    )
    for assay_family, rows in state.connection.execute(
        """
        SELECT COALESCE(a.assay_family, '<unresolved_assay_family>'), SUM(o.rows)
        FROM observation_assays AS o LEFT JOIN assays AS a USING(assay_id)
        GROUP BY COALESCE(a.assay_family, '<unresolved_assay_family>')
        ORDER BY 1
        """
    ):
        state.composition[("observations", "assay_family", str(assay_family))] += int(rows)
    state.connection.execute(
        """
        INSERT INTO coverage(dataset, dimension, value, rows)
        SELECT c.dataset, 'canonical_target_id', COALESCE(p.canonical_target_id, '<unresolved_target>'), SUM(c.rows)
        FROM coverage AS c LEFT JOIN protein_targets AS p ON c.value=p.protein_id
        WHERE c.dataset='observations' AND c.dimension='protein_id'
        GROUP BY c.dataset, COALESCE(p.canonical_target_id, '<unresolved_target>')
        """
    )
    state.commit()


def _task_scope(path: Path, build_root: Path) -> str:
    parts = path.relative_to(build_root).parts
    return parts[1] if len(parts) >= 3 and parts[0] == "tasks" else "unknown"


def _scan_tasks(state: _AnalysisState, binding: _InputBinding, batch_size: int) -> None:
    paths = _parquet_paths(binding, "tasks/")
    required = (
        "task_id",
        "task_type",
        "observation_id",
        "molecule_id",
        "protein_id",
        "assay_id",
        "source_id",
        "label_kind",
        "label_value",
        "label_relation",
        "label_lower_bound",
        "label_upper_bound",
        "label_unit",
        "required_modalities",
        "evidence_domain",
        "endpoint",
        "assay_family",
    )
    if not paths:
        raise RuntimeError("Canonical analysis requires emitted task shards")
    for path in sorted(paths):
        scope = _task_scope(path, binding.build_root)
        for frame in _batches([path], required=required, batch_size=batch_size):
            dataset = f"tasks/{scope}"
            state.add_missingness(dataset, frame)
            state.readiness[(scope, "eligible", "<all>")] += len(frame)
            for dimension, column in (
                ("task", "task_id"),
                ("task_type", "task_type"),
                ("domain", "evidence_domain"),
                ("endpoint", "endpoint"),
                ("unit", "label_unit"),
                ("relation", "label_relation"),
                ("assay_family", "assay_family"),
                ("source", "source_id"),
                ("observation_kind", "observation_kind"),
                ("label_kind", "label_kind"),
            ):
                if column in frame.columns:
                    state.add_composition(dataset, dimension, frame[column])
            for dimension, column in (
                ("molecule_id", "molecule_id"),
                ("protein_id", "protein_id"),
                ("canonical_target_id", "canonical_target_id"),
                ("assay_id", "assay_id"),
                ("source_id", "source_id"),
            ):
                if column in frame.columns:
                    state.add_coverage(dataset, dimension, frame[column])
            joint_counts: Counter[tuple[str, ...]] = Counter()
            for row in frame.itertuples(index=False):
                values = row._asdict()
                joint_counts[
                    (
                        dataset,
                        clean_text(values["task_id"]) or "<missing>",
                        clean_text(values["evidence_domain"]) or "<missing>",
                        clean_text(values["endpoint"]) or "<missing>",
                        clean_text(values["label_unit"]) or "<missing>",
                        clean_text(values["label_relation"]) or "<missing>",
                        clean_text(values["assay_family"]) or "<missing>",
                        clean_text(values["source_id"]) or "<missing>",
                        "<not_carried_by_task_schema>",
                        "<not_carried_by_task_schema>",
                        clean_text(values["observation_kind"]) or "<missing>",
                        clean_text(values["label_kind"]) or "<missing>",
                        clean_text(values.get("inclusion_status")) or "<missing>",
                    )
                ] += 1
            state.connection.executemany(
                """
                INSERT INTO joint_composition VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO UPDATE SET rows=rows+excluded.rows
                """,
                [(*key, int(count)) for key, count in joint_counts.items()],
            )
            direct, observations = _numeric_records(
                frame,
                dataset=dataset,
                task_scope=scope,
                value_column="label_value",
                unit_column="label_unit",
                relation_column="label_relation",
                lower_column="label_lower_bound",
                upper_column="label_upper_bound",
                assay_column="assay_family",
                assay_is_family=True,
            )
            if observations:
                raise AssertionError("Task numeric records unexpectedly require assay lookup")
            state.connection.executemany(
                "INSERT INTO numeric_values VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", direct
            )
            for row in frame.itertuples(index=False):
                values = row._asdict()
                identity = f"{scope}:{clean_text(values['task_id'])}:{clean_text(values['observation_id'])}"
                state.task_sample.add(
                    identity,
                    {
                        "task_scope": scope,
                        "task_id": clean_text(values["task_id"]),
                        "observation_id": clean_text(values["observation_id"]),
                        "task_type": clean_text(values["task_type"]),
                        "label_kind": clean_text(values["label_kind"]),
                        "label_relation": clean_text(values["label_relation"]),
                    },
                )
                if (
                    scope == "default"
                    and clean_text(values["evidence_domain"]).casefold() == "herg"
                    and _endpoint_key(values["endpoint"]) == "ic50"
                    and clean_text(values["assay_family"]) == "herg_functional"
                    and clean_text(values["label_kind"]) == "categorical"
                ):
                    label = clean_text(values.get("label_text")) or "<missing_label>"
                    state.herg_emitted[label] += 1
            state.commit()


def _scan_exclusions(state: _AnalysisState, binding: _InputBinding, batch_size: int) -> None:
    for path in _parquet_paths(binding, "task_exclusions/"):
        parts = path.relative_to(binding.build_root).parts
        scope = parts[1] if len(parts) >= 3 else "unknown"
        required = ("model_readiness_exclusion_reason",)
        for frame in _batches([path], required=required, batch_size=batch_size):
            dataset = f"task_exclusions/{scope}"
            state.add_missingness(dataset, frame)
            for reason in frame["model_readiness_exclusion_reason"].map(clean_text):
                combination = reason or "<missing>"
                state.readiness[(scope, "excluded_combination", combination)] += 1
                for item in reason.split(";"):
                    if item:
                        state.readiness[(scope, "excluded_reason", item)] += 1
            state.readiness[(scope, "excluded", "<all>")] += len(frame)


def _scan_development(state: _AnalysisState, binding: _InputBinding, batch_size: int) -> None:
    paths = _parquet_paths(binding, "molecule_development_annotations/")
    for frame in _batches(paths, required=("semantic_role", "molecule_id"), batch_size=batch_size):
        state.add_missingness("development_metadata", frame)
        roles = set(frame["semantic_role"].map(clean_text))
        if roles - {"development_metadata_not_outcome_or_model_label"}:
            raise RuntimeError("Development artifact violates metadata-only semantic role")
        for dimension in (
            "max_phase",
            "first_approval",
            "withdrawn_flag",
            "black_box_warning",
            "therapeutic_flag",
            "molecule_type",
            "semantic_role",
        ):
            if dimension in frame.columns:
                counts = (
                    frame[dimension]
                    .fillna("<missing>")
                    .astype(str)
                    .str.strip()
                    .replace("", "<missing>")
                    .value_counts()
                )
                state.development.update(
                    {(dimension, str(value)): int(count) for value, count in counts.items()}
                )
        state.add_coverage("development_metadata", "molecule_id", frame["molecule_id"])


def _scan_derivations(state: _AnalysisState, binding: _InputBinding, batch_size: int) -> None:
    paths = _parquet_paths(binding, "views/binding_free_energy_standard/")
    required = ("delta_g_kcal_mol", "temperature_source", "formula", "source_kd_value", "source_kd_unit")
    for frame in _batches(paths, required=required, batch_size=batch_size):
        state.add_missingness("kd_free_energy_sensitivity", frame)
        invalid = frame["source_kd_unit"].map(clean_text).ne("nM") | (
            pd.to_numeric(frame["source_kd_value"], errors="coerce") <= 0
        )
        if invalid.any():
            raise RuntimeError("Kd-derived sensitivity artifact includes a non-positive or non-nM Kd")
        for dimension in ("temperature_source", "formula", "standard_state"):
            if dimension in frame.columns:
                counts = (
                    frame[dimension]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .replace("", "<missing>")
                    .value_counts()
                )
                state.derivation.update(
                    {(dimension, str(value)): int(count) for value, count in counts.items()}
                )
        records: list[tuple[Any, ...]] = []
        for row in frame.itertuples(index=False):
            values = row._asdict()
            delta_g = _finite(values["delta_g_kcal_mol"])
            if delta_g is not None:
                records.append(
                    (
                        "kd_free_energy_sensitivity",
                        "derived_sensitivity",
                        "binding_affinity",
                        "standard_binding_free_energy",
                        "kcal/mol",
                        "=",
                        "binding",
                        "derived",
                        "exact_value",
                        None,
                        delta_g,
                    )
                )
        state.connection.executemany(
            "INSERT INTO numeric_values VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", records
        )
        state.commit()


def _finalize_numeric(state: _AnalysisState) -> None:
    state.connection.execute(
        """
        CREATE INDEX numeric_stratum_value ON numeric_values(
            dataset, task_scope, evidence_domain, endpoint, unit, relation,
            assay_family, observation_kind, measure_role, value
        )
        """
    )
    state.connection.execute(
        "CREATE INDEX numeric_temporal ON numeric_values(dataset, evidence_domain, endpoint, unit, document_year, value)"
    )
    state.connection.execute("CREATE INDEX duplicate_group ON duplicate_rows(group_kind, group_id)")
    state.connection.execute("CREATE INDEX global_conflict_group ON global_conflict_rows(group_id, value)")
    state.commit()


def _quantile(
    connection: sqlite3.Connection,
    where_values: tuple[str, ...],
    probability: float,
    count: int,
) -> float:
    position = (count - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    query = (
        "SELECT value FROM numeric_values WHERE dataset=? AND task_scope=? AND "
        "evidence_domain=? AND endpoint=? AND unit=? AND relation=? AND "
        "assay_family=? AND observation_kind=? AND measure_role=? ORDER BY value LIMIT 1 OFFSET ?"
    )
    low = float(connection.execute(query, (*where_values, lower)).fetchone()[0])
    if lower == upper:
        return low
    high = float(connection.execute(query, (*where_values, upper)).fetchone()[0])
    return low + (position - lower) * (high - low)


def _stratum_summaries(state: _AnalysisState) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    query = """
        SELECT dataset, task_scope, evidence_domain, endpoint, unit, relation,
               assay_family, observation_kind, measure_role, COUNT(*), MIN(value), MAX(value),
               AVG(value), AVG(value*value)
        FROM numeric_values
        GROUP BY dataset, task_scope, evidence_domain, endpoint, unit, relation,
                 assay_family, observation_kind, measure_role
        ORDER BY dataset, task_scope, evidence_domain, endpoint, unit, relation,
                 assay_family, observation_kind, measure_role
    """
    for result in state.connection.execute(query):
        keys = tuple(str(value) for value in result[:9])
        count = int(result[9])
        mean = float(result[12])
        variance = max(0.0, float(result[13]) - mean * mean)
        q01 = _quantile(state.connection, keys, 0.01, count)
        q25 = _quantile(state.connection, keys, 0.25, count)
        q50 = _quantile(state.connection, keys, 0.50, count)
        q75 = _quantile(state.connection, keys, 0.75, count)
        q99 = _quantile(state.connection, keys, 0.99, count)
        rows.append(
            {
                "dataset": keys[0],
                "task_scope": keys[1],
                "evidence_domain": keys[2],
                "endpoint": keys[3],
                "unit": keys[4],
                "relation": keys[5],
                "assay_family": keys[6],
                "observation_kind": keys[7],
                "measure_role": keys[8],
                "rows": count,
                "minimum": float(result[10]),
                "p01": q01,
                "q1": q25,
                "median": q50,
                "q3": q75,
                "p99": q99,
                "maximum": float(result[11]),
                "mean": mean,
                "population_sd": math.sqrt(variance),
                "iqr": q75 - q25,
                "quantile_method": "exact_linear_order_statistics_disk_backed",
            }
        )
    return rows


def _coverage_summaries(state: _AnalysisState) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    dimensions = state.connection.execute(
        "SELECT DISTINCT dataset, dimension FROM coverage ORDER BY dataset, dimension"
    ).fetchall()
    for dataset, dimension in dimensions:
        aggregate = state.connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(rows), 0),
                   COALESCE(SUM(CASE WHEN rows=1 THEN 1 ELSE 0 END), 0),
                   COALESCE(SUM(CASE WHEN rows BETWEEN 2 AND 5 THEN 1 ELSE 0 END), 0),
                   COALESCE(SUM(CASE WHEN rows BETWEEN 6 AND 10 THEN 1 ELSE 0 END), 0),
                   COALESCE(SUM(CASE WHEN rows>10 THEN 1 ELSE 0 END), 0),
                   COALESCE(SUM(CAST(rows AS REAL)*rows), 0.0)
            FROM coverage WHERE dataset=? AND dimension=?
            """,
            (dataset, dimension),
        ).fetchone()
        if aggregate is None:
            raise RuntimeError("Coverage aggregate unexpectedly missing")
        unique = int(aggregate[0])
        total = int(aggregate[1])
        weighted_rank = sum(
            index * int(count)
            for index, (count,) in enumerate(
                state.connection.execute(
                    "SELECT rows FROM coverage WHERE dataset=? AND dimension=? ORDER BY rows, value",
                    (dataset, dimension),
                ),
                start=1,
            )
        )
        top_values = [
            (str(value), int(count))
            for value, count in state.connection.execute(
                "SELECT value, rows FROM coverage WHERE dataset=? AND dimension=? ORDER BY rows DESC, value LIMIT 100",
                (dataset, dimension),
            )
        ]
        gini = (2.0 * weighted_rank) / (unique * total) - (unique + 1.0) / unique if unique and total else 0.0
        hhi = float(aggregate[6]) / (total * total) if total else 0.0
        summaries.append(
            {
                "dataset": dataset,
                "dimension": dimension,
                "population_rows": total,
                "unique_values": unique,
                "singleton_values": int(aggregate[2]),
                "values_with_2_to_5_rows": int(aggregate[3]),
                "values_with_6_to_10_rows": int(aggregate[4]),
                "values_with_over_10_rows": int(aggregate[5]),
                "top_1_share": sum(count for _, count in top_values[:1]) / total if total else 0.0,
                "top_10_share": sum(count for _, count in top_values[:10]) / total if total else 0.0,
                "top_100_share": sum(count for _, count in top_values[:100]) / total if total else 0.0,
                "hhi": hhi,
                "effective_value_count_inverse_hhi": 1.0 / hhi if hhi else 0.0,
                "gini_count_concentration": gini,
            }
        )
        for rank, (value, count) in enumerate(top_values[:DEFAULT_PLOT_CAP], start=1):
            top_rows.append(
                {
                    "dataset": dataset,
                    "dimension": dimension,
                    "rank": rank,
                    "value": value,
                    "rows": count,
                    "population_rows": total,
                    "share": count / total if total else 0.0,
                    "cap": DEFAULT_PLOT_CAP,
                }
            )
    return summaries, top_rows


def _duplicate_summaries(
    state: _AnalysisState, sample_cap: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for kind in ("dedup", "conflict"):
        histogram: Counter[int] = Counter()
        group_rows = 0
        repeated_groups = 0
        repeated_rows = 0
        max_group_size = 0
        sampled_groups = 0
        query = """
            SELECT group_id, COUNT(*) AS n, COUNT(DISTINCT molecule_id),
                   COUNT(DISTINCT protein_id), COUNT(DISTINCT source_id),
                   COUNT(DISTINCT endpoint || char(31) || unit), MIN(value), MAX(value)
            FROM duplicate_rows WHERE group_kind=? GROUP BY group_id ORDER BY group_id
        """
        for (
            group_id,
            count,
            molecules,
            proteins,
            sources,
            strata,
            minimum,
            maximum,
        ) in state.connection.execute(query, (kind,)):
            count = int(count)
            group_rows += count
            histogram[count] += 1
            max_group_size = max(max_group_size, count)
            if count > 1:
                repeated_groups += 1
                repeated_rows += count
                if sampled_groups < sample_cap:
                    samples.append(
                        {
                            "group_kind": kind,
                            "group_id": str(group_id),
                            "rows": count,
                            "distinct_molecules": int(molecules),
                            "distinct_proteins": int(proteins),
                            "distinct_sources": int(sources),
                            "distinct_endpoint_unit_strata": int(strata),
                            "minimum_numeric_value": minimum,
                            "maximum_numeric_value": maximum,
                            "positive_value_span_ratio": (
                                float(maximum) / float(minimum)
                                if minimum is not None and maximum is not None and float(minimum) > 0
                                else None
                            ),
                            "sample_policy": f"first_{sample_cap}_lexicographic_group_ids_across_each_kind",
                        }
                    )
                    sampled_groups += 1
        total_groups = sum(histogram.values())
        summaries.append(
            {
                "group_kind": kind,
                "summary_kind": "overall",
                "group_size": "all",
                "group_count": total_groups,
                "rows": group_rows,
                "repeated_group_count": repeated_groups,
                "rows_in_repeated_groups": repeated_rows,
                "maximum_group_size": max_group_size,
            }
        )
        for size, count in sorted(histogram.items()):
            summaries.append(
                {
                    "group_kind": kind,
                    "summary_kind": "size_histogram",
                    "group_size": size,
                    "group_count": count,
                    "rows": size * count,
                    "repeated_group_count": "",
                    "rows_in_repeated_groups": "",
                    "maximum_group_size": "",
                }
            )
    global_histogram: Counter[int] = Counter()
    global_rows = 0
    global_maximum = 0
    sampled_global = 0
    global_query = """
        SELECT group_id, COUNT(*) AS n, COUNT(DISTINCT source_id), MIN(value), MAX(value),
               MIN(molecule_id), MIN(protein_id), MIN(assay_id), MIN(endpoint), MIN(unit)
        FROM global_conflict_rows
        GROUP BY group_id
        HAVING COUNT(*)>=2 AND MIN(value)>0 AND MAX(value)/MIN(value)>=10.0
        ORDER BY group_id
    """
    for (
        group_id,
        count,
        sources,
        minimum,
        maximum,
        molecule,
        protein,
        assay,
        endpoint,
        unit,
    ) in state.connection.execute(global_query):
        count = int(count)
        global_histogram[count] += 1
        global_rows += count
        global_maximum = max(global_maximum, count)
        if sampled_global < sample_cap:
            samples.append(
                {
                    "group_kind": "global_tenfold_conflict",
                    "group_id": str(group_id),
                    "rows": count,
                    "distinct_molecules": 1,
                    "distinct_proteins": 1,
                    "distinct_sources": int(sources),
                    "distinct_endpoint_unit_strata": 1,
                    "minimum_numeric_value": float(minimum),
                    "maximum_numeric_value": float(maximum),
                    "positive_value_span_ratio": float(maximum) / float(minimum),
                    "molecule_id": str(molecule),
                    "protein_id": str(protein),
                    "assay_id": str(assay),
                    "endpoint": str(endpoint),
                    "unit": str(unit),
                    "sample_policy": f"first_{sample_cap}_lexicographic_group_ids_across_each_kind",
                }
            )
            sampled_global += 1
    summaries.append(
        {
            "group_kind": "global_tenfold_conflict",
            "summary_kind": "overall",
            "group_size": "all",
            "group_count": sum(global_histogram.values()),
            "rows": global_rows,
            "repeated_group_count": sum(global_histogram.values()),
            "rows_in_repeated_groups": global_rows,
            "maximum_group_size": global_maximum,
        }
    )
    for size, count in sorted(global_histogram.items()):
        summaries.append(
            {
                "group_kind": "global_tenfold_conflict",
                "summary_kind": "size_histogram",
                "group_size": size,
                "group_count": count,
                "rows": size * count,
                "repeated_group_count": "",
                "rows_in_repeated_groups": "",
                "maximum_group_size": "",
            }
        )
    return summaries, samples


def _herg_support(state: _AnalysisState) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    query = """
        SELECT COALESCE(a.assay_family, '<unresolved_assay_family>'), h.relation, h.unit,
               h.canonical_value, h.lower_bound, h.upper_bound, h.inclusion_status,
               h.observation_kind
        FROM herg_candidates AS h LEFT JOIN assays AS a USING(assay_id)
    """
    for family, relation, unit, value, _lower, _upper, status, kind in state.connection.execute(query):
        counts["all_hERG_IC50_observations"] += 1
        if str(family) != "herg_functional":
            counts["not_class__nonfunctional_or_unresolved_assay"] += 1
        elif str(unit) != "nM":
            counts["not_class__incompatible_or_missing_unit"] += 1
        elif str(status) != "included" or str(kind) not in {
            "experimental_raw",
            "experimental_summary",
            "curated_assertion",
        }:
            counts["not_class__not_included_experimental_evidence"] += 1
        elif str(relation) != "=":
            counts["not_class__censored_or_interval"] += 1
        elif value is None or not math.isfinite(float(value)):
            counts["not_class__missing_exact_value"] += 1
        elif float(value) <= 10_000.0:
            counts["classifier_candidate__blocker_le_10uM"] += 1
        elif float(value) >= 30_000.0:
            counts["classifier_candidate__nonblocker_ge_30uM"] += 1
        else:
            counts["not_class__exact_intermediate_10_to_30uM"] += 1
    rows = [
        {
            "population": "canonical_observations",
            "category": category,
            "rows": count,
            "threshold_low_nM": 10_000.0,
            "threshold_high_nM": 30_000.0,
            "class_policy": (
                "exact included experimental hERG functional IC50 in nM only; "
                "censored, intermediate, absent, incompatible, and excluded rows are not classes"
            ),
        }
        for category, count in sorted(counts.items())
    ]
    for label, count in sorted(state.herg_emitted.items()):
        rows.append(
            {
                "population": "emitted_model_ready_binary_task",
                "category": f"emitted_class__{label}",
                "rows": count,
                "threshold_low_nM": 10_000.0,
                "threshold_high_nM": 30_000.0,
                "class_policy": "post-model-input-readiness subset of exact candidate classes",
            }
        )
    return rows


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return deterministic Benjamini-Hochberg adjusted values in input order."""

    if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for value in p_values):
        raise ValueError("p-values must be finite values in [0, 1]")
    count = len(p_values)
    if count == 0:
        return []
    order = sorted(range(count), key=lambda index: (float(p_values[index]), index))
    adjusted = [1.0] * count
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        candidate = min(1.0, float(p_values[original_index]) * count / rank)
        running = min(running, candidate)
        adjusted[original_index] = running
    return adjusted


def _association_panel(
    temporal: Mapping[tuple[str, str, str, str, str, str], int],
) -> list[dict[str, Any]]:
    by_domain: defaultdict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for (layer, domain, _endpoint, year_bin, evidence_stage, _development_stage), count in temporal.items():
        if layer == "source" and year_bin != "<missing_or_invalid>" and evidence_stage != "<missing>":
            by_domain[domain][(year_bin, evidence_stage)] += int(count)
    rows: list[dict[str, Any]] = []
    tested_positions: list[int] = []
    p_values: list[float] = []
    for domain, counts in sorted(by_domain.items()):
        years = sorted({key[0] for key in counts})
        stages = sorted({key[1] for key in counts})
        total = sum(counts.values())
        base = {
            "panel": "document_decade_x_evidence_stage_within_domain",
            "evidence_domain": domain,
            "rows": total,
            "year_bins": len(years),
            "stage_levels": len(stages),
            "assumptions": (
                "descriptive source-snapshot association; independent-cell chi-square approximation; "
                "all expected cells >=5; no causal, temporal-trend, or population-generalization claim"
            ),
            "multiple_testing": "Benjamini-Hochberg across all tested evidence domains",
        }
        if total < 100 or len(years) < 2 or len(stages) < 2:
            rows.append(
                {
                    **base,
                    "status": "not_tested",
                    "reason": "requires_n>=100_and_at_least_2x2",
                    "chi_square": None,
                    "p_value": None,
                    "q_value": None,
                    "cramers_v_bias_corrected": None,
                }
            )
            continue
        table = np.asarray([[counts[(year, stage)] for stage in stages] for year in years], dtype=float)
        chi_square, p_value, _degrees, expected = chi2_contingency(table, correction=False)
        if float(expected.min()) < 5.0:
            rows.append(
                {
                    **base,
                    "status": "not_tested",
                    "reason": "minimum_expected_cell_below_5",
                    "chi_square": None,
                    "p_value": None,
                    "q_value": None,
                    "cramers_v_bias_corrected": None,
                }
            )
            continue
        row_count, column_count = table.shape
        phi_squared = float(chi_square) / total
        corrected_phi = max(0.0, phi_squared - ((column_count - 1) * (row_count - 1)) / (total - 1))
        corrected_rows = row_count - ((row_count - 1) ** 2) / (total - 1)
        corrected_columns = column_count - ((column_count - 1) ** 2) / (total - 1)
        denominator = min(corrected_rows - 1, corrected_columns - 1)
        effect = math.sqrt(corrected_phi / denominator) if denominator > 0 else 0.0
        rows.append(
            {
                **base,
                "status": "tested",
                "reason": "",
                "chi_square": float(chi_square),
                "p_value": float(p_value),
                "q_value": None,
                "cramers_v_bias_corrected": effect,
            }
        )
        tested_positions.append(len(rows) - 1)
        p_values.append(float(p_value))
    adjusted = benjamini_hochberg(p_values)
    for position, q_value in zip(tested_positions, adjusted, strict=True):
        rows[position]["q_value"] = q_value
    return rows


def _reconcile(state: _AnalysisState, binding: _InputBinding) -> dict[str, Any]:
    manifest = binding.manifest
    source_status: Counter[str] = Counter()
    source_reasons: Counter[str] = Counter()
    for (layer, _domain, status, reason), count in state.attrition.items():
        if layer != "source":
            continue
        if reason == "<all>":
            source_status[status] += count
        elif reason.startswith("reason:"):
            source_reasons[reason.removeprefix("reason:")] += count
    expected_attrition = manifest.get("canonical_attrition", {})
    if dict(sorted(source_status.items())) != {
        str(key): int(value)
        for key, value in sorted(expected_attrition.get("inclusion_status_counts", {}).items())
    }:
        raise RuntimeError("Observation inclusion-status census does not reconcile to build manifest")
    if dict(sorted(source_reasons.items())) != {
        str(key): int(value)
        for key, value in sorted(expected_attrition.get("exclusion_reason_counts", {}).items())
    }:
        raise RuntimeError("Observation exclusion-reason census does not reconcile to build manifest")
    readiness_manifest = manifest.get("model_readiness_policy", {}).get("stage_counts", {})
    readiness_policy = manifest.get("model_readiness_policy", {})
    readiness_rows: dict[str, dict[str, int]] = {}
    for scope in sorted(set(readiness_manifest) | {key[0] for key in state.readiness}):
        eligible = int(state.readiness[(scope, "eligible", "<all>")])
        excluded = int(state.readiness[(scope, "excluded", "<all>")])
        candidate = eligible + excluded
        declared = readiness_manifest.get(scope, {})
        expected = {stage: int(declared.get(stage, 0)) for stage in ("candidate", "eligible", "excluded")}
        observed = {"candidate": candidate, "eligible": eligible, "excluded": excluded}
        if expected != observed:
            raise RuntimeError(
                f"Model-readiness attrition does not reconcile for {scope}: {observed} != {expected}"
            )
        observed_reasons = {
            value: int(count)
            for (row_scope, kind, value), count in sorted(state.readiness.items())
            if row_scope == scope and kind == "excluded_reason"
        }
        observed_combinations = {
            value: int(count)
            for (row_scope, kind, value), count in sorted(state.readiness.items())
            if row_scope == scope and kind == "excluded_combination"
        }
        declared_reason_document = readiness_policy.get("reason_counts", {})
        declared_combination_document = readiness_policy.get("reason_combination_counts", {})
        if scope in declared_reason_document:
            declared_reasons = {
                str(key): int(value) for key, value in sorted(declared_reason_document[scope].items())
            }
            if observed_reasons != declared_reasons:
                raise RuntimeError(f"Model-readiness exclusion reasons do not reconcile for {scope}")
        if scope in declared_combination_document:
            declared_combinations = {
                str(key): int(value) for key, value in sorted(declared_combination_document[scope].items())
            }
            if observed_combinations != declared_combinations:
                raise RuntimeError(f"Model-readiness exclusion combinations do not reconcile for {scope}")
        readiness_rows[scope] = observed
    observation_rows = int(state.dataset_rows["observations"])
    expected_observations = int(binding.qc_report.get("counts", {}).get("observations", -1))
    if observation_rows != expected_observations:
        raise RuntimeError("Observation row census does not reconcile to bound QC report")
    joint_counts = {
        str(dataset): int(rows)
        for dataset, rows in state.connection.execute(
            "SELECT dataset, SUM(rows) FROM joint_composition GROUP BY dataset ORDER BY dataset"
        )
    }
    if joint_counts.get("observations", -1) != observation_rows:
        raise RuntimeError("Joint observation composition does not conserve rows")
    for scope in readiness_rows:
        dataset = f"tasks/{scope}"
        if joint_counts.get(dataset, 0) != int(state.dataset_rows[dataset]):
            raise RuntimeError(f"Joint task composition does not conserve rows for {scope}")
    development_rows = int(state.dataset_rows["development_metadata"])
    declared_development = manifest.get("molecule_development_annotations", {})
    if isinstance(declared_development, dict) and "rows" in declared_development:
        if development_rows != int(declared_development["rows"]):
            raise RuntimeError("Development metadata census does not reconcile to build manifest")
    derivation_rows = int(state.dataset_rows["kd_free_energy_sensitivity"])
    if "derived_binding_free_energy_rows" in manifest:
        if derivation_rows != int(manifest["derived_binding_free_energy_rows"]):
            raise RuntimeError("Kd-derived sensitivity census does not reconcile to build manifest")
    return {
        "observation_rows": observation_rows,
        "source_observation_rows": sum(source_status.values()),
        "derived_observation_rows": observation_rows - sum(source_status.values()),
        "inclusion_status_counts": dict(sorted(source_status.items())),
        "exclusion_reason_counts": dict(sorted(source_reasons.items())),
        "model_readiness_stage_counts": readiness_rows,
        "joint_composition_rows": joint_counts,
        "development_metadata_rows": development_rows,
        "kd_derived_free_energy_rows": derivation_rows,
    }


def _svg_bar_chart(
    path: Path, title: str, values: Sequence[tuple[str, int]], population: int, cap: int
) -> None:
    selected = list(values[:cap])
    width, height = 960, max(220, 80 + 28 * len(selected))
    left, right, top = 300, 30, 55
    plot_width = width - left - right
    maximum = max((count for _, count in selected), default=1)
    escaped_title = html.escape(title)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="30" font-family="sans-serif" font-size="18">{escaped_title}</text>',
        f'<text x="20" y="49" font-family="sans-serif" font-size="11">population={population}; displayed={len(selected)}; deterministic cap={cap}</text>',
    ]
    for index, (label, count) in enumerate(selected):
        y = top + index * 28
        bar_width = 0 if maximum <= 0 else round(plot_width * count / maximum, 3)
        lines.append(
            f'<text x="10" y="{y + 16}" font-family="monospace" font-size="11">{html.escape(label[:42])}</text>'
        )
        lines.append(f'<rect x="{left}" y="{y + 3}" width="{bar_width}" height="18" fill="#35618f"/>')
        lines.append(
            f'<text x="{left + bar_width + 5}" y="{y + 17}" font-family="monospace" font-size="11">{count}</text>'
        )
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _artifact(
    path: Path, root: Path, *, rows: int | None = None, schema: Sequence[str] | None = None
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if rows is not None:
        record["rows"] = rows
    if schema is not None:
        record["columns"] = list(schema)
        record["schema_sha256"] = hashlib.sha256(canonical_json(list(schema)).encode("utf-8")).hexdigest()
    return record


def _verify_output(root: Path, manifest: Mapping[str, Any]) -> None:
    declared = {str(record["path"]): record for record in manifest["artifacts"]}
    expected = sorted([*declared, "analysis_manifest.json"])
    actual = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    if actual != expected:
        raise RuntimeError("Statistical-analysis output membership is not exact")
    for relative, record in declared.items():
        path = root / relative
        if path.stat().st_size != int(record["size_bytes"]) or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Statistical-analysis artifact integrity mismatch: {relative}")


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"Duplicate JSON key is forbidden: {key!r}")
        payload[key] = value
    return payload


def _strict_json_document(path: Path, *, canonical_encoding: bool = False) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON document: {path}") from exc
    if canonical_encoding and text != canonical_json(payload) + "\n":
        raise ValueError(f"JSON document is not in the required canonical encoding: {path.name}")
    return payload


def _checked_unsymlinked_path(
    candidate: str | os.PathLike[str],
    *,
    label: str,
    directory: bool,
) -> Path:
    raw = Path(candidate)
    if ".." in raw.parts or any(ord(character) < 32 for character in os.fspath(raw)):
        raise ValueError(f"{label} path is malformed or contains parent traversal")
    lexical = Path(os.path.abspath(os.fspath(raw)))
    for part in (lexical, *lexical.parents):
        if part.is_symlink():
            raise ValueError(f"{label} path chain contains a symlink: {part}")
    if directory:
        if not lexical.is_dir():
            raise FileNotFoundError(lexical)
    elif not lexical.is_file():
        raise FileNotFoundError(lexical)
    return lexical


def _exact_keys(value: object, expected: set[str], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    actual = set(value)
    if actual != expected or any(not isinstance(key, str) for key in value):
        raise ValueError(
            f"{field} schema drift; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _sha256_digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _reject_nonportable_manifest_values(value: Any, field: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field} contains a non-string object key")
            _reject_nonportable_manifest_values(key, f"{field}.<key>")
            _reject_nonportable_manifest_values(item, f"{field}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonportable_manifest_values(item, f"{field}[{index}]")
    elif isinstance(value, str):
        if any(ord(character) < 32 for character in value):
            raise ValueError(f"{field} contains a control character")
        if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
            raise ValueError(f"{field} contains a forbidden absolute path value")


def _safe_artifact_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\\" in value:
        raise ValueError(f"Unsafe {field}: {value!r}")
    path = Path(value)
    if path.is_absolute() or PureWindowsPath(value).is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe {field}: {value!r}")
    normalized = path.as_posix()
    if normalized != value or normalized in {".", ""} or any(part in {"", "."} for part in path.parts):
        raise ValueError(f"Non-canonical {field}: {value!r}")
    return value


def _csv_rows(root: Path, relative: str) -> Iterator[dict[str, str]]:
    path = root / relative
    expected = list(_CSV_SCHEMAS[relative])
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames != expected:
                raise ValueError(f"CSV header drift: {relative}")
            for line_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Malformed CSV row at {relative}:{line_number}")
                yield {str(key): str(value) for key, value in row.items()}
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Invalid CSV artifact: {relative}") from exc


def _csv_count(value: str, field: str) -> int:
    if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise ValueError(f"{field} must be a canonical non-negative integer")
    return int(value)


def _verify_analysis_manifest(manifest: object) -> tuple[dict[str, Mapping[str, Any]], int]:
    document = _exact_keys(
        manifest,
        {
            "analysis_version",
            "zero_training",
            "training_actions",
            "input_binding",
            "reconciliation",
            "methods",
            "scientific_boundaries",
            "artifacts",
            "exact_recursive_membership",
        },
        "analysis manifest",
    )
    if document["analysis_version"] != ANALYSIS_VERSION:
        raise ValueError("Unsupported or drifted statistical-analysis version")
    if document["zero_training"] is not True or document["training_actions"] != []:
        raise ValueError("Statistical analysis no longer attests zero training")
    if document["scientific_boundaries"] != _SCIENTIFIC_BOUNDARIES:
        raise ValueError("Statistical-analysis scientific boundaries have drifted")

    input_binding = _exact_keys(
        document["input_binding"],
        {
            "canonical_build_root_name",
            "canonical_build_manifest_sha256",
            "canonical_qc_report_sha256",
            "canonical_qc_passed",
            "canonical_component_count",
            "canonical_component_inventory_sha256",
            "source_id",
            "snapshot_id",
        },
        "input_binding",
    )
    root_name = _safe_artifact_path(input_binding["canonical_build_root_name"], "canonical_build_root_name")
    if "/" in root_name:
        raise ValueError("canonical_build_root_name must be one portable path component")
    _sha256_digest(
        input_binding["canonical_build_manifest_sha256"],
        "input_binding.canonical_build_manifest_sha256",
    )
    _sha256_digest(
        input_binding["canonical_qc_report_sha256"],
        "input_binding.canonical_qc_report_sha256",
    )
    _sha256_digest(
        input_binding["canonical_component_inventory_sha256"],
        "input_binding.canonical_component_inventory_sha256",
    )
    if input_binding["canonical_qc_passed"] is not True:
        raise ValueError("Statistical analysis is not bound to an accepted canonical QC report")
    if (
        _nonnegative_integer(
            input_binding["canonical_component_count"], "input_binding.canonical_component_count"
        )
        < 1
    ):
        raise ValueError("Canonical component count must be positive")
    for identity in ("source_id", "snapshot_id"):
        if not isinstance(input_binding[identity], (str, type(None))):
            raise ValueError(f"input_binding.{identity} must be a string or null")

    methods = _exact_keys(
        document["methods"],
        {"row_accounting", "high_cardinality_state", "quantiles", "censored_values", "plots", "samples"},
        "methods",
    )
    fixed_methods = {
        "row_accounting": "exact Arrow batch census",
        "high_cardinality_state": (
            "disk-backed SQLite; Python memory bounded by one input batch and capped summaries"
        ),
        "quantiles": (
            "exact linear order statistics within domain/endpoint/unit/relation/assay-family/measure-role strata"
        ),
        "censored_values": (
            "bounds summarized as separate roles; never midpoint-imputed or pooled with exact values"
        ),
        "plots": f"deterministic SVG; top {DEFAULT_PLOT_CAP}; exact population embedded and tabulated",
    }
    if any(methods[key] != value for key, value in fixed_methods.items()):
        raise ValueError("Statistical-analysis methods have drifted")
    sample_match = re.fullmatch(
        r"deterministic smallest-SHA identities; cap ([1-9][0-9]*); exact populations declared",
        str(methods["samples"]),
    )
    if sample_match is None:
        raise ValueError("Statistical sample method is malformed or drifted")
    sample_cap = int(sample_match.group(1))

    artifacts_value = document["artifacts"]
    if not isinstance(artifacts_value, list):
        raise ValueError("artifacts must be a list")
    artifacts: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(artifacts_value):
        if not isinstance(value, Mapping):
            raise ValueError(f"artifacts[{index}] must be an object")
        relative = _safe_artifact_path(value.get("path"), f"artifacts[{index}].path")
        if relative in artifacts:
            raise ValueError(f"Duplicate statistical-analysis artifact: {relative}")
        expected_keys = {"path", "sha256", "size_bytes"}
        if relative in _CSV_SCHEMAS:
            expected_keys |= {"rows", "columns", "schema_sha256"}
        _exact_keys(value, expected_keys, f"artifacts[{index}]")
        _sha256_digest(value["sha256"], f"artifacts[{index}].sha256")
        if _nonnegative_integer(value["size_bytes"], f"artifacts[{index}].size_bytes") < 1:
            raise ValueError(f"Artifact must be non-empty: {relative}")
        if relative in _CSV_SCHEMAS:
            _nonnegative_integer(value["rows"], f"artifacts[{index}].rows")
            expected_columns = list(_CSV_SCHEMAS[relative])
            if value["columns"] != expected_columns:
                raise ValueError(f"Declared CSV schema drift: {relative}")
            expected_schema_digest = hashlib.sha256(
                canonical_json(expected_columns).encode("utf-8")
            ).hexdigest()
            if value["schema_sha256"] != expected_schema_digest:
                raise ValueError(f"Declared CSV schema digest drift: {relative}")
        artifacts[relative] = value
    expected_artifacts = set(_CSV_SCHEMAS) | _NON_CSV_ARTIFACTS
    if set(artifacts) != expected_artifacts:
        raise ValueError(
            "Statistical-analysis artifact schema drift; "
            f"missing={sorted(expected_artifacts - set(artifacts))}, "
            f"extra={sorted(set(artifacts) - expected_artifacts)}"
        )
    if list(artifacts) != sorted(artifacts):
        raise ValueError("Statistical-analysis artifacts must be sorted by path")
    membership = _exact_keys(
        document["exact_recursive_membership"], {"paths", "self_hash_policy"}, "exact_recursive_membership"
    )
    expected_paths = sorted([*artifacts, "analysis_manifest.json"])
    if membership["paths"] != expected_paths or membership["self_hash_policy"] != _SELF_HASH_POLICY:
        raise ValueError("Statistical-analysis recursive-membership declaration has drifted")
    _verify_reconciliation(document["reconciliation"])
    return artifacts, sample_cap


def _verify_reconciliation(value: object) -> None:
    reconciliation = _exact_keys(
        value,
        {
            "observation_rows",
            "source_observation_rows",
            "derived_observation_rows",
            "inclusion_status_counts",
            "exclusion_reason_counts",
            "model_readiness_stage_counts",
            "joint_composition_rows",
            "development_metadata_rows",
            "kd_derived_free_energy_rows",
        },
        "reconciliation",
    )
    observations = _nonnegative_integer(reconciliation["observation_rows"], "reconciliation.observation_rows")
    source = _nonnegative_integer(
        reconciliation["source_observation_rows"], "reconciliation.source_observation_rows"
    )
    derived = _nonnegative_integer(
        reconciliation["derived_observation_rows"], "reconciliation.derived_observation_rows"
    )
    if source + derived != observations:
        raise ValueError("Source and derived observation counts do not conserve total observations")
    status_counts = _count_mapping(reconciliation["inclusion_status_counts"], "inclusion_status_counts")
    _count_mapping(reconciliation["exclusion_reason_counts"], "exclusion_reason_counts")
    if sum(status_counts.values()) != source:
        raise ValueError("Inclusion-status counts do not conserve source observations")
    stages_document = reconciliation["model_readiness_stage_counts"]
    if not isinstance(stages_document, Mapping):
        raise ValueError("model_readiness_stage_counts must be an object")
    stages: dict[str, dict[str, int]] = {}
    for scope, counts_value in stages_document.items():
        if scope not in {"default", "derived_sensitivity"}:
            raise ValueError(f"Unsupported task scope in reconciliation: {scope!r}")
        counts = _exact_keys(counts_value, {"candidate", "eligible", "excluded"}, f"stage_counts.{scope}")
        normalized = {
            key: _nonnegative_integer(counts[key], f"stage_counts.{scope}.{key}")
            for key in ("candidate", "eligible", "excluded")
        }
        if normalized["candidate"] != normalized["eligible"] + normalized["excluded"]:
            raise ValueError(f"Model-readiness counts do not conserve candidates for {scope}")
        stages[str(scope)] = normalized
    if not stages:
        raise ValueError("Statistical analysis must contain at least one model-readiness scope")
    joint = _count_mapping(reconciliation["joint_composition_rows"], "joint_composition_rows")
    expected_joint_keys = {"observations", *(f"tasks/{scope}" for scope in stages)}
    if set(joint) != expected_joint_keys or joint.get("observations") != observations:
        raise ValueError("Joint-composition reconciliation keys or observation count have drifted")
    for scope, counts in stages.items():
        if joint[f"tasks/{scope}"] != counts["eligible"]:
            raise ValueError(f"Joint task rows do not equal eligible rows for {scope}")
    _nonnegative_integer(
        reconciliation["development_metadata_rows"], "reconciliation.development_metadata_rows"
    )
    _nonnegative_integer(
        reconciliation["kd_derived_free_energy_rows"], "reconciliation.kd_derived_free_energy_rows"
    )


def _count_mapping(value: object, field: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    result: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field} contains an invalid key")
        result[key] = _nonnegative_integer(count, f"{field}.{key}")
    return result


def _verify_artifact_membership_and_integrity(root: Path, artifacts: Mapping[str, Mapping[str, Any]]) -> None:
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Statistical-analysis output contains a symlink: {path}")
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            actual_files.add(relative)
        elif path.is_dir():
            actual_directories.add(relative)
        else:
            raise ValueError(f"Statistical-analysis output contains a special filesystem entry: {path}")
    expected_files = {*artifacts, "analysis_manifest.json"}
    expected_directories = {
        parent.as_posix()
        for relative in artifacts
        for parent in [Path(relative).parent]
        if parent.as_posix() != "."
    }
    if actual_files != expected_files or actual_directories != expected_directories:
        raise ValueError(
            "Statistical-analysis recursive membership mismatch; "
            f"unbound_files={sorted(actual_files - expected_files)}, "
            f"missing_files={sorted(expected_files - actual_files)}, "
            f"unbound_directories={sorted(actual_directories - expected_directories)}, "
            f"missing_directories={sorted(expected_directories - actual_directories)}"
        )
    for relative, record in sorted(artifacts.items()):
        path = root / relative
        if path.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"Statistical-analysis artifact size drift: {relative}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Statistical-analysis artifact SHA-256 drift: {relative}")
        if relative in _CSV_SCHEMAS:
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.reader(handle, strict=True)
                    header = next(reader)
                    if header != list(_CSV_SCHEMAS[relative]):
                        raise ValueError(f"CSV header drift: {relative}")
                    row_count = 0
                    for line_number, row in enumerate(reader, start=2):
                        if len(row) != len(header):
                            raise ValueError(f"Malformed CSV row at {relative}:{line_number}")
                        row_count += 1
            except StopIteration as exc:
                raise ValueError(f"CSV artifact has no header: {relative}") from exc
            except (OSError, UnicodeError, csv.Error) as exc:
                raise ValueError(f"Invalid CSV artifact: {relative}") from exc
            if row_count != int(record["rows"]):
                raise ValueError(f"CSV row-count drift: {relative}")
            expected_digest = hashlib.sha256(canonical_json(header).encode("utf-8")).hexdigest()
            if expected_digest != record["schema_sha256"]:
                raise ValueError(f"CSV schema digest drift: {relative}")
        elif relative.endswith(".json"):
            _strict_json_document(path, canonical_encoding=True)


def _verify_semantic_artifacts(root: Path, manifest: Mapping[str, Any], sample_cap: int) -> None:
    inference = _strict_json_document(root / "inference_policy.json", canonical_encoding=True)
    if inference != _INFERENCE_POLICY:
        raise ValueError("Inference policy has drifted beyond the accepted scientific boundary")
    samples = _strict_json_document(root / "deterministic_samples.json", canonical_encoding=True)
    sample_document = _exact_keys(
        samples,
        {
            "selection",
            "cap_per_population",
            "observations_population_rows",
            "observations",
            "tasks_population_rows",
            "tasks",
            "duplicate_group_population_note",
            "duplicate_groups",
        },
        "deterministic_samples",
    )
    if sample_document["selection"] != (
        "smallest SHA-256(identity), deterministic and capped; not a statistical random sample"
    ) or sample_document["duplicate_group_population_note"] != (
        "exact populations are in duplicate_conflict_summary.csv; displayed group rows are capped"
    ):
        raise ValueError("Deterministic sample policy has drifted")
    if sample_document["cap_per_population"] != sample_cap:
        raise ValueError("Deterministic sample cap disagrees with the analysis method")
    for population in ("observations", "tasks"):
        records = sample_document[population]
        if not isinstance(records, list) or len(records) > sample_cap:
            raise ValueError(f"Deterministic {population} sample exceeds its declared cap")
        total = _nonnegative_integer(
            sample_document[f"{population}_population_rows"],
            f"deterministic_samples.{population}_population_rows",
        )
        if total < len(records) or any(not isinstance(record, Mapping) for record in records):
            raise ValueError(f"Deterministic {population} sample is malformed")
    if not isinstance(sample_document["duplicate_groups"], list) or any(
        not isinstance(record, Mapping) for record in sample_document["duplicate_groups"]
    ):
        raise ValueError("Deterministic duplicate-group sample is malformed")

    reconciliation = manifest["reconciliation"]
    if not isinstance(reconciliation, Mapping):
        raise AssertionError("reconciliation was checked before semantic verification")
    expected_joint = _count_mapping(reconciliation["joint_composition_rows"], "joint_composition_rows")
    observed_joint: Counter[str] = Counter()
    for row in _csv_rows(root, "joint_composition.csv"):
        observed_joint[row["dataset"]] += _csv_count(row["rows"], "joint_composition.rows")
    if dict(sorted(observed_joint.items())) != dict(sorted(expected_joint.items())):
        raise ValueError("Joint-composition CSV does not reconcile to the analysis manifest")

    observed_status: Counter[str] = Counter()
    observed_reasons: Counter[str] = Counter()
    for row in _csv_rows(root, "attrition.csv"):
        count = _csv_count(row["rows"], "attrition.rows")
        if row["layer"] == "source" and row["reason"] == "<all>":
            observed_status[row["inclusion_status"]] += count
        elif row["layer"] == "source" and row["reason"].startswith("reason:"):
            observed_reasons[row["reason"].removeprefix("reason:")] += count
    if dict(sorted(observed_status.items())) != dict(
        sorted(_count_mapping(reconciliation["inclusion_status_counts"], "inclusion_status_counts").items())
    ) or dict(sorted(observed_reasons.items())) != dict(
        sorted(_count_mapping(reconciliation["exclusion_reason_counts"], "exclusion_reason_counts").items())
    ):
        raise ValueError("Attrition CSV does not reconcile to the analysis manifest")

    stages = reconciliation["model_readiness_stage_counts"]
    if not isinstance(stages, Mapping):
        raise AssertionError("stage counts were checked before semantic verification")
    observed_stage: Counter[tuple[str, str]] = Counter()
    allowed_stage_kinds = {"eligible", "excluded", "excluded_reason", "excluded_combination"}
    for row in _csv_rows(root, "model_input_exclusions.csv"):
        if row["stage_or_reason_kind"] not in allowed_stage_kinds:
            raise ValueError("Model-readiness CSV contains an unsupported stage/reason kind")
        count = _csv_count(row["rows"], "model_input_exclusions.rows")
        if row["stage_or_reason_kind"] in {"eligible", "excluded"} and row["value"] == "<all>":
            observed_stage[(row["task_scope"], row["stage_or_reason_kind"])] += count
    for scope, counts_value in stages.items():
        if not isinstance(counts_value, Mapping):
            raise AssertionError("stage scope was checked before semantic verification")
        for stage in ("eligible", "excluded"):
            if observed_stage[(str(scope), stage)] != int(counts_value[stage]):
                raise ValueError(f"Model-readiness CSV does not reconcile {scope}/{stage}")

    _verify_repeated_dimension_table(
        root,
        "development_metadata.csv",
        "semantic_role",
        "metadata_only_not_outcome_or_model_label",
        int(reconciliation["development_metadata_rows"]),
        required_dimensions={"semantic_role"},
    )
    _verify_repeated_dimension_table(
        root,
        "kd_free_energy_sensitivity.csv",
        "analysis_scope",
        "Kd_derived_free_energy_sensitivity_only",
        int(reconciliation["kd_derived_free_energy_rows"]),
        required_dimensions={"temperature_source", "formula"},
    )
    _verify_strata_semantics(root)
    _verify_herg_semantics(root)
    _verify_association_semantics(root)
    _verify_svg_semantics(root)


def _verify_repeated_dimension_table(
    root: Path,
    relative: str,
    boundary_column: str,
    boundary_value: str,
    population_rows: int,
    *,
    required_dimensions: set[str],
) -> None:
    totals: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for row in _csv_rows(root, relative):
        if row[boundary_column] != boundary_value:
            raise ValueError(f"Scientific role drift in {relative}")
        identity = (row["dimension"], row["value"])
        if identity in seen:
            raise ValueError(f"Duplicate dimension/value row in {relative}")
        seen.add(identity)
        totals[row["dimension"]] += _csv_count(row["rows"], f"{relative}.rows")
    if population_rows == 0:
        if totals:
            raise ValueError(f"{relative} declares rows for an empty population")
        return
    if not required_dimensions.issubset(totals):
        raise ValueError(f"{relative} lacks required exact population dimensions")
    if any(total != population_rows for total in totals.values()):
        raise ValueError(f"{relative} dimension totals do not conserve its population")


def _verify_strata_semantics(root: Path) -> None:
    seen: set[tuple[str, ...]] = set()
    lower_relations = {">", ">=", "interval"}
    upper_relations = {"<", "<=", "interval"}
    for row in _csv_rows(root, "compatible_stratum_summaries.csv"):
        key = tuple(row[column] for column in _CSV_SCHEMAS["compatible_stratum_summaries.csv"][:9])
        if key in seen:
            raise ValueError("Compatible-stratum table contains a duplicate stratum")
        seen.add(key)
        if _csv_count(row["rows"], "compatible_stratum_summaries.rows") < 1:
            raise ValueError("Compatible-stratum table contains an empty stratum")
        role = row["measure_role"]
        relation = row["relation"]
        if (
            (role == "exact_value" and relation != "=")
            or (role == "lower_censor_bound" and relation not in lower_relations)
            or (role == "upper_censor_bound" and relation not in upper_relations)
            or role not in {"exact_value", "lower_censor_bound", "upper_censor_bound"}
        ):
            raise ValueError("Exact and censor-bound strata have been pooled or mislabeled")
        if row["endpoint"] == "Kd" and row["unit"] == "kcal/mol":
            raise ValueError("Kd values may not be relabeled as binding free energy")
        if row["endpoint"] == "standard_binding_free_energy" and (
            row["unit"] != "kcal/mol" or row["observation_kind"] != "derived" or relation != "="
        ):
            raise ValueError("Binding-free-energy sensitivity stratum has drifted")
        if row["dataset"] == "kd_free_energy_sensitivity" and (
            row["task_scope"] != "derived_sensitivity"
            or row["evidence_domain"] != "binding_affinity"
            or row["endpoint"] != "standard_binding_free_energy"
            or row["unit"] != "kcal/mol"
            or row["observation_kind"] != "derived"
            or row["measure_role"] != "exact_value"
            or relation != "="
        ):
            raise ValueError("Kd-derived sensitivity analysis has crossed its scientific boundary")


def _verify_herg_semantics(root: Path) -> None:
    canonical_policy = (
        "exact included experimental hERG functional IC50 in nM only; "
        "censored, intermediate, absent, incompatible, and excluded rows are not classes"
    )
    emitted_policy = "post-model-input-readiness subset of exact candidate classes"
    canonical_categories = {
        "all_hERG_IC50_observations",
        "not_class__nonfunctional_or_unresolved_assay",
        "not_class__incompatible_or_missing_unit",
        "not_class__not_included_experimental_evidence",
        "not_class__censored_or_interval",
        "not_class__missing_exact_value",
        "classifier_candidate__blocker_le_10uM",
        "classifier_candidate__nonblocker_ge_30uM",
        "not_class__exact_intermediate_10_to_30uM",
    }
    emitted_categories = {"emitted_class__blocker", "emitted_class__nonblocker"}
    counts: dict[tuple[str, str], int] = {}
    for row in _csv_rows(root, "herg_10_30uM_support.csv"):
        try:
            low = float(row["threshold_low_nM"])
            high = float(row["threshold_high_nM"])
        except ValueError as exc:
            raise ValueError("hERG thresholds must be numeric") from exc
        if low != 10_000.0 or high != 30_000.0:
            raise ValueError("hERG 10/30 micromolar thresholds have drifted")
        population = row["population"]
        category = row["category"]
        if population == "canonical_observations":
            if category not in canonical_categories or row["class_policy"] != canonical_policy:
                raise ValueError("Canonical hERG class-exclusion policy has drifted")
        elif population == "emitted_model_ready_binary_task":
            if category not in emitted_categories or row["class_policy"] != emitted_policy:
                raise ValueError("Emitted hERG class policy has drifted")
        else:
            raise ValueError("hERG support table contains an unsupported population")
        key = (population, category)
        if key in counts:
            raise ValueError("hERG support table contains a duplicate category")
        counts[key] = _csv_count(row["rows"], "herg_10_30uM_support.rows")
    canonical_total = counts.get(("canonical_observations", "all_hERG_IC50_observations"), 0)
    classified_total = sum(
        count
        for (population, category), count in counts.items()
        if population == "canonical_observations" and category != "all_hERG_IC50_observations"
    )
    if canonical_total != classified_total:
        raise ValueError("Every canonical hERG IC50 row must belong to exactly one support/exclusion bucket")
    if counts.get(("emitted_model_ready_binary_task", "emitted_class__blocker"), 0) > counts.get(
        ("canonical_observations", "classifier_candidate__blocker_le_10uM"), 0
    ) or counts.get(("emitted_model_ready_binary_task", "emitted_class__nonblocker"), 0) > counts.get(
        ("canonical_observations", "classifier_candidate__nonblocker_ge_30uM"), 0
    ):
        raise ValueError("Emitted hERG classes exceed their exact candidate support")


def _verify_association_semantics(root: Path) -> None:
    assumptions = (
        "descriptive source-snapshot association; independent-cell chi-square approximation; "
        "all expected cells >=5; no causal, temporal-trend, or population-generalization claim"
    )
    multiple_testing = "Benjamini-Hochberg across all tested evidence domains"
    seen_domains: set[str] = set()
    for row in _csv_rows(root, "association_panel.csv"):
        if (
            row["panel"] != "document_decade_x_evidence_stage_within_domain"
            or row["assumptions"] != assumptions
            or row["multiple_testing"] != multiple_testing
            or not row["evidence_domain"]
            or row["evidence_domain"] in seen_domains
        ):
            raise ValueError("Association-panel scientific policy or identity has drifted")
        seen_domains.add(row["evidence_domain"])
        _csv_count(row["rows"], "association_panel.rows")
        _csv_count(row["year_bins"], "association_panel.year_bins")
        _csv_count(row["stage_levels"], "association_panel.stage_levels")
        if row["status"] == "tested":
            if row["reason"]:
                raise ValueError("Tested association panel must not carry an exclusion reason")
            try:
                chi_square = float(row["chi_square"])
                p_value = float(row["p_value"])
                q_value = float(row["q_value"])
                effect = float(row["cramers_v_bias_corrected"])
            except ValueError as exc:
                raise ValueError("Tested association panel contains a non-numeric statistic") from exc
            if (
                not all(math.isfinite(value) for value in (chi_square, p_value, q_value, effect))
                or chi_square < 0
                or effect < 0
                or not 0 <= p_value <= 1
                or not 0 <= q_value <= 1
            ):
                raise ValueError("Tested association statistics are outside their valid ranges")
        elif row["status"] == "not_tested":
            if row["reason"] not in {
                "requires_n>=100_and_at_least_2x2",
                "minimum_expected_cell_below_5",
            } or any(
                row[field] for field in ("chi_square", "p_value", "q_value", "cramers_v_bias_corrected")
            ):
                raise ValueError("Untested association panel lacks its exact reason/boundary")
        else:
            raise ValueError("Association panel has an unsupported status")


def _verify_svg_semantics(root: Path) -> None:
    expectations = {
        "plots/top_task_types.svg": "Top emitted task types",
        "plots/observation_domains.svg": "Canonical observation evidence domains",
    }
    for relative, title in expectations.items():
        text = (root / relative).read_text(encoding="utf-8")
        folded = text.casefold()
        if (
            title not in text
            or f"deterministic cap={DEFAULT_PLOT_CAP}" not in text
            or "<script" in folded
            or "<!doctype" in folded
            or "href=" in folded
        ):
            raise ValueError(f"Deterministic SVG semantics or safety boundary drift: {relative}")


def _verify_source_rebinding(
    manifest: Mapping[str, Any],
    canonical_build_root: str | os.PathLike[str],
    qc_report_path: str | os.PathLike[str],
) -> int:
    canonical = _checked_unsymlinked_path(canonical_build_root, label="canonical build", directory=True)
    qc_path = _checked_unsymlinked_path(qc_report_path, label="canonical QC report", directory=False)
    # This is intentionally the same frozen verifier used by corpus readiness.
    from .platform_corpus_readiness import _verify_canonical_corpus

    binding = _verify_canonical_corpus(canonical, qc_path)
    source = manifest["input_binding"]
    reconciliation = manifest["reconciliation"]
    if not isinstance(source, Mapping) or not isinstance(reconciliation, Mapping):
        raise AssertionError("manifest schema was checked before source rebinding")
    expected_binding: dict[str, Any] = {
        "canonical_build_root_name": binding.root.name,
        "canonical_build_manifest_sha256": binding.build_manifest_sha256,
        "canonical_qc_report_sha256": binding.qc_report_sha256,
        "canonical_qc_passed": True,
        "canonical_component_count": len(binding.build_manifest["component_inventory"]),
        "canonical_component_inventory_sha256": binding.component_inventory_sha256,
        "source_id": binding.build_manifest.get("source_id"),
        "snapshot_id": binding.build_manifest.get("snapshot_id"),
    }
    if any(source.get(key) != value for key, value in expected_binding.items()):
        raise ValueError("Statistical-analysis source binding no longer matches canonical inputs")

    qc = _strict_json_document(qc_path)
    if not isinstance(qc, Mapping):
        raise ValueError("Canonical QC report must be an object")
    qc_counts = qc.get("counts")
    if not isinstance(qc_counts, Mapping) or _nonnegative_integer(
        qc_counts.get("observations"), "canonical QC counts.observations"
    ) != int(reconciliation["observation_rows"]):
        raise ValueError("Statistical observation count no longer matches canonical QC")

    manifest_stages = (
        binding.build_manifest.get("model_readiness_policy", {}).get("stage_counts", {})
        if isinstance(binding.build_manifest.get("model_readiness_policy"), Mapping)
        else {}
    )
    if not isinstance(manifest_stages, Mapping):
        raise ValueError("Canonical model-readiness stage counts are malformed")
    normalized_stages: dict[str, dict[str, int]] = {}
    for scope, counts_value in manifest_stages.items():
        if not isinstance(counts_value, Mapping):
            raise ValueError(f"Canonical model-readiness counts are malformed for {scope}")
        normalized_stages[str(scope)] = {
            stage: _nonnegative_integer(counts_value.get(stage), f"canonical stage_counts.{scope}.{stage}")
            for stage in ("candidate", "eligible", "excluded")
        }
    if normalized_stages != reconciliation["model_readiness_stage_counts"]:
        raise ValueError("Statistical model-readiness counts no longer match the canonical manifest")

    task_rows: Counter[str] = Counter()
    for key, entry in binding.task_datasets.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"Canonical task dataset is malformed: {key}")
        scope = str(entry.get("task_scope", ""))
        task_rows[scope] += _nonnegative_integer(entry.get("row_count"), f"task_datasets.{key}.row_count")
    eligible_rows = {scope: counts["eligible"] for scope, counts in sorted(normalized_stages.items())}
    if dict(sorted(task_rows.items())) != eligible_rows:
        raise ValueError("Canonical task inventory rows no longer match model-readiness eligible counts")
    joint = reconciliation["joint_composition_rows"]
    if not isinstance(joint, Mapping) or any(
        joint.get(f"tasks/{scope}") != count for scope, count in task_rows.items()
    ):
        raise ValueError("Statistical task counts no longer match the canonical task inventory")

    attrition = binding.build_manifest.get("canonical_attrition")
    if not isinstance(attrition, Mapping):
        raise ValueError("Canonical manifest lacks its source attrition binding")
    expected_status = _count_mapping(attrition.get("inclusion_status_counts"), "canonical attrition statuses")
    expected_reasons = _count_mapping(attrition.get("exclusion_reason_counts"), "canonical attrition reasons")
    if (
        expected_status != reconciliation["inclusion_status_counts"]
        or expected_reasons != reconciliation["exclusion_reason_counts"]
    ):
        raise ValueError("Statistical source attrition no longer matches the canonical manifest")
    optional_counts = {
        "unique_activity_rows": "source_observation_rows",
        "derived_binding_free_energy_rows": "derived_observation_rows",
    }
    for canonical_field, analysis_field in optional_counts.items():
        if canonical_field in binding.build_manifest and _nonnegative_integer(
            binding.build_manifest[canonical_field], f"canonical manifest.{canonical_field}"
        ) != int(reconciliation[analysis_field]):
            raise ValueError(f"Statistical {analysis_field} no longer matches {canonical_field}")
    development = binding.build_manifest.get("molecule_development_annotations")
    if (
        isinstance(development, Mapping)
        and "rows" in development
        and _nonnegative_integer(development["rows"], "canonical molecule_development_annotations.rows")
        != int(reconciliation["development_metadata_rows"])
    ):
        raise ValueError("Statistical development-metadata count no longer matches canonical inputs")
    if "derived_binding_free_energy_rows" in binding.build_manifest and _nonnegative_integer(
        binding.build_manifest["derived_binding_free_energy_rows"],
        "canonical derived_binding_free_energy_rows",
    ) != int(reconciliation["kd_derived_free_energy_rows"]):
        raise ValueError("Statistical Kd-derived sensitivity count no longer matches canonical inputs")
    return len(binding.build_manifest["component_inventory"])


def verify_statistical_analysis(
    output_root: str | os.PathLike[str],
    *,
    canonical_build_root: str | os.PathLike[str] | None = None,
    qc_report_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Independently verify a completed zero-training statistical-analysis bundle."""

    if (canonical_build_root is None) != (qc_report_path is None):
        raise ValueError("canonical_build_root and qc_report_path must be supplied together")
    root = _checked_unsymlinked_path(output_root, label="statistical-analysis output", directory=True)
    if root.name.startswith(".") or root.name.endswith(".building"):
        raise ValueError("Refusing a provisional statistical-analysis directory")
    manifest_path = root / "analysis_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(manifest_path)
    manifest = _strict_json_document(manifest_path, canonical_encoding=True)
    _reject_nonportable_manifest_values(manifest)
    artifacts, sample_cap = _verify_analysis_manifest(manifest)
    _verify_artifact_membership_and_integrity(root, artifacts)
    if not isinstance(manifest, Mapping):
        raise AssertionError("analysis manifest was checked before artifact verification")
    _verify_semantic_artifacts(root, manifest, sample_cap)
    source_reverified = False
    canonical_component_count = int(manifest["input_binding"]["canonical_component_count"])
    if canonical_build_root is not None and qc_report_path is not None:
        canonical_component_count = _verify_source_rebinding(manifest, canonical_build_root, qc_report_path)
        source_reverified = True
    return {
        "analysis_version": ANALYSIS_VERSION,
        "status": "verified",
        "analysis_manifest_sha256": sha256_file(manifest_path),
        "artifact_count": len(artifacts),
        "canonical_component_count": canonical_component_count,
        "source_reverified": source_reverified,
        "zero_training": True,
        "training_actions": [],
        "scientific_boundaries_verified": True,
    }


def run_statistical_analysis(
    canonical_build_root: str | os.PathLike[str],
    qc_report_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sample_cap: int = DEFAULT_SAMPLE_CAP,
) -> dict[str, Any]:
    """Run a deterministic exact census and atomically promote its bound reports."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    destination = Path(output_root).resolve()
    building = destination.with_name(f".{destination.name}.building")
    if destination.exists():
        raise RuntimeError(f"Statistical-analysis destination already exists: {destination}")
    if building.exists():
        raise RuntimeError(
            f"Incomplete statistical-analysis build exists; inspect before retrying: {building}"
        )
    binding = _load_and_verify_input(Path(canonical_build_root), Path(qc_report_path))
    building.mkdir(parents=True)
    state_path = building / ".analysis_state.sqlite"
    state = _AnalysisState(state_path, sample_cap)
    try:
        _load_entities(state, binding, batch_size)
        _scan_observations(state, binding, batch_size)
        _scan_tasks(state, binding, batch_size)
        _scan_exclusions(state, binding, batch_size)
        _scan_development(state, binding, batch_size)
        _scan_derivations(state, binding, batch_size)
        _finalize_numeric(state)
        reconciliation = _reconcile(state, binding)

        composition_rows = [
            {"dataset": dataset, "dimension": dimension, "value": value, "rows": count}
            for (dataset, dimension, value), count in sorted(state.composition.items())
        ]
        joint_composition_rows = [
            {
                "dataset": str(row[0]),
                "task_id": str(row[1]),
                "evidence_domain": str(row[2]),
                "endpoint": str(row[3]),
                "unit": str(row[4]),
                "relation": str(row[5]),
                "assay_family": str(row[6]),
                "source_id": str(row[7]),
                "evidence_stage": str(row[8]),
                "development_stage": str(row[9]),
                "observation_kind": str(row[10]),
                "label_kind": str(row[11]),
                "inclusion_status": str(row[12]),
                "rows": int(row[13]),
            }
            for row in state.connection.execute(
                """
                SELECT dataset, task_id, evidence_domain, endpoint, unit, relation,
                       assay_family, source_id, evidence_stage, development_stage,
                       observation_kind, label_kind, inclusion_status, rows
                FROM joint_composition
                ORDER BY dataset, task_id, evidence_domain, endpoint, unit, relation,
                         assay_family, source_id, evidence_stage, development_stage,
                         observation_kind, label_kind, inclusion_status
                """
            )
        ]
        missingness_rows = [
            {
                "dataset": dataset,
                "field": field,
                "rows": state.denominators[(dataset, field)],
                "missing_rows": state.missing[(dataset, field)],
                "present_rows": state.denominators[(dataset, field)] - state.missing[(dataset, field)],
                "missing_fraction": state.missing[(dataset, field)] / state.denominators[(dataset, field)]
                if state.denominators[(dataset, field)]
                else 0.0,
            }
            for dataset, field in sorted(state.denominators)
        ]
        attrition_rows = [
            {
                "layer": layer,
                "evidence_domain": domain,
                "inclusion_status": status,
                "reason": reason,
                "rows": count,
            }
            for (layer, domain, status, reason), count in sorted(state.attrition.items())
        ]
        readiness_rows = [
            {"task_scope": scope, "stage_or_reason_kind": kind, "value": value, "rows": count}
            for (scope, kind, value), count in sorted(state.readiness.items())
        ]
        temporal_rows = [
            {
                "layer": layer,
                "evidence_domain": domain,
                "endpoint": endpoint,
                "document_decade": year_bin,
                "evidence_stage": evidence_stage,
                "development_stage": development_stage,
                "rows": count,
            }
            for (layer, domain, endpoint, year_bin, evidence_stage, development_stage), count in sorted(
                state.temporal_stage.items()
            )
        ]
        development_rows = [
            {
                "dimension": dimension,
                "value": value,
                "rows": count,
                "semantic_role": "metadata_only_not_outcome_or_model_label",
            }
            for (dimension, value), count in sorted(state.development.items())
        ]
        derivation_rows = [
            {
                "dimension": dimension,
                "value": value,
                "rows": count,
                "analysis_scope": "Kd_derived_free_energy_sensitivity_only",
            }
            for (dimension, value), count in sorted(state.derivation.items())
        ]
        strata = _stratum_summaries(state)
        coverage, top_coverage = _coverage_summaries(state)
        duplicate_rows, duplicate_samples = _duplicate_summaries(state, sample_cap)
        herg_rows = _herg_support(state)
        association_rows = _association_panel(state.temporal_stage)
        samples = {
            "selection": "smallest SHA-256(identity), deterministic and capped; not a statistical random sample",
            "cap_per_population": sample_cap,
            "observations_population_rows": state.observation_sample.population_rows,
            "observations": state.observation_sample.records(),
            "tasks_population_rows": state.task_sample.population_rows,
            "tasks": state.task_sample.records(),
            "duplicate_group_population_note": "exact populations are in duplicate_conflict_summary.csv; displayed group rows are capped",
            "duplicate_groups": duplicate_samples,
        }

        table_specs: list[tuple[str, list[dict[str, Any]], list[str]]] = [
            ("composition.csv", composition_rows, ["dataset", "dimension", "value", "rows"]),
            (
                "joint_composition.csv",
                joint_composition_rows,
                [
                    "dataset",
                    "task_id",
                    "evidence_domain",
                    "endpoint",
                    "unit",
                    "relation",
                    "assay_family",
                    "source_id",
                    "evidence_stage",
                    "development_stage",
                    "observation_kind",
                    "label_kind",
                    "inclusion_status",
                    "rows",
                ],
            ),
            (
                "missingness.csv",
                missingness_rows,
                ["dataset", "field", "rows", "missing_rows", "present_rows", "missing_fraction"],
            ),
            (
                "attrition.csv",
                attrition_rows,
                ["layer", "evidence_domain", "inclusion_status", "reason", "rows"],
            ),
            (
                "model_input_exclusions.csv",
                readiness_rows,
                ["task_scope", "stage_or_reason_kind", "value", "rows"],
            ),
            (
                "compatible_stratum_summaries.csv",
                strata,
                [
                    "dataset",
                    "task_scope",
                    "evidence_domain",
                    "endpoint",
                    "unit",
                    "relation",
                    "assay_family",
                    "observation_kind",
                    "measure_role",
                    "rows",
                    "minimum",
                    "p01",
                    "q1",
                    "median",
                    "q3",
                    "p99",
                    "maximum",
                    "mean",
                    "population_sd",
                    "iqr",
                    "quantile_method",
                ],
            ),
            (
                "coverage_long_tail.csv",
                coverage,
                [
                    "dataset",
                    "dimension",
                    "population_rows",
                    "unique_values",
                    "singleton_values",
                    "values_with_2_to_5_rows",
                    "values_with_6_to_10_rows",
                    "values_with_over_10_rows",
                    "top_1_share",
                    "top_10_share",
                    "top_100_share",
                    "hhi",
                    "effective_value_count_inverse_hhi",
                    "gini_count_concentration",
                ],
            ),
            (
                "coverage_top_entities.csv",
                top_coverage,
                ["dataset", "dimension", "rank", "value", "rows", "population_rows", "share", "cap"],
            ),
            (
                "duplicate_conflict_summary.csv",
                duplicate_rows,
                [
                    "group_kind",
                    "summary_kind",
                    "group_size",
                    "group_count",
                    "rows",
                    "repeated_group_count",
                    "rows_in_repeated_groups",
                    "maximum_group_size",
                ],
            ),
            (
                "temporal_stage_composition.csv",
                temporal_rows,
                [
                    "layer",
                    "evidence_domain",
                    "endpoint",
                    "document_decade",
                    "evidence_stage",
                    "development_stage",
                    "rows",
                ],
            ),
            (
                "herg_10_30uM_support.csv",
                herg_rows,
                ["population", "category", "rows", "threshold_low_nM", "threshold_high_nM", "class_policy"],
            ),
            (
                "kd_free_energy_sensitivity.csv",
                derivation_rows,
                ["dimension", "value", "rows", "analysis_scope"],
            ),
            ("development_metadata.csv", development_rows, ["dimension", "value", "rows", "semantic_role"]),
            (
                "association_panel.csv",
                association_rows,
                [
                    "panel",
                    "evidence_domain",
                    "rows",
                    "year_bins",
                    "stage_levels",
                    "status",
                    "reason",
                    "chi_square",
                    "p_value",
                    "q_value",
                    "cramers_v_bias_corrected",
                    "assumptions",
                    "multiple_testing",
                ],
            ),
        ]
        artifacts: list[dict[str, Any]] = []
        for relative, rows, columns in table_specs:
            path = building / relative
            row_count = _csv(path, rows, columns)
            artifacts.append(_artifact(path, building, rows=row_count, schema=columns))
        sample_path = building / "deterministic_samples.json"
        _json(sample_path, samples)
        artifacts.append(_artifact(sample_path, building))
        inference = {
            "analysis_kind": "exact descriptive census plus one explicitly screened association panel",
            "effect_size": "bias-corrected Cramer's V for tested document-decade/evidence-stage tables",
            "uncertainty_policy": "no confidence intervals: rows exhaust the accepted snapshot and are not IID biological samples",
            "multiple_testing": "Benjamini-Hochberg over tested domains; untested panels retain explicit reasons",
            "prohibited_interpretations": [
                "causal effects",
                "clinical efficacy or clinical cardiotoxicity",
                "hERG as QT, torsades, or patient risk",
                "p-values as biological-population evidence",
            ],
        }
        inference_path = building / "inference_policy.json"
        _json(inference_path, inference)
        artifacts.append(_artifact(inference_path, building))

        task_values = sorted(
            (
                (value, count)
                for (dataset, dimension, value), count in state.composition.items()
                if dataset.startswith("tasks/") and dimension == "task_type"
            ),
            key=lambda item: (-item[1], item[0]),
        )
        domain_values = sorted(
            (
                (value, count)
                for (dataset, dimension, value), count in state.composition.items()
                if dataset == "observations" and dimension == "evidence_domain"
            ),
            key=lambda item: (-item[1], item[0]),
        )
        plot_one = building / "plots" / "top_task_types.svg"
        _svg_bar_chart(
            plot_one,
            "Top emitted task types",
            task_values,
            sum(count for _, count in task_values),
            DEFAULT_PLOT_CAP,
        )
        artifacts.append(_artifact(plot_one, building))
        plot_two = building / "plots" / "observation_domains.svg"
        _svg_bar_chart(
            plot_two,
            "Canonical observation evidence domains",
            domain_values,
            sum(count for _, count in domain_values),
            DEFAULT_PLOT_CAP,
        )
        artifacts.append(_artifact(plot_two, building))

        state.close()
        state_path.unlink(missing_ok=True)
        artifacts = sorted(artifacts, key=lambda record: str(record["path"]))
        manifest = {
            "analysis_version": ANALYSIS_VERSION,
            "zero_training": True,
            "training_actions": [],
            "input_binding": {
                "canonical_build_root_name": binding.build_root.name,
                "canonical_build_manifest_sha256": binding.manifest_sha256,
                "canonical_qc_report_sha256": binding.qc_report_sha256,
                "canonical_qc_passed": True,
                "canonical_component_count": len(binding.component_records),
                "canonical_component_inventory_sha256": hashlib.sha256(
                    canonical_json(list(binding.component_records)).encode("utf-8")
                ).hexdigest(),
                "source_id": binding.manifest.get("source_id"),
                "snapshot_id": binding.manifest.get("snapshot_id"),
            },
            "reconciliation": reconciliation,
            "methods": {
                "row_accounting": "exact Arrow batch census",
                "high_cardinality_state": "disk-backed SQLite; Python memory bounded by one input batch and capped summaries",
                "quantiles": "exact linear order statistics within domain/endpoint/unit/relation/assay-family/measure-role strata",
                "censored_values": "bounds summarized as separate roles; never midpoint-imputed or pooled with exact values",
                "plots": f"deterministic SVG; top {DEFAULT_PLOT_CAP}; exact population embedded and tabulated",
                "samples": f"deterministic smallest-SHA identities; cap {sample_cap}; exact populations declared",
            },
            "scientific_boundaries": [
                "No incompatible endpoint, unit, relation, assay-family, or exact/censor-bound stratum is pooled.",
                "hERG exact 10/30 micromolar support is not QT, torsades, cardiotoxicity, or clinical risk.",
                "Censored, intermediate, absent, incompatible, and excluded hERG rows are not classifier classes.",
                "Binding free energy is an opt-in exact-positive-Kd sensitivity derivation, never IC50-derived.",
                "Development annotations are metadata only and are neither outcomes nor labels.",
                "Association statistics describe the accepted source snapshot and support no causal or clinical claim.",
            ],
            "artifacts": artifacts,
            "exact_recursive_membership": {
                "paths": sorted([str(record["path"]) for record in artifacts] + ["analysis_manifest.json"]),
                "self_hash_policy": "analysis_manifest.json is necessarily excluded from its own artifact hash inventory",
            },
        }
        _json(building / "analysis_manifest.json", manifest)
        _verify_output(building, manifest)
        os.replace(building, destination)
        _verify_output(destination, manifest)
        return manifest
    except BaseException:
        try:
            state.close()
        except Exception:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    argument_list = list(sys.argv[1:] if argv is None else argv)
    if argument_list and argument_list[0] == "verify-existing":
        verifier = argparse.ArgumentParser(
            description="Verify an existing zero-training statistical-analysis bundle"
        )
        verifier.add_argument(
            "--output-root",
            type=Path,
            default=Path("research/reports/platform/statistical_analysis"),
        )
        verifier.add_argument("--canonical-build-root", type=Path)
        verifier.add_argument("--qc-report", type=Path)
        verification_arguments = verifier.parse_args(argument_list[1:])
        result = verify_statistical_analysis(
            verification_arguments.output_root,
            canonical_build_root=verification_arguments.canonical_build_root,
            qc_report_path=verification_arguments.qc_report,
        )
        print(canonical_json(result))
        return 0
    parser = argparse.ArgumentParser(description="Run the zero-training canonical statistical census")
    parser.add_argument(
        "--canonical-build-root", type=Path, default=Path("research/data/platform/canonical/full_chembl37")
    )
    parser.add_argument("--qc-report", type=Path, default=Path("research/reports/platform/qc_report.json"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("research/reports/platform/statistical_analysis")
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sample-cap", type=int, default=DEFAULT_SAMPLE_CAP)
    arguments = parser.parse_args(argument_list)
    manifest = run_statistical_analysis(
        arguments.canonical_build_root,
        arguments.qc_report,
        arguments.output_root,
        batch_size=arguments.batch_size,
        sample_cap=arguments.sample_cap,
    )
    print(canonical_json(manifest))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_VERSION",
    "benjamini_hochberg",
    "main",
    "run_statistical_analysis",
    "verify_statistical_analysis",
]
