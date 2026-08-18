"""Expand the frozen hERG benchmark with label-blind pre-HPC challenges.

The release reads only identifiers, structures, split routing, assay/document
context, and policy booleans.  It never projects target relations, target
values, censoring bounds, target classes, native values, or native labels.

The v1.5 challenges are supplemental.  They do not replace v1.4, disclose a
lockbox label, train a model, create a gold standard, or support a superiority
claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import __version__ as RDKIT_VERSION

from .features import nearest_neighbor_tanimoto

SCHEMA_VERSION = "platform-herg-benchmark-freeze/1.5"
MANIFEST_NAME = "benchmark_freeze_v1_5_manifest.json"
MEMBERSHIP_NAME = "frozen_challenge_memberships_v1_5.parquet"
REGISTRY_NAME = "benchmark_challenge_registry_v1_5.parquet"
CONTEXT_EVIDENCE_NAME = "context_metadata_evidence_v1_5.parquet"
SIMILARITY_AUDIT_NAME = "low_similarity_audit_v1_5.parquet"
BLOCKER_EVIDENCE_NAME = "benchmark_blocker_evidence_v1_5.parquet"
REPORT_NAME = "BENCHMARK_FREEZE_V1_5.md"

Q2_TASK = "Q2_FUNCTIONAL_ASSAY_AWARE"
Q1_TASK = "Q1_QUANTITATIVE_PIC50"
CHEMBL_SOURCE = "chembl_herg_specialized_view"
QUANTITATIVE_SOURCE = "quantitative_pic50_release"
AUTO_PATCH = "automated_patch_clamp"
MANUAL_PATCH = "manual_patch_clamp"
PARTITIONS = ("train", "validation", "test")

TASK_COLUMNS = (
    "membership_id",
    "task_id",
    "quality_level",
    "source_artifact",
    "record_id",
    "observation_id",
    "structure_id",
    "target_scope",
    "source_family",
    "measurement_technology",
    "model_split",
    "scaffold_group_id",
    "eligible",
    "use_as_training_label",
    "clinical_context_only",
)
OBSERVATION_COLUMNS = ("observation_id", "source_family", "source_record_id")
STRUCTURE_COLUMNS = (
    "structure_id",
    "standardized_smiles",
    "model_split",
    "scaffold_group_id",
)
CANONICAL_CONTEXT_COLUMNS = (
    "source_record_id",
    "observation_id",
    "assay_id",
    "document_id",
    "document_year",
)
EXACT_TASK_COLUMNS = ("source_record_id",)

FORBIDDEN_READ_COLUMNS = frozenset(
    {
        "target_relation_pic50",
        "target_pic50_point",
        "target_pic50_lower_bound",
        "target_pic50_upper_bound",
        "target_class",
        "native_relation",
        "native_value",
        "native_label",
        "potency_relation_pic50",
        "potency_pic50_point",
        "potency_pic50_lower_bound",
        "potency_pic50_upper_bound",
        "derived_binary_label",
        "label_kind",
        "label_value",
        "label_text",
        "label_relation",
        "label_lower_bound",
        "label_upper_bound",
        "label_unit",
    }
)

MEMBERSHIP_COLUMNS = (
    "challenge_id",
    "challenge_membership_id",
    "challenge_split",
    "membership_id",
    "task_id",
    "quality_level",
    "source_artifact",
    "record_id",
    "observation_id",
    "structure_id",
    "target_scope",
    "source_family",
    "measurement_technology",
    "base_model_split",
    "scaffold_group_id",
    "context_group_kind",
    "context_group_id",
    "document_year",
    "nearest_reference_tanimoto",
    "similarity_threshold",
    "clinical_context_only",
)

CHALLENGE_LIMITATIONS = {
    "Q2_ASSAY_GROUP_HOLDOUT": (
        "Metadata-group sensitivity; the leakage-purged test is one ChEMBL assay group, not a "
        "prospective panel."
    ),
    "Q2_DOCUMENT_GROUP_HOLDOUT": (
        "Metadata-group sensitivity; test and validation each contain one large document group "
        "within one ChEMBL release."
    ),
    "Q2_DOCUMENT_YEAR_TEMPORAL_COMPLETE_CASE": (
        "Complete-case document-year stress with 41% undated rows and a dominant late year; "
        "document year is not assay date."
    ),
    "Q2_CHEMBL37_EXACT_TEMPORAL_COMPLETE_CASE": (
        "Narrow exact-task temporal sensitivity ending in 2014 with a small test surface; not a "
        "current prospective panel."
    ),
    "Q2_AUTOMATED_PATCH_MODALITY_HOLDOUT": (
        "Automated-patch test versus non-automated/unspecified training context; protocol detail remains incomplete."
    ),
    "Q2_LOW_SIMILARITY_060_SCAFFOLD": (
        "Internal scaffold split with Morgan Tanimoto <0.60; representation dependent and not external."
    ),
    "Q2_NO_NEAR_DUPLICATE_080_SCAFFOLD": (
        "Internal scaffold split with Morgan Tanimoto <0.80; absence of near duplicates is not external validation."
    ),
}


class HergBenchmarkFreezeV15Error(RuntimeError):
    """Raised when a v1.5 benchmark-freeze invariant fails."""


@dataclass(frozen=True)
class FreezeV15Config:
    """Frozen feature-only routing settings; no seed or threshold search is allowed."""

    analysis_date: str = "2026-08-09"
    group_split_seed: int = 20260805
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    temporal_rule: str = "whole_year_nearest_cumulative_70_85_reserve_one_year_each_v1"
    fingerprint_backend: str = "rdkit"
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2048
    low_similarity_threshold: float = 0.60
    near_duplicate_threshold: float = 0.80
    minimum_holdout_structures: int = 30

    def validate(self) -> None:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.analysis_date):
            raise HergBenchmarkFreezeV15Error("analysis_date must use YYYY-MM-DD")
        if abs(self.train_fraction + self.validation_fraction + self.test_fraction - 1.0) > 1e-12:
            raise HergBenchmarkFreezeV15Error("split fractions must sum to one")
        if not 0 < self.low_similarity_threshold < self.near_duplicate_threshold < 1:
            raise HergBenchmarkFreezeV15Error("similarity thresholds must satisfy 0 < low < near < 1")
        if self.fingerprint_backend != "rdkit":
            raise HergBenchmarkFreezeV15Error("v1.5 requires the explicit RDKit backend")
        if self.fingerprint_bits < 128 or self.fingerprint_bits & (self.fingerprint_bits - 1):
            raise HergBenchmarkFreezeV15Error("fingerprint_bits must be a power of two >= 128")
        if not 1 <= self.fingerprint_radius <= 8:
            raise HergBenchmarkFreezeV15Error("fingerprint_radius must be between one and eight")
        if self.minimum_holdout_structures < 1:
            raise HergBenchmarkFreezeV15Error("minimum_holdout_structures must be positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_hash(path: Path) -> str:
    return hashlib.sha256(pq.read_schema(path).serialize().to_pybytes()).hexdigest()


def _manifest_hash(payload: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _guard_projection(columns: tuple[str, ...], source: str) -> tuple[str, ...]:
    normalized = tuple(str(column) for column in columns)
    forbidden = set(normalized) & FORBIDDEN_READ_COLUMNS
    if forbidden:
        raise HergBenchmarkFreezeV15Error(
            f"forbidden outcome columns requested from {source}: {sorted(forbidden)}"
        )
    return normalized


def _read_parquet(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    requested = _guard_projection(columns, path.as_posix())
    if not path.is_file():
        raise HergBenchmarkFreezeV15Error(f"missing input: {path}")
    available = set(pq.read_schema(path).names)
    missing = set(requested) - available
    if missing:
        raise HergBenchmarkFreezeV15Error(f"missing columns in {path}: {sorted(missing)}")
    return pd.read_parquet(path, columns=list(requested))


def _canonical_source_record_ids(source_records: pd.Series) -> pd.Series:
    extracted = source_records.astype("string").str.extract(r"^ACTIVITY:(\d+)$", expand=False)
    if extracted.isna().any():
        raise HergBenchmarkFreezeV15Error("master ChEMBL source record does not match ACTIVITY:<digits>")
    return "ChEMBL:activity:" + extracted


def _load_q2_context(
    *, master_root: Path, canonical_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    tasks = _read_parquet(master_root / "task_membership.parquet", TASK_COLUMNS)
    eligible = tasks["eligible"].fillna(False) & tasks["use_as_training_label"].fillna(False)
    q2 = tasks.loc[tasks["task_id"].eq(Q2_TASK) & eligible].copy()
    if q2.empty or q2["membership_id"].duplicated().any() or q2["observation_id"].duplicated().any():
        raise HergBenchmarkFreezeV15Error("eligible Q2 membership is empty or not one row per observation")

    observations = _read_parquet(master_root / "observation_master.parquet", OBSERVATION_COLUMNS)
    observations = observations.loc[observations["observation_id"].isin(q2["observation_id"])].copy()
    if len(observations) != len(q2) or observations["observation_id"].duplicated().any():
        raise HergBenchmarkFreezeV15Error("Q2-to-master observation join is not one-to-one")
    if not observations["source_family"].eq(CHEMBL_SOURCE).all():
        raise HergBenchmarkFreezeV15Error("eligible Q2 is not entirely the declared ChEMBL source")
    observations["canonical_source_record_id"] = _canonical_source_record_ids(
        observations["source_record_id"]
    )
    q2 = q2.merge(
        observations[["observation_id", "canonical_source_record_id"]],
        on="observation_id",
        how="inner",
        validate="one_to_one",
    )

    observation_parts = sorted((canonical_root / "observations").glob("*.parquet"))
    if not observation_parts:
        raise HergBenchmarkFreezeV15Error("canonical observation parts are missing")
    needed = set(q2["canonical_source_record_id"].astype(str))
    context_parts: list[pd.DataFrame] = []
    for path in observation_parts:
        frame = _read_parquet(path, CANONICAL_CONTEXT_COLUMNS)
        selected = frame.loc[frame["source_record_id"].isin(needed)].copy()
        if not selected.empty:
            context_parts.append(selected)
    if not context_parts:
        raise HergBenchmarkFreezeV15Error("no Q2 context rows were found in the canonical corpus")
    context = pd.concat(context_parts, ignore_index=True)
    if context["source_record_id"].duplicated().any() or set(context["source_record_id"]) != needed:
        raise HergBenchmarkFreezeV15Error(
            "canonical Q2 context coverage is not exactly one row per source record"
        )
    context["document_year"] = pd.to_numeric(context["document_year"], errors="coerce").astype("Int64")
    years = context["document_year"].dropna().astype(int)
    if not years.empty and (int(years.min()) < 1800 or int(years.max()) > 2100):
        raise HergBenchmarkFreezeV15Error("canonical document year is outside the accepted range")
    q2 = q2.merge(
        context.rename(columns={"observation_id": "canonical_observation_id"}),
        left_on="canonical_source_record_id",
        right_on="source_record_id",
        how="inner",
        validate="one_to_one",
    )
    q2 = q2.drop(columns=["source_record_id"])
    return q2, tasks, observation_parts


def _stable_digest(kind: str, value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    return f"{kind}:{hashlib.sha256(text.encode()).hexdigest()[:24]}"


def _balanced_group_split(
    frame: pd.DataFrame,
    *,
    group_column: str,
    group_kind: str,
    config: FreezeV15Config,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    known = frame.loc[frame[group_column].notna() & frame[group_column].astype(str).str.strip().ne("")].copy()
    unknown_rows = int(len(frame) - len(known))
    counts = known.groupby(group_column, sort=False).size()
    if len(counts) < 3:
        raise HergBenchmarkFreezeV15Error(f"{group_kind} holdout requires at least three groups")
    targets = {
        "train": config.train_fraction * len(known),
        "validation": config.validation_fraction * len(known),
        "test": config.test_fraction * len(known),
    }
    loads = dict.fromkeys(PARTITIONS, 0)
    assignments: dict[object, str] = {}
    ordered = sorted(
        counts.items(),
        key=lambda item: (
            -int(item[1]),
            hashlib.sha256(f"{config.group_split_seed}|{group_kind}|{item[0]}".encode()).hexdigest(),
        ),
    )
    for group_value, raw_count in ordered:
        count = int(raw_count)

        def score(partition: str, group_count: int = count) -> tuple[float, str]:
            squared_error = 0.0
            for candidate in PARTITIONS:
                proposed = loads[candidate] + (group_count if candidate == partition else 0)
                squared_error += ((proposed - targets[candidate]) / max(targets[candidate], 1.0)) ** 2
            return squared_error, partition

        chosen = min(PARTITIONS, key=score)
        assignments[group_value] = chosen
        loads[chosen] += count
    known["challenge_split"] = known[group_column].map(assignments)
    known["context_group_kind"] = group_kind
    known["context_group_id"] = known[group_column].map(lambda value: _stable_digest(group_kind, value))
    metadata = {
        "algorithm": "descending_group_size_minimum_normalized_squared_deficit_v1",
        "seed": config.group_split_seed,
        "seed_search_performed": False,
        "population_rows": int(len(frame)),
        "known_context_rows": int(len(known)),
        "unknown_context_rows": unknown_rows,
        "groups": int(len(counts)),
        "pre_purge_split_rows": {partition: int(loads[partition]) for partition in PARTITIONS},
    }
    return known, metadata


def _temporal_boundaries(year_counts: pd.Series) -> tuple[int, int]:
    normalized = year_counts.sort_index()
    years = [int(year) for year in normalized.index]
    if len(years) < 3:
        raise HergBenchmarkFreezeV15Error("strict temporal split requires at least three distinct years")
    cumulative = normalized.cumsum()
    total = int(normalized.sum())
    train_index = min(
        range(0, len(years) - 2),
        key=lambda index: (abs(float(cumulative.iloc[index]) / total - 0.70), years[index]),
    )
    validation_index = min(
        range(train_index + 1, len(years) - 1),
        key=lambda index: (abs(float(cumulative.iloc[index]) / total - 0.85), years[index]),
    )
    return years[train_index], years[validation_index]


def _temporal_split(frame: pd.DataFrame, *, config: FreezeV15Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    known = frame.loc[frame["document_year"].notna()].copy()
    year_counts = known["document_year"].astype(int).value_counts().sort_index()
    train_end, validation_end = _temporal_boundaries(year_counts)
    years = known["document_year"].astype(int)
    known["challenge_split"] = np.select(
        [years.le(train_end), years.le(validation_end)],
        ["train", "validation"],
        default="test",
    )
    known["context_group_kind"] = "document_year"
    known["context_group_id"] = years.map(lambda year: f"year:{year}")
    metadata = {
        "algorithm": config.temporal_rule,
        "population_rows": int(len(frame)),
        "known_context_rows": int(len(known)),
        "unknown_context_rows": int(len(frame) - len(known)),
        "distinct_years": int(len(year_counts)),
        "train_max_year": train_end,
        "validation_max_year": validation_end,
        "pre_purge_split_rows": {
            partition: int(known["challenge_split"].eq(partition).sum()) for partition in PARTITIONS
        },
    }
    return known, metadata


def _purge_cross_partition_identities(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not set(frame["challenge_split"]).issubset(PARTITIONS):
        raise HergBenchmarkFreezeV15Error("purge received an unsupported split route")
    bad_structures = set(
        frame.groupby("structure_id", observed=True)["challenge_split"]
        .nunique()
        .loc[lambda values: values > 1]
        .index
    )
    bad_scaffolds = set(
        frame.groupby("scaffold_group_id", observed=True)["challenge_split"]
        .nunique()
        .loc[lambda values: values > 1]
        .index
    )
    structure_mask = frame["structure_id"].isin(bad_structures)
    scaffold_mask = frame["scaffold_group_id"].isin(bad_scaffolds)
    purge_mask = structure_mask | scaffold_mask
    kept = frame.loc[~purge_mask].copy()
    evidence = {
        "cross_partition_structures": int(len(bad_structures)),
        "cross_partition_scaffolds": int(len(bad_scaffolds)),
        "rows_touching_cross_partition_structure": int(structure_mask.sum()),
        "rows_touching_cross_partition_scaffold": int(scaffold_mask.sum()),
        "purged_rows_union": int(purge_mask.sum()),
        "post_purge_rows": int(len(kept)),
        "post_purge_split_rows": {
            partition: int(kept["challenge_split"].eq(partition).sum()) for partition in PARTITIONS
        },
    }
    _assert_partition_exclusivity(kept)
    return kept, evidence


def _assert_partition_exclusivity(frame: pd.DataFrame) -> None:
    for identity in ("structure_id", "scaffold_group_id"):
        counts = (
            frame.dropna(subset=[identity, "challenge_split"])
            .groupby(["challenge_id", identity], observed=True)["challenge_split"]
            .nunique()
            if "challenge_id" in frame.columns
            else frame.dropna(subset=[identity, "challenge_split"])
            .groupby(identity, observed=True)["challenge_split"]
            .nunique()
        )
        if not counts.empty and int(counts.max()) > 1:
            raise HergBenchmarkFreezeV15Error(f"{identity} crosses challenge partitions")


def _membership_frame(
    routed: pd.DataFrame,
    *,
    challenge_id: str,
    nearest: pd.Series | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    frame = routed.copy()
    frame["challenge_id"] = challenge_id
    frame["challenge_membership_id"] = frame["membership_id"].map(
        lambda membership_id: (
            "HCH15-" + hashlib.sha256(f"{challenge_id}|{membership_id}".encode()).hexdigest()[:24].upper()
        )
    )
    frame["base_model_split"] = frame["model_split"]
    frame["nearest_reference_tanimoto"] = (
        pd.Series(pd.NA, index=frame.index, dtype="Float64")
        if nearest is None
        else pd.to_numeric(nearest.reindex(frame.index), errors="coerce").astype("Float64")
    )
    frame["similarity_threshold"] = (
        pd.Series(pd.NA, index=frame.index, dtype="Float64")
        if threshold is None
        else pd.Series(float(threshold), index=frame.index, dtype="Float64")
    )
    frame["document_year"] = pd.to_numeric(frame.get("document_year"), errors="coerce").astype("Int64")
    missing = set(MEMBERSHIP_COLUMNS) - set(frame.columns)
    if missing:
        raise HergBenchmarkFreezeV15Error(f"membership assembly is missing columns: {sorted(missing)}")
    result = frame[list(MEMBERSHIP_COLUMNS)].copy()
    forbidden = set(result.columns) & FORBIDDEN_READ_COLUMNS
    if forbidden:
        raise HergBenchmarkFreezeV15Error(f"outcome columns entered membership output: {sorted(forbidden)}")
    return result


def _similarity_audit(
    q2: pd.DataFrame,
    *,
    structure_path: Path,
    config: FreezeV15Config,
) -> tuple[pd.DataFrame, dict[float, pd.DataFrame], dict[str, Any]]:
    structures = _read_parquet(structure_path, STRUCTURE_COLUMNS)
    q2_structures = q2[["structure_id", "model_split", "scaffold_group_id"]].drop_duplicates()
    consistency = q2_structures.groupby("structure_id", observed=True).agg(
        split_count=("model_split", "nunique"), scaffold_count=("scaffold_group_id", "nunique")
    )
    if int(consistency[["split_count", "scaffold_count"]].to_numpy().max()) > 1:
        raise HergBenchmarkFreezeV15Error("Q2 structure routing is inconsistent")
    q2_structures = q2_structures.drop_duplicates("structure_id").merge(
        structures[["structure_id", "standardized_smiles"]],
        on="structure_id",
        how="left",
        validate="one_to_one",
    )
    if q2_structures["standardized_smiles"].isna().any():
        raise HergBenchmarkFreezeV15Error("Q2 similarity population has missing structures")
    q2_structures = q2_structures.sort_values("structure_id", kind="stable").reset_index(drop=True)
    train = q2_structures.loc[q2_structures["model_split"].eq("train")].copy()
    validation = q2_structures.loc[q2_structures["model_split"].eq("validation")].copy()
    test = q2_structures.loc[q2_structures["model_split"].eq("test")].copy()
    if train.empty or validation.empty or test.empty:
        raise HergBenchmarkFreezeV15Error("Q2 similarity challenge requires all three partitions")

    validation_maxima, validation_neighbors, backend = nearest_neighbor_tanimoto(
        validation["standardized_smiles"],
        train["standardized_smiles"],
        backend=config.fingerprint_backend,
        n_bits=config.fingerprint_bits,
        radius=config.fingerprint_radius,
    )
    if backend != "rdkit_morgan":
        raise HergBenchmarkFreezeV15Error("similarity audit did not resolve to RDKit Morgan")
    validation_neighbor_ids = [
        None if index < 0 else str(train.iloc[int(index)]["structure_id"]) for index in validation_neighbors
    ]

    audit = pd.concat(
        [
            validation[["structure_id", "model_split", "scaffold_group_id"]],
            test[["structure_id", "model_split", "scaffold_group_id"]],
        ],
        ignore_index=True,
    ).rename(columns={"model_split": "base_model_split"})
    challenge_routes: dict[float, pd.DataFrame] = {}
    for suffix, threshold in (
        ("060", config.low_similarity_threshold),
        ("080", config.near_duplicate_threshold),
    ):
        qualified_validation = validation.loc[validation_maxima < threshold].copy()
        reference = pd.concat([train, qualified_validation], ignore_index=True).sort_values(
            "structure_id", kind="stable"
        )
        test_maxima, test_neighbors, test_backend = nearest_neighbor_tanimoto(
            test["standardized_smiles"],
            reference["standardized_smiles"],
            backend=config.fingerprint_backend,
            n_bits=config.fingerprint_bits,
            radius=config.fingerprint_radius,
        )
        if test_backend != backend:
            raise HergBenchmarkFreezeV15Error("similarity backend changed between partitions")
        test_neighbor_ids = [
            None if index < 0 else str(reference.iloc[int(index)]["structure_id"]) for index in test_neighbors
        ]
        nearest_values = np.concatenate([validation_maxima, test_maxima])
        nearest_ids = validation_neighbor_ids + test_neighbor_ids
        qualifies = nearest_values < threshold
        audit[f"nearest_reference_structure_id_{suffix}"] = nearest_ids
        audit[f"nearest_reference_tanimoto_{suffix}"] = nearest_values
        audit[f"qualifies_{suffix}"] = qualifies
        audit[f"reference_population_{suffix}"] = [
            *("train" for _ in range(len(validation))),
            *(f"train_plus_qualified_validation_{suffix}" for _ in range(len(test))),
        ]
        qualification = dict(zip(audit["structure_id"], qualifies, strict=True))
        similarity = dict(zip(audit["structure_id"], nearest_values, strict=True))
        routed = q2.loc[
            q2["model_split"].eq("train")
            | (
                q2["model_split"].isin(("validation", "test"))
                & q2["structure_id"].map(qualification).fillna(False)
            )
        ].copy()
        routed["challenge_split"] = routed["model_split"]
        routed["context_group_kind"] = "master_scaffold_similarity"
        routed["context_group_id"] = routed["scaffold_group_id"].map(
            lambda value: _stable_digest("scaffold", value)
        )
        routed["_nearest"] = routed["structure_id"].map(similarity)
        challenge_routes[threshold] = routed

    audit["fingerprint_backend"] = backend
    audit["fingerprint_radius"] = config.fingerprint_radius
    audit["fingerprint_bits"] = config.fingerprint_bits
    audit["rdkit_version"] = RDKIT_VERSION
    audit = audit.sort_values(["base_model_split", "structure_id"], kind="stable").reset_index(drop=True)
    metadata = {
        "backend": backend,
        "radius": config.fingerprint_radius,
        "bits": config.fingerprint_bits,
        "rdkit_version": RDKIT_VERSION,
        "population_structures": int(len(q2_structures)),
        "train_structures": int(len(train)),
        "validation_structures": int(len(validation)),
        "test_structures": int(len(test)),
        "thresholds": {
            "060": config.low_similarity_threshold,
            "080": config.near_duplicate_threshold,
        },
        "threshold_selection": (
            "fixed in the v1.5 feature-only contract before artifact materialization; "
            "0.80 also matches the earlier deep-leakage near-duplicate contract"
        ),
        "label_columns_read": [],
    }
    return audit, challenge_routes, metadata


def _assert_context_group_exclusivity(frame: pd.DataFrame) -> None:
    grouped = frame.dropna(subset=["context_group_id"]).groupby(
        ["challenge_id", "context_group_id"], observed=True
    )["challenge_split"]
    for (challenge_id, _), values in grouped:
        splits = set(values.astype(str))
        if challenge_id == "Q2_AUTOMATED_PATCH_MODALITY_HOLDOUT":
            if "test" in splits and len(splits) > 1:
                raise HergBenchmarkFreezeV15Error("held-out modality crosses into development data")
        elif len(splits) > 1:
            raise HergBenchmarkFreezeV15Error("context group crosses challenge partitions")


def _context_evidence(q2: pd.DataFrame, exact_q2: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for surface_id, frame in (("eligible_q2", q2), ("chembl37_exact_q2_overlap", exact_q2)):
        for field in ("canonical_source_record_id", "assay_id", "document_id", "document_year"):
            nonmissing = frame[field].notna() & frame[field].astype(str).str.strip().ne("")
            values = frame.loc[nonmissing, field]
            numeric = pd.to_numeric(values, errors="coerce").dropna()
            rows.append(
                {
                    "surface_id": surface_id,
                    "field": field,
                    "total_rows": int(len(frame)),
                    "nonmissing_rows": int(nonmissing.sum()),
                    "missing_rows": int((~nonmissing).sum()),
                    "coverage_fraction": float(nonmissing.mean()) if len(frame) else 0.0,
                    "unique_nonmissing": int(values.nunique()),
                    "minimum_numeric": None if numeric.empty else float(numeric.min()),
                    "maximum_numeric": None if numeric.empty else float(numeric.max()),
                    "evidence_scope": "exhaustive_label_blind_projection",
                }
            )
    return pd.DataFrame(rows).sort_values(["surface_id", "field"], kind="stable").reset_index(drop=True)


def _candidate_with_purge(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        return frame.copy(), {
            "cross_partition_structures": 0,
            "cross_partition_scaffolds": 0,
            "rows_touching_cross_partition_structure": 0,
            "rows_touching_cross_partition_scaffold": 0,
            "purged_rows_union": 0,
            "post_purge_rows": 0,
            "post_purge_split_rows": dict.fromkeys(PARTITIONS, 0),
        }
    return _purge_cross_partition_identities(frame)


def _blocker_evidence(tasks: pd.DataFrame, config: FreezeV15Config) -> pd.DataFrame:
    eligible = tasks["eligible"].fillna(False) & tasks["use_as_training_label"].fillna(False)
    q1 = tasks.loc[tasks["task_id"].eq(Q1_TASK) & eligible].copy()
    q1["challenge_split"] = pd.NA
    q1_release = q1["source_family"].eq(QUANTITATIVE_SOURCE)
    q1.loc[q1_release & q1["model_split"].isin(("train", "validation")), "challenge_split"] = q1.loc[
        q1_release & q1["model_split"].isin(("train", "validation")), "model_split"
    ]
    q1.loc[q1["source_family"].eq(CHEMBL_SOURCE), "challenge_split"] = "test"
    q1_candidate, _ = _candidate_with_purge(q1.loc[q1["challenge_split"].notna()].copy())
    q1_test_structures = int(
        q1_candidate.loc[q1_candidate["challenge_split"].eq("test"), "structure_id"].nunique()
    )

    q2 = tasks.loc[tasks["task_id"].eq(Q2_TASK) & eligible].copy()
    modality = q2.loc[
        (q2["measurement_technology"].eq(AUTO_PATCH) & q2["model_split"].isin(("train", "validation")))
        | q2["measurement_technology"].eq(MANUAL_PATCH)
    ].copy()
    modality["challenge_split"] = modality["model_split"]
    modality.loc[modality["measurement_technology"].eq(MANUAL_PATCH), "challenge_split"] = "test"
    modality_candidate, _ = _candidate_with_purge(modality)
    manual_test_structures = int(
        modality_candidate.loc[modality_candidate["challenge_split"].eq("test"), "structure_id"].nunique()
    )
    q2_source_count = int(q2["source_family"].nunique())

    rows = [
        {
            "challenge_id": "Q1_CROSS_SOURCE_HOLDOUT",
            "status": "blocked_underpowered_after_leakage_purge",
            "criterion": "leakage_purged_test_structures",
            "observed_value": str(q1_test_structures),
            "required_value": f">={config.minimum_holdout_structures}",
            "evidence_scope": "exhaustive_label_blind_candidate_routing",
            "blocker": "The independent ChEMBL test surface is too small after structure/scaffold isolation.",
            "next_action": "Admit another independently governed quantitative source or expand adjudicated ChEMBL coverage.",
        },
        {
            "challenge_id": "Q2_MANUAL_VS_AUTOMATED_MODALITY_HOLDOUT",
            "status": "blocked_underpowered_after_leakage_purge",
            "criterion": "manual_patch_test_structures",
            "observed_value": str(manual_test_structures),
            "required_value": f">={config.minimum_holdout_structures}",
            "evidence_scope": "exhaustive_label_blind_candidate_routing",
            "blocker": "Manual-patch structures largely overlap automated-patch structures.",
            "next_action": "Acquire a sealed manual-patch panel with chemically independent structures.",
        },
        {
            "challenge_id": "Q2_SOURCE_FAMILY_HOLDOUT",
            "status": "blocked_single_source",
            "criterion": "eligible_q2_source_families",
            "observed_value": str(q2_source_count),
            "required_value": ">=2",
            "evidence_scope": "exhaustive_master_routing_metadata",
            "blocker": "All eligible Q2 memberships come from one source family.",
            "next_action": "Admit an independently governed functional hERG source before source-holdout claims.",
        },
        {
            "challenge_id": "LOW_SIMILARITY_EXTERNAL_HOLDOUT",
            "status": "blocked_no_external_panel",
            "criterion": "frozen_external_panel_rows",
            "observed_value": "0",
            "required_value": ">0 with independent provenance",
            "evidence_scope": "local_artifact_inventory",
            "blocker": "The v1.5 low-similarity challenges are internal; no external panel is frozen.",
            "next_action": "Freeze an external panel before computing panel-to-training similarities.",
        },
        {
            "challenge_id": "STRICT_ASSAY_DATE_TEMPORAL_HOLDOUT",
            "status": "blocked_no_assay_date",
            "criterion": "actual_assay_or_test_date_field",
            "observed_value": "absent",
            "required_value": "complete governed assay/test date",
            "evidence_scope": "master_and_canonical_schema_audit",
            "blocker": "Document year is available only as a mixed source/publication clock.",
            "next_action": "Collect actual assay or campaign dates for governed temporal validation.",
        },
        {
            "challenge_id": "PROSPECTIVE_MANUAL_PATCH_GOLD",
            "status": "blocked_no_prospective_adjudication",
            "criterion": "sealed_adjudicated_panel",
            "observed_value": "absent",
            "required_value": "independent prospective panel",
            "evidence_scope": "local_artifact_inventory",
            "blocker": "No independently adjudicated prospective manual-patch panel exists locally.",
            "next_action": "Create and seal the experimental protocol and panel before measurements are opened.",
        },
    ]
    return pd.DataFrame(rows).sort_values(["challenge_id", "criterion"], kind="stable").reset_index(drop=True)


def _registry(
    memberships: pd.DataFrame,
    blockers: pd.DataFrame,
    challenge_metadata: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for challenge_id, group in memberships.groupby("challenge_id", sort=True):
        metadata = challenge_metadata[challenge_id]
        counts = group["challenge_split"].value_counts().to_dict()
        structures = group.groupby("challenge_split")["structure_id"].nunique().to_dict()
        context_groups = group.groupby("challenge_split")["context_group_id"].nunique().to_dict()
        rows.append(
            {
                "challenge_id": challenge_id,
                "status": "materialized_label_blind_membership",
                "rows": int(len(group)),
                "structures": int(group["structure_id"].nunique()),
                "scaffolds": int(group["scaffold_group_id"].nunique()),
                "train_rows": int(counts.get("train", 0)),
                "validation_rows": int(counts.get("validation", 0)),
                "test_rows": int(counts.get("test", 0)),
                "train_structures": int(structures.get("train", 0)),
                "validation_structures": int(structures.get("validation", 0)),
                "test_structures": int(structures.get("test", 0)),
                "train_context_groups": int(context_groups.get("train", 0)),
                "validation_context_groups": int(context_groups.get("validation", 0)),
                "test_context_groups": int(context_groups.get("test", 0)),
                "purged_rows": int(metadata.get("purge", {}).get("purged_rows_union", 0)),
                "labels_embedded": False,
                "test_labels_opened": False,
                "structure_exclusive": True,
                "scaffold_exclusive": True,
                "ready_for_superiority_claim": False,
                "limitation_or_blocker": CHALLENGE_LIMITATIONS[challenge_id],
            }
        )
    for challenge_id, group in blockers.groupby("challenge_id", sort=True):
        rows.append(
            {
                "challenge_id": challenge_id,
                "status": str(group.iloc[0]["status"]),
                "rows": 0,
                "structures": 0,
                "scaffolds": 0,
                "train_rows": 0,
                "validation_rows": 0,
                "test_rows": 0,
                "train_structures": 0,
                "validation_structures": 0,
                "test_structures": 0,
                "train_context_groups": 0,
                "validation_context_groups": 0,
                "test_context_groups": 0,
                "purged_rows": 0,
                "labels_embedded": False,
                "test_labels_opened": False,
                "structure_exclusive": False,
                "scaffold_exclusive": False,
                "ready_for_superiority_claim": False,
                "limitation_or_blocker": " ".join(group["blocker"].astype(str)),
            }
        )
    return pd.DataFrame(rows).sort_values("challenge_id", kind="stable").reset_index(drop=True)


def _assemble_release(
    *,
    master_root: Path,
    canonical_root: Path,
    exact_task_path: Path,
    config: FreezeV15Config,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], list[Path]]:
    config.validate()
    q2, tasks, observation_parts = _load_q2_context(master_root=master_root, canonical_root=canonical_root)
    membership_parts: list[pd.DataFrame] = []
    challenge_metadata: dict[str, dict[str, Any]] = {}

    for challenge_id, column, kind in (
        ("Q2_ASSAY_GROUP_HOLDOUT", "assay_id", "assay"),
        ("Q2_DOCUMENT_GROUP_HOLDOUT", "document_id", "document"),
    ):
        routed, metadata = _balanced_group_split(q2, group_column=column, group_kind=kind, config=config)
        routed, purge = _purge_cross_partition_identities(routed)
        metadata["purge"] = purge
        challenge_metadata[challenge_id] = metadata
        membership_parts.append(_membership_frame(routed, challenge_id=challenge_id))

    routed, metadata = _temporal_split(q2, config=config)
    routed, purge = _purge_cross_partition_identities(routed)
    metadata["purge"] = purge
    challenge_id = "Q2_DOCUMENT_YEAR_TEMPORAL_COMPLETE_CASE"
    challenge_metadata[challenge_id] = metadata
    membership_parts.append(_membership_frame(routed, challenge_id=challenge_id))

    exact = _read_parquet(exact_task_path, EXACT_TASK_COLUMNS)
    exact_keys = set(exact["source_record_id"].dropna().astype(str))
    exact_q2 = q2.loc[q2["canonical_source_record_id"].isin(exact_keys)].copy()
    if exact_q2.empty:
        raise HergBenchmarkFreezeV15Error("exact ChEMBL task has no overlap with eligible Q2")
    routed, metadata = _temporal_split(exact_q2, config=config)
    routed, purge = _purge_cross_partition_identities(routed)
    metadata["purge"] = purge
    metadata["canonical_exact_task_rows"] = int(len(exact))
    metadata["eligible_q2_overlap_rows"] = int(len(exact_q2))
    challenge_id = "Q2_CHEMBL37_EXACT_TEMPORAL_COMPLETE_CASE"
    challenge_metadata[challenge_id] = metadata
    membership_parts.append(_membership_frame(routed, challenge_id=challenge_id))

    modality = q2.loc[
        q2["measurement_technology"].eq(AUTO_PATCH)
        | (~q2["measurement_technology"].eq(AUTO_PATCH) & q2["model_split"].isin(("train", "validation")))
    ].copy()
    modality["challenge_split"] = modality["model_split"]
    modality.loc[modality["measurement_technology"].eq(AUTO_PATCH), "challenge_split"] = "test"
    modality["context_group_kind"] = "measurement_technology"
    modality["context_group_id"] = np.where(
        modality["measurement_technology"].eq(AUTO_PATCH),
        "modality:automated_patch_clamp",
        "modality:non_automated_training_context",
    )
    pre_counts = {partition: int(modality["challenge_split"].eq(partition).sum()) for partition in PARTITIONS}
    modality, purge = _purge_cross_partition_identities(modality)
    challenge_id = "Q2_AUTOMATED_PATCH_MODALITY_HOLDOUT"
    challenge_metadata[challenge_id] = {
        "algorithm": "all_automated_patch_to_test_nonautomated_base_train_validation_v1",
        "pre_purge_split_rows": pre_counts,
        "purge": purge,
    }
    membership_parts.append(_membership_frame(modality, challenge_id=challenge_id))

    similarity_audit, similarity_routes, similarity_metadata = _similarity_audit(
        q2, structure_path=master_root / "structure_master.parquet", config=config
    )
    for challenge_id, threshold in (
        ("Q2_LOW_SIMILARITY_060_SCAFFOLD", config.low_similarity_threshold),
        ("Q2_NO_NEAR_DUPLICATE_080_SCAFFOLD", config.near_duplicate_threshold),
    ):
        routed = similarity_routes[threshold]
        challenge_metadata[challenge_id] = {
            "algorithm": "sequential_validation_then_test_maximum_training_reference_tanimoto_v1",
            "threshold": threshold,
            "purge": {"purged_rows_union": 0},
        }
        membership_parts.append(
            _membership_frame(
                routed,
                challenge_id=challenge_id,
                nearest=routed["_nearest"],
                threshold=threshold,
            )
        )

    memberships = pd.concat(membership_parts, ignore_index=True)
    split_rank = {partition: index for index, partition in enumerate(PARTITIONS)}
    memberships["_split_rank"] = memberships["challenge_split"].map(split_rank)
    memberships = memberships.sort_values(
        ["challenge_id", "_split_rank", "structure_id", "membership_id"], kind="stable"
    ).drop(columns=["_split_rank"])
    memberships = memberships.reset_index(drop=True)
    if tuple(memberships.columns) != MEMBERSHIP_COLUMNS:
        raise HergBenchmarkFreezeV15Error("membership column order changed")
    _assert_partition_exclusivity(memberships)
    _assert_context_group_exclusivity(memberships)

    blockers = _blocker_evidence(tasks, config)
    registry = _registry(memberships, blockers, challenge_metadata)
    context = _context_evidence(q2, exact_q2)
    frames = {
        MEMBERSHIP_NAME: memberships,
        REGISTRY_NAME: registry,
        CONTEXT_EVIDENCE_NAME: context,
        SIMILARITY_AUDIT_NAME: similarity_audit,
        BLOCKER_EVIDENCE_NAME: blockers,
    }
    metadata = {
        "challenge_metadata": challenge_metadata,
        "similarity": similarity_metadata,
        "counts": {
            "materialized_challenges": int(
                registry["status"].eq("materialized_label_blind_membership").sum()
            ),
            "blocked_challenges": int((~registry["status"].eq("materialized_label_blind_membership")).sum()),
            "membership_rows": int(len(memberships)),
            "eligible_q2_rows": int(len(q2)),
            "eligible_q2_structures": int(q2["structure_id"].nunique()),
            "exact_q2_overlap_rows": int(len(exact_q2)),
        },
    }
    source_paths = [
        master_root / "task_membership.parquet",
        master_root / "observation_master.parquet",
        master_root / "structure_master.parquet",
        master_root / "herg_master_manifest.json",
        canonical_root / "build_manifest.json",
        exact_task_path,
        *observation_parts,
    ]
    return frames, metadata, source_paths


def _source_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HergBenchmarkFreezeV15Error(f"source input is missing: {path}")
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix == ".parquet":
        record["rows"] = pq.read_metadata(path).num_rows
        record["arrow_schema_sha256"] = _schema_hash(path)
    return record


def _source_records(paths: list[Path]) -> list[dict[str, Any]]:
    unique = sorted({path.resolve() for path in paths}, key=lambda path: path.as_posix())
    return [_source_record(path) for path in unique]


def _artifact_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix == ".parquet":
        record["rows"] = pq.read_metadata(path).num_rows if rows is None else int(rows)
        record["arrow_schema_sha256"] = _schema_hash(path)
    return record


def _report_text(frames: dict[str, pd.DataFrame], metadata: dict[str, Any]) -> str:
    registry = frames[REGISTRY_NAME]
    materialized = registry.loc[registry["status"].eq("materialized_label_blind_membership")]
    blocked = registry.loc[~registry["status"].eq("materialized_label_blind_membership")]
    lines = [
        "# hERG pre-HPC benchmark freeze v1.5",
        "",
        "This supplemental release is label-blind. It reads only routing metadata, structures, "
        "assay/document context, and policy booleans. It embeds no target relation, value, bound, "
        "class, native value, or native label; opens no test label; and trains no model.",
        "",
        "## Materialized challenges",
        "",
        "| Challenge | Rows | Train structures | Validation structures | Test structures | Purged rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in materialized.itertuples(index=False):
        lines.append(
            f"| {row.challenge_id} | {row.rows:,} | {row.train_structures:,} | "
            f"{row.validation_structures:,} | {row.test_structures:,} | {row.purged_rows:,} |"
        )
    lines.extend(
        [
            "",
            "Assay/document partitions use deterministic group-size balancing with a fixed seed "
            "and no seed search, followed by complete removal of every structure or master "
            "scaffold that crossed a proposed partition. Temporal challenges use whole document "
            "years and exclude undated rows before the same leakage purge.",
            "",
            "The two chemical-distance challenges use RDKit Morgan fingerprints "
            f"(radius {metadata['similarity']['radius']}, {metadata['similarity']['bits']} bits; "
            f"RDKit {metadata['similarity']['rdkit_version']}). Validation structures are compared "
            "with training; test structures are compared with training plus the threshold-qualified "
            "validation structures. The thresholds are strict `<0.60` and `<0.80`.",
            "",
            "## Explicit blockers",
            "",
            "| Challenge | Status | Blocker |",
            "|---|---|---|",
        ]
    )
    for row in blocked.itertuples(index=False):
        lines.append(f"| {row.challenge_id} | {row.status} | {row.limitation_or_blocker} |")
    lines.extend(
        [
            "",
            "## Scientific limits",
            "",
            "- Every materialized challenge remains internal to the assembled public corpus.",
            "- Document year is not assay date, synthesis date, or first disclosure date.",
            "- Similarity depends on standardization, Morgan radius, bit width, and threshold.",
            "- Context-group balancing uses row counts only; it never inspects outcome prevalence.",
            "- These memberships are sensitivity benchmarks, not a gold standard and not evidence of "
            "predictive superiority or clinical safety.",
            "",
            "## Reproducibility",
            "",
            f"The manifest binds {len(metadata['source_inputs'])} source files by full SHA-256, "
            "size, Parquet row count/schema where applicable, the implementation hash, every output "
            "hash/schema, and a full deterministic source replay. v1.4 is unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def build_benchmark_freeze_v15(
    *,
    master_root: Path,
    canonical_root: Path,
    exact_task_path: Path,
    output_root: Path,
    config: FreezeV15Config | None = None,
) -> dict[str, Any]:
    config = config or FreezeV15Config()
    frames, metadata, source_paths = _assemble_release(
        master_root=master_root,
        canonical_root=canonical_root,
        exact_task_path=exact_task_path,
        config=config,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), output_root / name)

    source_inputs = _source_records(source_paths)
    metadata["source_inputs"] = source_inputs
    report_path = output_root / REPORT_NAME
    report_path.write_text(_report_text(frames, metadata), encoding="utf-8")

    artifacts = {
        name: _artifact_record(output_root / name, rows=len(frame)) for name, frame in frames.items()
    }
    artifacts[REPORT_NAME] = _artifact_record(report_path)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "configuration": asdict(config),
        "roots": {
            "master_root": str(master_root.resolve()),
            "canonical_root": str(canonical_root.resolve()),
            "exact_task_path": str(exact_task_path.resolve()),
        },
        "projection_contract": {
            "task_membership": list(TASK_COLUMNS),
            "observation_master": list(OBSERVATION_COLUMNS),
            "structure_master": list(STRUCTURE_COLUMNS),
            "canonical_observations": list(CANONICAL_CONTEXT_COLUMNS),
            "canonical_exact_task": list(EXACT_TASK_COLUMNS),
            "forbidden_read_columns": sorted(FORBIDDEN_READ_COLUMNS),
        },
        "source_inputs": source_inputs,
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__)),
        },
        "artifacts": artifacts,
        "counts": metadata["counts"],
        "challenge_metadata": metadata["challenge_metadata"],
        "similarity_contract": metadata["similarity"],
        "scientific_contract": {
            "labels_embedded": False,
            "target_relations_read": False,
            "target_values_read": False,
            "target_bounds_read": False,
            "target_classes_read": False,
            "native_values_or_labels_read": False,
            "test_labels_opened": False,
            "training_performed": False,
            "adjudicated_gold_standard_created": False,
            "external_holdout_created": False,
            "predictive_superiority_established": False,
            "seed_search_performed": False,
        },
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    (output_root / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def _verify_source_record(record: dict[str, Any]) -> None:
    path = Path(record["path"])
    if not path.is_file() or path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
        raise HergBenchmarkFreezeV15Error(f"source rebinding failed: {path}")
    if path.suffix == ".parquet" and (
        pq.read_metadata(path).num_rows != record["rows"]
        or _schema_hash(path) != record["arrow_schema_sha256"]
    ):
        raise HergBenchmarkFreezeV15Error(f"source Parquet contract changed: {path}")


def validate_benchmark_freeze_v15(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise HergBenchmarkFreezeV15Error("manifest schema version mismatch")
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise HergBenchmarkFreezeV15Error("manifest self-hash mismatch")
    implementation = Path(manifest["implementation"]["path"])
    if not implementation.is_file() or _sha256(implementation) != manifest["implementation"]["sha256"]:
        raise HergBenchmarkFreezeV15Error("implementation binding changed")
    for record in manifest["source_inputs"]:
        _verify_source_record(record)
    for name, expected in manifest["artifacts"].items():
        path = output_root / name
        if (
            not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or _sha256(path) != expected["sha256"]
        ):
            raise HergBenchmarkFreezeV15Error(f"artifact verification failed: {name}")
        if path.suffix == ".parquet" and (
            pq.read_metadata(path).num_rows != expected["rows"]
            or _schema_hash(path) != expected["arrow_schema_sha256"]
        ):
            raise HergBenchmarkFreezeV15Error(f"artifact Parquet contract changed: {name}")

    config = FreezeV15Config(**manifest["configuration"])
    roots = manifest["roots"]
    frames, metadata, source_paths = _assemble_release(
        master_root=Path(roots["master_root"]),
        canonical_root=Path(roots["canonical_root"]),
        exact_task_path=Path(roots["exact_task_path"]),
        config=config,
    )
    replayed_sources = _source_records(source_paths)
    if replayed_sources != manifest["source_inputs"]:
        raise HergBenchmarkFreezeV15Error("source inventory replay mismatch")
    for name, expected in frames.items():
        observed = pd.read_parquet(output_root / name)
        try:
            pd.testing.assert_frame_equal(observed, expected, check_dtype=False, check_like=False)
        except AssertionError as exc:
            raise HergBenchmarkFreezeV15Error(f"source replay mismatch: {name}") from exc
    metadata["source_inputs"] = replayed_sources
    if (output_root / REPORT_NAME).read_text(encoding="utf-8") != _report_text(frames, metadata):
        raise HergBenchmarkFreezeV15Error("report replay mismatch")
    if metadata["counts"] != manifest["counts"]:
        raise HergBenchmarkFreezeV15Error("count replay mismatch")
    if metadata["challenge_metadata"] != manifest["challenge_metadata"]:
        raise HergBenchmarkFreezeV15Error("challenge metadata replay mismatch")
    if metadata["similarity"] != manifest["similarity_contract"]:
        raise HergBenchmarkFreezeV15Error("similarity contract replay mismatch")
    expected_scientific_contract = {
        "adjudicated_gold_standard_created": False,
        "external_holdout_created": False,
        "labels_embedded": False,
        "native_values_or_labels_read": False,
        "predictive_superiority_established": False,
        "seed_search_performed": False,
        "target_bounds_read": False,
        "target_classes_read": False,
        "target_relations_read": False,
        "target_values_read": False,
        "test_labels_opened": False,
        "training_performed": False,
    }
    if manifest["scientific_contract"] != expected_scientific_contract:
        raise HergBenchmarkFreezeV15Error("scientific contract weakened")
    memberships = frames[MEMBERSHIP_NAME]
    if tuple(memberships.columns) != MEMBERSHIP_COLUMNS:
        raise HergBenchmarkFreezeV15Error("membership schema mismatch")
    _assert_partition_exclusivity(memberships)
    _assert_context_group_exclusivity(memberships)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-root", type=Path)
    parser.add_argument("--canonical-root", type=Path)
    parser.add_argument("--exact-task-path", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_benchmark_freeze_v15(args.output_root)
    else:
        for name in ("master_root", "canonical_root", "exact_task_path"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required unless --validate-only is set")
        build_benchmark_freeze_v15(
            master_root=args.master_root,
            canonical_root=args.canonical_root,
            exact_task_path=args.exact_task_path,
            output_root=args.output_root,
        )
        validate_benchmark_freeze_v15(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
