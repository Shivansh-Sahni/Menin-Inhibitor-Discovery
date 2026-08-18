"""Freeze label-blind pre-HPC hERG benchmark membership surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "platform-herg-benchmark-freeze/1.0"
MEMBERSHIP_COLUMNS = [
    "challenge_id",
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
    "clinical_context_only",
]
INPUT_COLUMNS = [
    *[column for column in MEMBERSHIP_COLUMNS if column != "challenge_id"],
    "eligible",
    "use_as_training_label",
]


class HergBenchmarkFreezeError(RuntimeError):
    """Raised when a benchmark-freeze invariant fails."""


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
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _materialized_challenges(memberships: pd.DataFrame) -> dict[str, pd.Series]:
    eligible_label = memberships["eligible"] & memberships["use_as_training_label"]
    q2 = memberships["task_id"].eq("Q2_FUNCTIONAL_ASSAY_AWARE") & eligible_label
    technology = memberships["measurement_technology"].fillna("")
    return {
        "Q0_OFFICIAL_SCAFFOLD": memberships["task_id"].eq("Q0_WEAK_FIXED_DOSE_BINARY") & eligible_label,
        "Q1_OFFICIAL_SCAFFOLD": memberships["task_id"].eq("Q1_QUANTITATIVE_PIC50") & eligible_label,
        "Q2_OFFICIAL_SCAFFOLD": q2,
        "CONFIRMED_WT_SENSITIVITY": eligible_label & memberships["target_scope"].eq("wild_type"),
        "Q2_PATCH_CLAMP_TRANSPORT": q2 & technology.str.contains("patch_clamp"),
        "Q2_MANUAL_PATCH_STRESS": q2 & technology.eq("manual_patch_clamp"),
        "Q2_AUTOMATED_PATCH_STRESS": q2 & technology.eq("automated_patch_clamp"),
        "Q2_CHEMBL_SOURCE_STRESS": q2 & memberships["source_family"].eq("chembl_herg_specialized_view"),
        "QT_TRANSLATION_CONTEXT": memberships["task_id"].eq("C1_QT_CONTEXT_EVALUATION")
        & memberships["eligible"],
    }


def _read_memberships(path: Path) -> pd.DataFrame:
    """Read only routing metadata; target values and classes stay unopened."""
    return pd.read_parquet(path, columns=INPUT_COLUMNS)


def _freeze_memberships(memberships: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for challenge_id, mask in _materialized_challenges(memberships).items():
        part = memberships.loc[
            mask, [column for column in MEMBERSHIP_COLUMNS if column != "challenge_id"]
        ].copy()
        part.insert(0, "challenge_id", challenge_id)
        parts.append(part)
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(
            ["challenge_id", "model_split", "structure_id", "membership_id"],
            na_position="last",
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _assert_partition_exclusivity(frozen: pd.DataFrame) -> None:
    for identity in ["structure_id", "scaffold_group_id"]:
        split_counts = (
            frozen.dropna(subset=[identity, "model_split"])
            .groupby(["challenge_id", identity], observed=True)["model_split"]
            .nunique()
        )
        if not split_counts.empty and int(split_counts.max()) > 1:
            raise HergBenchmarkFreezeError(f"{identity} crosses partitions inside a challenge")


def _registry_rows(frozen: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for challenge_id, group in frozen.groupby("challenge_id", sort=True):
        splits = group["model_split"].fillna("unassigned").value_counts().to_dict()
        rows.append(
            {
                "challenge_id": challenge_id,
                "status": "materialized_label_blind_membership",
                "rows": int(len(group)),
                "structures": int(group["structure_id"].dropna().nunique()),
                "train_rows": int(splits.get("train", 0)),
                "validation_rows": int(splits.get("validation", 0)),
                "test_rows": int(splits.get("test", 0)),
                "unassigned_rows": int(splits.get("unassigned", 0)),
                "labels_embedded": False,
                "adjudicated_gold_standard": False,
                "ready_for_superiority_claim": False,
                "blocker": "Requires frozen model, prespecified metric, and external/prospective evidence.",
            }
        )
    rows.extend(
        [
            {
                "challenge_id": "STRICT_TEMPORAL_HOLDOUT",
                "status": "blocked_not_materialized",
                "rows": 0,
                "structures": 0,
                "train_rows": 0,
                "validation_rows": 0,
                "test_rows": 0,
                "unassigned_rows": 0,
                "labels_embedded": False,
                "adjudicated_gold_standard": False,
                "ready_for_superiority_claim": False,
                "blocker": "Document year is not complete in the master task surface.",
            },
            {
                "challenge_id": "LOW_SIMILARITY_EXTERNAL_HOLDOUT",
                "status": "blocked_not_materialized",
                "rows": 0,
                "structures": 0,
                "train_rows": 0,
                "validation_rows": 0,
                "test_rows": 0,
                "unassigned_rows": 0,
                "labels_embedded": False,
                "adjudicated_gold_standard": False,
                "ready_for_superiority_claim": False,
                "blocker": "Similarity threshold and external panel must be frozen before feature computation.",
            },
            {
                "challenge_id": "PROSPECTIVE_MANUAL_PATCH_GOLD",
                "status": "blocked_not_materialized",
                "rows": 0,
                "structures": 0,
                "train_rows": 0,
                "validation_rows": 0,
                "test_rows": 0,
                "unassigned_rows": 0,
                "labels_embedded": False,
                "adjudicated_gold_standard": False,
                "ready_for_superiority_claim": False,
                "blocker": "Requires independent experimental adjudication and a sealed prospective panel.",
            },
        ]
    )
    return sorted(rows, key=lambda row: row["challenge_id"])


def build_benchmark_freeze(*, master_root: Path, output_root: Path) -> dict[str, Any]:
    task_path = master_root / "task_membership.parquet"
    master_manifest_path = master_root / "herg_master_manifest.json"
    if not task_path.is_file() or not master_manifest_path.is_file():
        raise HergBenchmarkFreezeError("master release is incomplete")

    memberships = _read_memberships(task_path)
    frozen = _freeze_memberships(memberships)

    forbidden = {
        "target_pic50_point",
        "target_pic50_lower_bound",
        "target_pic50_upper_bound",
        "target_class",
        "native_value",
        "native_label",
    }
    if forbidden.intersection(frozen.columns):
        raise HergBenchmarkFreezeError("label/value columns entered membership surface")
    _assert_partition_exclusivity(frozen)

    registry = pd.DataFrame(_registry_rows(frozen))
    output_root.mkdir(parents=True, exist_ok=True)
    membership_path = output_root / "frozen_challenge_memberships.parquet"
    registry_path = output_root / "benchmark_challenge_registry.parquet"
    pq.write_table(pa.Table.from_pandas(frozen, preserve_index=False), membership_path)
    pq.write_table(pa.Table.from_pandas(registry, preserve_index=False), registry_path)

    artifacts = {}
    for path, rows in [(membership_path, len(frozen)), (registry_path, len(registry))]:
        artifacts[path.name] = {
            "path": path.name,
            "rows": int(rows),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "arrow_schema_sha256": _schema_hash(path),
        }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "master_manifest_sha256": _sha256(master_manifest_path),
        "input": {
            "path": str(task_path.resolve()),
            "rows": int(len(memberships)),
            "bytes": task_path.stat().st_size,
            "sha256": _sha256(task_path),
        },
        "artifacts": artifacts,
        "counts": {
            "materialized_challenges": int(registry["status"].str.startswith("materialized").sum()),
            "blocked_challenges": int(registry["status"].str.startswith("blocked").sum()),
            "membership_rows": int(len(frozen)),
        },
        "scientific_contract": {
            "labels_embedded": False,
            "adjudicated_gold_standard_created": False,
            "test_labels_opened_by_builder": False,
            "training_performed": False,
            "predictive_superiority_established": False,
        },
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    manifest_path = output_root / "benchmark_freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def validate_benchmark_freeze(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "benchmark_freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise HergBenchmarkFreezeError("manifest self-hash mismatch")
    source = Path(manifest["input"]["path"])
    expected_source = manifest["input"]
    if (
        not source.is_file()
        or source.stat().st_size != expected_source["bytes"]
        or _sha256(source) != expected_source["sha256"]
        or pq.read_metadata(source).num_rows != expected_source["rows"]
    ):
        raise HergBenchmarkFreezeError("source membership rebinding failed")
    master_manifest = source.parent / "herg_master_manifest.json"
    if not master_manifest.is_file() or _sha256(master_manifest) != manifest["master_manifest_sha256"]:
        raise HergBenchmarkFreezeError("master manifest rebinding failed")
    for name, expected in manifest["artifacts"].items():
        path = output_root / name
        if (
            not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or _sha256(path) != expected["sha256"]
            or pq.read_metadata(path).num_rows != expected["rows"]
            or _schema_hash(path) != expected["arrow_schema_sha256"]
        ):
            raise HergBenchmarkFreezeError(f"artifact verification failed: {name}")
    frozen = pd.read_parquet(output_root / "frozen_challenge_memberships.parquet")
    if set(frozen.columns) != set(MEMBERSHIP_COLUMNS):
        raise HergBenchmarkFreezeError("membership schema mismatch")
    _assert_partition_exclusivity(frozen)
    source_memberships = _read_memberships(source)
    expected_frozen = _freeze_memberships(source_memberships)
    try:
        pd.testing.assert_frame_equal(frozen, expected_frozen, check_dtype=False, check_like=False)
    except AssertionError as exc:
        raise HergBenchmarkFreezeError("challenge membership replay mismatch") from exc
    registry = pd.read_parquet(output_root / "benchmark_challenge_registry.parquet")
    expected_registry = pd.DataFrame(_registry_rows(expected_frozen))
    try:
        pd.testing.assert_frame_equal(registry, expected_registry, check_dtype=False, check_like=False)
    except AssertionError as exc:
        raise HergBenchmarkFreezeError("challenge registry replay mismatch") from exc
    expected_counts = {
        "materialized_challenges": int(expected_registry["status"].str.startswith("materialized").sum()),
        "blocked_challenges": int(expected_registry["status"].str.startswith("blocked").sum()),
        "membership_rows": int(len(expected_frozen)),
    }
    if manifest["counts"] != expected_counts:
        raise HergBenchmarkFreezeError("manifest count replay mismatch")
    if manifest["scientific_contract"] != {
        "adjudicated_gold_standard_created": False,
        "labels_embedded": False,
        "predictive_superiority_established": False,
        "test_labels_opened_by_builder": False,
        "training_performed": False,
    }:
        raise HergBenchmarkFreezeError("scientific contract weakened")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-root", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_benchmark_freeze(args.output_root)
    else:
        if args.master_root is None:
            parser.error("--master-root is required unless --validate-only is set")
        build_benchmark_freeze(master_root=args.master_root, output_root=args.output_root)
        validate_benchmark_freeze(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
