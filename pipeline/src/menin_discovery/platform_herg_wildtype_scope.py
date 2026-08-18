"""Freeze an additive wild-type hERG scope for paper-facing analyses.

The source observation ledger is never rewritten.  Confirmed wild-type records
and records with no reported variant are admitted, but remain distinguishable.
Any explicit mutant/variant record is excluded from every model-facing index.
This deliberately permissive policy preserves useful scale without representing
``wild_type_or_unspecified`` observations as experimentally confirmed wild type.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "platform-herg-wildtype-scope/1.0"
INDEX_NAME = "wildtype_observation_index.parquet"
EXCLUSIONS_NAME = "explicit_mutant_exclusions.parquet"
MANIFEST_NAME = "wildtype_scope_manifest.json"

_REQUIRED = (
    "observation_id",
    "source_family",
    "structure_id",
    "target_variant",
    "assay_id",
    "native_aux_json",
    "pic50_value",
    "derived_binary_label",
)

_INDEX_SCHEMA = pa.schema(
    [
        pa.field("observation_id", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string()),
        pa.field("source_family", pa.large_string(), nullable=False),
        pa.field("assay_id", pa.large_string()),
        pa.field("target_variant_original", pa.large_string(), nullable=False),
        pa.field("wildtype_scope", pa.large_string(), nullable=False),
        pa.field("admission_status", pa.large_string(), nullable=False),
        pa.field("admission_reason", pa.large_string(), nullable=False),
        pa.field("pic50_value", pa.float64()),
        pa.field("derived_binary_label", pa.int8()),
    ]
)

_EXCLUSION_SCHEMA = pa.schema(
    [
        pa.field("observation_id", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string()),
        pa.field("source_family", pa.large_string(), nullable=False),
        pa.field("assay_id", pa.large_string()),
        pa.field("target_variant_original", pa.large_string(), nullable=False),
        pa.field("exclusion_reason", pa.large_string(), nullable=False),
        pa.field("native_aux_json", pa.large_string(), nullable=False),
    ]
)


class HergWildtypeScopeError(RuntimeError):
    """Raised when the scope input or generated artifact fails closed."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_with_digest(body: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["manifest_sha256"] = hashlib.sha256(_canonical_json_bytes(result)).hexdigest()
    return result


def _checked_ledger(path: Path) -> Path:
    if path.is_symlink() or not path.is_file() or path.suffix.casefold() != ".parquet":
        raise HergWildtypeScopeError(f"Missing, unsafe, or non-Parquet observation ledger: {path}")
    path = path.resolve()
    names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(set(_REQUIRED) - names)
    if missing:
        raise HergWildtypeScopeError(f"Observation ledger is missing required columns: {missing}")
    return path


def _clean_required(value: object, *, column: str) -> str:
    if value is None or not str(value).strip():
        raise HergWildtypeScopeError(f"Input column {column!r} contains null or blank values")
    return str(value).strip()


def _classify_variant(value: object) -> tuple[str, str, str]:
    variant = _clean_required(value, column="target_variant")
    if variant == "wild_type":
        return "confirmed_wild_type", "admitted", "source_explicit_wild_type"
    if variant == "wild_type_or_unspecified":
        return "wild_type_or_unspecified", "admitted", "no_explicit_mutant_evidence"
    if variant == "mutant_or_variant":
        return "explicit_mutant_or_variant", "excluded", "explicit_mutant_or_variant"
    raise HergWildtypeScopeError(f"Unrecognized target_variant value: {variant!r}")


def _rows(ledger_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    table = pq.read_table(ledger_path, columns=list(_REQUIRED))
    if table.num_rows == 0:
        raise HergWildtypeScopeError("Observation ledger is empty")
    admitted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[str] = set()
    variant_counts: Counter[str] = Counter()
    source_scope_counts: Counter[str] = Counter()
    for source in table.to_pylist():
        observation_id = _clean_required(source["observation_id"], column="observation_id")
        if observation_id in seen:
            raise HergWildtypeScopeError(f"Duplicate observation_id: {observation_id}")
        seen.add(observation_id)
        source_family = _clean_required(source["source_family"], column="source_family")
        variant = _clean_required(source["target_variant"], column="target_variant")
        scope, status, reason = _classify_variant(variant)
        variant_counts[variant] += 1
        source_scope_counts[f"{source_family}|{scope}"] += 1
        common = {
            "observation_id": observation_id,
            "structure_id": source["structure_id"],
            "source_family": source_family,
            "assay_id": source["assay_id"],
            "target_variant_original": variant,
        }
        if status == "admitted":
            admitted.append(
                {
                    **common,
                    "wildtype_scope": scope,
                    "admission_status": status,
                    "admission_reason": reason,
                    "pic50_value": source["pic50_value"],
                    "derived_binary_label": source["derived_binary_label"],
                }
            )
        else:
            excluded.append(
                {
                    **common,
                    "exclusion_reason": reason,
                    "native_aux_json": _clean_required(source["native_aux_json"], column="native_aux_json"),
                }
            )
    admitted.sort(key=lambda row: row["observation_id"])
    excluded.sort(key=lambda row: row["observation_id"])
    counts = {
        "input_observations": table.num_rows,
        "admitted_observations": len(admitted),
        "excluded_explicit_mutant_observations": len(excluded),
        "admitted_unique_structures": len({row["structure_id"] for row in admitted if row["structure_id"]}),
        "variant_counts": dict(sorted(variant_counts.items())),
        "source_scope_counts": dict(sorted(source_scope_counts.items())),
    }
    if len(admitted) + len(excluded) != table.num_rows:
        raise HergWildtypeScopeError("Admission accounting does not reconcile to input rows")
    return admitted, excluded, counts


def build_herg_wildtype_scope(*, observation_ledger_path: Path, output_root: Path) -> dict[str, Any]:
    """Create deterministic wild-type admission and explicit-exclusion indexes."""

    ledger = _checked_ledger(observation_ledger_path)
    output_root = output_root.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        admitted, excluded, counts = _rows(ledger)
        pq.write_table(
            pa.Table.from_pylist(admitted, schema=_INDEX_SCHEMA), temporary / INDEX_NAME, compression="zstd"
        )
        pq.write_table(
            pa.Table.from_pylist(excluded, schema=_EXCLUSION_SCHEMA),
            temporary / EXCLUSIONS_NAME,
            compression="zstd",
        )
        artifacts = {
            name: {
                "sha256": _sha256_file(temporary / name),
                "rows": pq.ParquetFile(temporary / name).metadata.num_rows,
            }
            for name in (INDEX_NAME, EXCLUSIONS_NAME)
        }
        manifest = _manifest_with_digest(
            {
                "schema_version": SCHEMA_VERSION,
                "input": {"path": str(ledger), "sha256": _sha256_file(ledger)},
                "policy": {
                    "primary_target": "human wild-type KCNH2/hERG",
                    "confirmed_wild_type": "admit",
                    "wild_type_or_unspecified": "admit_with_explicit_uncertainty_label",
                    "mutant_or_variant": "exclude",
                    "scope_claim": "permissive wild-type scope; unspecified is not confirmed wild type",
                },
                "counts": counts,
                "artifacts": artifacts,
            }
        )
        (temporary / MANIFEST_NAME).write_bytes(_canonical_json_bytes(manifest) + b"\n")
        if output_root.exists():
            shutil.rmtree(output_root)
        os.replace(temporary, output_root)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_herg_wildtype_scope(output_root: Path) -> dict[str, Any]:
    """Validate hashes, schemas, row accounting, and mutant exclusion."""

    root = output_root.resolve()
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise HergWildtypeScopeError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_digest = manifest.pop("manifest_sha256", None)
    actual_digest = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
    if expected_digest != actual_digest:
        raise HergWildtypeScopeError("Manifest digest mismatch")
    manifest["manifest_sha256"] = expected_digest
    for name, schema in ((INDEX_NAME, _INDEX_SCHEMA), (EXCLUSIONS_NAME, _EXCLUSION_SCHEMA)):
        path = root / name
        artifact = manifest["artifacts"][name]
        if _sha256_file(path) != artifact["sha256"]:
            raise HergWildtypeScopeError(f"Artifact digest mismatch: {name}")
        table = pq.read_table(path)
        if table.schema != schema:
            raise HergWildtypeScopeError(f"Artifact schema mismatch: {name}")
        if table.num_rows != artifact["rows"]:
            raise HergWildtypeScopeError(f"Artifact row-count mismatch: {name}")
    admitted = pq.read_table(root / INDEX_NAME, columns=["target_variant_original", "admission_status"])
    excluded = pq.read_table(root / EXCLUSIONS_NAME, columns=["target_variant_original"])
    if "mutant_or_variant" in set(admitted.column("target_variant_original").to_pylist()):
        raise HergWildtypeScopeError("Explicit mutant observation leaked into admitted index")
    if set(excluded.column("target_variant_original").to_pylist()) - {"mutant_or_variant"}:
        raise HergWildtypeScopeError("Non-mutant observation appears in explicit mutant exclusions")
    if set(admitted.column("admission_status").to_pylist()) != {"admitted"}:
        raise HergWildtypeScopeError("Admitted index contains an invalid status")
    counts = manifest["counts"]
    if (
        counts["admitted_observations"] + counts["excluded_explicit_mutant_observations"]
        != counts["input_observations"]
    ):
        raise HergWildtypeScopeError("Manifest row accounting does not reconcile")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    manifest = (
        validate_herg_wildtype_scope(args.output_root)
        if args.validate_only
        else build_herg_wildtype_scope(
            observation_ledger_path=args.observation_ledger,
            output_root=args.output_root,
        )
    )
    print(json.dumps(manifest, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
