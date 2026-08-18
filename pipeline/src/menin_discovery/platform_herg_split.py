"""Deterministic, leakage-resistant split for the barebones hERG backbone.

The input is the four-column ``structure_consensus_binary.parquet`` produced by
``platform_herg_hierarchy``.  Whole Bemis--Murcko scaffold groups are assigned
by one fixed SHA-256 rule to 80/10/10 train/validation/test partitions.  There
is deliberately no label balancing, seed search, or repeated split selection:
the output preserves the severe class imbalance that is present in the source.

Murcko extraction returns an empty scaffold for acyclic molecules.  Such rows
are *not* combined into one enormous group; their exact canonical structure is
used as an explicitly reported proxy group.  Invalid structures use the same
exact-structure fallback and are reported separately by method in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .features import is_exact_smiles_proxy_method, scaffold_key

SCHEMA_VERSION = "platform-herg-scaffold-split/1.0"
ARTIFACT_NAME = "structure_consensus_binary_scaffold_split.parquet"
PARTITIONS = ("train", "validation", "test")
SPLIT_FRACTIONS = {"train": 0.80, "validation": 0.10, "test": 0.10}
SPLIT_SALT = "platform-herg-aid720551-scaffold-split-v1"

_REQUIRED_INPUT_COLUMNS = (
    "structure_id",
    "standardized_smiles",
    "standard_inchi_key",
    "herg_blocker_label",
)
_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("standardized_smiles", pa.large_string(), nullable=False),
        pa.field("standard_inchi_key", pa.large_string(), nullable=False),
        pa.field("herg_blocker_label", pa.int8(), nullable=False),
        pa.field("split", pa.large_string(), nullable=False),
        pa.field("scaffold_group_id", pa.large_string(), nullable=False),
    ]
)


class HergSplitError(RuntimeError):
    """Raised when a split input or generated artifact fails closed."""


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


def _clean_text(value: object, *, column: str) -> str:
    if value is None:
        raise HergSplitError(f"Input column {column!r} contains null values")
    text = str(value).strip()
    if not text:
        raise HergSplitError(f"Input column {column!r} contains blank values")
    return text


def _group_id(method: str, key: str) -> str:
    digest = hashlib.sha256(f"{method}\x1f{key}".encode()).hexdigest().upper()
    return f"HSCF-{digest}"


def _assigned_split(group_id: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_SALT}\x1f{group_id}".encode()).digest()
    draw = int.from_bytes(digest[:8], byteorder="big", signed=False) / 2**64
    if draw < SPLIT_FRACTIONS["train"]:
        return "train"
    if draw < SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["validation"]:
        return "validation"
    return "test"


def _checked_input(path: Path) -> Path:
    if path.is_symlink() or not path.is_file() or path.suffix.casefold() != ".parquet":
        raise HergSplitError(f"Missing, unsafe, or non-Parquet hERG consensus input: {path}")
    return path.resolve()


def _read_and_split(path: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    schema_names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(set(_REQUIRED_INPUT_COLUMNS) - schema_names)
    if missing:
        raise HergSplitError(f"Consensus input is missing required columns: {missing}")
    table = pq.read_table(path, columns=list(_REQUIRED_INPUT_COLUMNS))
    if table.num_rows < 3:
        raise HergSplitError("At least three consensus structures are required")

    rows: list[dict[str, Any]] = []
    method_counts: Counter[str] = Counter()
    seen_structure_ids: set[str] = set()
    seen_smiles: set[str] = set()
    seen_inchi_keys: set[str] = set()
    for source in table.to_pylist():
        structure_id = _clean_text(source["structure_id"], column="structure_id")
        smiles = _clean_text(source["standardized_smiles"], column="standardized_smiles")
        inchi_key = _clean_text(source["standard_inchi_key"], column="standard_inchi_key")
        try:
            label = int(source["herg_blocker_label"])
        except (TypeError, ValueError) as error:
            raise HergSplitError("hERG blocker labels must be integers in {0, 1}") from error
        if label not in {0, 1}:
            raise HergSplitError("hERG blocker labels must be integers in {0, 1}")
        if structure_id in seen_structure_ids:
            raise HergSplitError(f"Duplicate structure_id in consensus input: {structure_id}")
        if smiles in seen_smiles:
            raise HergSplitError(f"Duplicate standardized_smiles in consensus input: {smiles}")
        if inchi_key in seen_inchi_keys:
            raise HergSplitError(f"Duplicate standard_inchi_key in consensus input: {inchi_key}")
        seen_structure_ids.add(structure_id)
        seen_smiles.add(smiles)
        seen_inchi_keys.add(inchi_key)

        key, method = scaffold_key(smiles)
        if not key:
            raise HergSplitError(f"Scaffold grouping returned an empty key for {structure_id}")
        group_id = _group_id(method, key)
        method_counts[method] += 1
        rows.append(
            {
                "structure_id": structure_id,
                "standardized_smiles": smiles,
                "standard_inchi_key": inchi_key,
                "herg_blocker_label": label,
                "split": _assigned_split(group_id),
                "scaffold_group_id": group_id,
            }
        )
    rows.sort(key=lambda row: row["structure_id"])
    if {str(row["split"]) for row in rows} != set(PARTITIONS):
        raise HergSplitError(
            "Fixed hash split did not populate all three partitions; input is too small or too group-concentrated"
        )
    return rows, method_counts


def _qc_counts(rows: Sequence[Mapping[str, Any]], method_counts: Mapping[str, int]) -> dict[str, Any]:
    group_partitions: dict[str, set[str]] = defaultdict(set)
    structure_partitions: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    per_split: dict[str, Any] = {}
    for row in rows:
        split = str(row["split"])
        group_partitions[str(row["scaffold_group_id"])].add(split)
        exact = (
            str(row["structure_id"]),
            str(row["standardized_smiles"]),
            str(row["standard_inchi_key"]),
        )
        structure_partitions[exact].add(split)
    for split in PARTITIONS:
        selected = [row for row in rows if row["split"] == split]
        labels = Counter(int(row["herg_blocker_label"]) for row in selected)
        per_split[split] = {
            "rows": len(selected),
            "fraction": len(selected) / len(rows),
            "groups": len({str(row["scaffold_group_id"]) for row in selected}),
            "class_counts": {"0": labels[0], "1": labels[1]},
        }
    labels = Counter(int(row["herg_blocker_label"]) for row in rows)
    return {
        "rows": len(rows),
        "groups": len(group_partitions),
        "class_counts": {"0": labels[0], "1": labels[1]},
        "per_split": per_split,
        "scaffold_method_row_counts": dict(sorted(method_counts.items())),
        "exact_structure_proxy_rows": sum(
            int(count) for method, count in method_counts.items() if is_exact_smiles_proxy_method(method)
        ),
        "acyclic_exact_proxy_rows": int(method_counts.get("bemis_murcko_with_exact_acyclic", 0)),
        "scaffold_group_overlap_count": sum(len(splits) > 1 for splits in group_partitions.values()),
        "exact_structure_overlap_count": sum(len(splits) > 1 for splits in structure_partitions.values()),
    }


def build_herg_scaffold_split(
    *,
    consensus_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Build one model-ready split Parquet plus a reproducibility manifest."""

    source = _checked_input(Path(consensus_path))
    output = Path(output_root)
    if output.exists():
        raise HergSplitError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        rows, method_counts = _read_and_split(source)
        artifact_path = staging / ARTIFACT_NAME
        table = pa.Table.from_pylist(rows, schema=_OUTPUT_SCHEMA)
        pq.write_table(
            table,
            artifact_path,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
            data_page_version="1.0",
            version="2.6",
        )
        qc = _qc_counts(rows, method_counts)
        manifest = _manifest_with_digest(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": "herg_aid720551_consensus_scaffold_split",
                "input": {
                    "path": str(source),
                    "bytes": source.stat().st_size,
                    "sha256": _sha256_file(source),
                    "rows": table.num_rows,
                },
                "split_policy": {
                    "algorithm": "fixed_sha256_whole_scaffold_group_v1",
                    "salt": SPLIT_SALT,
                    "fractions": SPLIT_FRACTIONS,
                    "hash_draw": "first_64_bits_unsigned_big_endian_divided_by_2^64",
                    "boundaries": "train:[0,0.8), validation:[0.8,0.9), test:[0.9,1)",
                    "group_definition": "Bemis-Murcko; exact canonical structure proxy for acyclic or invalid structures",
                    "label_stratification": False,
                    "seed_search": False,
                    "class_balance_preserved": True,
                },
                "qc": qc,
                "artifact": {
                    "path": ARTIFACT_NAME,
                    "rows": table.num_rows,
                    "bytes": artifact_path.stat().st_size,
                    "sha256": _sha256_file(artifact_path),
                    "arrow_schema_sha256": hashlib.sha256(
                        _OUTPUT_SCHEMA.serialize().to_pybytes()
                    ).hexdigest(),
                },
            }
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        validate_herg_scaffold_split(staging)
        os.replace(staging, output)
        return validate_herg_scaffold_split(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_herg_scaffold_split(output_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Recompute hashes, groups, assignments, counts, and leakage audits."""

    root = Path(output_root)
    manifest_path = root / "manifest.json"
    artifact_path = root / ARTIFACT_NAME
    if (
        root.is_symlink()
        or not root.is_dir()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
        or artifact_path.is_symlink()
        or not artifact_path.is_file()
    ):
        raise HergSplitError(f"Missing or unsafe hERG split output: {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HergSplitError(f"Unreadable hERG split manifest: {manifest_path}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise HergSplitError("Unexpected hERG split manifest schema")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if hashlib.sha256(_canonical_json_bytes(body)).hexdigest() != manifest.get("manifest_sha256"):
        raise HergSplitError("hERG split manifest digest mismatch")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict) or artifact.get("path") != ARTIFACT_NAME:
        raise HergSplitError("Unexpected hERG split artifact declaration")
    if {path.name for path in root.iterdir()} != {"manifest.json", ARTIFACT_NAME} or any(
        path.is_symlink() or not path.is_file() for path in root.iterdir()
    ):
        raise HergSplitError("hERG split output contains unexpected or unsafe members")
    source_binding = manifest.get("input")
    if not isinstance(source_binding, dict):
        raise HergSplitError("hERG split input binding is malformed")
    source_path = Path(str(source_binding.get("path", "")))
    if (
        source_path.is_symlink()
        or not source_path.is_file()
        or source_path.stat().st_size != int(source_binding.get("bytes", -1))
        or _sha256_file(source_path) != source_binding.get("sha256")
    ):
        raise HergSplitError(f"hERG split input binding mismatch: {source_path}")
    if artifact_path.stat().st_size != int(artifact.get("bytes", -1)) or _sha256_file(
        artifact_path
    ) != artifact.get("sha256"):
        raise HergSplitError("hERG split artifact hash mismatch")
    parquet = pq.ParquetFile(artifact_path)
    if parquet.schema_arrow != _OUTPUT_SCHEMA:
        raise HergSplitError("hERG split artifact schema mismatch")
    if parquet.metadata is None or parquet.metadata.num_rows != int(artifact.get("rows", -1)):
        raise HergSplitError("hERG split artifact row-count mismatch")
    if parquet.metadata.num_rows != int(source_binding.get("rows", -1)):
        raise HergSplitError("hERG split source/output row-count mismatch")

    rows = pq.read_table(artifact_path).to_pylist()
    if rows != sorted(rows, key=lambda row: row["structure_id"]):
        raise HergSplitError("hERG split artifact is not deterministically sorted")
    method_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_smiles: set[str] = set()
    seen_keys: set[str] = set()
    for row in rows:
        structure_id = _clean_text(row["structure_id"], column="structure_id")
        smiles = _clean_text(row["standardized_smiles"], column="standardized_smiles")
        inchi_key = _clean_text(row["standard_inchi_key"], column="standard_inchi_key")
        if int(row["herg_blocker_label"]) not in {0, 1}:
            raise HergSplitError("Split artifact contains an invalid label")
        if structure_id in seen_ids or smiles in seen_smiles or inchi_key in seen_keys:
            raise HergSplitError("Split artifact contains a duplicate exact structure")
        seen_ids.add(structure_id)
        seen_smiles.add(smiles)
        seen_keys.add(inchi_key)
        key, method = scaffold_key(smiles)
        method_counts[method] += 1
        expected_group = _group_id(method, key)
        if row["scaffold_group_id"] != expected_group:
            raise HergSplitError(f"Scaffold group drift for {structure_id}")
        if row["split"] != _assigned_split(expected_group):
            raise HergSplitError(f"Fixed hash assignment drift for {structure_id}")
    observed_qc = _qc_counts(rows, method_counts)
    if observed_qc != manifest.get("qc"):
        raise HergSplitError("hERG split QC/count mismatch")
    if observed_qc["scaffold_group_overlap_count"] or observed_qc["exact_structure_overlap_count"]:
        raise HergSplitError("hERG split leakage audit failed")
    source_rows = {
        str(row["structure_id"]): (
            str(row["standardized_smiles"]),
            str(row["standard_inchi_key"]),
            int(row["herg_blocker_label"]),
        )
        for row in pq.read_table(source_path, columns=list(_REQUIRED_INPUT_COLUMNS)).to_pylist()
    }
    output_rows = {
        str(row["structure_id"]): (
            str(row["standardized_smiles"]),
            str(row["standard_inchi_key"]),
            int(row["herg_blocker_label"]),
        )
        for row in rows
    }
    if source_rows != output_rows or len(source_rows) != len(rows):
        raise HergSplitError("hERG split changed or omitted source structures or labels")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.validate_only:
        validate_herg_scaffold_split(args.output_root)
    else:
        if args.consensus is None:
            raise SystemExit("build mode requires --consensus")
        build_herg_scaffold_split(consensus_path=args.consensus, output_root=args.output_root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
