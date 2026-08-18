"""Deterministic file manifests and provenance verification.

Manifests are portable: file paths are relative to a supplied root and the
content-derived ``dataset_sha256``/default build identifier do not depend on
absolute locations or file modification times.  ``created_at`` can be supplied
explicitly (or through ``SOURCE_DATE_EPOCH``) when byte-identical manifest JSON
is required across builds.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import mimetypes
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

PathLike = str | os.PathLike[str]
MANIFEST_VERSION = "1.0"
DEFAULT_TABULAR_EXTENSIONS = frozenset(
    {".csv", ".tsv", ".tab", ".json", ".jsonl", ".ndjson", ".parquet", ".xlsx", ".xls"}
)
PORTABLE_MEDIA_TYPES = {
    ".csv": "text/csv",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".ndjson": "application/x-ndjson",
    ".parquet": "application/vnd.apache.parquet",
    ".tab": "text/tab-separated-values",
    ".tsv": "text/tab-separated-values",
    ".txt": "text/plain",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


@dataclass(frozen=True)
class VerificationIssue:
    """One manifest or file verification failure."""

    code: str
    message: str
    path: str | None = None
    expected: Any = None
    actual: Any = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["expected"] = _json_value(result["expected"])
        result["actual"] = _json_value(result["actual"])
        return result


@dataclass
class ManifestVerification:
    """Machine-readable result returned by :func:`verify_manifest`."""

    stage: str
    checked_files: int = 0
    expected_files: int = 0
    issues: list[VerificationIssue] = field(default_factory=list)
    verified_at: str = field(default_factory=lambda: _utc_timestamp())

    @property
    def valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_verification_version": "1.0",
            "stage": self.stage,
            "verified_at": self.verified_at,
            "valid": self.valid,
            "checked_files": self.checked_files,
            "expected_files": self.expected_files,
            "issue_count": len(self.issues),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def issues_frame(self) -> pd.DataFrame:
        records = []
        for issue in self.issues:
            record = issue.to_dict()
            for column in ("expected", "actual"):
                value = record[column]
                if isinstance(value, (dict, list)):
                    record[column] = json.dumps(value, sort_keys=True, separators=(",", ":"))
            records.append(record)
        return pd.DataFrame(
            records,
            columns=["code", "message", "path", "expected", "actual"],
        )

    def write_json(self, path: PathLike) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return output

    def write_csv(self, path: PathLike) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.issues_frame().to_csv(output, index=False)
        return output


def sha256_file(path: PathLike, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the hexadecimal SHA-256 digest for a file."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_file(path: PathLike, *, relative_path: str | None = None) -> dict[str, Any]:
    """Build a manifest entry with hash, size, media type and table metadata."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    stat_before = file_path.stat()
    digest = sha256_file(file_path)
    metadata = _tabular_metadata(file_path)
    stat_after = file_path.stat()
    if (stat_before.st_size, stat_before.st_mtime_ns) != (
        stat_after.st_size,
        stat_after.st_mtime_ns,
    ):
        raise RuntimeError(f"File changed while it was being inspected: {file_path}")

    media_type = PORTABLE_MEDIA_TYPES.get(file_path.suffix.casefold())
    if media_type is None:
        media_type, _ = mimetypes.guess_type(file_path.name, strict=False)
    entry: dict[str, Any] = {
        "path": relative_path or file_path.name,
        "sha256": digest,
        "size_bytes": int(stat_after.st_size),
        "media_type": media_type or "application/octet-stream",
    }
    entry.update(metadata)
    return entry


def create_manifest(
    paths: PathLike | Iterable[PathLike] | None = None,
    *,
    root: PathLike | None = None,
    stage: str = "data",
    include: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
    created_at: str | datetime | None = None,
    build_id: str | None = None,
    root_label: str = ".",
    upstream: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Create a portable, content-addressed manifest.

    ``paths`` may be a directory, a file, or an iterable of files/directories.
    When omitted, ``root`` is scanned recursively.  Include/exclude patterns are
    matched against POSIX-style paths relative to ``root``.  Entries and schemas
    are always sorted, making the content digest stable across machines.
    """

    root_path, files = _resolve_root_and_files(paths, root)
    include_patterns = tuple(include or ("*",))
    exclude_patterns = tuple(exclude)
    selected: list[tuple[str, Path]] = []
    for file_path in files:
        relative = _relative_posix(file_path, root_path)
        if not any(fnmatch.fnmatchcase(relative, pattern) for pattern in include_patterns):
            continue
        if any(fnmatch.fnmatchcase(relative, pattern) for pattern in exclude_patterns):
            continue
        selected.append((relative, file_path))
    selected.sort(key=lambda item: item[0])

    entries = [inspect_file(file_path, relative_path=relative) for relative, file_path in selected]
    dataset_payload = {
        "stage": str(stage),
        "files": entries,
        "upstream": _normalized_upstream(upstream),
    }
    dataset_sha256 = _sha256_json(dataset_payload)
    resolved_build_id = str(build_id) if build_id else f"build-{dataset_sha256[:16]}"
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "stage": str(stage),
        "created_at": _utc_timestamp(created_at),
        "build_id": resolved_build_id,
        "root": str(root_label),
        "file_count": len(entries),
        "total_size_bytes": int(sum(entry["size_bytes"] for entry in entries)),
        "dataset_sha256": dataset_sha256,
        "upstream": _normalized_upstream(upstream),
        "files": entries,
    }
    return _finalize_manifest(manifest)


def create_directory_manifest(
    directory: PathLike,
    *,
    stage: str,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
    created_at: str | datetime | None = None,
    build_id: str | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for recursively manifesting one directory."""

    return create_manifest(
        directory,
        stage=stage,
        include=include,
        exclude=exclude,
        created_at=created_at,
        build_id=build_id,
    )


def create_data_manifests(
    raw_root: PathLike,
    processed_root: PathLike,
    *,
    created_at: str | datetime | None = None,
    build_id: str | None = None,
    raw_include: Sequence[str] | None = None,
    processed_include: Sequence[str] | None = None,
    raw_exclude: Sequence[str] = (),
    processed_exclude: Sequence[str] = (),
    output_directory: PathLike | None = None,
) -> dict[str, dict[str, Any]]:
    """Create linked raw and processed manifests for one data build.

    If no build identifier is supplied, a deterministic identifier is derived
    jointly from both stage content digests.  The processed manifest records the
    raw dataset digest as its upstream dependency.
    """

    timestamp = _utc_timestamp(created_at)
    raw = create_manifest(
        raw_root,
        stage="raw",
        include=raw_include,
        exclude=raw_exclude,
        created_at=timestamp,
        build_id=build_id or "pending",
    )
    processed = create_manifest(
        processed_root,
        stage="processed",
        include=processed_include,
        exclude=processed_exclude,
        created_at=timestamp,
        build_id=build_id or "pending",
        upstream=(
            {
                "stage": "raw",
                "dataset_sha256": raw["dataset_sha256"],
            },
        ),
    )
    resolved_build_id = (
        build_id
        or "build-"
        + _sha256_json(
            {
                "raw": raw["dataset_sha256"],
                "processed": processed["dataset_sha256"],
            }
        )[:16]
    )
    raw["build_id"] = resolved_build_id
    processed["build_id"] = resolved_build_id
    raw = _finalize_manifest(raw)
    processed = _finalize_manifest(processed)
    manifests = {"raw": raw, "processed": processed}
    if output_directory is not None:
        write_data_manifests(manifests, output_directory)
    return manifests


def write_manifest(manifest: Mapping[str, Any], path: PathLike) -> Path:
    """Write canonical, stable JSON for a manifest."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def write_data_manifests(manifests: Mapping[str, Mapping[str, Any]], directory: PathLike) -> dict[str, Path]:
    """Write stage manifests as ``<stage>_manifest.json`` files."""

    output_dir = Path(directory)
    written: dict[str, Path] = {}
    for stage, manifest in sorted(manifests.items()):
        written[stage] = write_manifest(manifest, output_dir / f"{stage}_manifest.json")
    return written


def load_manifest(path: PathLike) -> dict[str, Any]:
    """Load a JSON manifest and ensure it is an object."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Manifest JSON must contain an object")
    return value


def verify_manifest(
    manifest: Mapping[str, Any] | PathLike,
    *,
    root: PathLike,
    check_metadata: bool = True,
    allow_extra: bool = True,
    verified_at: str | datetime | None = None,
) -> ManifestVerification:
    """Verify manifest integrity and the current files below ``root``.

    Verification checks the manifest's own digest, dataset digest, aggregate
    counts, path safety, file presence, size, SHA-256, and (by default) recorded
    row counts/schema.  Set ``allow_extra=False`` to reject unmanifested files.
    """

    manifest_path: Path | None = None
    if isinstance(manifest, Mapping):
        document = dict(manifest)
    else:
        manifest_path = Path(manifest).resolve()
        document = load_manifest(manifest_path)

    result = ManifestVerification(
        stage=str(document.get("stage", "unknown")),
        verified_at=_utc_timestamp(verified_at),
    )
    files_value = document.get("files")
    entries = files_value if isinstance(files_value, list) else []
    result.expected_files = len(entries)

    _verify_manifest_structure(document, result)
    expected_manifest_sha = document.get("manifest_sha256")
    actual_manifest_sha = _manifest_sha256(document)
    if expected_manifest_sha != actual_manifest_sha:
        result.issues.append(
            VerificationIssue(
                code="manifest_digest_mismatch",
                message="Manifest content does not match manifest_sha256.",
                expected=expected_manifest_sha,
                actual=actual_manifest_sha,
            )
        )

    if isinstance(files_value, list):
        expected_dataset_sha = document.get("dataset_sha256")
        actual_dataset_sha = _sha256_json(
            {
                "stage": str(document.get("stage", "data")),
                "files": files_value,
                "upstream": _normalized_upstream(document.get("upstream", ())),
            }
        )
        if expected_dataset_sha != actual_dataset_sha:
            result.issues.append(
                VerificationIssue(
                    code="dataset_digest_mismatch",
                    message="Manifest file entries do not match dataset_sha256.",
                    expected=expected_dataset_sha,
                    actual=actual_dataset_sha,
                )
            )

    declared_count = document.get("file_count")
    if declared_count != len(entries):
        result.issues.append(
            VerificationIssue(
                code="file_count_mismatch",
                message="file_count does not equal the number of file entries.",
                expected=declared_count,
                actual=len(entries),
            )
        )
    declared_size = document.get("total_size_bytes")
    entry_size = sum(
        entry.get("size_bytes", 0)
        for entry in entries
        if isinstance(entry, Mapping) and isinstance(entry.get("size_bytes"), int)
    )
    if declared_size != entry_size:
        result.issues.append(
            VerificationIssue(
                code="total_size_mismatch",
                message="total_size_bytes does not equal the sum of file entries.",
                expected=declared_size,
                actual=entry_size,
            )
        )

    root_path = Path(root).resolve()
    if not root_path.is_dir():
        result.issues.append(
            VerificationIssue(
                code="missing_root",
                message="Verification root does not exist or is not a directory.",
                path=str(root),
            )
        )
        return result

    seen_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            result.issues.append(
                VerificationIssue(
                    code="invalid_file_entry",
                    message="Every item in files must be an object.",
                    actual=type(entry).__name__,
                )
            )
            continue
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative:
            result.issues.append(
                VerificationIssue(
                    code="invalid_file_path",
                    message="File entry path must be a non-empty string.",
                    actual=relative,
                )
            )
            continue
        if relative in seen_paths:
            result.issues.append(
                VerificationIssue(
                    code="duplicate_manifest_path",
                    message="Manifest contains the same relative path more than once.",
                    path=relative,
                )
            )
            continue
        seen_paths.add(relative)
        file_path = (root_path / relative).resolve()
        if not _is_within(file_path, root_path):
            result.issues.append(
                VerificationIssue(
                    code="unsafe_manifest_path",
                    message="Manifest path escapes the verification root.",
                    path=relative,
                )
            )
            continue
        if not file_path.is_file():
            result.issues.append(
                VerificationIssue(
                    code="missing_file",
                    message="Manifested file is absent.",
                    path=relative,
                )
            )
            continue

        result.checked_files += 1
        actual_size = int(file_path.stat().st_size)
        expected_size = entry.get("size_bytes")
        if actual_size != expected_size:
            result.issues.append(
                VerificationIssue(
                    code="size_mismatch",
                    message="File size differs from the manifest.",
                    path=relative,
                    expected=expected_size,
                    actual=actual_size,
                )
            )
        actual_hash = sha256_file(file_path)
        expected_hash = entry.get("sha256")
        if actual_hash != expected_hash:
            result.issues.append(
                VerificationIssue(
                    code="sha256_mismatch",
                    message="File content differs from the manifest.",
                    path=relative,
                    expected=expected_hash,
                    actual=actual_hash,
                )
            )
        if check_metadata and any(
            key in entry for key in ("format", "row_count", "column_count", "schema", "metadata_error")
        ):
            current_metadata = _tabular_metadata(file_path)
            for key in ("format", "row_count", "column_count", "schema", "metadata_error"):
                if key in entry and entry.get(key) != current_metadata.get(key):
                    result.issues.append(
                        VerificationIssue(
                            code=f"{key}_mismatch",
                            message=f"Current table {key} differs from the manifest.",
                            path=relative,
                            expected=entry.get(key),
                            actual=current_metadata.get(key),
                        )
                    )

    if not allow_extra:
        current_files = {
            _relative_posix(path, root_path)
            for path in _walk_files(root_path)
            if manifest_path is None or path.resolve() != manifest_path
        }
        for relative in sorted(current_files - seen_paths):
            result.issues.append(
                VerificationIssue(
                    code="unexpected_file",
                    message="File is present below the root but absent from the manifest.",
                    path=relative,
                )
            )

    result.issues.sort(key=lambda issue: (issue.code, issue.path or ""))
    return result


def verify_data_manifests(
    manifests: Mapping[str, Mapping[str, Any] | PathLike],
    *,
    raw_root: PathLike,
    processed_root: PathLike,
    check_metadata: bool = True,
    allow_extra: bool = True,
    verified_at: str | datetime | None = None,
) -> dict[str, ManifestVerification]:
    """Verify linked raw and processed manifests with their respective roots."""

    roots = {"raw": raw_root, "processed": processed_root}
    results: dict[str, ManifestVerification] = {}
    for stage, root in roots.items():
        if stage not in manifests:
            results[stage] = ManifestVerification(
                stage=stage,
                issues=[
                    VerificationIssue(
                        code="missing_stage_manifest",
                        message=f"No {stage!r} manifest was supplied.",
                    )
                ],
                verified_at=_utc_timestamp(verified_at),
            )
            continue
        results[stage] = verify_manifest(
            manifests[stage],
            root=root,
            check_metadata=check_metadata,
            allow_extra=allow_extra,
            verified_at=verified_at,
        )

    raw_document = _manifest_document(manifests.get("raw"))
    processed_document = _manifest_document(manifests.get("processed"))
    if raw_document and processed_document:
        raw_digest = raw_document.get("dataset_sha256")
        upstream = processed_document.get("upstream", [])
        linked = any(
            isinstance(item, Mapping)
            and item.get("stage") == "raw"
            and item.get("dataset_sha256") == raw_digest
            for item in upstream
        )
        if not linked:
            results["processed"].issues.append(
                VerificationIssue(
                    code="upstream_manifest_mismatch",
                    message="Processed manifest does not reference the supplied raw dataset digest.",
                    expected=raw_digest,
                    actual=upstream,
                )
            )
        if raw_document.get("build_id") != processed_document.get("build_id"):
            results["processed"].issues.append(
                VerificationIssue(
                    code="build_id_mismatch",
                    message="Raw and processed manifests have different build identifiers.",
                    expected=raw_document.get("build_id"),
                    actual=processed_document.get("build_id"),
                )
            )
    return results


def _resolve_root_and_files(
    paths: PathLike | Iterable[PathLike] | None,
    root: PathLike | None,
) -> tuple[Path, list[Path]]:
    if paths is None:
        if root is None:
            raise ValueError("Either paths or root must be supplied")
        items = [Path(root)]
    elif isinstance(paths, (str, os.PathLike)):
        items = [Path(paths)]
    else:
        items = [Path(item) for item in paths]
    if not items:
        if root is None:
            raise ValueError("root is required when paths is empty")
        items = []

    root_path = Path(root).resolve() if root is not None else _infer_root(items)
    files: list[Path] = []
    for item in items:
        candidate_unresolved = (
            item if item.is_absolute() else (root_path / item if root is not None else item)
        )
        if candidate_unresolved.is_symlink():
            raise ValueError(f"Symlinks are not supported in manifests: {candidate_unresolved}")
        candidate = candidate_unresolved.resolve()
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        if not _is_within(candidate, root_path):
            raise ValueError(f"Path is outside manifest root: {candidate}")
        if candidate.is_dir():
            files.extend(_walk_files(candidate))
        elif candidate.is_file():
            files.append(candidate)

    unique = sorted({file.resolve() for file in files}, key=lambda path: _relative_posix(path, root_path))
    return root_path, unique


def _infer_root(items: Sequence[Path]) -> Path:
    if len(items) == 1 and items[0].exists() and items[0].is_dir():
        return items[0].resolve()
    absolute = [item.resolve() for item in items]
    parents = [item if item.is_dir() else item.parent for item in absolute]
    return Path(os.path.commonpath([str(parent) for parent in parents])).resolve()


def _walk_files(directory: Path) -> list[Path]:
    files: list[Path] = []
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not supported in manifests: {path}")
        if path.is_file():
            files.append(path.resolve())
    return files


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path is outside manifest root: {path}") from exc


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _tabular_metadata(path: Path) -> dict[str, Any]:
    suffix = path.suffix.casefold()
    if suffix not in DEFAULT_TABULAR_EXTENSIONS:
        return {}
    try:
        if suffix in {".csv", ".tsv", ".tab"}:
            frame = pd.read_csv(
                path,
                sep="\t" if suffix in {".tsv", ".tab"} else ",",
                low_memory=False,
            )
            return _frame_metadata(frame, suffix.lstrip("."))
        if suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(path)
            return _frame_metadata(frame, suffix.lstrip("."))
        if suffix == ".parquet":
            frame = pd.read_parquet(path)
            return _frame_metadata(frame, "parquet")
        if suffix in {".jsonl", ".ndjson"}:
            records = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        records.append(json.loads(line))
            return _json_records_metadata(records, suffix.lstrip("."))
        value = json.loads(path.read_text(encoding="utf-8"))
        return _json_metadata(value)
    except Exception as exc:  # A malformed/unsupported table is itself provenance metadata.
        return {
            "format": suffix.lstrip("."),
            "metadata_error": type(exc).__name__,
        }


def _frame_metadata(frame: pd.DataFrame, format_name: str) -> dict[str, Any]:
    schema = []
    for column in sorted(frame.columns, key=lambda value: str(value)):
        series = frame[column]
        schema.append(
            {
                "name": str(column),
                "dtype": _portable_dtype(series),
                "nullable": bool(series.isna().any()),
            }
        )
    return {
        "format": format_name,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "schema": schema,
    }


def _portable_dtype(series: pd.Series) -> str:
    dtype = series.dtype
    if ptypes.is_bool_dtype(dtype):
        return "boolean"
    if ptypes.is_integer_dtype(dtype):
        return "integer"
    if ptypes.is_float_dtype(dtype):
        return "number"
    if ptypes.is_datetime64_any_dtype(dtype):
        return "datetime"
    if ptypes.is_string_dtype(dtype):
        return "string"
    return "object"


def _json_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        return _json_records_metadata(value, "json")
    if isinstance(value, dict):
        # Recognize column-oriented JSON only when all values are equal-length lists.
        if value and all(isinstance(item, list) for item in value.values()):
            lengths = {len(item) for item in value.values()}
            if len(lengths) == 1:
                frame = pd.DataFrame(value)
                return _frame_metadata(frame, "json")
        schema = [
            {
                "name": str(key),
                "dtype": _json_type(item),
                "nullable": item is None,
            }
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        ]
        return {
            "format": "json",
            "row_count": 1,
            "column_count": len(schema),
            "schema": schema,
        }
    return {
        "format": "json",
        "row_count": 1,
        "column_count": 1,
        "schema": [{"name": "value", "dtype": _json_type(value), "nullable": value is None}],
    }


def _json_records_metadata(records: list[Any], format_name: str) -> dict[str, Any]:
    if not records:
        return {"format": format_name, "row_count": 0, "column_count": 0, "schema": []}
    if not all(isinstance(record, dict) for record in records):
        types = sorted({_json_type(record) for record in records})
        return {
            "format": format_name,
            "row_count": len(records),
            "column_count": 1,
            "schema": [
                {
                    "name": "value",
                    "dtype": "|".join(types),
                    "nullable": any(record is None for record in records),
                }
            ],
        }
    keys = sorted({str(key) for record in records for key in record})
    schema = []
    for key in keys:
        values = [record.get(key) for record in records]
        types = sorted({_json_type(value) for value in values if value is not None})
        schema.append(
            {
                "name": key,
                "dtype": "|".join(types) if types else "null",
                "nullable": any(key not in record or record.get(key) is None for record in records),
            }
        )
    return {
        "format": format_name,
        "row_count": len(records),
        "column_count": len(schema),
        "schema": schema,
    }


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _normalized_upstream(upstream: Any) -> list[dict[str, Any]]:
    if upstream is None:
        return []
    if isinstance(upstream, Mapping):
        values = [upstream]
    else:
        values = list(upstream)
    normalized = [
        {str(key): _json_value(value) for key, value in sorted(item.items())}
        for item in values
        if isinstance(item, Mapping)
    ]
    normalized.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    return normalized


def _verify_manifest_structure(document: Mapping[str, Any], result: ManifestVerification) -> None:
    required = (
        "manifest_version",
        "stage",
        "created_at",
        "build_id",
        "root",
        "file_count",
        "total_size_bytes",
        "dataset_sha256",
        "files",
        "manifest_sha256",
    )
    for field_name in required:
        if field_name not in document:
            result.issues.append(
                VerificationIssue(
                    code="missing_manifest_field",
                    message=f"Required manifest field {field_name!r} is absent.",
                    path=field_name,
                )
            )
    if document.get("manifest_version") != MANIFEST_VERSION:
        result.issues.append(
            VerificationIssue(
                code="unsupported_manifest_version",
                message="Manifest version is not supported by this verifier.",
                expected=MANIFEST_VERSION,
                actual=document.get("manifest_version"),
            )
        )
    expected_types = {
        "manifest_version": str,
        "stage": str,
        "created_at": str,
        "build_id": str,
        "root": str,
        "file_count": int,
        "total_size_bytes": int,
        "dataset_sha256": str,
        "files": list,
        "manifest_sha256": str,
    }
    for field_name, expected_type in expected_types.items():
        if field_name in document and not isinstance(document[field_name], expected_type):
            result.issues.append(
                VerificationIssue(
                    code="invalid_manifest_field_type",
                    message=f"Manifest field {field_name!r} has the wrong type.",
                    path=field_name,
                    expected=expected_type.__name__,
                    actual=type(document[field_name]).__name__,
                )
            )
    for field_name in ("dataset_sha256", "manifest_sha256"):
        value = document.get(field_name)
        if not isinstance(value, str) or not _is_sha256(value):
            result.issues.append(
                VerificationIssue(
                    code="invalid_digest",
                    message=f"{field_name} must be a 64-character hexadecimal SHA-256 digest.",
                    path=field_name,
                    actual=value,
                )
            )


def _is_sha256(value: str) -> bool:
    """Return whether *value* is a canonical lowercase SHA-256 hex digest."""

    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _finalize_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(manifest)
    result.pop("manifest_sha256", None)
    result["manifest_sha256"] = _manifest_sha256(result)
    return result


def _manifest_sha256(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return _sha256_json(payload)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_document(value: Mapping[str, Any] | PathLike | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    return load_manifest(value)


def _utc_timestamp(value: str | datetime | None = None) -> str:
    if value is None:
        source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
        if source_date_epoch:
            moment = datetime.fromtimestamp(int(source_date_epoch), tz=timezone.utc)
        else:
            moment = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        moment = value
    else:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    return str(value)


# Intuitive alias for callers that prefer "build" terminology.
build_manifest = create_manifest


__all__ = [
    "MANIFEST_VERSION",
    "ManifestVerification",
    "VerificationIssue",
    "build_manifest",
    "create_data_manifests",
    "create_directory_manifest",
    "create_manifest",
    "inspect_file",
    "load_manifest",
    "sha256_file",
    "verify_data_manifests",
    "verify_manifest",
    "write_data_manifests",
    "write_manifest",
]
