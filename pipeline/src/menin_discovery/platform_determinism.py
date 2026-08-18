"""Independent content-equivalence verification for two canonical builds."""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .platform_data_schema import canonical_json
from .platform_data_sources import sha256_file

SCHEMA_VERSION = "platform_canonical_determinism_verification_v1"
BUILD_NONDETERMINISTIC_FIELDS = ("built_at_utc",)
QC_NONDETERMINISTIC_FIELDS = ("generated_at_utc", "build_manifest_sha256")


def _document(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _relative(value: object) -> str:
    text = str(value).strip()
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe artifact path: {text!r}")
    return path.as_posix()


def _verified_root(value: str | os.PathLike[str], label: str) -> Path:
    raw = Path(os.path.abspath(value))
    current = Path(raw.anchor)
    for part in raw.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} path chain contains a symlink: {current}")
    resolved = raw.resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    for path in resolved.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"{label} tree contains a symlink: {path}")
    return resolved


def _normalized(document: Mapping[str, Any], ignored: Sequence[str]) -> dict[str, Any]:
    result = copy.deepcopy(dict(document))
    for field in ignored:
        if field not in result:
            raise ValueError(f"Declared nondeterministic field is absent: {field}")
        result.pop(field)
    return result


def _verify_build(root: Path) -> tuple[dict[str, Any], str, dict[str, dict[str, Any]]]:
    resolved = _verified_root(root, "canonical build")
    manifest_path = resolved / "build_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(manifest_path)
    manifest = _document(manifest_path)
    raw_inventory = manifest.get("component_inventory")
    if not isinstance(raw_inventory, list) or not raw_inventory:
        raise ValueError("Canonical build lacks a component inventory")
    inventory: dict[str, dict[str, Any]] = {}
    for raw in raw_inventory:
        if not isinstance(raw, Mapping):
            raise ValueError("Malformed canonical component inventory record")
        relative = _relative(raw.get("path"))
        if relative in inventory:
            raise ValueError(f"Duplicate canonical component: {relative}")
        inventory[relative] = dict(raw)
    actual: set[str] = set()
    for path in resolved.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Canonical build contains a symlink: {path}")
        if path.is_file() and path != manifest_path:
            actual.add(path.relative_to(resolved).as_posix())
    if actual != set(inventory):
        raise ValueError(
            "Canonical component membership differs from its manifest; "
            f"unbound={sorted(actual - set(inventory))}, missing={sorted(set(inventory) - actual)}"
        )
    for relative, record in sorted(inventory.items()):
        path = resolved / relative
        if path.stat().st_size != int(record.get("size_bytes", -1)):
            raise ValueError(f"Canonical component size mismatch: {relative}")
        if sha256_file(path) != str(record.get("sha256", "")):
            raise ValueError(f"Canonical component hash mismatch: {relative}")
    return manifest, sha256_file(manifest_path), inventory


def _artifact_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        if {"path", "sha256", "size_bytes"}.issubset(value):
            records.append(dict(value))
        else:
            for child in value.values():
                records.extend(_artifact_records(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(_artifact_records(child))
    return records


def _verify_report_records(root: Path, qc: Mapping[str, Any]) -> set[str]:
    expected: set[str] = set()
    for record in _artifact_records({"artifacts": qc.get("artifacts"), "figures": qc.get("figures")}):
        relative = _relative(record["path"])
        if relative in expected:
            raise ValueError(f"Duplicate QC artifact record: {relative}")
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"QC artifact size mismatch: {relative}")
        if sha256_file(path) != str(record["sha256"]):
            raise ValueError(f"QC artifact hash mismatch: {relative}")
        expected.add(relative)
    return expected


def compare_canonical_builds(
    build_a: str | os.PathLike[str],
    reports_a: str | os.PathLike[str],
    build_b: str | os.PathLike[str],
    reports_b: str | os.PathLike[str],
) -> dict[str, Any]:
    """Fail closed unless two builds differ only in two declared timestamp bindings."""

    root_a = _verified_root(build_a, "canonical build A")
    root_b = _verified_root(build_b, "canonical build B")
    if root_a == root_b:
        raise ValueError("Determinism verification requires two distinct build roots")
    manifest_a, manifest_sha_a, inventory_a = _verify_build(root_a)
    manifest_b, manifest_sha_b, inventory_b = _verify_build(root_b)
    if set(inventory_a) != set(inventory_b):
        raise ValueError("Canonical builds have different component memberships")
    component_bytes = 0
    for relative in sorted(inventory_a):
        path_a = root_a / relative
        path_b = root_b / relative
        if path_a.stat().st_size != path_b.stat().st_size:
            raise ValueError(f"Canonical builds differ in component bytes: {relative}")
        digest_a = sha256_file(path_a)
        digest_b = sha256_file(path_b)
        if digest_a != digest_b:
            raise ValueError(f"Canonical builds differ in component content: {relative}")
        component_bytes += path_a.stat().st_size
    normalized_manifest_a = _normalized(manifest_a, BUILD_NONDETERMINISTIC_FIELDS)
    normalized_manifest_b = _normalized(manifest_b, BUILD_NONDETERMINISTIC_FIELDS)
    if normalized_manifest_a != normalized_manifest_b:
        raise ValueError("Canonical build manifests differ beyond built_at_utc")

    report_root_a = _verified_root(reports_a, "canonical reports A")
    report_root_b = _verified_root(reports_b, "canonical reports B")
    if report_root_a == report_root_b:
        raise ValueError("Determinism verification requires two distinct report roots")
    qc_path_a = report_root_a / "qc_report.json"
    qc_path_b = report_root_b / "qc_report.json"
    qc_a = _document(qc_path_a)
    qc_b = _document(qc_path_b)
    for qc, manifest_sha, label in (
        (qc_a, manifest_sha_a, "A"),
        (qc_b, manifest_sha_b, "B"),
    ):
        if qc.get("qc_passed") is not True:
            raise ValueError(f"Canonical QC {label} did not pass")
        if qc.get("build_manifest_sha256") != manifest_sha:
            raise ValueError(f"Canonical QC {label} is not bound to its build manifest")
    declared_a = _verify_report_records(report_root_a, qc_a)
    declared_b = _verify_report_records(report_root_b, qc_b)
    if declared_a != declared_b:
        raise ValueError("Canonical QC reports declare different artifact memberships")
    generated = declared_b | {
        "qc_report.json",
        "eda_summary.json",
        "data_bulk_canonical_manifest.json",
    }
    observed_b = {
        path.relative_to(report_root_b).as_posix() for path in report_root_b.rglob("*") if path.is_file()
    }
    if observed_b != generated:
        raise ValueError(
            "Determinism report root has unexpected membership; "
            f"unbound={sorted(observed_b - generated)}, missing={sorted(generated - observed_b)}"
        )
    if not generated.issubset(
        {path.relative_to(report_root_a).as_posix() for path in report_root_a.rglob("*") if path.is_file()}
    ):
        raise ValueError("Primary report root is missing generated canonical QC artifacts")
    for relative in sorted(generated - {"qc_report.json", "data_bulk_canonical_manifest.json"}):
        if sha256_file(report_root_a / relative) != sha256_file(report_root_b / relative):
            raise ValueError(f"Canonical QC artifact differs across builds: {relative}")
    normalized_qc_a = _normalized(qc_a, QC_NONDETERMINISTIC_FIELDS)
    normalized_qc_b = _normalized(qc_b, QC_NONDETERMINISTIC_FIELDS)
    if normalized_qc_a != normalized_qc_b:
        raise ValueError("Canonical QC reports differ beyond their declared timestamp/build binding")
    report_manifest_a = _document(report_root_a / "data_bulk_canonical_manifest.json")
    report_manifest_b = _document(report_root_b / "data_bulk_canonical_manifest.json")
    if sha256_file(report_root_a / "data_bulk_canonical_manifest.json") != manifest_sha_a:
        raise ValueError("Canonical report manifest A is not the physical build manifest A")
    if sha256_file(report_root_b / "data_bulk_canonical_manifest.json") != manifest_sha_b:
        raise ValueError("Canonical report manifest B is not the physical build manifest B")
    if _normalized(report_manifest_a, BUILD_NONDETERMINISTIC_FIELDS) != _normalized(
        report_manifest_b, BUILD_NONDETERMINISTIC_FIELDS
    ):
        raise ValueError("Canonical report manifests differ beyond built_at_utc")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_content_equivalent",
        "content_equivalent": True,
        "canonical_component_count": len(inventory_a),
        "canonical_component_bytes": component_bytes,
        "qc_generated_artifact_count": len(generated),
        "build_manifest_a_sha256": manifest_sha_a,
        "build_manifest_b_sha256": manifest_sha_b,
        "normalized_build_manifest_sha256": sha256_text(canonical_json(normalized_manifest_a)),
        "qc_report_a_sha256": sha256_file(qc_path_a),
        "qc_report_b_sha256": sha256_file(qc_path_b),
        "normalized_qc_report_sha256": sha256_text(canonical_json(normalized_qc_a)),
        "ignored_nondeterministic_fields": {
            "build_manifest": list(BUILD_NONDETERMINISTIC_FIELDS),
            "qc_report": list(QC_NONDETERMINISTIC_FIELDS),
        },
        "large_model_training_started": False,
        "substantive_training_started": False,
    }


def sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_determinism_report(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(dict(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-a", type=Path, required=True)
    parser.add_argument("--reports-a", type=Path, required=True)
    parser.add_argument("--build-b", type=Path, required=True)
    parser.add_argument("--reports-b", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/reports/platform/canonical_determinism_verification.json"),
    )
    arguments = parser.parse_args(argv)
    result = compare_canonical_builds(
        arguments.build_a,
        arguments.reports_a,
        arguments.build_b,
        arguments.reports_b,
    )
    write_determinism_report(arguments.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["SCHEMA_VERSION", "compare_canonical_builds", "main", "write_determinism_report"]
