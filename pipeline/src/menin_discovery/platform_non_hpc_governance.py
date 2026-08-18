"""Deterministic non-HPC governance, release, and resource-readiness audit.

This module closes automatable governance work without granting rights, choosing
a scientific task, opening a test lockbox, downloading a large checkpoint, or
starting substantive training. Human decisions remain explicit failed gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "platform_non_hpc_governance_v1"
MANIFEST_SCHEMA_VERSION = "platform_non_hpc_governance_manifest_v1"
NO_TRAINING = False
TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
SCAN_ROOTS = (".github", "docs", "packages", "pipeline", "README.md", "CITATION.cff")
EXCLUDED_WALK_DIRS = {
    ".git",
    ".venv",
    ".tmp",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
PERSONAL_PATH_PATTERN = re.compile(
    r"(?:/Users/[A-Za-z0-9._-]+/|/home/[A-Za-z0-9._-]+/|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+\\)"
)
SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_personal_token": re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
    "openai_style_secret": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str, field: str) -> str:
    if not value or "\\" in value or any(ord(character) < 32 for character in value):
        raise ValueError(f"Unsafe {field}: {value!r}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"Unsafe {field}: {value!r}")
    return value


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if payload.get("substantive_training_started") is not NO_TRAINING:
        raise ValueError("Every governance JSON must state substantive_training_started=false")
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _run_git(project_root: Path, arguments: Sequence[str]) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _git_status(project_root: Path) -> list[dict[str, str]]:
    raw = _run_git(project_root, ["status", "--porcelain=v1", "-z", "--untracked-files=all"])
    items = raw.split(b"\0")
    records: list[dict[str, str]] = []
    index = 0
    while index < len(items):
        item = items[index]
        index += 1
        if not item:
            continue
        if len(item) < 4:
            raise ValueError("Malformed git status record")
        code = item[:2].decode("ascii")
        path = item[3:].decode("utf-8", errors="surrogateescape")
        record = {"code": code, "path": path}
        if "R" in code or "C" in code:
            if index >= len(items) or not items[index]:
                raise ValueError("Malformed git rename/copy status record")
            record["original_path"] = items[index].decode("utf-8", errors="surrogateescape")
            index += 1
        records.append(record)
    return records


def _git_visible_paths(project_root: Path) -> list[str]:
    raw = _run_git(project_root, ["ls-files", "-co", "--exclude-standard", "-z"])
    return sorted({item.decode("utf-8", errors="surrogateescape") for item in raw.split(b"\0") if item})


def _under_scan_root(relative: str) -> bool:
    return any(relative == root or relative.startswith(f"{root}/") for root in SCAN_ROOTS)


def _scan_text_surfaces(project_root: Path, visible_paths: Sequence[str]) -> dict[str, Any]:
    personal_paths: list[dict[str, Any]] = []
    secrets: list[dict[str, Any]] = []
    scanned = 0
    for relative in visible_paths:
        if not _under_scan_root(relative):
            continue
        path = project_root / relative
        if path.is_symlink() or not path.is_file() or path.suffix.casefold() not in TEXT_SUFFIXES:
            continue
        if path.stat().st_size > 5 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        scanned += 1
        for line_number, line in enumerate(text.splitlines(), start=1):
            if PERSONAL_PATH_PATTERN.search(line):
                personal_paths.append({"path": relative, "line": line_number})
            for category, pattern in SECRET_PATTERNS.items():
                if pattern.search(line):
                    secrets.append({"category": category, "path": relative, "line": line_number})
    return {
        "files_scanned": scanned,
        "personal_path_findings": personal_paths,
        "high_confidence_secret_findings": secrets,
    }


def _walk_repository(project_root: Path) -> dict[str, Any]:
    large_files: list[dict[str, Any]] = []
    symlinks: list[str] = []
    special_files: list[str] = []
    transient_paths: list[str] = []
    cache_paths: list[str] = []
    for current, directories, filenames in os.walk(project_root, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(name for name in directories if name not in EXCLUDED_WALK_DIRS)
        for name in sorted(directories + filenames):
            path = current_path / name
            relative = path.relative_to(project_root).as_posix()
            if path.is_symlink():
                symlinks.append(relative)
                continue
            if name.startswith(".building") or name.endswith((".building", ".partial")):
                transient_paths.append(relative)
            if name in {".coverage", "coverage.xml"} or name.endswith(".pyc"):
                cache_paths.append(relative)
            if name in {".cache", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"}:
                cache_paths.append(relative)
            try:
                mode = path.lstat().st_mode
            except FileNotFoundError:
                continue
            if path.is_file():
                size = path.stat().st_size
                if size > 50 * 1024 * 1024:
                    ignored = (
                        subprocess.run(
                            ["git", "check-ignore", "--quiet", "--", relative],
                            cwd=project_root,
                            check=False,
                        ).returncode
                        == 0
                    )
                    large_files.append({"ignored": ignored, "path": relative, "size_bytes": size})
            elif not path.is_dir() and not stat.S_ISLNK(mode):
                special_files.append(relative)
    return {
        "large_files_over_50_mib": large_files,
        "symlinks": sorted(symlinks),
        "special_files": sorted(special_files),
        "transient_build_or_partial_paths": sorted(transient_paths),
        "cache_or_bytecode_paths": sorted(set(cache_paths)),
    }


def _tree_summary(root: Path) -> dict[str, int]:
    files = 0
    bytes_total = 0
    if not root.exists():
        return {"file_count": 0, "size_bytes": 0}
    for current, directories, names in os.walk(root, followlinks=False):
        directories[:] = sorted(name for name in directories if name not in EXCLUDED_WALK_DIRS)
        current_path = Path(current)
        for name in names:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                continue
            files += 1
            bytes_total += path.stat().st_size
    return {"file_count": files, "size_bytes": bytes_total}


def _tmp_summary(project_root: Path) -> dict[str, Any]:
    root = project_root / ".tmp"
    summary = _tree_summary(root)
    symlinks = 0
    if root.exists():
        for current, directories, names in os.walk(root, followlinks=False):
            current_path = Path(current)
            symlinks += sum((current_path / name).is_symlink() for name in directories + names)
    return {"policy": "preserve_user_scratch_excluded_from_release", "symlink_count": symlinks, **summary}


def load_and_validate_config(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != "platform-non-hpc-readiness-1.0":
        raise ValueError("Unsupported non-HPC readiness configuration")
    training = document.get("training")
    external = document.get("external_admission")
    release = document.get("release")
    compute = document.get("compute")
    if not all(isinstance(item, Mapping) for item in (training, external, release, compute)):
        raise ValueError("Configuration is missing a required policy mapping")
    assert isinstance(training, Mapping)
    assert isinstance(external, Mapping)
    assert isinstance(release, Mapping)
    assert isinstance(compute, Mapping)
    if training.get("substantive_large_model_training_authorized") is not False:
        raise ValueError("Substantive training must remain unauthorized")
    if training.get("substantive_large_model_training_started") is not False:
        raise ValueError("Substantive training must remain unstarted")
    if int(training.get("allowed_smoke_maximum_parameters", 0)) > 100_000:
        raise ValueError("Smoke parameter cap exceeds the governing boundary")
    if int(training.get("allowed_smoke_maximum_steps", 0)) > 2:
        raise ValueError("Smoke step cap exceeds the governing boundary")
    if int(external.get("canonical_observations_admitted", -1)) != 0:
        raise ValueError("External canonical admission is not authorized")
    if int(external.get("model_labels_admitted", -1)) != 0:
        raise ValueError("External model-label admission is not authorized")
    if release.get("public_release_approved") is not False:
        raise ValueError("Public release cannot be marked approved by this configuration")
    if compute.get("hpc_allocation_approved") is not False:
        raise ValueError("HPC allocation requires a separate human authorization")
    return document


def build_governance_report(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    if not (root / ".git").exists():
        raise ValueError("project_root must be a Git working tree")
    config = load_and_validate_config(config_path)
    status = _git_status(root)
    visible_paths = _git_visible_paths(root)
    status_counts = Counter(record["code"] for record in status)
    staged = [record for record in status if record["code"] != "??" and record["code"][0] not in {" ", "?"}]
    walk = _walk_repository(root)
    text_scan = _scan_text_surfaces(root, visible_paths)
    top_level_notices = sorted(
        path.name
        for path in root.iterdir()
        if path.is_file()
        and (
            path.name.casefold().startswith("license")
            or path.name.casefold().startswith("copying")
            or path.name.casefold().startswith("notice")
        )
    )
    nonignored_large = [item for item in walk["large_files_over_50_mib"] if not item["ignored"]]
    automatable_pass = (
        not staged
        and not nonignored_large
        and not text_scan["personal_path_findings"]
        and not text_scan["high_confidence_secret_findings"]
        and not walk["special_files"]
        and not walk["transient_build_or_partial_paths"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_date": str(config["evidence_date"]),
        "config": {
            "path": config_path.resolve().relative_to(root).as_posix(),
            "sha256": _sha256(config_path),
            "training_policy_validated": True,
        },
        "environment": {
            "machine": platform.machine(),
            "operating_system": platform.platform(),
            "python": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
            "filesystem_total_bytes": shutil.disk_usage(root).total,
        },
        "git_release_inventory": {
            "status_counts": dict(sorted(status_counts.items())),
            "status_record_count": len(status),
            "status_records": status,
            "staged_record_count": len(staged),
            "staged_records": staged,
            "git_visible_path_count": len(visible_paths),
        },
        "release_hygiene": {
            "automatable_checks_passed": automatable_pass,
            "top_level_license_copying_notice_files": top_level_notices,
            "repository_license_present": any(
                name.casefold().startswith(("license", "copying")) for name in top_level_notices
            ),
            "nonignored_large_file_count": len(nonignored_large),
            "text_scan": text_scan,
            **walk,
            "preserved_tmp_surface": _tmp_summary(root),
        },
        "artifact_surfaces": {
            "platform_data": _tree_summary(root / "research/data/platform"),
            "platform_models": _tree_summary(root / "research/models/platform"),
            "platform_reports": _tree_summary(root / "research/reports/platform"),
        },
        "decision_gates": {
            "automatable_release_hygiene": automatable_pass,
            "clean_committed_candidate": len(status) == 0,
            "repository_license_approved": False,
            "source_and_model_redistribution_approved": False,
            "external_canonical_admission_approved": False,
            "intended_task_and_claim_approved": False,
            "exact_checkpoint_and_overlap_approved": False,
            "hpc_and_operations_approved": False,
            "independent_external_validation_accepted": False,
            "public_release_ready": False,
            "scientific_task_claim_ready": False,
            "substantive_large_model_training_ready": False,
            "substantive_large_model_training_authorized": False,
        },
        "human_actions_remaining": [
            "review and approve the exact staged migration inventory",
            "select repository, documentation, model, and per-source redistribution licenses",
            "approve artifact storage and public release scope",
            "approve one intended scientific task, claim, and leakage thresholds",
            "freeze exact model checkpoint, hashes, terms, cutoff, and overlap evidence",
            "approve HPC budget, monitoring, checkpoint/resume, retention, and responsible use",
            "accept independent external or prospective validation before broad claims",
        ],
        "large_model_training_started": False,
        "substantive_training_started": False,
        "training_actions": [],
    }


def _render_summary(report: Mapping[str, Any]) -> str:
    git = report["git_release_inventory"]
    hygiene = report["release_hygiene"]
    gates = report["decision_gates"]
    surfaces = report["artifact_surfaces"]
    lines = [
        "# Non-HPC governance and release-readiness snapshot",
        "",
        f"Evidence date: {report['evidence_date']}.",
        "",
        "## Outcome",
        "",
        "All automatable work in this report is diagnostic. It grants no license, scientific approval, "
        "public-release approval, checkpoint approval, HPC allocation, or training authorization.",
        "",
        f"- Git status records: {git['status_record_count']}; staged records: {git['staged_record_count']}.",
        f"- Repository license present: {str(hygiene['repository_license_present']).lower()}.",
        f"- Nonignored files over 50 MiB: {hygiene['nonignored_large_file_count']}.",
        f"- Personal-path findings: {len(hygiene['text_scan']['personal_path_findings'])}.",
        f"- High-confidence secret findings: {len(hygiene['text_scan']['high_confidence_secret_findings'])}.",
        f"- Public release ready: {str(gates['public_release_ready']).lower()}.",
        f"- Scientific task claim ready: {str(gates['scientific_task_claim_ready']).lower()}.",
        "- Substantive large-model training ready/authorized/started: false / false / false.",
        "",
        "## Platform storage",
        "",
    ]
    for name, summary in surfaces.items():
        lines.append(f"- `{name}`: {summary['file_count']} files / {summary['size_bytes']} bytes.")
    lines.extend(["", "## Human gates", ""])
    lines.extend(f"- {action}." for action in report["human_actions_remaining"])
    lines.append("")
    return "\n".join(lines)


def _write_release_inventory(path: Path, report: Mapping[str, Any]) -> None:
    rows = report["git_release_inventory"]["status_records"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["code", "path", "original_path"])
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "code": row["code"],
                        "path": row["path"],
                        "original_path": row.get("original_path", ""),
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_governance_bundle(project_root: Path, config_path: Path, output_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    output = output_root.resolve()
    try:
        output.relative_to(root)
    except ValueError as error:
        raise ValueError("output_root must stay inside project_root") from error
    report = build_governance_report(root, config_path.resolve())
    report_path = output / "non_hpc_governance_report.json"
    summary_path = output / "non_hpc_governance_summary.md"
    inventory_path = output / "release_inventory.csv"
    _atomic_json(report_path, report)
    _atomic_text(summary_path, _render_summary(report))
    _write_release_inventory(inventory_path, report)
    components = {}
    for path in (report_path, summary_path, inventory_path):
        relative = path.relative_to(root).as_posix()
        components[relative] = {"path": relative, "sha256": _sha256(path), "size_bytes": path.stat().st_size}
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "components": components,
        "component_count": len(components),
        "scientific_task_claim_ready": False,
        "substantive_large_model_training_ready": False,
        "substantive_large_model_training_authorized": False,
        "large_model_training_started": False,
        "substantive_training_started": False,
        "training_actions": [],
    }
    _atomic_json(output / "acceptance_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("pipeline/config/non_hpc_readiness.yaml"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/reports/platform/non_hpc_completion"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    manifest = materialize_governance_bundle(
        arguments.project_root,
        arguments.config,
        arguments.output_root,
    )
    json.dump(manifest, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
