#!/usr/bin/env python3
"""Fail-closed CLI for the persisted DailyMed PK candidate scanner artifacts.

The two ``*_exact.py`` files beside this module are byte-identical copies of
the programs used for the completed six-part scan and merge.  This wrapper
provides a safe scan entry point and an independent, read-only validation
replay for the frozen evidence index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "pk-expansion-dailymed-replay/1.0"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[5]
DEFAULT_EVIDENCE = (
    PROJECT_ROOT
    / "research/data/platform/raw/external_public/pk_expansion/avicenna"
    / "dailymed_pk_candidate_evidence"
)
DEFAULT_SOURCE_MANIFEST = (
    PROJECT_ROOT
    / "research/data/platform/raw/external_public/dailymed_spl_v2_human_rx"
    / "dailymed_spl_v2_human_rx_manifest.json"
)
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "dailymed_pk_candidate_validation_replay.json"
EXACT_SCANNER = SCRIPT_DIR / "dailymed_pk_candidate_scanner_exact.py"
EXACT_MERGER = SCRIPT_DIR / "dailymed_pk_candidate_merge_validator_exact.py"


class ValidationError(RuntimeError):
    """Raised when an artifact contract is missing or inconsistent."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(PROJECT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"cannot parse JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"expected JSON object: {path}")
    return value


def json_lines(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        handle = path.open(encoding="utf-8")
    except Exception as exc:
        raise ValidationError(f"cannot open JSONL {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except Exception as exc:
                raise ValidationError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValidationError(f"{path}:{line_number}: expected object")
            yield line_number, value


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValidationError("required files missing: " + ", ".join(missing))


def artifact_manifest_checks(evidence: Path, manifest: dict[str, Any]) -> dict[str, bool]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValidationError("raw manifest artifacts must be a list")
    declared_paths: set[str] = set()
    total_bytes = 0
    checks: dict[str, bool] = {}
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValidationError("raw manifest contains malformed artifact entry")
        relative = item["path"]
        if relative in declared_paths:
            raise ValidationError(f"duplicate raw manifest artifact: {relative}")
        declared_paths.add(relative)
        path = evidence / relative
        if not path.is_file():
            raise ValidationError(f"declared artifact is missing: {path}")
        actual_bytes = path.stat().st_size
        actual_sha = sha256(path)
        checks[f"artifact::{relative}::bytes"] = actual_bytes == item.get("bytes")
        checks[f"artifact::{relative}::sha256"] = actual_sha == item.get("sha256")
        total_bytes += actual_bytes
    checks["manifest_artifact_count"] = len(artifacts) == manifest.get("artifact_count")
    checks["manifest_total_bytes"] = total_bytes == manifest.get("total_bytes")
    checks["manifest_canonical_rows_zero"] = manifest.get("canonical_rows") == 0
    checks["manifest_training_labels_zero"] = manifest.get("training_labels") == 0
    return checks


def replay_validation(evidence: Path, source_manifest_path: Path) -> dict[str, Any]:
    required = {
        "manifest": evidence / "manifest.json",
        "summary": evidence / "scan_summary.json",
        "validation": evidence / "validation.json",
        "documents": evidence / "document_inventory.jsonl",
        "sections": evidence / "section_candidates.jsonl",
        "tables": evidence / "table_candidates.jsonl",
        "latest": evidence / "latest_available_candidate_documents.jsonl",
        "source_manifest": source_manifest_path,
        "exact_scanner": EXACT_SCANNER,
        "exact_merger_validator": EXACT_MERGER,
    }
    require_files(required.values())
    manifest = read_json(required["manifest"])
    summary = read_json(required["summary"])
    prior_validation = read_json(required["validation"])
    source_manifest = read_json(required["source_manifest"])
    expected = summary.get("counts")
    if not isinstance(expected, dict):
        raise ValidationError("scan summary counts must be an object")

    checks = artifact_manifest_checks(evidence, manifest)
    document_keys: set[tuple[Any, Any, Any]] = set()
    candidate_document_keys: set[tuple[Any, Any, Any]] = set()
    document_ids: set[Any] = set()
    candidate_documents = 0
    parsed_documents = 0
    parse_errors = 0
    clinical_documents = 0
    explicit_pk_documents = 0
    fda_documents = 0
    candidate_fda_documents = 0
    for _, row in json_lines(required["documents"]):
        key = (row.get("archive"), row.get("outer_member"), row.get("inner_xml_member"))
        if key in document_keys:
            raise ValidationError(f"duplicate document locator: {key}")
        document_keys.add(key)
        if row.get("parse_status") == "error":
            parse_errors += 1
            continue
        parsed_documents += 1
        if row.get("document_id"):
            document_ids.add(row["document_id"])
        clinical_documents += bool(row.get("has_clinical_pharmacology_section"))
        explicit_pk_documents += bool(row.get("has_explicit_pharmacokinetics_section"))
        fda_documents += bool(row.get("drugsfda_matches"))
        if int(row.get("candidate_section_count") or 0) > 0:
            candidate_documents += 1
            candidate_document_keys.add(key)
            candidate_fda_documents += bool(row.get("drugsfda_matches"))

    candidate_ids: set[Any] = set()
    section_rows = 0
    declared_tables = 0
    evidence_spans = 0
    tier_a_document_keys: set[tuple[Any, Any, Any]] = set()
    tier_counts: Counter[str] = Counter()
    for _, row in json_lines(required["sections"]):
        section_rows += 1
        candidate_id = row.get("candidate_id")
        if not candidate_id or candidate_id in candidate_ids:
            raise ValidationError(f"missing/duplicate candidate ID at section row {section_rows}")
        candidate_ids.add(candidate_id)
        key = (row.get("archive"), row.get("outer_member"), row.get("inner_xml_member"))
        if key not in candidate_document_keys:
            raise ValidationError(f"orphan/noncandidate section document locator: {key}")
        declared_tables += int(row.get("table_count") or 0)
        evidence_spans += len(row.get("evidence_spans") or [])
        flags = row.get("context_completeness_flags")
        if not isinstance(flags, dict):
            raise ValidationError(f"section {candidate_id} lacks context flags")
        if flags.get("all_core_context_flags_in_same_section"):
            tier = "A_all_core_machine_detected_unverified"
            tier_a_document_keys.add(key)
        elif (
            flags.get("unit_like_mentioned")
            and flags.get("matrix_mentioned")
            and flags.get("human_population_mentioned")
            and (flags.get("route_mentioned") or flags.get("dose_mentioned"))
        ):
            tier = "B_endpoint_unit_matrix_human_and_partial_administration_context"
        else:
            tier = "C_quantitative_endpoint_candidate_with_major_context_gap"
        tier_counts[tier] += 1

    table_keys: set[tuple[Any, Any]] = set()
    table_rows = 0
    table_hash_rows = 0
    for _, row in json_lines(required["tables"]):
        table_rows += 1
        candidate_id = row.get("candidate_id")
        if candidate_id not in candidate_ids:
            raise ValidationError(f"table row {table_rows} has orphan candidate ID")
        key = (candidate_id, row.get("table_index_in_section"))
        if key in table_keys:
            raise ValidationError(f"duplicate table key: {key}")
        table_keys.add(key)
        if row.get("table_xml_sha256") and row.get("normalized_text_sha256"):
            table_hash_rows += 1

    latest_rows = sum(1 for _ in json_lines(required["latest"]))
    observed = {
        "outer_members_scanned": len(document_keys),
        "inner_xml_members_parsed": parsed_documents,
        "parse_errors": parse_errors,
        "documents_with_clinical_pharmacology_section": clinical_documents,
        "documents_with_explicit_pharmacokinetics_section": explicit_pk_documents,
        "candidate_document_versions": candidate_documents,
        "candidate_sections": section_rows,
        "candidate_tables": table_rows,
        "unique_document_ids": len(document_ids),
        "latest_available_candidate_documents": latest_rows,
        "documents_with_exact_drugsfda_application_match": fda_documents,
        "candidate_documents_with_exact_drugsfda_application_match": candidate_fda_documents,
        "candidate_documents_with_at_least_one_tier_A_section": len(tier_a_document_keys),
        "bounded_evidence_spans": evidence_spans,
        "canonical_rows": 0,
        "training_labels": 0,
    }
    for name, value in observed.items():
        checks[f"summary_count::{name}"] = value == expected.get(name)
    checks["summary_context_tiers"] = dict(tier_counts) == summary.get("context_completeness_tiers")
    checks["declared_table_count_matches_rows"] = declared_tables == table_rows
    checks["all_table_hashes_present"] = table_hash_rows == table_rows
    checks["prior_validation_all_passed"] = prior_validation.get("all_passed") is True
    checks["prior_validation_checks_all_true"] = (
        isinstance(prior_validation.get("checks"), dict)
        and bool(prior_validation["checks"])
        and all(value is True for value in prior_validation["checks"].values())
    )
    checks["source_manifest_member_count"] = (
        source_manifest.get("expected_and_verified_file_member_count") == len(document_keys)
    )
    checks["source_manifest_binding"] = (
        summary.get("source_bindings", {}).get("dailymed_manifest", {}).get("sha256")
        == sha256(source_manifest_path)
    )
    all_passed = bool(checks) and all(checks.values())
    result = {
        "schema_version": SCHEMA,
        "mode": "validation_only_replay",
        "all_passed": all_passed,
        "fail_closed": True,
        "inputs": {
            "evidence_manifest": binding(required["manifest"]),
            "source_manifest": binding(required["source_manifest"]),
        },
        "implementation_bindings": {
            "exact_scanner": binding(EXACT_SCANNER),
            "exact_merge_validator": binding(EXACT_MERGER),
            "replay_cli": binding(Path(__file__).resolve()),
        },
        "observed_counts": observed,
        "context_completeness_tiers": dict(tier_counts),
        "checks": checks,
    }
    return result


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_scan(args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit <= 0:
        raise ValidationError("--limit must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValidationError(f"scan output must be absent or empty: {args.out}")
    command = [sys.executable, str(EXACT_SCANNER), "--out", str(args.out)]
    if args.archive:
        command.extend(["--archive", args.archive])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise ValidationError(f"exact scanner failed with exit code {completed.returncode}")
    summary = read_json(args.out / "scan_summary.json")
    manifest = read_json(args.out / "manifest.json")
    counts = summary.get("counts", {})
    checks = artifact_manifest_checks(args.out, manifest)
    checks.update({
        "scanner_exit_zero": True,
        "parse_errors_zero": counts.get("parse_errors") == 0,
        "all_scanned_members_parsed": counts.get("outer_members_scanned") == counts.get("inner_xml_members_parsed"),
        "canonical_rows_zero": counts.get("canonical_rows") == 0,
        "training_labels_zero": counts.get("training_labels") == 0,
    })
    receipt = {
        "schema_version": SCHEMA,
        "mode": "bounded_scan" if args.limit is not None or args.archive else "full_scan",
        "all_passed": all(checks.values()),
        "fail_closed": True,
        "implementation_bindings": {
            "exact_scanner": binding(EXACT_SCANNER),
            "wrapper_cli": binding(Path(__file__).resolve()),
        },
        "output": str(args.out.resolve()),
        "counts": counts,
        "checks": checks,
    }
    atomic_write_json(args.receipt, receipt)
    if not receipt["all_passed"]:
        raise ValidationError("scan completed but post-scan contract failed")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def run_validate(args: argparse.Namespace) -> int:
    result = replay_validation(args.evidence_dir, args.source_manifest)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_passed"]:
        raise ValidationError("validation replay failed")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan", help="run the byte-identical scanner with post-scan fail-closed checks")
    scan.add_argument("--out", type=Path, required=True, help="new/empty output directory")
    scan.add_argument("--archive", help="one frozen DailyMed archive filename")
    scan.add_argument("--limit", type=int, help="bounded document count for a smoke run")
    scan.add_argument("--receipt", type=Path, required=True, help="JSON run receipt")
    scan.set_defaults(func=run_scan)
    validate = commands.add_parser("validate", help="read-only full evidence validation replay")
    validate.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    validate.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    validate.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate.set_defaults(func=run_validate)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        return args.func(args)
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: unexpected failure: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
