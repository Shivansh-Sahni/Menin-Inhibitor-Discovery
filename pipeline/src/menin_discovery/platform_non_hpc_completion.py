"""Lead-owned verification for the completed bounded non-HPC expansion.

This gate replays every supplemental workstream verifier, binds the literature,
governance, configuration, integration, and legacy mechanical reports, and
preserves a strict distinction between completed CPU engineering and scientific,
clinical, rights, release, checkpoint, HPC, or training approval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .platform_clinical_results import verify_candidates
from .platform_context_splits import verify_context_split_candidates, verify_context_split_report
from .platform_deep_leakage import verify_deep_leakage_audit
from .platform_external_admission import verify_external_admission_analysis
from .platform_final_verification import _scan_no_training_json
from .platform_non_hpc_governance import load_and_validate_config
from .platform_pkdb_candidates import verify as verify_pkdb_candidates
from .platform_regulatory_records import verify_regulatory_records
from .platform_structure_metadata import verify_structure_metadata

SCHEMA_VERSION = "platform_non_hpc_completion_v1"
REPORT_PATH = Path("research/reports/platform/non_hpc_completion/final_non_hpc_completion_verification.json")
CONFIG_PATH = Path("pipeline/config/non_hpc_readiness.yaml")
LEDGER_PATH = Path("docs/project/pretraining_readiness_ledger.md")

LITERATURE_PATHS = (
    Path("research/reports/platform/literature_decisions/comprehensive_literature_review.md"),
    Path("research/reports/platform/literature_decisions/decision_recommendations.md"),
    Path("research/reports/platform/literature_decisions/model_candidate_decision_matrix.csv"),
    Path("research/reports/platform/literature_decisions/source_bibliography.json"),
)

CRITICAL_PATHS = (
    CONFIG_PATH,
    LEDGER_PATH,
    Path("README.md"),
    Path("Makefile"),
    Path("pipeline/src/menin_discovery/platform_cli.py"),
    Path("pipeline/tests/test_platform_integration.py"),
    Path("pipeline/src/menin_discovery/platform_external_admission.py"),
    Path("pipeline/tests/test_platform_external_admission.py"),
    Path("research/reports/platform/external_admission_analysis/external_admission_analysis_manifest.json"),
    Path("pipeline/src/menin_discovery/platform_deep_leakage.py"),
    Path("pipeline/tests/test_platform_deep_leakage.py"),
    Path("research/reports/platform/deep_leakage_analysis/manifest.json"),
    Path("pipeline/src/menin_discovery/platform_structure_metadata.py"),
    Path("pipeline/tests/test_platform_structure_metadata.py"),
    Path("research/data/platform/interim/structure_metadata/full_chembl37/structure_metadata_manifest.json"),
    Path("research/reports/platform/structure_metadata/full_chembl37/structure_metadata_report.json"),
    Path("pipeline/src/menin_discovery/platform_context_splits.py"),
    Path("pipeline/tests/test_platform_context_splits.py"),
    Path("research/data/platform/splits/context_holdout_candidates/acceptance.json"),
    Path("research/reports/platform/context_split_analysis/manifest.json"),
    Path("pipeline/src/menin_discovery/platform_clinical_results.py"),
    Path("pipeline/tests/test_platform_clinical_results.py"),
    Path(
        "research/data/platform/interim/clinical_results_candidates/clinical_results_candidates_manifest.json"
    ),
    Path("research/reports/platform/clinical_results_analysis/clinical_results_analysis.md"),
    Path("research/reports/platform/clinical_results_analysis/verification_report.json"),
    Path("pipeline/src/menin_discovery/platform_regulatory_records.py"),
    Path("pipeline/tests/test_platform_regulatory_records.py"),
    Path(
        "research/data/platform/interim/regulatory_record_candidates/"
        "drugs_at_fda_20260804/regulatory_record_candidates_manifest.json"
    ),
    Path(
        "research/reports/platform/regulatory_record_analysis/"
        "drugs_at_fda_20260804/regulatory_record_analysis_report.json"
    ),
    Path("pipeline/src/menin_discovery/platform_pkdb_candidates.py"),
    Path("pipeline/tests/test_platform_pkdb_candidates.py"),
    Path("research/data/platform/raw/external_public/pkdb_public_2026_08_05/manifest.json"),
    Path("research/data/platform/interim/pkdb_candidates/manifest.json"),
    Path("research/reports/platform/pkdb_candidate_analysis/report.json"),
    Path("research/reports/platform/pkdb_candidate_analysis/summary.md"),
    Path("research/reports/platform/non_hpc_completion/acceptance_manifest.json"),
    Path("research/reports/platform/non_hpc_completion/non_hpc_governance_report.json"),
    Path("research/reports/platform/non_hpc_completion/non_hpc_governance_summary.md"),
    Path("research/reports/platform/non_hpc_completion/release_inventory.csv"),
    Path("research/reports/platform/non_hpc_completion/non_hpc_execution_summary.md"),
    Path("research/reports/platform/non_hpc_completion/governance_and_release_decision_packet.md"),
    Path("research/reports/platform/non_hpc_completion/compute_and_operations_plan.md"),
    Path("research/reports/platform/non_hpc_completion/external_and_prospective_validation_protocol.md"),
    Path("research/reports/platform/non_hpc_completion/non_hpc_decision_register.csv"),
    Path("research/reports/final_verification/platform_final_artifact_verification.json"),
    Path("research/reports/final_verification/platform_final_artifact_verification_2026_08_05_replay.json"),
    *LITERATURE_PATHS,
)

NO_TRAINING_ROOTS = (
    Path("research/reports/platform/external_admission_analysis"),
    Path("research/reports/platform/deep_leakage_analysis"),
    Path("research/data/platform/interim/structure_metadata/full_chembl37"),
    Path("research/reports/platform/structure_metadata/full_chembl37"),
    Path("research/data/platform/splits/context_holdout_candidates"),
    Path("research/reports/platform/context_split_analysis"),
    Path("research/data/platform/interim/clinical_results_candidates"),
    Path("research/reports/platform/clinical_results_analysis"),
    Path("research/data/platform/interim/regulatory_record_candidates/drugs_at_fda_20260804"),
    Path("research/reports/platform/regulatory_record_analysis/drugs_at_fda_20260804"),
    Path("research/data/platform/interim/pkdb_candidates"),
    Path("research/reports/platform/pkdb_candidate_analysis"),
)


class NonHPCCompletionError(RuntimeError):
    """Raised when a bounded completion invariant is not reproducible."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_with_sha256(document: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(document)
    output.pop("document_sha256", None)
    output["document_sha256"] = hashlib.sha256(canonical_json_bytes(output)).hexdigest()
    return output


def verify_document_sha256(document: Mapping[str, Any]) -> bool:
    expected = document.get("document_sha256")
    body = dict(document)
    body.pop("document_sha256", None)
    return isinstance(expected, str) and expected == hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _safe_project_file(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise NonHPCCompletionError(f"unsafe project-relative path: {relative}")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise NonHPCCompletionError(f"missing, non-regular, or symlinked critical artifact: {relative}")
    return path


def _file_record(root: Path, relative: Path) -> dict[str, Any]:
    path = _safe_project_file(root, relative)
    return {
        "path": relative.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NonHPCCompletionError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise NonHPCCompletionError(f"JSON artifact is not an object: {path}")
    return value


def _validate_literature(root: Path) -> dict[str, Any]:
    bibliography_path = _safe_project_file(root, LITERATURE_PATHS[3])
    bibliography = _load_json(bibliography_path)
    sources = bibliography.get("sources")
    if not isinstance(sources, list) or len(sources) != 49:
        raise NonHPCCompletionError("literature bibliography must contain exactly 49 sources")
    source_ids = [item.get("source_id") for item in sources if isinstance(item, dict)]
    if len(source_ids) != 49 or len(set(source_ids)) != 49:
        raise NonHPCCompletionError("literature source identifiers are incomplete or duplicated")
    matrix_path = _safe_project_file(root, LITERATURE_PATHS[2])
    with matrix_path.open("r", encoding="utf-8", newline="") as handle:
        models = list(csv.DictReader(handle))
    candidate_ids = [row.get("candidate_id") for row in models]
    if len(models) != 17 or len(set(candidate_ids)) != 17 or any(not value for value in candidate_ids):
        raise NonHPCCompletionError("model decision matrix must contain 17 unique candidates")
    probable = [
        row for row in models if row.get("decision") == "probable_meeting_reference_and_later_comparator"
    ]
    if len(probable) != 1 or probable[0].get("candidate_name") != "RoseTTAFold All-Atom":
        raise NonHPCCompletionError("probable meeting-reference decision drifted")
    for path in LITERATURE_PATHS[:2]:
        if _safe_project_file(root, path).stat().st_size < 1000:
            raise NonHPCCompletionError(f"literature decision document is unexpectedly short: {path}")
    return {
        "status": "verified",
        "source_count": len(sources),
        "model_candidate_count": len(models),
        "probable_meeting_reference": "RoseTTAFold All-Atom",
        "identification_confidence": "moderate_high_not_certain",
        "checkpoint_downloaded": False,
    }


def _verify_governance(root: Path) -> dict[str, Any]:
    manifest_path = _safe_project_file(
        root, Path("research/reports/platform/non_hpc_completion/acceptance_manifest.json")
    )
    manifest = _load_json(manifest_path)
    components = manifest.get("components")
    if not isinstance(components, dict) or len(components) != manifest.get("component_count"):
        raise NonHPCCompletionError("governance component inventory is malformed")
    for relative, raw in components.items():
        if not isinstance(raw, dict) or raw.get("path") != relative:
            raise NonHPCCompletionError("governance component path binding failed")
        record = _file_record(root, Path(relative))
        if record["sha256"] != raw.get("sha256") or record["bytes"] != raw.get("size_bytes"):
            raise NonHPCCompletionError(f"governance component changed: {relative}")
    for key in (
        "scientific_task_claim_ready",
        "substantive_large_model_training_ready",
        "substantive_large_model_training_authorized",
        "large_model_training_started",
        "substantive_training_started",
    ):
        if manifest.get(key) is not False:
            raise NonHPCCompletionError(f"governance gate became true: {key}")
    if manifest.get("training_actions") != []:
        raise NonHPCCompletionError("governance training actions are nonempty")
    return {
        "status": "verified",
        "manifest_sha256": sha256_file(manifest_path),
        "component_count": len(components),
        "scientific_task_claim_ready": False,
        "substantive_training_started": False,
    }


def _verify_old_and_replay_reports(root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for relative in (
        Path("research/reports/final_verification/platform_final_artifact_verification.json"),
        Path(
            "research/reports/final_verification/platform_final_artifact_verification_2026_08_05_replay.json"
        ),
    ):
        path = _safe_project_file(root, relative)
        document = _load_json(path)
        if document.get("mechanical_artifact_verification") != "passed":
            raise NonHPCCompletionError(f"legacy mechanical verifier did not pass: {relative}")
        boundary = document.get("readiness_boundary")
        if not isinstance(boundary, dict) or boundary.get("scientific_task_claim_ready") is not False:
            raise NonHPCCompletionError(f"scientific readiness became true: {relative}")
        records.append(_file_record(root, relative))
    return {"status": "verified", "reports": records}


def _run_workstream_verifiers(root: Path) -> dict[str, Any]:
    clinical_output = root / "research/data/platform/interim/clinical_results_candidates"
    return {
        "external_admission": verify_external_admission_analysis(
            root / "research/reports/platform/external_admission_analysis"
        ),
        "deep_leakage": verify_deep_leakage_audit(root / "research/reports/platform/deep_leakage_analysis"),
        "structure_metadata": verify_structure_metadata(
            root / "research/data/platform/raw/structure_metadata/sifts_2026_08_03",
            root / "research/data/platform/interim/structure_metadata/full_chembl37",
            root / "research/reports/platform/structure_metadata/full_chembl37",
        ),
        "context_splits": {
            "data": verify_context_split_candidates(
                root / "research/data/platform/splits/context_holdout_candidates"
            ),
            "report": verify_context_split_report(
                root / "research/reports/platform/context_split_analysis",
                root / "research/data/platform/splits/context_holdout_candidates",
            ),
        },
        "clinical_results": verify_candidates(
            clinical_output,
            source_root=root / "research/data/platform/raw/external_public/clinicaltrials_gov_v2",
            code_path=root / "pipeline/src/menin_discovery/platform_clinical_results.py",
        ),
        "regulatory_records": verify_regulatory_records(
            root / "research/data/platform/raw/external_public/drugs_at_fda_bulk",
            root / "research/data/platform/interim/regulatory_record_candidates/drugs_at_fda_20260804",
            root / "research/reports/platform/regulatory_record_analysis/drugs_at_fda_20260804",
        ),
        "pkdb": verify_pkdb_candidates(
            root / "research/data/platform/raw/external_public/pkdb_public_2026_08_05",
            root / "research/data/platform/interim/pkdb_candidates",
            root / "research/reports/platform/pkdb_candidate_analysis",
        ),
    }


def build_completion_report(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    if not (root / ".git").is_dir():
        raise NonHPCCompletionError("project root is not a Git worktree")
    config = load_and_validate_config(_safe_project_file(root, CONFIG_PATH))
    if str(config.get("evidence_date")) != "2026-08-05":
        raise NonHPCCompletionError("non-HPC evidence date drifted")
    workstreams = _run_workstream_verifiers(root)
    literature = _validate_literature(root)
    governance = _verify_governance(root)
    prior_mechanical_verification = _verify_old_and_replay_reports(root)
    no_training = _scan_no_training_json([root / path for path in NO_TRAINING_ROOTS])
    critical = {
        path.as_posix(): _file_record(root, path)
        for path in (*CRITICAL_PATHS, Path(__file__).resolve().relative_to(root))
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "evidence_date": "2026-08-05",
        "status": "bounded_non_hpc_mechanical_work_complete_human_gates_open",
        "scope": {
            "cpu_data_engineering_complete": True,
            "cpu_evidence_review_complete": True,
            "cpu_leakage_and_split_analysis_complete": True,
            "candidate_clinical_regulatory_pk_inventories_complete": True,
            "scientific_task_claim_ready": False,
            "clinical_validation_complete": False,
            "external_or_prospective_validation_complete": False,
            "public_release_ready": False,
            "substantive_large_model_training_ready": False,
            "substantive_large_model_training_authorized": False,
            "substantive_training_started": False,
        },
        "literature": literature,
        "workstream_verification": workstreams,
        "governance": governance,
        "prior_mechanical_verification": prior_mechanical_verification,
        "no_training_scan": no_training,
        "critical_artifacts": critical,
        "critical_artifact_count": len(critical),
        "remaining_human_or_external_gates": [
            "scientific owner approval of intended task, estimand, claim, metrics, and leakage thresholds",
            "source, repository, documentation, checkpoint, output, and redistribution rights approval",
            "manual adjudication of clinical QT/QTc and PK endpoint candidates and group/intervention linkage",
            "curated Drugs@FDA ingredient-to-structure resolution and regulatory-context review",
            "documented reproducible PK-DB output access plus record-level reuse-rights review",
            "task-specific structure/construct/ligand/method/quality reconciliation before coordinate use",
            "exact checkpoint revision, hashes, cutoff, corpus overlap, endpoint compatibility, and terms freeze",
            "HPC budget, measured throughput, monitoring, recovery, retention, failure, and responsible-use approval",
            "independent external or prospective validation before broad scientific or clinical claims",
            "clean committed release candidate, top-level license, artifact storage, and clean-clone verification",
            "professor confirmation that RoseTTAFold All-Atom was the model referenced in the meeting",
        ],
        "canonical_external_observations_admitted": 0,
        "model_labels_admitted_from_new_external_layers": 0,
        "training_actions": [],
        "large_model_training_started": False,
        "substantive_training_started": False,
    }
    return document_with_sha256(report)


def _write_immutable(path: Path, document: Mapping[str, Any]) -> None:
    payload = json.dumps(dict(document), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise NonHPCCompletionError(f"completion report path is a symlink: {path}")
    if path.exists():
        if path.is_file() and path.read_text(encoding="utf-8") == payload:
            return
        raise NonHPCCompletionError("refusing to overwrite a non-identical completion report")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_non_hpc_completion(project_root: Path, output_report: Path = REPORT_PATH) -> dict[str, Any]:
    root = project_root.resolve()
    report = build_completion_report(root)
    destination = output_report.resolve() if output_report.is_absolute() else (root / output_report).resolve()
    try:
        destination.relative_to(root)
    except ValueError as error:
        raise NonHPCCompletionError("completion report must stay inside project root") from error
    _write_immutable(destination, report)
    return report


def verify_non_hpc_completion(project_root: Path, report_path: Path = REPORT_PATH) -> dict[str, Any]:
    root = project_root.resolve()
    path = report_path.resolve() if report_path.is_absolute() else (root / report_path).resolve()
    retained = _load_json(_safe_project_file(root, path.relative_to(root)))
    if retained.get("schema_version") != SCHEMA_VERSION or not verify_document_sha256(retained):
        raise NonHPCCompletionError("completion report schema or self-hash failed")
    expected = build_completion_report(root)
    if retained != expected:
        raise NonHPCCompletionError("completion report no longer matches verified project artifacts")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "verified",
        "report_sha256": sha256_file(path),
        "critical_artifact_count": retained["critical_artifact_count"],
        "workstream_count": len(retained["workstream_verification"]),
        "scientific_task_claim_ready": False,
        "substantive_large_model_training_ready": False,
        "substantive_large_model_training_authorized": False,
        "substantive_training_started": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-report", type=Path, default=REPORT_PATH)
    parser.add_argument("--verify-existing", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = (
        verify_non_hpc_completion(args.project_root, args.output_report)
        if args.verify_existing
        else materialize_non_hpc_completion(args.project_root, args.output_report)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
