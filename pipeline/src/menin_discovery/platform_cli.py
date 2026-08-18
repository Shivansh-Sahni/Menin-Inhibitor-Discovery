"""Umbrella command surface for the public protein--molecule platform.

Every mutating command maps to one already-tested transactional API.  There is
deliberately no large-model training command: this surface ends at immutable
model-readiness bundles and capped diagnostic baselines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

NO_SUBSTANTIVE_TRAINING = False
DEFAULT_PROJECT_ROOT = Path(".")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=_json_default))


def _manifest_record(project_root: Path, relative: str) -> dict[str, Any]:
    path = (project_root / relative).resolve()
    record: dict[str, Any] = {"path": relative, "exists": path.is_file()}
    if not path.is_file():
        return record
    if path.is_symlink():
        record["status"] = "rejected_symlink"
        return record
    record.update({"size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        record["json_status"] = "invalid"
        return record
    record["json_status"] = "parsed"
    if isinstance(document, dict):
        for key in (
            "schema_version",
            "manifest_sha256",
            "source_version",
            "source_id",
            "release_id",
            "snapshot_status",
            "status",
            "substantive_training_started",
            "large_model_training_started",
        ):
            if key in document:
                output_key = "declared_manifest_sha256" if key == "manifest_sha256" else key
                record[output_key] = document[key]
    return record


def platform_status(project_root: Path) -> dict[str, Any]:
    """Return a read-only manifest inventory without claiming scientific readiness."""

    root = project_root.resolve()
    manifest_paths = (
        "research/data/platform/raw/chembl_37_bulk/archive_manifest.json",
        "research/data/platform/raw/chembl_37_bulk/extracted/extraction_manifest.json",
        "research/data/platform/interim/chembl_37_bulk/activity_export_manifest.json",
        "research/data/platform/interim/chembl_37_bulk/schema_normalization_receipt.json",
        "research/data/platform/interim/chembl_37_bulk/specialized_views/specialized_views_manifest.json",
        "research/data/platform/interim/external_public_normalized/external_public_normalized_manifest.json",
        "research/data/platform/canonical/full_chembl37/build_manifest.json",
        "research/reports/platform/qc_report.json",
        "research/reports/platform/canonical_determinism_verification.json",
        "research/data/platform/raw/external_public/bindingdb_curated_202608/bindingdb_curated_202608_manifest.json",
        "research/data/platform/raw/external_public/uniprotkb_targeted_2026_02/uniprotkb_targeted_2026_02_manifest.json",
        "research/data/platform/raw/external_public/clinicaltrials_gov_v2/clinicaltrials_gov_v2_manifest.json",
        "research/data/platform/raw/external_public/drugs_at_fda_bulk/drugs_at_fda_bulk_manifest.json",
        "research/data/platform/raw/external_public/dailymed_spl_v2_human_rx/dailymed_spl_v2_human_rx_manifest.json",
        "research/models/platform/pretraining_static_manifest.json",
        "research/reports/platform/statistical_analysis/analysis_manifest.json",
        "research/reports/platform/statistical_analysis_determinism_b/analysis_manifest.json",
        "research/data/platform/splits/full_chembl37/acceptance.json",
        "research/data/platform/splits/determinism_build_b/full_chembl37/acceptance.json",
        "research/models/platform/corpus_readiness/full_chembl37/acceptance.json",
        "research/reports/platform/external_admission_analysis/external_admission_analysis_manifest.json",
        "research/reports/platform/deep_leakage_analysis/manifest.json",
        "research/data/platform/raw/structure_metadata/sifts_2026_08_03/acquisition_manifest.json",
        "research/data/platform/interim/structure_metadata/full_chembl37/structure_metadata_manifest.json",
        "research/reports/platform/structure_metadata/full_chembl37/structure_metadata_report.json",
        "research/data/platform/splits/context_holdout_candidates/acceptance.json",
        "research/reports/platform/context_split_analysis/manifest.json",
        "research/data/platform/interim/clinical_results_candidates/clinical_results_candidates_manifest.json",
        "research/reports/platform/clinical_results_analysis/verification_report.json",
        "research/data/platform/interim/regulatory_record_candidates/drugs_at_fda_20260804/regulatory_record_candidates_manifest.json",
        "research/reports/platform/regulatory_record_analysis/drugs_at_fda_20260804/regulatory_record_analysis_report.json",
        "research/data/platform/raw/external_public/pkdb_public_2026_08_05/manifest.json",
        "research/data/platform/interim/pkdb_candidates/manifest.json",
        "research/reports/platform/pkdb_candidate_analysis/manifest.json",
        "research/reports/platform/non_hpc_completion/acceptance_manifest.json",
        "research/reports/platform/non_hpc_completion/final_non_hpc_completion_verification.json",
        "research/reports/final_verification/platform_final_artifact_verification.json",
        "research/reports/final_verification/platform_final_artifact_verification_2026_08_05_replay.json",
    )
    transaction_marker = (
        root / "research/data/platform/interim/chembl_37_bulk/schema_normalization_transaction.json"
    )
    canonical_building = root / "research/data/platform/canonical/.full_chembl37.building"
    return {
        "schema_version": "protein_molecule_platform_status_v1",
        "project_root": root.as_posix(),
        "manifest_inventory": [_manifest_record(root, relative) for relative in manifest_paths],
        "fail_closed_markers": {
            "schema_normalization_transaction_present": transaction_marker.exists(),
            "canonical_building_directory_present": canonical_building.exists(),
        },
        "interpretation": (
            "inventory_only_not_a_readiness_decision; use bound verifiers, QC, and the final "
            "cross-workstream audit"
        ),
        "large_model_training_started": NO_SUBSTANTIVE_TRAINING,
        "substantive_training_started": NO_SUBSTANTIVE_TRAINING,
    }


def _binary_mapping(values: Sequence[str]) -> tuple[tuple[str, int], ...]:
    parsed: list[tuple[str, int]] = []
    for value in values:
        match = re.fullmatch(r"(.+)=(0|1)", value)
        if match is None or not match.group(1).strip():
            raise ValueError("Each --binary-label must be CLASS=0 or CLASS=1")
        parsed.append((match.group(1).strip(), int(match.group(2))))
    return tuple(parsed)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Repository root used to resolve relative paths (default: current directory).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Inventory known manifests without changing files.")
    subparsers.add_parser("contracts", help="Print the integrated no-training API contracts.")

    normalize = subparsers.add_parser(
        "normalize-chembl-exports",
        help="Crash-safely rewrite frozen ChEMBL Parquet exports to explicit schemas.",
    )
    normalize.add_argument(
        "--database",
        type=Path,
        default=Path("research/data/platform/raw/chembl_37_bulk/extracted/chembl_37.db"),
    )
    normalize.add_argument("--interim-root", type=Path, default=Path("research/data/platform/interim"))

    canonical = subparsers.add_parser(
        "canonicalize-chembl", help="Build and QC-promote the canonical ChEMBL corpus."
    )
    canonical.add_argument("--raw-root", type=Path, default=Path("research/data/platform/raw"))
    canonical.add_argument("--interim-root", type=Path, default=Path("research/data/platform/interim"))
    canonical.add_argument("--canonical-root", type=Path, default=Path("research/data/platform/canonical"))
    canonical.add_argument("--reports-root", type=Path, default=Path("research/reports/platform"))

    determinism = subparsers.add_parser(
        "verify-canonical-determinism",
        help="Compare two independently materialized canonical builds and bound QC outputs.",
    )
    determinism.add_argument(
        "--build-a",
        type=Path,
        default=Path("research/data/platform/canonical/full_chembl37"),
    )
    determinism.add_argument("--reports-a", type=Path, default=Path("research/reports/platform"))
    determinism.add_argument(
        "--build-b",
        type=Path,
        default=Path("research/data/platform/determinism_build_b/full_chembl37"),
    )
    determinism.add_argument(
        "--reports-b",
        type=Path,
        default=Path("research/reports/platform/determinism_build_b"),
    )
    determinism.add_argument(
        "--output",
        type=Path,
        default=Path("research/reports/platform/canonical_determinism_verification.json"),
    )

    acquire = subparsers.add_parser(
        "acquire-external", help="Acquire one explicitly selected immutable public source."
    )
    acquire.add_argument(
        "source",
        choices=(
            "bindingdb",
            "clinicaltrials",
            "drugsfda",
            "dailymed",
            "dailymed-part",
            "uniprot",
        ),
    )
    acquire.add_argument(
        "--raw-root",
        type=Path,
        default=Path("research/data/platform/raw/external_public"),
    )
    acquire.add_argument("--part-number", type=int)
    acquire.add_argument("--chembl-target-components", type=Path)
    acquire.add_argument("--bindingdb-mapping", type=Path)

    verify_external = subparsers.add_parser(
        "verify-external", help="Rehash and validate one external source manifest."
    )
    verify_external.add_argument("--source-root", type=Path, required=True)
    verify_external.add_argument("--manifest", type=Path, required=True)

    normalize_external = subparsers.add_parser(
        "normalize-external",
        help="Normalize frozen external-source bundles without admitting labels.",
    )
    normalize_external.add_argument(
        "--raw-root",
        type=Path,
        default=Path("research/data/platform/raw/external_public"),
    )
    normalize_external.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/data/platform/interim/external_public_normalized"),
    )
    normalize_external.add_argument(
        "--report-root",
        type=Path,
        default=Path("research/reports/platform/external_normalization"),
    )

    verify_normalized = subparsers.add_parser(
        "verify-external-normalized",
        help="Rehash and semantically reconcile the normalized external bundle.",
    )
    verify_normalized.add_argument(
        "--raw-root",
        type=Path,
        default=Path("research/data/platform/raw/external_public"),
    )
    verify_normalized.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/data/platform/interim/external_public_normalized"),
    )
    verify_normalized.add_argument(
        "--report-root",
        type=Path,
        default=Path("research/reports/platform/external_normalization"),
    )
    verify_normalized.add_argument(
        "--refresh-report",
        action="store_true",
        help="Refresh the portable report only after full existing-output verification.",
    )

    external_admission = subparsers.add_parser(
        "analyze-external-admission",
        help="Quantify external linkage, overlap, conflict, rights, and extraction readiness without admitting labels.",
    )
    external_admission.add_argument(
        "--normalized-root",
        type=Path,
        default=Path("research/data/platform/interim/external_public_normalized"),
    )
    external_admission.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("research/data/platform/canonical/full_chembl37"),
    )
    external_admission.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/reports/platform/external_admission_analysis"),
    )

    verify_external_admission = subparsers.add_parser(
        "verify-external-admission",
        help="Verify the immutable external-admission candidate analysis and zero-label boundary.",
    )
    verify_external_admission.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/reports/platform/external_admission_analysis"),
    )

    deep_leakage = subparsers.add_parser(
        "analyze-deep-leakage",
        help="Run the bounded label-blind ligand/protein/context leakage audit and task decision.",
    )
    deep_leakage.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("research/data/platform/canonical/full_chembl37"),
    )
    deep_leakage.add_argument(
        "--qc-report",
        type=Path,
        default=Path("research/reports/platform/qc_report.json"),
    )
    deep_leakage.add_argument(
        "--split-root",
        type=Path,
        default=Path("research/data/platform/splits/full_chembl37"),
    )
    deep_leakage.add_argument(
        "--corpus-acceptance",
        type=Path,
        default=Path("research/models/platform/corpus_readiness/full_chembl37/acceptance.json"),
    )
    deep_leakage.add_argument(
        "--model-registry",
        type=Path,
        default=Path("research/models/platform/model_candidate_registry.json"),
    )
    deep_leakage.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/reports/platform/deep_leakage_analysis"),
    )

    verify_deep_leakage = subparsers.add_parser(
        "verify-deep-leakage",
        help="Verify the deep-leakage artifact topology, hashes, scopes, and sealed-label boundary.",
    )
    verify_deep_leakage.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/reports/platform/deep_leakage_analysis"),
    )

    structure_metadata = subparsers.add_parser(
        "prepare-structure-metadata",
        help="Build exact UniProt-to-PDB metadata coverage from the frozen SIFTS snapshot; no coordinates.",
    )
    structure_metadata.add_argument(
        "--raw-root",
        type=Path,
        default=Path("research/data/platform/raw/structure_metadata/sifts_2026_08_03"),
    )
    structure_metadata.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("research/data/platform/canonical/full_chembl37"),
    )
    structure_metadata.add_argument(
        "--external-root",
        type=Path,
        default=Path("research/data/platform/interim/external_public_normalized"),
    )
    structure_metadata.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/data/platform/interim/structure_metadata/full_chembl37"),
    )
    structure_metadata.add_argument(
        "--report-root",
        type=Path,
        default=Path("research/reports/platform/structure_metadata/full_chembl37"),
    )

    verify_structure_metadata = subparsers.add_parser(
        "verify-structure-metadata",
        help="Verify the SIFTS/wwPDB acquisition, normalized metadata, and zero-coordinate boundary.",
    )
    verify_structure_metadata.add_argument(
        "--raw-root",
        type=Path,
        default=Path("research/data/platform/raw/structure_metadata/sifts_2026_08_03"),
    )
    verify_structure_metadata.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/data/platform/interim/structure_metadata/full_chembl37"),
    )
    verify_structure_metadata.add_argument(
        "--report-root",
        type=Path,
        default=Path("research/reports/platform/structure_metadata/full_chembl37"),
    )

    context_splits = subparsers.add_parser(
        "prepare-context-splits",
        help="Materialize supplemental label-blind assay, document, and strict-temporal split candidates.",
    )
    context_splits.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("research/data/platform/canonical/full_chembl37"),
    )
    context_splits.add_argument(
        "--qc-report",
        type=Path,
        default=Path("research/reports/platform/qc_report.json"),
    )
    context_splits.add_argument(
        "--split-root",
        type=Path,
        default=Path("research/data/platform/splits/full_chembl37"),
    )
    context_splits.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/data/platform/splits/context_holdout_candidates"),
    )
    context_splits.add_argument(
        "--report-root",
        type=Path,
        default=Path("research/reports/platform/context_split_analysis"),
    )

    verify_context_splits = subparsers.add_parser(
        "verify-context-splits",
        help="Verify supplemental context splits without reading labels or replacing official assignments.",
    )
    verify_context_splits.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/data/platform/splits/context_holdout_candidates"),
    )
    verify_context_splits.add_argument(
        "--report-root",
        type=Path,
        default=Path("research/reports/platform/context_split_analysis"),
    )

    clinical_results = subparsers.add_parser(
        "prepare-clinical-results",
        help="Build immutable candidate-only ClinicalTrials.gov QT/QTc and PK inventories.",
    )
    clinical_results.add_argument(
        "--source-root",
        type=Path,
        default=Path("research/data/platform/raw/external_public/clinicaltrials_gov_v2"),
    )
    clinical_results.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/data/platform/interim/clinical_results_candidates"),
    )
    verify_clinical_results = subparsers.add_parser(
        "verify-clinical-results",
        help="Verify clinical candidate source/code bindings, closed topology, and zero-label boundary.",
    )
    verify_clinical_results.add_argument(
        "--source-root",
        type=Path,
        default=Path("research/data/platform/raw/external_public/clinicaltrials_gov_v2"),
    )
    verify_clinical_results.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/data/platform/interim/clinical_results_candidates"),
    )

    regulatory_records = subparsers.add_parser(
        "prepare-regulatory-records",
        help="Normalize frozen Drugs@FDA records as candidate metadata without biomedical labels.",
    )
    verify_regulatory_records_parser = subparsers.add_parser(
        "verify-regulatory-records",
        help="Verify Drugs@FDA record candidates, anomalies, code/source binding, and zero labels.",
    )
    for command in (regulatory_records, verify_regulatory_records_parser):
        command.add_argument(
            "--raw-root",
            type=Path,
            default=Path("research/data/platform/raw/external_public/drugs_at_fda_bulk"),
        )
        command.add_argument(
            "--output-root",
            type=Path,
            default=Path("research/data/platform/interim/regulatory_record_candidates/drugs_at_fda_20260804"),
        )
        command.add_argument(
            "--report-root",
            type=Path,
            default=Path("research/reports/platform/regulatory_record_analysis/drugs_at_fda_20260804"),
        )

    acquire_pkdb = subparsers.add_parser(
        "acquire-pkdb-candidates",
        help="Acquire one bounded official PK-DB metadata snapshot; no result labels are admitted.",
    )
    acquire_pkdb.add_argument(
        "--raw-root",
        type=Path,
        default=Path("research/data/platform/raw/external_public/pkdb_public_2026_08_05"),
    )
    prepare_pkdb = subparsers.add_parser(
        "prepare-pkdb-candidates",
        help="Normalize the frozen PK-DB metadata snapshot under a fail-closed zero-admission policy.",
    )
    verify_pkdb = subparsers.add_parser(
        "verify-pkdb-candidates",
        help="Verify PK-DB source/output/code bindings and byte-identical normalized replay.",
    )
    for command in (prepare_pkdb, verify_pkdb):
        command.add_argument(
            "--raw-root",
            type=Path,
            default=Path("research/data/platform/raw/external_public/pkdb_public_2026_08_05"),
        )
        command.add_argument(
            "--interim-root",
            type=Path,
            default=Path("research/data/platform/interim/pkdb_candidates"),
        )
        command.add_argument(
            "--report-root",
            type=Path,
            default=Path("research/reports/platform/pkdb_candidate_analysis"),
        )

    analyze = subparsers.add_parser(
        "analyze-canonical",
        help="Run the exact, zero-training statistical census of an accepted corpus.",
    )
    analyze.add_argument(
        "--canonical-build-root",
        type=Path,
        default=Path("research/data/platform/canonical/full_chembl37"),
    )
    analyze.add_argument(
        "--qc-report",
        type=Path,
        default=Path("research/reports/platform/qc_report.json"),
    )
    analyze.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/reports/platform/statistical_analysis"),
    )
    analyze.add_argument("--batch-size", type=int, default=65_536)
    analyze.add_argument("--sample-cap", type=int, default=100)

    verify_analysis = subparsers.add_parser(
        "verify-statistical-analysis",
        help="Verify the statistical census and rebind it to canonical data/QC.",
    )
    verify_analysis.add_argument(
        "--canonical-build-root",
        type=Path,
        default=Path("research/data/platform/canonical/full_chembl37"),
    )
    verify_analysis.add_argument(
        "--qc-report",
        type=Path,
        default=Path("research/reports/platform/qc_report.json"),
    )
    verify_analysis.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/reports/platform/statistical_analysis"),
    )

    split_suite = subparsers.add_parser(
        "prepare-split-suite",
        help="Materialize every fixed-seed, feature-only official split candidate.",
    )
    split_suite.add_argument(
        "--canonical-build-root",
        type=Path,
        default=Path("research/data/platform/canonical/full_chembl37"),
    )
    split_suite.add_argument(
        "--qc-report",
        type=Path,
        default=Path("research/reports/platform/qc_report.json"),
    )
    split_suite.add_argument(
        "--output-directory",
        type=Path,
        default=Path("research/data/platform/splits/full_chembl37"),
    )
    split_suite.add_argument("--seed", type=int, default=20260804)
    split_suite.add_argument("--train-fraction", type=float, default=0.70)
    split_suite.add_argument("--validation-fraction", type=float, default=0.15)
    split_suite.add_argument("--test-fraction", type=float, default=0.15)
    split_suite.add_argument("--batch-size", type=int, default=50_000)
    split_suite.add_argument("--near-sample-cap-per-partition", type=int, default=256)
    split_suite.add_argument("--chemical-tanimoto-threshold", type=float, default=0.80)
    split_suite.add_argument("--chemical-fingerprint-bits", type=int, default=2_048)
    split_suite.add_argument("--chemical-fingerprint-radius", type=int, default=2)
    split_suite.add_argument("--protein-kmer-size", type=int, default=3)
    split_suite.add_argument("--protein-jaccard-threshold", type=float, default=0.80)

    verify_splits = subparsers.add_parser(
        "verify-split-suite",
        help="Regenerate and verify every official split against canonical data/QC.",
    )
    verify_splits.add_argument(
        "--canonical-build-root",
        type=Path,
        default=Path("research/data/platform/canonical/full_chembl37"),
    )
    verify_splits.add_argument(
        "--qc-report",
        type=Path,
        default=Path("research/reports/platform/qc_report.json"),
    )
    verify_splits.add_argument(
        "--output-directory",
        type=Path,
        default=Path("research/data/platform/splits/full_chembl37"),
    )

    corpus = subparsers.add_parser(
        "prepare-corpus-readiness",
        help="Preflight every task, fix splits, and run only capped diagnostics.",
    )
    corpus.add_argument(
        "--canonical-build-root",
        type=Path,
        default=Path("research/data/platform/canonical/full_chembl37"),
    )
    corpus.add_argument(
        "--qc-report",
        type=Path,
        default=Path("research/reports/platform/qc_report.json"),
    )
    corpus.add_argument(
        "--output-directory",
        type=Path,
        default=Path("research/models/platform/corpus_readiness/full_chembl37"),
    )
    corpus.add_argument("--seed", type=int, default=20260804)
    corpus.add_argument("--preflight-batch-size", type=int, default=50_000)
    corpus.add_argument("--split-batch-size", type=int, default=50_000)
    corpus.add_argument("--serialization-batch-size", type=int, default=8_192)
    corpus.add_argument("--loader-batch-size", type=int, default=8)
    corpus.add_argument("--loader-maximum-batches", type=int, default=4)
    corpus.add_argument("--minimum-rows", type=int, default=4)
    corpus.add_argument("--minimum-unique-molecules", type=int, default=3)
    corpus.add_argument("--minimum-unique-proteins", type=int, default=1)
    corpus.add_argument("--diagnostic-maximum-train-examples", type=int, default=10_000)
    corpus.add_argument("--diagnostic-maximum-validation-examples", type=int, default=2_500)
    corpus.add_argument("--diagnostic-fingerprint-bits", type=int, default=2_048)
    corpus.add_argument("--diagnostic-fingerprint-radius", type=int, default=2)

    verify_corpus = subparsers.add_parser(
        "verify-corpus-readiness",
        help="Verify corpus-readiness artifacts and rebind them to canonical data/QC.",
    )
    verify_corpus.add_argument(
        "--canonical-build-root",
        type=Path,
        default=Path("research/data/platform/canonical/full_chembl37"),
    )
    verify_corpus.add_argument(
        "--qc-report",
        type=Path,
        default=Path("research/reports/platform/qc_report.json"),
    )
    verify_corpus.add_argument(
        "--output-directory",
        type=Path,
        default=Path("research/models/platform/corpus_readiness/full_chembl37"),
    )

    final_verification = subparsers.add_parser(
        "verify-final-artifacts",
        help=(
            "Replay all cross-workstream mechanical gates and bind accepted bytes; never authorizes training."
        ),
    )
    final_verification.add_argument(
        "--output-report",
        type=Path,
        default=Path("research/reports/final_verification/platform_final_artifact_verification.json"),
    )

    governance = subparsers.add_parser(
        "audit-non-hpc-governance",
        help="Materialize local resource, release-hygiene, and human-gate evidence without granting approval.",
    )
    governance.add_argument(
        "--config",
        type=Path,
        default=Path("pipeline/config/non_hpc_readiness.yaml"),
    )
    governance.add_argument(
        "--output-root",
        type=Path,
        default=Path("research/reports/platform/non_hpc_completion"),
    )

    completion = subparsers.add_parser(
        "verify-non-hpc-completion",
        help="Replay and bind every bounded non-HPC workstream without granting scientific/training approval.",
    )
    completion.add_argument(
        "--report",
        type=Path,
        default=Path(
            "research/reports/platform/non_hpc_completion/final_non_hpc_completion_verification.json"
        ),
    )

    static = subparsers.add_parser(
        "prepare-static", help="Write feature/model registries; never download or fit a model."
    )
    static.add_argument("--evidence-checked-date", required=True)
    static.add_argument(
        "--feature-directory",
        type=Path,
        default=Path("research/data/platform/features/static"),
    )
    static.add_argument("--model-directory", type=Path, default=Path("research/models/platform"))

    integrate = subparsers.add_parser(
        "integrate-task",
        help="Create an immutable fixed-split/model-ready bundle with bounded loader smoke.",
    )
    integrate.add_argument("--task-dataset", type=Path, required=True)
    integrate.add_argument("--output-directory", type=Path, required=True)
    integrate.add_argument("--split-name", default="molecule_hash_stream_v1")
    integrate.add_argument(
        "--split-strategy",
        choices=(
            "molecule_grouped",
            "scaffold",
            "source_holdout",
            "protein_holdout",
            "target_holdout",
            "double_cold",
        ),
        default="molecule_grouped",
    )
    integrate.add_argument(
        "--intended-use",
        default="new molecule within the observed public task domain at platform scale",
    )
    integrate.add_argument("--seed", type=int, default=20260804)
    integrate.add_argument(
        "--task-eligibility-mode",
        choices=("default", "derived_sensitivity"),
        default="default",
    )

    diagnose = subparsers.add_parser(
        "diagnose-task",
        help="Run capped train/validation-only features, baselines, and negative controls.",
    )
    diagnose.add_argument("--integration-bundle", type=Path, required=True)
    diagnose.add_argument("--output-directory", type=Path, required=True)
    diagnose.add_argument("--integration-acceptance-sha256", required=True)
    diagnose.add_argument("--seed", type=int, default=20260804)
    diagnose.add_argument("--maximum-train-examples", type=int, default=50_000)
    diagnose.add_argument("--maximum-validation-examples", type=int, default=10_000)
    diagnose.add_argument("--fingerprint-bits", type=int, default=2_048)
    diagnose.add_argument("--fingerprint-radius", type=int, default=2)
    diagnose.add_argument(
        "--binary-label",
        action="append",
        default=[],
        metavar="CLASS=0|1",
        help="Repeat exactly twice for nonnumeric binary labels.",
    )
    return parser


def _contracts() -> dict[str, Any]:
    from .platform_data_bulk import bulk_integration_contract
    from .platform_data_bulk_canonical import bulk_canonicalization_contract
    from .platform_pretraining import candidate_model_registry

    return {
        "schema_version": "protein_molecule_platform_contracts_v1",
        "bulk_source": bulk_integration_contract(),
        "canonicalization": bulk_canonicalization_contract(),
        "frozen_model_candidates": [candidate.key for candidate in candidate_model_registry()],
        "training_command_exposed": False,
        "large_model_training_started": NO_SUBSTANTIVE_TRAINING,
        "substantive_training_started": NO_SUBSTANTIVE_TRAINING,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    if not project_root.is_dir():
        raise NotADirectoryError(project_root)

    if args.command == "status":
        _print_json(platform_status(project_root))
        return 0
    if args.command == "contracts":
        _print_json(_contracts())
        return 0
    if args.command == "normalize-chembl-exports":
        from .platform_data_bulk import normalize_chembl37_export_schemas

        result = normalize_chembl37_export_schemas(
            _resolved(project_root, args.database),
            _resolved(project_root, args.interim_root),
        )
        _print_json(result)
        return 0
    if args.command == "canonicalize-chembl":
        from .platform_data_bulk_canonical import materialize_chembl37_specialized_canonical

        result = materialize_chembl37_specialized_canonical(
            _resolved(project_root, args.raw_root),
            _resolved(project_root, args.interim_root),
            _resolved(project_root, args.canonical_root),
            _resolved(project_root, args.reports_root),
        )
        _print_json(result)
        return 0
    if args.command == "verify-canonical-determinism":
        from .platform_determinism import (
            compare_canonical_builds,
            write_determinism_report,
        )

        result = compare_canonical_builds(
            _resolved(project_root, args.build_a),
            _resolved(project_root, args.reports_a),
            _resolved(project_root, args.build_b),
            _resolved(project_root, args.reports_b),
        )
        write_determinism_report(_resolved(project_root, args.output), result)
        _print_json(result)
        return 0
    if args.command == "acquire-external":
        from .platform_external_acquisition import main as external_main

        delegated = [args.source, "--raw-root", _resolved(project_root, args.raw_root).as_posix()]
        if args.part_number is not None:
            delegated.extend(("--part-number", str(args.part_number)))
        if args.chembl_target_components is not None:
            delegated.extend(
                (
                    "--chembl-target-components",
                    _resolved(project_root, args.chembl_target_components).as_posix(),
                )
            )
        if args.bindingdb_mapping is not None:
            delegated.extend(
                (
                    "--bindingdb-mapping",
                    _resolved(project_root, args.bindingdb_mapping).as_posix(),
                )
            )
        return external_main(delegated)
    if args.command == "verify-external":
        from .platform_external_acquisition import verify_source_acquisition_manifest

        source_root = _resolved(project_root, args.source_root)
        manifest_path = _resolved(project_root, args.manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("External manifest must be a JSON object")
        _print_json(verify_source_acquisition_manifest(source_root, manifest))
        return 0
    if args.command == "normalize-external":
        from .platform_external_normalization import build_external_normalization

        result = build_external_normalization(
            _resolved(project_root, args.raw_root),
            _resolved(project_root, args.output_root),
            _resolved(project_root, args.report_root),
        )
        _print_json(result)
        return 0
    if args.command == "verify-external-normalized":
        from .platform_external_normalization import (
            refresh_external_normalization_report,
            verify_external_normalized_output,
        )

        output_root = _resolved(project_root, args.output_root)
        result = verify_external_normalized_output(
            output_root,
            _resolved(project_root, args.raw_root),
        )
        if args.refresh_report:
            report = refresh_external_normalization_report(
                output_root,
                _resolved(project_root, args.report_root),
                result,
            )
            result["refreshed_report_manifest_sha256"] = report["manifest_sha256"]
        _print_json(result)
        return 0
    if args.command == "analyze-external-admission":
        from .platform_external_admission import run_external_admission_analysis

        result = run_external_admission_analysis(
            _resolved(project_root, args.normalized_root),
            _resolved(project_root, args.canonical_root),
            _resolved(project_root, args.output_root),
        )
        _print_json(result)
        return 0
    if args.command == "verify-external-admission":
        from .platform_external_admission import verify_external_admission_analysis

        result = verify_external_admission_analysis(_resolved(project_root, args.output_root))
        _print_json(result)
        return 0
    if args.command == "analyze-deep-leakage":
        from .platform_deep_leakage import materialize_deep_leakage_audit

        result = materialize_deep_leakage_audit(
            _resolved(project_root, args.canonical_root),
            _resolved(project_root, args.qc_report),
            _resolved(project_root, args.split_root),
            _resolved(project_root, args.corpus_acceptance),
            _resolved(project_root, args.model_registry),
            _resolved(project_root, args.output_root),
        )
        _print_json(result)
        return 0
    if args.command == "verify-deep-leakage":
        from .platform_deep_leakage import verify_deep_leakage_audit

        result = verify_deep_leakage_audit(_resolved(project_root, args.output_root))
        _print_json(result)
        return 0
    if args.command == "prepare-structure-metadata":
        from .platform_structure_metadata import build_structure_metadata

        result = build_structure_metadata(
            _resolved(project_root, args.raw_root),
            _resolved(project_root, args.canonical_root),
            _resolved(project_root, args.external_root),
            _resolved(project_root, args.output_root),
            _resolved(project_root, args.report_root),
        )
        _print_json(result)
        return 0
    if args.command == "verify-structure-metadata":
        from .platform_structure_metadata import verify_structure_metadata

        result = verify_structure_metadata(
            _resolved(project_root, args.raw_root),
            _resolved(project_root, args.output_root),
            _resolved(project_root, args.report_root),
        )
        _print_json(result)
        return 0
    if args.command == "prepare-context-splits":
        from .platform_context_splits import materialize_context_split_candidates

        result = materialize_context_split_candidates(
            _resolved(project_root, args.canonical_root),
            _resolved(project_root, args.qc_report),
            _resolved(project_root, args.split_root),
            _resolved(project_root, args.output_root),
            _resolved(project_root, args.report_root),
        )
        _print_json(result)
        return 0
    if args.command == "verify-context-splits":
        from .platform_context_splits import (
            verify_context_split_candidates,
            verify_context_split_report,
        )

        result = {
            "data": verify_context_split_candidates(_resolved(project_root, args.output_root)),
            "report": verify_context_split_report(
                _resolved(project_root, args.report_root),
                _resolved(project_root, args.output_root),
            ),
        }
        _print_json(result)
        return 0
    if args.command == "prepare-clinical-results":
        from .platform_clinical_results import build_candidates

        result = build_candidates(
            _resolved(project_root, args.source_root),
            _resolved(project_root, args.output_root),
            code_path=project_root / "pipeline/src/menin_discovery/platform_clinical_results.py",
        )
        _print_json(result)
        return 0
    if args.command == "verify-clinical-results":
        from .platform_clinical_results import verify_candidates

        result = verify_candidates(
            _resolved(project_root, args.output_root),
            source_root=_resolved(project_root, args.source_root),
            code_path=project_root / "pipeline/src/menin_discovery/platform_clinical_results.py",
        )
        _print_json(result)
        return 0
    if args.command in {"prepare-regulatory-records", "verify-regulatory-records"}:
        from .platform_regulatory_records import build_regulatory_records, verify_regulatory_records

        paths = (
            _resolved(project_root, args.raw_root),
            _resolved(project_root, args.output_root),
            _resolved(project_root, args.report_root),
        )
        result = (
            verify_regulatory_records(*paths)
            if args.command == "verify-regulatory-records"
            else build_regulatory_records(*paths)
        )
        _print_json(result)
        return 0
    if args.command == "acquire-pkdb-candidates":
        from .platform_pkdb_candidates import acquire

        result = acquire(_resolved(project_root, args.raw_root))
        _print_json(result)
        return 0
    if args.command in {"prepare-pkdb-candidates", "verify-pkdb-candidates"}:
        from .platform_pkdb_candidates import normalize, verify

        paths = (
            _resolved(project_root, args.raw_root),
            _resolved(project_root, args.interim_root),
            _resolved(project_root, args.report_root),
        )
        result = verify(*paths) if args.command == "verify-pkdb-candidates" else normalize(*paths)
        _print_json(result)
        return 0
    if args.command == "analyze-canonical":
        from .platform_statistical_analysis import run_statistical_analysis

        result = run_statistical_analysis(
            _resolved(project_root, args.canonical_build_root),
            _resolved(project_root, args.qc_report),
            _resolved(project_root, args.output_root),
            batch_size=args.batch_size,
            sample_cap=args.sample_cap,
        )
        _print_json(result)
        return 0
    if args.command == "verify-statistical-analysis":
        from .platform_statistical_analysis import verify_statistical_analysis

        result = verify_statistical_analysis(
            _resolved(project_root, args.output_root),
            canonical_build_root=_resolved(project_root, args.canonical_build_root),
            qc_report_path=_resolved(project_root, args.qc_report),
        )
        _print_json(result)
        return 0
    if args.command == "prepare-split-suite":
        from .platform_split_suite import SplitSuiteConfig, materialize_split_suite

        result = materialize_split_suite(
            _resolved(project_root, args.canonical_build_root),
            _resolved(project_root, args.qc_report),
            _resolved(project_root, args.output_directory),
            SplitSuiteConfig(
                seed=args.seed,
                train_fraction=args.train_fraction,
                validation_fraction=args.validation_fraction,
                test_fraction=args.test_fraction,
                batch_size=args.batch_size,
                near_sample_cap_per_partition=args.near_sample_cap_per_partition,
                chemical_tanimoto_threshold=args.chemical_tanimoto_threshold,
                chemical_fingerprint_bits=args.chemical_fingerprint_bits,
                chemical_fingerprint_radius=args.chemical_fingerprint_radius,
                protein_kmer_size=args.protein_kmer_size,
                protein_jaccard_threshold=args.protein_jaccard_threshold,
            ),
        )
        _print_json(result)
        return 0
    if args.command == "verify-split-suite":
        from .platform_split_suite import verify_split_suite

        result = verify_split_suite(
            _resolved(project_root, args.output_directory),
            canonical_build_root=_resolved(project_root, args.canonical_build_root),
            qc_report_path=_resolved(project_root, args.qc_report),
        )
        _print_json(result)
        return 0
    if args.command == "prepare-corpus-readiness":
        from .platform_corpus_readiness import (
            CorpusReadinessConfig,
            materialize_corpus_readiness_bundle,
        )

        result = materialize_corpus_readiness_bundle(
            _resolved(project_root, args.canonical_build_root),
            _resolved(project_root, args.qc_report),
            _resolved(project_root, args.output_directory),
            CorpusReadinessConfig(
                seed=args.seed,
                preflight_batch_size=args.preflight_batch_size,
                split_batch_size=args.split_batch_size,
                serialization_batch_size=args.serialization_batch_size,
                loader_batch_size=args.loader_batch_size,
                loader_maximum_batches=args.loader_maximum_batches,
                minimum_rows=args.minimum_rows,
                minimum_unique_molecules=args.minimum_unique_molecules,
                minimum_unique_proteins=args.minimum_unique_proteins,
                diagnostic_maximum_train_examples=(args.diagnostic_maximum_train_examples),
                diagnostic_maximum_validation_examples=(args.diagnostic_maximum_validation_examples),
                diagnostic_fingerprint_bits=args.diagnostic_fingerprint_bits,
                diagnostic_fingerprint_radius=args.diagnostic_fingerprint_radius,
            ),
        )
        _print_json(result)
        return 0
    if args.command == "verify-corpus-readiness":
        from .platform_corpus_readiness import verify_corpus_readiness_bundle

        result = verify_corpus_readiness_bundle(
            _resolved(project_root, args.output_directory),
            canonical_build_root=_resolved(project_root, args.canonical_build_root),
            qc_report_path=_resolved(project_root, args.qc_report),
        )
        _print_json(result)
        return 0
    if args.command == "verify-final-artifacts":
        from .platform_final_verification import (
            FinalVerificationPaths,
            run_final_artifact_verification,
        )

        result = run_final_artifact_verification(
            project_root,
            FinalVerificationPaths(output_report=args.output_report.as_posix()),
        )
        _print_json(result)
        return 0
    if args.command == "audit-non-hpc-governance":
        from .platform_non_hpc_governance import materialize_governance_bundle

        result = materialize_governance_bundle(
            project_root,
            _resolved(project_root, args.config),
            _resolved(project_root, args.output_root),
        )
        _print_json(result)
        return 0
    if args.command == "verify-non-hpc-completion":
        from .platform_non_hpc_completion import verify_non_hpc_completion

        result = verify_non_hpc_completion(project_root, args.report)
        _print_json(result)
        return 0
    if args.command == "prepare-static":
        from .platform_pretraining import materialize_static_readiness_registries

        result = materialize_static_readiness_registries(
            feature_directory=_resolved(project_root, args.feature_directory),
            model_directory=_resolved(project_root, args.model_directory),
            evidence_checked_date=args.evidence_checked_date,
        )
        _print_json(result)
        return 0
    if args.command == "integrate-task":
        from .platform_model_integration import (
            TaskIntegrationConfig,
            materialize_task_integration_bundle,
        )

        result = materialize_task_integration_bundle(
            _resolved(project_root, args.task_dataset),
            _resolved(project_root, args.output_directory),
            TaskIntegrationConfig(
                split_name=args.split_name,
                split_strategy=args.split_strategy,
                intended_use=args.intended_use,
                seed=args.seed,
                task_eligibility_mode=args.task_eligibility_mode,
            ),
        )
        _print_json(result)
        return 0
    if args.command == "diagnose-task":
        from .platform_model_integration import (
            DiagnosticConfig,
            materialize_capped_diagnostic_bundle,
        )

        result = materialize_capped_diagnostic_bundle(
            _resolved(project_root, args.integration_bundle),
            _resolved(project_root, args.output_directory),
            integration_acceptance_sha256=args.integration_acceptance_sha256,
            config=DiagnosticConfig(
                seed=args.seed,
                maximum_train_examples=args.maximum_train_examples,
                maximum_validation_examples=args.maximum_validation_examples,
                fingerprint_bits=args.fingerprint_bits,
                fingerprint_radius=args.fingerprint_radius,
                binary_label_mapping=_binary_mapping(args.binary_label),
            ),
        )
        _print_json(result)
        return 0
    raise AssertionError(f"Unhandled platform command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_argument_parser", "main", "platform_status"]
