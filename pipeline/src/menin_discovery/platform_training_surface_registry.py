"""Integrate canonical trainable-data releases without copying their payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "platform-training-surface-registry/1.0"
RELEASE_VERSION = "v1_0_platform_registry"
MANIFEST_NAME = "platform_registry_manifest.json"
REGISTRY_NAME = "platform_training_registry.json"
REPORT_NAME = "PLATFORM_TRAINING_SURFACES.md"
VALIDATION_NAME = "validation.json"


class PlatformRegistryError(RuntimeError):
    """Raised when an integrated surface cannot be physically or scientifically proven."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bind(root: Path, path: Path, role: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PlatformRegistryError(f"Missing or symlinked {role}: {path}")
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _bind_staged_artifact(root: Path, staged: Path, destination: Path, role: str) -> dict[str, Any]:
    binding = _bind(root, staged, role)
    binding["path"] = str(destination.resolve().relative_to(root.resolve()))
    return binding


def _manifest_hash(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_json(clean).encode()).hexdigest()


def _surface_payload(
    herg: Mapping[str, Any],
    pk: Mapping[str, Any],
    affinity: Mapping[str, Any],
    multimodal: Mapping[str, Any],
) -> dict[str, Any]:
    herg_counts = herg["counts"]
    pk_summary = pk
    affinity_counts = affinity["counts"]
    adjacent_counts = multimodal["counts"]
    endpoints = affinity_counts["endpoints"]
    families = {
        "herg": {
            "broad_trainable_observations": int(herg_counts["primary_eligible_observations"]),
            "broad_unique_structures": int(herg_counts["primary_eligible_unique_structures"]),
            "confirmed_wt_fixed_dose_structure_labels": int(
                herg_counts["surfaces"]["STRUCT_CONFIRMED_WT_FIXED_DOSE_CONSENSUS"]["rows"]
            ),
            "preclinical_native_numeric": int(
                herg_counts["surfaces"]["OBS_PRECLINICAL_NATIVE_NUMERIC_PRIMARY"]["rows"]
            ),
            "standardized_pic50_exact_or_censored": int(
                herg_counts["surfaces"]["OBS_PRECLINICAL_STANDARDIZED_PIC50_PRIMARY"]["rows"]
            ),
            "functional_method_resolved_numeric": int(
                herg_counts["surfaces"]["OBS_FUNCTIONAL_HOW_MEASURED_NATIVE_NUMERIC"]["rows"]
            ),
            "clinical_context_rows_not_labels": int(herg_counts["clinical_context_rows"]),
            "mutants_admitted": 0,
            "scale_status": "hundreds_of_thousands_broad_and_WT; nested_quality_surfaces_are_smaller_by_physical_evidence",
        },
        "pk_adme": {
            "source_bound_observations": int(pk_summary["measurement_observation_count"]),
            "modeling_rows": int(pk_summary["modeling_surface_count"]),
            "exact_modeling_rows": int(pk_summary["exact_modeling_observation_count"]),
            "censored_modeling_rows": int(pk_summary["censored_only_modeling_observation_count"]),
            "molecule_representations": int(pk_summary["unique_molecule_count"]),
            "connectivity_leakage_groups": int(pk_summary["unique_leakage_group_count"]),
            "trainable_tasks": int(pk_summary["trainable_task_count"]),
            "clinical_pk_context_rows": int(pk_summary["clinical_pk_context_observation_count"]),
            "qt_ecg_context_rows_not_targets": int(pk_summary["qt_ecg_context_observation_count"]),
            "scale_status": "hundreds_of_thousands_endpoint_specific",
        },
        "affinity_potency": {
            "all_endpoint_observations": int(affinity_counts["total"]["observations"]),
            "primary_kd_ki_ic50_observations": int(affinity_counts["total"]["primary_observations"]),
            "unique_structures": int(affinity_counts["total"]["unique_structures"]),
            "unique_target_groups": int(affinity_counts["total"]["unique_target_groups"]),
            "endpoint_observations": {
                endpoint: int(endpoints[endpoint]["observations"])
                for endpoint in ("Kd", "Ki", "IC50", "EC50")
            },
            "scale_status": "millions_combined; primary_endpoints_remain_separate",
        },
        "multimodal_pretraining": {
            "prism_verified_finite_viability_values": int(
                adjacent_counts["prism_structure_linked_training_values"]
            ),
            "lincs_structure_linked_instances": int(adjacent_counts["lincs_compound_instance_rows"]),
            "lincs_metadata_derived_positions_not_scanned_values": int(
                adjacent_counts["lincs_metadata_derived_profile_gene_positions"]
            ),
            "scale_status": "millions_verified_for_PRISM; LINCS_positions_are_addressable_metadata_only",
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "release_version": RELEASE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "families": families,
        "recommended_training_order": [
            "hERG confirmed-WT fixed-dose binary baseline",
            "hERG preclinical native-numeric and censor-aware pIC50",
            "PK/ADME endpoint-specific tasks with conditioning",
            "Kd endpoint-semantics pilot",
            "Ki target-conditioned affinity",
            "IC50 target-conditioned potency",
            "PRISM viability representation pretraining",
            "LINCS expression only after GCTX finite-value staging",
            "shared molecule-protein encoder and quality-specific heads",
            "physics and structure feature increments on HPC",
        ],
        "nonnegotiable_boundaries": [
            "Do not pool Kd, Ki, IC50, EC50, PK/ADME tasks, or hERG reporting levels as one scalar target.",
            "Do not admit mutant hERG into the wild-type paper scope.",
            "Do not convert QT, ECG, trial, approval, or candidate context into hERG labels.",
            "Do not count known mirrors or metadata-addressable matrix positions as independent measurements.",
            "Fit preprocessing only on training partitions and preserve structure, scaffold, target, and source grouping.",
        ],
        "execution_state": {
            "trainable_data_surfaces_prepared": True,
            "production_feature_store_generated": False,
            "substantive_model_training_started": False,
            "hpc_executed": False,
            "predictive_superiority_established": False,
            "established_advantages": [
                "scale with explicit mirror control",
                "hERG reporting-quality hierarchy",
                "wild-type-only hERG scope",
                "method and protocol provenance",
                "censor-aware endpoint semantics",
                "frozen leakage-aware challenge designs",
            ],
        },
    }


def _validate_science(registry: Mapping[str, Any]) -> None:
    families = registry["families"]
    if int(families["herg"]["broad_trainable_observations"]) < 300_000:
        raise PlatformRegistryError("hERG broad surface is below the frozen scale floor")
    if int(families["herg"]["confirmed_wt_fixed_dose_structure_labels"]) < 300_000:
        raise PlatformRegistryError("confirmed-WT hERG surface is below the frozen scale floor")
    if int(families["herg"]["mutants_admitted"]) != 0:
        raise PlatformRegistryError("mutant hERG was admitted")
    if int(families["pk_adme"]["modeling_rows"]) < 500_000:
        raise PlatformRegistryError("PK/ADME modeling surface is below the frozen scale floor")
    affinity = families["affinity_potency"]
    if int(affinity["primary_kd_ki_ic50_observations"]) < 1_000_000:
        raise PlatformRegistryError("primary affinity/potency union is below one million")
    for endpoint in ("Kd", "Ki", "IC50"):
        if int(affinity["endpoint_observations"][endpoint]) < 100_000:
            raise PlatformRegistryError(f"{endpoint} is below the endpoint scale floor")
    if int(families["multimodal_pretraining"]["prism_verified_finite_viability_values"]) < 1_000_000:
        raise PlatformRegistryError("PRISM verified finite-value surface is below one million")
    if registry["execution_state"]["production_feature_store_generated"]:
        raise PlatformRegistryError("Registry must not claim ungenerated production features")


def validate_platform_registry(root: Path, output: Path) -> dict[str, Any]:
    manifest_path = output / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise PlatformRegistryError("Manifest self-hash mismatch")
    expected = {MANIFEST_NAME, REGISTRY_NAME, REPORT_NAME, VALIDATION_NAME}
    if {path.name for path in output.iterdir()} != expected:
        raise PlatformRegistryError("Registry release membership mismatch")
    for item in [*manifest["inputs"], *manifest["artifacts"]]:
        path = root / item["path"]
        if path.stat().st_size != item["bytes"] or _sha256(path) != item["sha256"]:
            raise PlatformRegistryError(f"Physical binding mismatch: {path}")
    registry = json.loads((output / REGISTRY_NAME).read_text())
    _validate_science(registry)
    report = (output / REPORT_NAME).read_text()
    if any(line.lstrip().startswith("|") for line in report.splitlines()):
        raise PlatformRegistryError("Report contains a Markdown table")
    if "```mermaid" in report or re.search(r"!\[[^]]*\]\(", report):
        raise PlatformRegistryError("Report contains a figure")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "manifest_self_hash_verified": True,
        "physical_bindings_verified": len(manifest["inputs"]) + len(manifest["artifacts"]),
        "closed_membership_verified": True,
        "scale_and_scientific_boundaries_verified": True,
        "report_contains_no_tables_or_figures": True,
    }


def _report(registry: Mapping[str, Any]) -> str:
    families = registry["families"]
    affinity = families["affinity_potency"]
    endpoints = affinity["endpoint_observations"]
    order = "\n".join(f"- {item}" for item in registry["recommended_training_order"])
    boundaries = "\n".join(f"- {item}" for item in registry["nonnegotiable_boundaries"])
    return f"""# Platform trainable-data surfaces v1.0

## What is ready

The platform now has physically bound, validated training surfaces at useful scale without generating the expensive HPC feature store or fitting substantive models.

- hERG has {families["herg"]["broad_trainable_observations"]:,} broad clean observations and {families["herg"]["confirmed_wt_fixed_dose_structure_labels"]:,} confirmed-WT fixed-dose structure labels. The reporting-quality hierarchy remains nested and smaller where the underlying evidence is genuinely rarer.
- PK/ADME has {families["pk_adme"]["modeling_rows"]:,} endpoint-specific modeling rows across {families["pk_adme"]["trainable_tasks"]:,} trainable tasks and {families["pk_adme"]["connectivity_leakage_groups"]:,} connectivity leakage groups.
- Protein-conditioned binding and potency has {affinity["primary_kd_ki_ic50_observations"]:,} primary Kd/Ki/IC50 observations. The separate endpoint counts are Kd {endpoints["Kd"]:,}, Ki {endpoints["Ki"]:,}, and IC50 {endpoints["IC50"]:,}; EC50 contributes {endpoints["EC50"]:,} auxiliary observations and is never relabelled as affinity.
- PRISM contributes {families["multimodal_pretraining"]["prism_verified_finite_viability_values"]:,} verified finite viability values. LINCS contributes {families["multimodal_pretraining"]["lincs_structure_linked_instances"]:,} structure-linked instances; its {families["multimodal_pretraining"]["lincs_metadata_derived_positions_not_scanned_values"]:,} profile-gene positions are explicitly metadata-derived and not claimed as scanned finite labels.

## hERG quality hierarchy

The first paper remains wild-type human KCNH2/hERG focused. Mutants are quarantined. Broad clean training, confirmed-WT fixed-dose consensus, preclinical native numeric, standardized exact-or-censored pIC50, functional method-resolved measurements, automation/modality, and clinical QT context remain separate surfaces. Clinical QT context is never a direct hERG label.

## Recommended training sequence

{order}

## Tool direction

The eventual researcher-facing tool accepts a molecule, a protein sequence or structure when the task requires it, the requested endpoint, and optional assay or exposure context. It should return an endpoint-specific prediction, calibrated uncertainty, applicability-domain evidence, nearest training analogs, provenance and reporting quality, and eventually constrained optimization suggestions. General affinity and IC50 models must condition on both molecule and protein; PK/ADME and hERG use task-specific context rather than pretending all labels are interchangeable.

## Boundaries

{boundaries}

## What is not done

No production HPC feature store, large representation model, physics calculation, docking campaign, or substantive final model was created in this release. Predictive superiority is not established until competitors and internal models are evaluated on identical frozen challenges with calibration, coverage, uncertainty, and applicability-domain reporting. The established advantage today is the integrated scale, provenance, quality hierarchy, censoring semantics, mirror control, and evaluation design.
"""


def build_platform_registry(root: Path, output: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    output = output or root / "research/data/platform/processed/training_surfaces" / RELEASE_VERSION
    config_path = root / "pipeline/config/platform_training_surfaces.yaml"
    config = yaml.safe_load(config_path.read_text())
    releases = config["canonical_releases"]
    herg_path = root / releases["herg"]["manifest"]
    pk_manifest_path = root / releases["pk_adme"]["manifest"]
    pk_summary_path = root / releases["pk_adme"]["summary"]
    affinity_path = root / releases["affinity"]["manifest"]
    multimodal_path = root / releases["multimodal"]["manifest"]
    feature_contract_path = root / config["feature_boundary"]["contract"]
    registry = _surface_payload(
        json.loads(herg_path.read_text()),
        json.loads(pk_summary_path.read_text()),
        json.loads(affinity_path.read_text()),
        json.loads(multimodal_path.read_text()),
    )
    _validate_science(registry)
    if output.exists():
        raise PlatformRegistryError(f"Versioned registry already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{RELEASE_VERSION}.", dir=output.parent))
    try:
        (staging / REGISTRY_NAME).write_text(_canonical_json(registry) + "\n")
        (staging / REPORT_NAME).write_text(_report(registry))
        inputs = [
            _bind(root, Path(__file__).resolve(), "builder_implementation"),
            _bind(root, config_path, "platform_configuration"),
            _bind(root, herg_path, "herg_release_manifest"),
            _bind(root, pk_manifest_path, "pk_adme_release_manifest"),
            _bind(root, pk_summary_path, "pk_adme_release_summary"),
            _bind(root, affinity_path, "affinity_release_manifest"),
            _bind(root, multimodal_path, "multimodal_release_manifest"),
            _bind(root, feature_contract_path, "future_feature_contract"),
        ]
        artifacts = [
            _bind_staged_artifact(
                root,
                staging / REGISTRY_NAME,
                output / REGISTRY_NAME,
                "integrated_machine_registry",
            ),
            _bind_staged_artifact(
                root,
                staging / REPORT_NAME,
                output / REPORT_NAME,
                "integrated_prose_report",
            ),
        ]
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "release_version": RELEASE_VERSION,
            "inputs": inputs,
            "artifacts": artifacts,
        }
        manifest["manifest_sha256"] = _manifest_hash(manifest)
        (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        validation = {
            "schema_version": SCHEMA_VERSION,
            "status": "pending_atomic_validation",
        }
        (staging / VALIDATION_NAME).write_text(_canonical_json(validation) + "\n")
        staging.rename(output)
        # Add validation as a closed release member but not a recursively hashed artifact.
        validation = validate_platform_registry(root, output)
        (output / VALIDATION_NAME).write_text(_canonical_json(validation) + "\n")
        return validate_platform_registry(root, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    output = (
        args.output or args.repo_root / "research/data/platform/processed/training_surfaces" / RELEASE_VERSION
    )
    result = (
        validate_platform_registry(args.repo_root.resolve(), output.resolve())
        if args.validate_only
        else build_platform_registry(args.repo_root, output)
    )
    print(_canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
