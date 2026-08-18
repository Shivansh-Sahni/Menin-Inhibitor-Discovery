"""Lead-owned cross-workstream mechanical verification and evidence binding.

This module deliberately distinguishes byte/schema/source verification from a
scientific or governance authorization to train.  It replays every public
artifact verifier, requires independent canonical/statistical/split builds,
and writes one deterministic report that binds the accepted physical bytes.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .platform_corpus_readiness import verify_corpus_readiness_bundle
from .platform_data_schema import canonical_json
from .platform_data_sources import sha256_file
from .platform_determinism import compare_canonical_builds
from .platform_external_normalization import verify_external_normalized_output
from .platform_pretraining import materialize_static_readiness_registries
from .platform_split_suite import verify_split_suite
from .platform_statistical_analysis import verify_statistical_analysis

SCHEMA_VERSION = "platform_final_artifact_verification_v1"
NO_SUBSTANTIVE_TRAINING = False
FINAL_REPORT_NAMESPACE = Path("research/reports/final_verification")
PLATFORM_CONFIG_SHA256 = "b00971d4d5d4aaf9d765793cc03c23dd73171a2097a3a5dc1b0f2b832fbd1058"
DEPENDENCY_AUDIT_SHA256 = "c223e9769bae512294ab5e1760ba6668084e1e8f6126bf390bf62847a85bb7cf"
DEPENDENCY_AUDIT_COMMAND = (
    ".venv/bin/python -m pip_audit --strict --no-deps --disable-pip "
    "--progress-spinner off --cache-dir .cache/pip-audit --format json "
    "--requirement pipeline/environments/requirements.lock"
)


@dataclass(frozen=True)
class FinalVerificationPaths:
    """Portable project-relative inputs to the final mechanical gate."""

    external_raw: str = "research/data/platform/raw/external_public"
    external_normalized: str = "research/data/platform/interim/external_public_normalized"
    canonical_primary: str = "research/data/platform/canonical/full_chembl37"
    reports_primary: str = "research/reports/platform"
    canonical_secondary: str = "research/data/platform/determinism_build_b/full_chembl37"
    reports_secondary: str = "research/reports/platform/determinism_build_b"
    canonical_determinism_report: str = "research/reports/platform/canonical_determinism_verification.json"
    statistical_primary: str = "research/reports/platform/statistical_analysis"
    statistical_secondary: str = "research/reports/platform/statistical_analysis_determinism_b"
    split_primary: str = "research/data/platform/splits/full_chembl37"
    split_secondary: str = "research/data/platform/splits/determinism_build_b/full_chembl37"
    corpus_readiness: str = "research/models/platform/corpus_readiness/full_chembl37"
    static_manifest: str = "research/models/platform/pretraining_static_manifest.json"
    platform_config: str = "pipeline/config/platform.yaml"
    dependency_audit: str = "research/reports/platform/audit/dependency_vulnerability_audit.json"
    output_report: str = "research/reports/final_verification/platform_final_artifact_verification.json"


_FALSE_TRAINING_FIELDS = frozenset(
    {
        "large_model_training_started",
        "substantive_large_model_training_authorized",
        "substantive_model_training_performed",
        "substantive_training_authorized",
        "substantive_training_started",
        "training_authorized",
        "enabled_for_training",
        "training_command_exposed",
    }
)
_TRUE_ZERO_TRAINING_FIELDS = frozenset({"zero_training", "zero_training_flag"})
_POSITIVE_TRAINING_KEY = re.compile(
    r"^(?:.*_)?training_(?:enabled|allowed|permitted|authorized|started|performed|ready)$"
    r"|^(?:enable|allow|permit|authorize|start|perform)_(?:.*_)?training$"
)

_STATIC_ARTIFACT_PATHS = {
    "baseline_robustness_matrix_csv": "baseline_robustness_matrix.csv",
    "feature_registry_csv": "../../data/platform/features/static/feature_registry.csv",
    "feature_registry_metadata": ("../../data/platform/features/static/feature_registry_metadata.json"),
    "model_candidate_registry_csv": "model_candidate_registry.csv",
    "model_candidate_registry_json": "model_candidate_registry.json",
    "model_metric_registry_csv": "model_metric_registry.csv",
}

_EXTERNAL_SOURCE_BINDINGS = {
    "bindingdb_curated_202608": ("bindingdb_curated_202608/bindingdb_curated_202608_manifest.json"),
    "clinicaltrials_gov_v2": ("clinicaltrials_gov_v2/clinicaltrials_gov_v2_manifest.json"),
    "dailymed_spl_v2_human_rx": ("dailymed_spl_v2_human_rx/dailymed_spl_v2_human_rx_manifest.json"),
    "drugs_at_fda_bulk": "drugs_at_fda_bulk/drugs_at_fda_bulk_manifest.json",
    "uniprotkb_targeted_2026_02": ("uniprotkb_targeted_2026_02/uniprotkb_targeted_2026_02_manifest.json"),
}
_EXTERNAL_MANIFEST_SHA256 = {
    "bindingdb_curated_202608": ("f430fdf4f5740708f5ac77089b9c0091604010338020c8ed83f80ee541d5c189"),
    "clinicaltrials_gov_v2": ("42707517b8afd0d74b8c3ad1abcd4457fec273596ac7001d710ba37a80d9a6ea"),
    "dailymed_spl_v2_human_rx": ("f2a2c2fca158bfd524c3d1d8acd4edb7bcf8c2db3a4212bc6595cf7cef359886"),
    "drugs_at_fda_bulk": ("406f87317a0f16b7c99a38dd5ec34aa86f0174f4158b6c97a6a17413528bcab8"),
    "uniprotkb_targeted_2026_02": ("781072c1959bf41dcfeccf3ebf6a66bbfc8ab3fba53624fbd551ac52d373d3cf"),
}

_CHILD_RESULT_KEYS = {
    "canonical determinism": frozenset(
        {
            "schema_version",
            "status",
            "content_equivalent",
            "canonical_component_count",
            "canonical_component_bytes",
            "qc_generated_artifact_count",
            "build_manifest_a_sha256",
            "build_manifest_b_sha256",
            "normalized_build_manifest_sha256",
            "qc_report_a_sha256",
            "qc_report_b_sha256",
            "normalized_qc_report_sha256",
            "ignored_nondeterministic_fields",
            "large_model_training_started",
            "substantive_training_started",
        }
    ),
    "external normalization": frozenset(
        {
            "status",
            "schema_version",
            "output_root",
            "manifest_declared_sha256",
            "manifest_physical_sha256",
            "manifest_physical_bytes",
            "inventory_entries",
            "parquet_artifacts",
            "aggregate_artifact_bytes",
            "aggregate_parquet_rows",
            "input_verification",
            "verified_input_count",
            "semantic_verification",
            "zero_label_training_and_identity_replacement_contract",
        }
    ),
    "primary statistical analysis": frozenset(
        {
            "analysis_version",
            "status",
            "analysis_manifest_sha256",
            "artifact_count",
            "canonical_component_count",
            "source_reverified",
            "zero_training",
            "training_actions",
            "scientific_boundaries_verified",
        }
    ),
    "secondary statistical analysis": frozenset(
        {
            "analysis_version",
            "status",
            "analysis_manifest_sha256",
            "artifact_count",
            "canonical_component_count",
            "source_reverified",
            "zero_training",
            "training_actions",
            "scientific_boundaries_verified",
        }
    ),
    "statistical determinism": frozenset(
        {"status", "file_count", "directory_count", "aggregate_bytes", "inventory_sha256"}
    ),
    "primary split suite": frozenset(
        {
            "schema_version",
            "status",
            "acceptance_file_sha256",
            "component_count",
            "accounting",
            "source_reverified",
            "label_values_read",
            "test_labels_disclosed",
            "large_model_training_started",
            "substantive_training_started",
        }
    ),
    "secondary split suite": frozenset(
        {
            "schema_version",
            "status",
            "acceptance_file_sha256",
            "component_count",
            "accounting",
            "source_reverified",
            "label_values_read",
            "test_labels_disclosed",
            "large_model_training_started",
            "substantive_training_started",
        }
    ),
    "split determinism": frozenset(
        {"status", "file_count", "directory_count", "aggregate_bytes", "inventory_sha256"}
    ),
    "corpus readiness": frozenset(
        {
            "schema_version",
            "status",
            "acceptance_file_sha256",
            "component_count",
            "task_counts",
            "source_reverified",
            "test_lockboxes_opened_or_hashed",
            "large_model_training_started",
            "substantive_training_started",
        }
    ),
    "static readiness": frozenset(
        {"status", "manifest", "artifact_count", "artifacts", "substantive_training_started"}
    ),
    "platform configuration": frozenset({"status", "config"}),
    "no-training scan": frozenset(
        {
            "status",
            "json_documents_scanned",
            "jsonl_or_routed_test_payloads_opened",
            "large_model_training_started",
            "substantive_training_started",
        }
    ),
}

_HUMAN_OR_EXTERNAL_BLOCKERS = (
    {
        "blocker_id": "external_evidence_canonical_admission_and_clinical_outcomes",
        "required_resolution": (
            "Resolve source-specific rights/admission, ambiguity and quarantine, cross-source "
            "duplicate/conflict reconciliation, molecule linkage, and genuinely reported "
            "clinical, PK, QT/QTc, and outcome evidence. External normalized candidates "
            "currently admit zero canonical observations or model labels, while the accepted "
            "canonical build is ChEMBL-only."
        ),
    },
    {
        "blocker_id": "public_release_hygiene_and_artifact_storage",
        "required_resolution": (
            "Human review of the complete staged inventory, secrets/PII/private-correspondence "
            "scan, and an explicit content-addressed storage/redistribution decision for large "
            "local artifacts."
        ),
    },
    {
        "blocker_id": "repository_and_redistribution_licenses",
        "required_resolution": (
            "Human approval of repository license and every conditional source/model redistribution term."
        ),
    },
    {
        "blocker_id": "task_intended_use_and_leakage_thresholds",
        "required_resolution": (
            "Scientific owner must freeze intended use and accept exhaustive/indexed ligand, "
            "protein, source, assay, and temporal leakage thresholds."
        ),
    },
    {
        "blocker_id": "large_model_checkpoint_and_overlap",
        "required_resolution": (
            "Freeze exact checkpoint revisions/hashes, weight terms, training cutoff, and "
            "corpus-overlap audit."
        ),
    },
    {
        "blocker_id": "authorized_compute_and_operations",
        "required_resolution": (
            "Approve accelerator allocation, measured budget, monitoring, checkpoint/resume, "
            "and responsible-compute controls."
        ),
    },
    {
        "blocker_id": "independent_external_validation",
        "required_resolution": (
            "Obtain appropriately designed external/prospective evidence before broad "
            "translational, safety, or product claims."
        ),
    },
    {
        "blocker_id": "unidentified_100k_structure_model",
        "required_resolution": (
            "Identify and review the meeting-referenced model reportedly trained on about 100,000 structures."
        ),
    },
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"Duplicate JSON key is forbidden: {key!r}")
        document[key] = value
    return document


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _json_document(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON artifact: {path}") from exc


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys at every mapping depth."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ValueError("Platform configuration contains an unhashable YAML key") from exc
        if duplicate:
            raise ValueError(f"Duplicate YAML key is forbidden: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _yaml_document(path: Path) -> Any:
    try:
        return yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid YAML artifact: {path}") from exc


def _checked_path(path: Path, *, directory: bool, label: str) -> Path:
    raw = Path(path)
    if ".." in raw.parts or any(ord(character) < 32 for character in os.fspath(raw)):
        raise ValueError(f"{label} contains parent traversal or control characters")
    lexical = Path(os.path.abspath(raw))
    for member in (lexical, *lexical.parents):
        if member.is_symlink():
            raise ValueError(f"{label} path chain contains a symlink: {member}")
    if directory and not lexical.is_dir():
        raise NotADirectoryError(lexical)
    if not directory and not lexical.is_file():
        raise FileNotFoundError(lexical)
    return lexical


def _project_path(project_root: Path, relative: str, *, directory: bool) -> Path:
    candidate = Path(relative)
    if (
        candidate.is_absolute()
        or "\\" in relative
        or ".." in candidate.parts
        or candidate.as_posix() != relative
    ):
        raise ValueError(f"Final-verification path is not portable: {relative!r}")
    resolved = _checked_path(
        project_root / candidate,
        directory=directory,
        label=f"final-verification input {relative}",
    )
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"Final-verification path escapes project root: {relative}") from exc
    return resolved


def _file_record(path: Path, project_root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(project_root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _tree_records(
    root: Path,
) -> tuple[list[dict[str, Any]], set[str], set[tuple[int, int]]]:
    records: list[dict[str, Any]] = []
    directories: set[str] = set()
    inodes: set[tuple[int, int]] = set()
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"Compared artifact tree contains a symlink: {path}")
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            directories.add(relative)
        elif path.is_file():
            status = path.stat(follow_symlinks=False)
            inode = (status.st_dev, status.st_ino)
            if status.st_nlink != 1 or inode in inodes:
                raise ValueError(f"Compared artifact tree contains a hardlinked file: {path}")
            inodes.add(inode)
            records.append(
                {
                    "path": relative,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        else:
            raise ValueError(f"Compared artifact tree contains a special entry: {path}")
    return records, directories, inodes


def compare_exact_artifact_trees(
    primary: str | os.PathLike[str], secondary: str | os.PathLike[str]
) -> dict[str, Any]:
    """Require two generated artifact trees to be byte-identical and symlink-free."""

    left = _checked_path(Path(primary), directory=True, label="primary artifact tree")
    right = _checked_path(Path(secondary), directory=True, label="secondary artifact tree")
    if left == right:
        raise ValueError("Independent artifact-tree comparison requires distinct roots")
    left_records, left_directories, left_inodes = _tree_records(left)
    right_records, right_directories, right_inodes = _tree_records(right)
    if left_inodes & right_inodes:
        raise ValueError("Independent artifact trees share physical file inodes")
    if left_directories != right_directories:
        raise ValueError("Independent artifact trees have different directory membership")
    if left_records != right_records:
        raise ValueError("Independent artifact trees are not byte-identical")
    return {
        "status": "passed_byte_identical",
        "file_count": len(left_records),
        "directory_count": len(left_directories),
        "aggregate_bytes": sum(int(record["size_bytes"]) for record in left_records),
        "inventory_sha256": sha256_file_from_text(canonical_json(left_records)),
    }


def sha256_file_from_text(value: str) -> str:
    """Return a SHA-256 digest without creating an intermediate file."""

    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _verify_static_manifest(manifest_path: Path, project_root: Path) -> dict[str, Any]:
    raw = _json_document(manifest_path)
    if not isinstance(raw, Mapping):
        raise ValueError("Static pretraining manifest must be an object")
    if set(raw) != {
        "schema_version",
        "evidence_checked_date",
        "artifacts",
        "environment",
        "substantive_training_started",
    }:
        raise ValueError("Static pretraining manifest top-level schema drifted")
    evidence_checked_date = raw.get("evidence_checked_date")
    if (
        raw.get("schema_version") != "static_pretraining_readiness_manifest_v1"
        or raw.get("substantive_training_started") is not False
        or not isinstance(evidence_checked_date, str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}", evidence_checked_date) is None
    ):
        raise ValueError("Static pretraining manifest violates its no-training contract")
    artifacts = raw.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(_STATIC_ARTIFACT_PATHS):
        raise ValueError("Static pretraining manifest must bind the exact six artifacts")

    with tempfile.TemporaryDirectory(prefix="platform-static-verification-") as temporary:
        replay_root = Path(temporary).resolve()
        replay_feature_directory = replay_root / "research/data/platform/features/static"
        replay_model_directory = replay_root / "research/models/platform"
        materialize_static_readiness_registries(
            feature_directory=replay_feature_directory,
            model_directory=replay_model_directory,
            evidence_checked_date=evidence_checked_date,
        )
        replay_manifest_path = replay_model_directory / "pretraining_static_manifest.json"
        replay_manifest = _json_document(replay_manifest_path)
        if raw != replay_manifest:
            raise ValueError("Static pretraining manifest differs from deterministic regeneration")

        verified: list[dict[str, Any]] = []
        for name, expected_relative in sorted(_STATIC_ARTIFACT_PATHS.items()):
            raw_record = artifacts.get(name)
            if not isinstance(raw_record, Mapping) or set(raw_record) != {"path", "sha256"}:
                raise ValueError(f"Static artifact record schema drift: {name}")
            relative = raw_record.get("path")
            if relative != expected_relative:
                raise ValueError(f"Static artifact path binding drift: {name}")
            artifact = _checked_path(
                Path(os.path.abspath(manifest_path.parent / expected_relative)),
                directory=False,
                label=f"static artifact {name}",
            )
            try:
                artifact.relative_to(project_root)
            except ValueError as exc:
                raise ValueError(f"Static artifact escapes project root: {name}") from exc
            replay_artifact = _checked_path(
                Path(os.path.abspath(replay_model_directory / expected_relative)),
                directory=False,
                label=f"regenerated static artifact {name}",
            )
            if artifact.read_bytes() != replay_artifact.read_bytes():
                raise ValueError(f"Static artifact differs from deterministic regeneration: {name}")
            if sha256_file(artifact) != raw_record.get("sha256"):
                raise ValueError(f"Static artifact hash drift: {name}")
            verified.append({"name": name, **_file_record(artifact, project_root)})
    return {
        "status": "verified",
        "manifest": _file_record(manifest_path, project_root),
        "artifact_count": len(verified),
        "artifacts": verified,
        "substantive_training_started": False,
    }


def _verify_platform_config(path: Path, project_root: Path) -> dict[str, Any]:
    document = _yaml_document(path)
    if not isinstance(document, Mapping):
        raise ValueError("Platform configuration must be an object")
    project = document.get("project")
    pretraining = document.get("pretraining_interface")
    release = document.get("release")
    tasks = document.get("tasks")
    clinical = tasks.get("clinical") if isinstance(tasks, Mapping) else None
    if (
        document.get("schema_version") != "protein-molecule-platform-config-1.0"
        or not isinstance(project, Mapping)
        or project.get("substantive_large_model_training_authorized") is not False
        or not isinstance(pretraining, Mapping)
        or pretraining.get("substantive_training_authorized") is not False
        or not isinstance(clinical, Mapping)
        or clinical.get("enabled_for_training") is not False
        or not isinstance(release, Mapping)
        or release.get("public_only") is not True
        or release.get("allowed_access_classes") != ["public_redistributable"]
    ):
        raise ValueError("Platform configuration violates no-training/public-only policy")
    _scan_training_value(document, location=path.as_posix())
    if sha256_file(path) != PLATFORM_CONFIG_SHA256:
        raise ValueError("Frozen platform configuration identity drifted")
    return {"status": "verified", "config": _file_record(path, project_root)}


def _verify_dependency_audit(path: Path, project_root: Path) -> dict[str, Any]:
    document = _json_document(path)
    if not isinstance(document, Mapping) or set(document) != {
        "audit_command",
        "audited_at_utc",
        "dependency_count",
        "fixes",
        "input",
        "limitations",
        "result",
        "results",
        "schema_version",
        "substantive_training_started",
        "tool",
        "vulnerability_count",
    }:
        raise ValueError("Dependency vulnerability audit schema drifted")
    if sha256_file(path) != DEPENDENCY_AUDIT_SHA256:
        raise ValueError("Frozen dependency vulnerability audit identity drifted")
    tool = document.get("tool")
    input_record = document.get("input")
    results = document.get("results")
    if (
        document.get("schema_version") != "dependency_vulnerability_audit_v1"
        or document.get("audit_command") != DEPENDENCY_AUDIT_COMMAND
        or document.get("result") != "no_known_vulnerabilities_found"
        or type(document.get("dependency_count")) is not int
        or document.get("dependency_count") != 53
        or type(document.get("vulnerability_count")) is not int
        or document.get("vulnerability_count") != 0
        or document.get("fixes") != []
        or document.get("substantive_training_started") is not False
        or tool
        != {
            "name": "pip-audit",
            "version": "2.10.1",
            "vulnerability_service": "pypi",
        }
        or not isinstance(input_record, Mapping)
        or input_record.get("path") != "pipeline/environments/requirements.lock"
        or not isinstance(results, list)
        or len(results) != 53
    ):
        raise ValueError("Dependency vulnerability audit did not record an accepted exact-lock run")
    lock_path = _project_path(
        project_root,
        "pipeline/environments/requirements.lock",
        directory=False,
    )
    if input_record.get("sha256") != sha256_file(lock_path):
        raise ValueError("Dependency vulnerability audit lock binding drifted")
    locked_identities: set[tuple[str, str]] = set()
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)", stripped)
        if match is None:
            raise ValueError(f"Dependency lock contains a non-exact requirement: {stripped}")
        identity = (
            re.sub(r"[-_.]+", "-", match.group(1)).casefold(),
            match.group(2),
        )
        if identity in locked_identities:
            raise ValueError(f"Dependency lock contains a duplicate identity: {identity[0]}")
        locked_identities.add(identity)
    if len(locked_identities) != 53:
        raise ValueError("Dependency lock no longer contains the exact 53 audited identities")
    identities: set[tuple[str, str]] = set()
    for result in results:
        if not isinstance(result, Mapping) or set(result) != {
            "name",
            "version",
            "vulnerabilities",
        }:
            raise ValueError("Dependency vulnerability result schema drifted")
        name, version = result.get("name"), result.get("version")
        normalized_identity = (
            (
                re.sub(r"[-_.]+", "-", name).casefold(),
                version,
            )
            if isinstance(name, str) and isinstance(version, str)
            else ("", "")
        )
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            or result.get("vulnerabilities") != []
            or normalized_identity in identities
        ):
            raise ValueError("Dependency vulnerability result identity or status drifted")
        identities.add(normalized_identity)
    if identities != locked_identities:
        raise ValueError("Dependency audit identities do not exactly match the dependency lock")
    return {
        "status": "verified_no_known_vulnerabilities",
        "dependency_count": len(identities),
        "vulnerability_count": 0,
        "point_in_time_only": True,
        "audit": _file_record(path, project_root),
        "lock": _file_record(lock_path, project_root),
    }


def _scan_training_value(value: Any, *, location: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized_key = key.casefold() if isinstance(key, str) else key
            requires_false = normalized_key in _FALSE_TRAINING_FIELDS or (
                isinstance(normalized_key, str)
                and (
                    normalized_key.endswith("training_authorized")
                    or normalized_key.endswith("enabled_for_training")
                    or _POSITIVE_TRAINING_KEY.fullmatch(normalized_key) is not None
                )
            )
            if requires_false and child is not False:
                raise ValueError(f"Training boundary became true at {location}.{key}")
            if normalized_key in _TRUE_ZERO_TRAINING_FIELDS and child is not True:
                raise ValueError(f"Zero-training boundary became false at {location}.{key}")
            if normalized_key == "training_actions" and child != []:
                raise ValueError(f"Training actions are nonempty at {location}.{key}")
            _scan_training_value(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_training_value(child, location=f"{location}[{index}]")


def _scan_no_training_json(roots: Sequence[Path]) -> dict[str, Any]:
    seen: set[Path] = set()
    scanned = 0
    for root in roots:
        for path in sorted(root.rglob("*.json"), key=lambda item: item.as_posix()):
            if path in seen:
                continue
            seen.add(path)
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"No-training scan encountered an unsafe JSON path: {path}")
            _scan_training_value(_json_document(path), location=path.as_posix())
            scanned += 1
    return {
        "status": "passed",
        "json_documents_scanned": scanned,
        "jsonl_or_routed_test_payloads_opened": False,
        "large_model_training_started": False,
        "substantive_training_started": False,
    }


def _write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Final verification report path is a symlink: {path}")
    payload = json.dumps(dict(document), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if not path.is_file():
            raise ValueError(f"Final verification report is not a regular file: {path}")
        if path.read_text(encoding="utf-8") == payload:
            return
        raise RuntimeError(
            "Refusing to overwrite a non-identical final verification report; "
            "preserve or explicitly relocate the prior evidence first"
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"Stale final-verification transaction exists: {temporary}")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_output_path(project_root: Path, relative: str) -> Path:
    raw = Path(relative)
    if (
        not relative
        or relative != relative.strip()
        or "\\" in relative
        or raw.is_absolute()
        or ".." in raw.parts
        or raw.as_posix() != relative
        or any(part in {"", "."} for part in raw.parts)
        or any(ord(character) < 32 for character in relative)
    ):
        raise ValueError("Final verification output path is not canonical and portable")
    if raw.parts[: len(FINAL_REPORT_NAMESPACE.parts)] != FINAL_REPORT_NAMESPACE.parts:
        raise ValueError("Final verification output must stay in the designated final-report namespace")
    output = Path(os.path.abspath(project_root / raw))
    try:
        output.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("Final verification output escapes project root") from exc
    current = project_root
    for part in raw.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"Final verification output path chain contains a symlink: {current}")
        if current.exists() and not current.is_dir():
            raise ValueError(f"Final verification output parent is not a directory: {current}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _checked_path(output.parent, directory=True, label="final verification output parent")
    if output.exists() and (output.is_symlink() or not output.is_file()):
        raise ValueError(f"Final verification output is not a regular file: {output}")
    if output.exists() and output.stat(follow_symlinks=False).st_nlink != 1:
        raise ValueError("Final verification output is hardlinked")
    return output


def _protect_verified_inputs(output: Path, resolved: Mapping[str, Path], *, project_root: Path) -> None:
    """Reject report locations that could mutate any source or verified artifact tree."""

    for field, candidate in resolved.items():
        if candidate.is_dir():
            try:
                output.relative_to(candidate)
            except ValueError:
                pass
            else:
                raise ValueError(f"Final verification output overlaps immutable input root: {field}")
        elif output == candidate:
            raise ValueError(f"Final verification output aliases immutable input: {field}")
    namespace = project_root / FINAL_REPORT_NAMESPACE
    if output == namespace:
        raise ValueError("Final verification output must name a file inside its namespace")


def _verify_generated_storage_is_unaliased(roots: Sequence[Path]) -> dict[str, Any]:
    """Reject symlinks, special files, hardlinks, and repeated inodes in generated evidence."""

    seen_paths: set[Path] = set()
    inode_owner: dict[tuple[int, int], Path] = {}
    files_checked = 0
    for root in roots:
        for path in (root, *sorted(root.rglob("*"), key=lambda item: item.as_posix())):
            lexical = Path(os.path.abspath(path))
            if lexical in seen_paths:
                continue
            seen_paths.add(lexical)
            if lexical.is_symlink():
                raise ValueError(f"Generated evidence contains a symlink: {lexical}")
            if lexical.is_dir():
                continue
            if not lexical.is_file():
                raise ValueError(f"Generated evidence contains a special entry: {lexical}")
            status = lexical.stat(follow_symlinks=False)
            inode = (status.st_dev, status.st_ino)
            owner = inode_owner.get(inode)
            if status.st_nlink != 1 or (owner is not None and owner != lexical):
                raise ValueError(f"Generated evidence file is hardlinked or inode-aliased: {lexical}")
            inode_owner[inode] = lexical
            files_checked += 1
    return {
        "status": "passed",
        "regular_files_checked": files_checked,
        "hardlinks_or_inode_aliases": 0,
    }


def _verify_external_source_bindings(normalized_root: Path, raw_root: Path) -> dict[str, str]:
    manifest = _json_document(normalized_root / "external_public_normalized_manifest.json")
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("inputs"), list):
        raise ValueError("External normalized manifest lacks input bindings")
    observed: dict[str, str] = {}
    for raw in manifest["inputs"]:
        if not isinstance(raw, Mapping):
            raise ValueError("External normalized input binding is malformed")
        source_id = raw.get("source_id")
        manifest_path = raw.get("manifest_path")
        if not isinstance(source_id, str) or not isinstance(manifest_path, str):
            raise ValueError("External normalized source identity is malformed")
        if source_id in observed:
            raise ValueError(f"Duplicate external normalized source identity: {source_id}")
        expected_sha256 = _EXTERNAL_MANIFEST_SHA256.get(source_id)
        raw_manifest = _checked_path(
            raw_root / manifest_path,
            directory=False,
            label=f"frozen external source manifest {source_id}",
        )
        if (
            expected_sha256 is None
            or sha256_file(raw_manifest) != expected_sha256
            or raw.get("physical_manifest_sha256") != expected_sha256
        ):
            raise ValueError(f"Frozen external source-manifest identity drifted: {source_id}")
        observed[source_id] = manifest_path
    if observed != _EXTERNAL_SOURCE_BINDINGS:
        raise ValueError("External normalized manifest does not bind the exact five sources")
    return dict(sorted(observed.items()))


def _require_contract(
    label: str,
    result: Any,
    required: Mapping[str, Any],
) -> None:
    if not isinstance(result, Mapping):
        raise ValueError(f"{label} verifier returned a non-object result")
    _scan_training_value(result, location=f"child-verifier.{label}")
    for field, expected in required.items():
        observed = result.get(field)
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(
                f"{label} verifier contract failed at {field}: expected {expected!r}, observed {observed!r}"
            )
    expected_keys = _CHILD_RESULT_KEYS.get(label)
    if expected_keys is None or set(result) != expected_keys:
        raise ValueError(
            f"{label} verifier result schema drifted: "
            f"expected={sorted(expected_keys or ())}, observed={sorted(map(str, result))}"
        )


def run_final_artifact_verification(
    project_root: str | os.PathLike[str],
    paths: FinalVerificationPaths | None = None,
) -> dict[str, Any]:
    """Replay every final mechanical gate and bind its accepted physical bytes."""

    configuration = paths or FinalVerificationPaths()
    root = _checked_path(Path(project_root), directory=True, label="project root")
    resolved = {
        field: _project_path(
            root,
            value,
            directory=field
            not in {
                "canonical_determinism_report",
                "static_manifest",
                "platform_config",
                "dependency_audit",
                "output_report",
            },
        )
        for field, value in asdict(configuration).items()
        if field != "output_report"
    }
    output = _safe_output_path(root, configuration.output_report)
    _protect_verified_inputs(output, resolved, project_root=root)
    external_manifest_paths = tuple(
        resolved["external_raw"] / manifest_path
        for _, manifest_path in sorted(_EXTERNAL_SOURCE_BINDINGS.items())
    )
    dependency_lock_path = _project_path(root, "pipeline/environments/requirements.lock", directory=False)
    storage_integrity = _verify_generated_storage_is_unaliased(
        [
            resolved["external_normalized"],
            resolved["canonical_primary"],
            resolved["canonical_secondary"],
            resolved["reports_primary"],
            resolved["reports_secondary"],
            resolved["statistical_primary"],
            resolved["statistical_secondary"],
            resolved["split_primary"],
            resolved["split_secondary"],
            resolved["corpus_readiness"],
            resolved["static_manifest"].parent,
            root / "research/data/platform/features/static",
            *external_manifest_paths,
            resolved["platform_config"],
            dependency_lock_path,
        ]
    )

    canonical_result = compare_canonical_builds(
        resolved["canonical_primary"],
        resolved["reports_primary"],
        resolved["canonical_secondary"],
        resolved["reports_secondary"],
    )
    _require_contract(
        "canonical determinism",
        canonical_result,
        {
            "schema_version": "platform_canonical_determinism_verification_v1",
            "status": "passed_content_equivalent",
            "content_equivalent": True,
            "large_model_training_started": False,
            "substantive_training_started": False,
        },
    )
    recorded_determinism = _json_document(resolved["canonical_determinism_report"])
    if recorded_determinism != canonical_result:
        raise ValueError("Recorded canonical determinism report differs from fresh verification")

    qc_primary = resolved["reports_primary"] / "qc_report.json"
    external_result = dict(
        verify_external_normalized_output(resolved["external_normalized"], resolved["external_raw"])
    )
    external_result["output_root"] = configuration.external_normalized
    _require_contract(
        "external normalization",
        external_result,
        {
            "status": "passed",
            "schema_version": "platform-external-normalization/1.0",
            "inventory_entries": 9,
            "input_verification": "passed_full_recursive_bundle_verification",
            "verified_input_count": 5,
            "zero_label_training_and_identity_replacement_contract": "passed",
        },
    )
    verified_input_count = external_result.get("verified_input_count")
    if type(verified_input_count) is not int or verified_input_count != 5:
        raise ValueError("External normalization verifier did not rebind the exact five sources")
    semantic_verification = external_result.get("semantic_verification")
    if (
        not isinstance(semantic_verification, Mapping)
        or semantic_verification.get("all_admission_prohibitions_recomputed") is not True
    ):
        raise ValueError("External normalization standard-topology semantic replay did not pass")
    external_source_bindings = _verify_external_source_bindings(
        resolved["external_normalized"], resolved["external_raw"]
    )
    statistical_primary_result = verify_statistical_analysis(
        resolved["statistical_primary"],
        canonical_build_root=resolved["canonical_primary"],
        qc_report_path=qc_primary,
    )
    statistical_contract = {
        "analysis_version": "platform-statistical-analysis-v1",
        "status": "verified",
        "source_reverified": True,
        "zero_training": True,
        "training_actions": [],
        "scientific_boundaries_verified": True,
    }
    _require_contract("primary statistical analysis", statistical_primary_result, statistical_contract)
    statistical_secondary_result = verify_statistical_analysis(
        resolved["statistical_secondary"],
        canonical_build_root=resolved["canonical_primary"],
        qc_report_path=qc_primary,
    )
    _require_contract("secondary statistical analysis", statistical_secondary_result, statistical_contract)
    statistical_determinism = compare_exact_artifact_trees(
        resolved["statistical_primary"], resolved["statistical_secondary"]
    )
    _require_contract(
        "statistical determinism",
        statistical_determinism,
        {"status": "passed_byte_identical"},
    )
    split_primary_result = verify_split_suite(
        resolved["split_primary"],
        canonical_build_root=resolved["canonical_primary"],
        qc_report_path=qc_primary,
    )
    split_contract = {
        "schema_version": "platform_split_suite_v1",
        "status": "verified",
        "source_reverified": True,
        "label_values_read": False,
        "test_labels_disclosed": False,
        "large_model_training_started": False,
        "substantive_training_started": False,
    }
    _require_contract("primary split suite", split_primary_result, split_contract)
    split_secondary_result = verify_split_suite(
        resolved["split_secondary"],
        canonical_build_root=resolved["canonical_primary"],
        qc_report_path=qc_primary,
    )
    _require_contract("secondary split suite", split_secondary_result, split_contract)
    split_determinism = compare_exact_artifact_trees(resolved["split_primary"], resolved["split_secondary"])
    _require_contract("split determinism", split_determinism, {"status": "passed_byte_identical"})
    corpus_result = verify_corpus_readiness_bundle(
        resolved["corpus_readiness"],
        canonical_build_root=resolved["canonical_primary"],
        qc_report_path=qc_primary,
    )
    _require_contract(
        "corpus readiness",
        corpus_result,
        {
            "schema_version": "platform_corpus_readiness_bundle_v1",
            "status": "verified",
            "source_reverified": True,
            "test_lockboxes_opened_or_hashed": False,
            "large_model_training_started": False,
            "substantive_training_started": False,
        },
    )
    static_result = _verify_static_manifest(resolved["static_manifest"], root)
    _require_contract(
        "static readiness",
        static_result,
        {"status": "verified", "substantive_training_started": False},
    )
    config_result = _verify_platform_config(resolved["platform_config"], root)
    _require_contract("platform configuration", config_result, {"status": "verified"})
    dependency_result = _verify_dependency_audit(resolved["dependency_audit"], root)
    no_training_result = _scan_no_training_json(
        [
            resolved["external_normalized"],
            resolved["canonical_primary"],
            resolved["canonical_secondary"],
            resolved["reports_primary"],
            resolved["reports_secondary"],
            resolved["statistical_primary"],
            resolved["statistical_secondary"],
            resolved["split_primary"],
            resolved["split_secondary"],
            resolved["corpus_readiness"],
            resolved["static_manifest"].parent,
        ]
    )
    _require_contract(
        "no-training scan",
        no_training_result,
        {
            "status": "passed",
            "jsonl_or_routed_test_payloads_opened": False,
            "large_model_training_started": False,
            "substantive_training_started": False,
        },
    )

    critical_paths = (
        *external_manifest_paths,
        resolved["external_normalized"] / "external_public_normalized_manifest.json",
        resolved["canonical_primary"] / "build_manifest.json",
        qc_primary,
        resolved["canonical_secondary"] / "build_manifest.json",
        resolved["reports_secondary"] / "qc_report.json",
        resolved["canonical_determinism_report"],
        resolved["statistical_primary"] / "analysis_manifest.json",
        resolved["statistical_secondary"] / "analysis_manifest.json",
        resolved["split_primary"] / "acceptance.json",
        resolved["split_secondary"] / "acceptance.json",
        resolved["corpus_readiness"] / "acceptance.json",
        resolved["static_manifest"],
        resolved["platform_config"],
        resolved["dependency_audit"],
        dependency_lock_path,
    )
    critical_artifacts = {
        path.relative_to(root).as_posix(): _file_record(path, root) for path in critical_paths
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mechanical_artifact_verification": "passed",
        "workstream_verification": {
            "generated_storage_integrity": storage_integrity,
            "external_normalized": external_result,
            "external_source_bindings": external_source_bindings,
            "canonical_determinism": canonical_result,
            "statistical_primary": statistical_primary_result,
            "statistical_secondary": statistical_secondary_result,
            "statistical_determinism": statistical_determinism,
            "split_primary": split_primary_result,
            "split_secondary": split_secondary_result,
            "split_determinism": split_determinism,
            "corpus_readiness": corpus_result,
            "static_readiness": static_result,
            "platform_configuration": config_result,
            "dependency_vulnerability_audit": dependency_result,
            "no_training_scan": no_training_result,
        },
        "critical_artifacts": dict(sorted(critical_artifacts.items())),
        "human_or_external_blockers": list(_HUMAN_OR_EXTERNAL_BLOCKERS),
        "readiness_boundary": {
            "artifact_integrity_and_source_rebinding_verified": True,
            "scientific_task_claim_ready": False,
            "substantive_large_model_training_ready": False,
            "substantive_large_model_training_authorized": False,
            "reason": (
                "Mechanical verification cannot waive the listed scientific, rights, model, "
                "compute, validation, and governance blockers."
            ),
        },
        "large_model_training_started": NO_SUBSTANTIVE_TRAINING,
        "substantive_training_started": NO_SUBSTANTIVE_TRAINING,
        "training_actions": [],
    }
    _scan_training_value(report, location="final-verification-report")
    _write_json_atomic(output, report)
    if output.stat(follow_symlinks=False).st_nlink != 1:
        raise RuntimeError("Final verification report became hardlinked during publication")
    reread = _json_document(output)
    if reread != report:
        raise RuntimeError("Final artifact verification report failed write/read equality")
    return {
        **report,
        "report_path": output.relative_to(root).as_posix(),
        "report_physical_sha256": sha256_file(output),
        "report_size_bytes": output.stat().st_size,
    }


__all__ = [
    "FinalVerificationPaths",
    "SCHEMA_VERSION",
    "compare_exact_artifact_trees",
    "run_final_artifact_verification",
]
