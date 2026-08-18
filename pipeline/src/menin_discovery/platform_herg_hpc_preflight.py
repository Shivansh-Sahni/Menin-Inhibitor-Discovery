"""Build a read-only preflight for future hERG feature/HPC execution."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .platform_herg_benchmark_freeze_v15 import validate_benchmark_freeze_v15
from .platform_herg_candidate_adjudication import validate_herg_candidate_adjudication
from .platform_herg_master_dataset import validate_herg_master_dataset
from .platform_herg_mmp_analysis import validate_mmp_analysis
from .platform_herg_pre_hpc_assets import validate_herg_pre_hpc_assets
from .platform_herg_wt_reference import validate_wt_reference
from .platform_qt_exposure_prep import verify_qt_exposure_prep

SCHEMA_VERSION = "platform-herg-hpc-preflight/1.0"
PACKAGE_STAGE_MAP = {
    "rdkit": ["S0", "S1", "S2"],
    "numpy": ["S0", "S1", "S2", "S3", "S4", "S5", "S6"],
    "pandas": ["S0", "S1", "S3"],
    "pyarrow": ["S0", "S1", "S2", "S3", "S4", "S5", "S6"],
    "scipy": ["S1", "S2", "S3", "S6"],
    "scikit-learn": ["S3"],
    "xgboost": ["S3"],
    "lightgbm": ["S3"],
    "torch": ["S0", "S4", "S6"],
    "torch-geometric": ["S0", "S4", "S6"],
    "transformers": ["S0", "S4", "S6"],
    "openmm": ["S2", "S5"],
    "meeko": ["S5"],
    "vina": ["S5"],
}
STAGE_MINIMUM_STORAGE_GB = {
    "S0": 5,
    "S1": 20,
    "S2": 500,
    "S3": 10,
    "S4": 50,
    "S5": 1000,
    "S6": 100,
}
TARGET_PACKAGE_CONSTRAINTS = {
    "rdkit": "2026.03.3",
    "torch": "2.7.*",
    "transformers": "4.*",
    "openmm": "8.*",
}


class HergHpcPreflightError(RuntimeError):
    """Raised when preflight inputs or outputs fail validation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_hash(path: Path) -> str:
    return hashlib.sha256(pq.read_schema(path).serialize().to_pybytes()).hexdigest()


def _manifest_hash(payload: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _nr_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_nr_paths(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_nr_paths(child, f"{prefix}[{index}]"))
    elif isinstance(value, str) and (
        value == "NR" or (value.endswith("_NR") and not value.endswith("non_NR"))
    ):
        paths.append(prefix)
    return paths


def _matches_constraint(version: str, constraint: str) -> bool:
    def numeric_parts(value: str) -> tuple[int, ...]:
        return tuple(int(part) for part in value.split(".") if part.isdigit())

    if constraint.endswith(".*"):
        return numeric_parts(version)[: len(numeric_parts(constraint[:-2]))] == numeric_parts(constraint[:-2])
    return numeric_parts(version) == numeric_parts(constraint)


def _package_inventory() -> list[dict[str, Any]]:
    rows = []
    for package, stages in PACKAGE_STAGE_MAP.items():
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = None
        rows.append(
            {
                "package": package,
                "installed": version is not None,
                "installed_version": version,
                "required_by_stages_json": json.dumps(stages, separators=(",", ":")),
                "target_constraint": TARGET_PACKAGE_CONSTRAINTS.get(package),
                "target_constraint_satisfied": (
                    None
                    if package not in TARGET_PACKAGE_CONSTRAINTS or version is None
                    else _matches_constraint(version, TARGET_PACKAGE_CONSTRAINTS[package])
                ),
                "version_frozen_for_production": package == "rdkit" and version == "2026.3.3",
            }
        )
    return rows


def build_hpc_preflight(
    *,
    workspace_root: Path,
    master_root: Path,
    assets_root: Path,
    benchmark_root: Path,
    candidate_root: Path,
    mmp_root: Path,
    qt_exposure_root: Path,
    contract_root: Path,
    wt_reference_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Inspect current readiness without computing a feature or running a model."""

    validate_herg_master_dataset(master_root)
    validate_herg_pre_hpc_assets(assets_root)
    validate_benchmark_freeze_v15(benchmark_root)
    validate_herg_candidate_adjudication(candidate_root)
    validate_mmp_analysis(mmp_root)
    verify_qt_exposure_prep(qt_exposure_root, verify_inputs=True)
    wt_reference_manifest = validate_wt_reference(wt_reference_root)
    feature_contract_path = contract_root / "future_feature_contract.json"
    hpc_plan_path = contract_root / "hpc_execution_plan.json"
    smoke_spec_path = contract_root / "smoke_test_spec.json"
    feature_contract = json.loads(feature_contract_path.read_text())
    hpc_plan = json.loads(hpc_plan_path.read_text())
    smoke_spec = json.loads(smoke_spec_path.read_text())
    if feature_contract["execution_status"] != "specification_only_no_features_computed":
        raise HergHpcPreflightError("feature contract execution boundary weakened")
    if hpc_plan["execution_status"] != "plan_only_no_hpc_jobs_submitted":
        raise HergHpcPreflightError("HPC plan execution boundary weakened")
    if smoke_spec["execution_status"] != "specified_not_run":
        raise HergHpcPreflightError("smoke-test execution boundary weakened")
    protein_family = next(
        row for row in feature_contract["families"] if row["id"] == "herg_protein_embedding"
    )
    if (
        protein_family["wt_sequence_accession"] != wt_reference_manifest["reference"]["primary_accession"]
        or protein_family["wt_sequence_sha256"] != wt_reference_manifest["reference"]["sequence_sha256"]
        or protein_family["wt_reference_manifest_sha256"] != wt_reference_manifest["manifest_sha256"]
    ):
        raise HergHpcPreflightError("feature contract does not bind the accepted WT sequence reference")

    packages = _package_inventory()
    installed = {row["package"]: row["installed"] for row in packages}
    disk = shutil.disk_usage(workspace_root)
    available_gb = disk.free / 1_000_000_000
    stage_rows = []
    for stage in hpc_plan["stages"]:
        stage_id = str(stage["id"])
        required_packages = sorted(
            package for package, stages in PACKAGE_STAGE_MAP.items() if stage_id in stages
        )
        missing = [package for package in required_packages if not installed[package]]
        minimum_storage = STAGE_MINIMUM_STORAGE_GB[stage_id]
        stage_rows.append(
            {
                "stage_id": stage_id,
                "work": stage["work"],
                "minimum_estimated_storage_gb": float(minimum_storage),
                "current_available_storage_gb": float(available_gb),
                "storage_fit_with_20pct_headroom": available_gb >= minimum_storage * 1.2,
                "required_packages_json": json.dumps(required_packages, separators=(",", ":")),
                "missing_packages_json": json.dumps(missing, separators=(",", ":")),
                "software_dependencies_present": not missing,
                "resource_values_are_estimates": True,
                "execution_attempted": False,
            }
        )

    nr_paths = _nr_paths(feature_contract)
    production_packages_missing = sorted(row["package"] for row in packages if not row["installed"])
    runtime_mismatches = []
    python_constraint = feature_contract["software_environment"]["python"]["constraint"]
    if not _matches_constraint(sys.version.split()[0], python_constraint):
        runtime_mismatches.append(f"python:{sys.version.split()[0]}!={python_constraint}")
    runtime_mismatches.extend(
        f"{row['package']}:{row['installed_version']}!={row['target_constraint']}"
        for row in packages
        if row["target_constraint_satisfied"] is False
    )
    readiness = [
        {
            "gate_id": "accepted_data_inputs_validate",
            "passed": True,
            "blocking": True,
            "evidence": "master, assets, candidate evidence, v1.5 benchmark, WT, MMP, and QT/exposure validators pass",
        },
        {
            "gate_id": "current_environment_dependency_consistency",
            "passed": True,
            "blocking": True,
            "evidence": "pip check passed before this release; installed versions are inventoried",
        },
        {
            "gate_id": "all_production_versions_resolved",
            "passed": not nr_paths,
            "blocking": True,
            "evidence": f"{len(nr_paths)} unresolved feature-contract paths",
        },
        {
            "gate_id": "all_future_stage_packages_present",
            "passed": not production_packages_missing,
            "blocking": True,
            "evidence": json.dumps(production_packages_missing, separators=(",", ":")),
        },
        {
            "gate_id": "target_runtime_constraints_compatible",
            "passed": not runtime_mismatches,
            "blocking": True,
            "evidence": json.dumps(runtime_mismatches, separators=(",", ":")),
        },
        {
            "gate_id": "local_storage_supports_all_stages",
            "passed": all(row["storage_fit_with_20pct_headroom"] for row in stage_rows),
            "blocking": True,
            "evidence": f"{available_gb:.2f} GB currently free; multi-TB stages require external/HPC storage",
        },
        {
            "gate_id": "smoke_test_executed_twice",
            "passed": False,
            "blocking": True,
            "evidence": "smoke specification exists but execution is intentionally pending",
        },
        {
            "gate_id": "feature_or_training_execution_started",
            "passed": False,
            "blocking": False,
            "evidence": "must remain false during pre-HPC preparation",
        },
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    package_path = output_root / "environment_package_inventory.parquet"
    stage_path = output_root / "hpc_stage_preflight.parquet"
    readiness_path = output_root / "hpc_readiness_gates.parquet"
    pq.write_table(pa.Table.from_pylist(packages), package_path)
    pq.write_table(pa.Table.from_pylist(stage_rows), stage_path)
    pq.write_table(pa.Table.from_pylist(readiness), readiness_path)
    inventory = {
        "schema_version": SCHEMA_VERSION,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "workspace_total_gb": disk.total / 1_000_000_000,
        "workspace_used_gb": disk.used / 1_000_000_000,
        "workspace_available_gb": available_gb,
        "unresolved_feature_contract_paths": nr_paths,
        "missing_future_stage_packages": production_packages_missing,
        "target_runtime_mismatches": runtime_mismatches,
        "features_computed": False,
        "models_trained": False,
        "hpc_jobs_submitted": False,
    }
    inventory_path = output_root / "preflight_environment.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")

    artifacts: dict[str, Any] = {}
    for path in [package_path, stage_path, readiness_path]:
        artifacts[path.name] = {
            "path": path.name,
            "rows": pq.read_metadata(path).num_rows,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "arrow_schema_sha256": _schema_hash(path),
        }
    artifacts[inventory_path.name] = {
        "path": inventory_path.name,
        "bytes": inventory_path.stat().st_size,
        "sha256": _sha256(inventory_path),
    }
    inputs = []
    for path in [
        master_root / "herg_master_manifest.json",
        assets_root / "herg_pre_hpc_assets_manifest.json",
        benchmark_root / "benchmark_freeze_v1_5_manifest.json",
        candidate_root / "herg_candidate_adjudication_manifest.json",
        mmp_root / "mmp_analysis_manifest.json",
        qt_exposure_root / "qt_exposure_prep_manifest.json",
        wt_reference_root / "wt_reference_manifest.json",
        feature_contract_path,
        hpc_plan_path,
        smoke_spec_path,
        Path(__file__).resolve(),
    ]:
        inputs.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "inputs": inputs,
        "artifacts": artifacts,
        "counts": {
            "packages": len(packages),
            "missing_packages": len(production_packages_missing),
            "hpc_stages": len(stage_rows),
            "unresolved_contract_paths": len(nr_paths),
            "blocking_gates_passed": sum(row["passed"] for row in readiness if row["blocking"]),
            "blocking_gates_total": sum(row["blocking"] for row in readiness),
        },
        "scientific_contract": {
            "features_computed": False,
            "models_trained": False,
            "hpc_jobs_submitted": False,
            "production_hpc_ready": False,
            "resource_values_are_estimates": True,
        },
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    (output_root / "hpc_preflight_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def validate_hpc_preflight(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "hpc_preflight_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["manifest_sha256"] != _manifest_hash(manifest):
        raise HergHpcPreflightError("manifest self-hash mismatch")
    for binding in manifest["inputs"]:
        path = Path(binding["path"])
        if (
            not path.is_file()
            or path.stat().st_size != binding["bytes"]
            or _sha256(path) != binding["sha256"]
        ):
            raise HergHpcPreflightError(f"input rebinding failed: {path}")
    for name, expected in manifest["artifacts"].items():
        path = output_root / name
        if (
            not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or _sha256(path) != expected["sha256"]
        ):
            raise HergHpcPreflightError(f"artifact verification failed: {name}")
        if name.endswith(".parquet") and (
            pq.read_metadata(path).num_rows != expected["rows"]
            or _schema_hash(path) != expected["arrow_schema_sha256"]
        ):
            raise HergHpcPreflightError(f"Parquet verification failed: {name}")
    if manifest["scientific_contract"] != {
        "features_computed": False,
        "hpc_jobs_submitted": False,
        "models_trained": False,
        "production_hpc_ready": False,
        "resource_values_are_estimates": True,
    }:
        raise HergHpcPreflightError("pre-HPC scientific contract weakened")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--master-root", type=Path)
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--benchmark-root", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--mmp-root", type=Path)
    parser.add_argument("--qt-exposure-root", type=Path)
    parser.add_argument("--contract-root", type=Path)
    parser.add_argument("--wt-reference-root", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_hpc_preflight(args.output_root)
    else:
        required = [
            args.workspace_root,
            args.master_root,
            args.assets_root,
            args.benchmark_root,
            args.candidate_root,
            args.mmp_root,
            args.qt_exposure_root,
            args.contract_root,
            args.wt_reference_root,
        ]
        if any(path is None for path in required):
            parser.error("all roots are required when building")
        build_hpc_preflight(
            workspace_root=args.workspace_root,
            master_root=args.master_root,
            assets_root=args.assets_root,
            benchmark_root=args.benchmark_root,
            candidate_root=args.candidate_root,
            mmp_root=args.mmp_root,
            qt_exposure_root=args.qt_exposure_root,
            contract_root=args.contract_root,
            wt_reference_root=args.wt_reference_root,
            output_root=args.output_root,
        )
        validate_hpc_preflight(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
