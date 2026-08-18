"""Command-line interface for Menin-Edit scoring and optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from .config import load_config
from .data import build_multi_endpoint_mmp_evidence, load_historical_lab_workbook
from .engine import MeninEditEngine
from .explanations import explain_path, render_markdown_report
from .local_models import LocalRegressionConfig, train_local_regression
from .schemas import (
    ConstraintOperator,
    ConstraintReference,
    ConstraintScope,
    ConstraintSpec,
    ObjectiveSpec,
    OptimizationRequest,
    SearchSpec,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "default.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="menin-edit",
        description="Explainable, constrained, stepwise Menin inhibitor optimization.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Validate configuration and summarize reusable assets")
    audit.add_argument("--load-models", action="store_true", help="Also verify and load every enabled model")

    score = subparsers.add_parser("score", help="Score one molecule with every enabled endpoint")
    score.add_argument("--smiles", required=True)
    score.add_argument("--output", type=Path)

    optimize = subparsers.add_parser("optimize", help="Run bounded stepwise edit optimization")
    source = optimize.add_mutually_exclusive_group(required=True)
    source.add_argument("--smiles")
    source.add_argument("--request", type=Path)
    optimize.add_argument("--output-dir", type=Path)
    optimize.add_argument("--top", type=int, default=10)

    for name, help_text in (
        ("lab-audit", "Audit a historical lab workbook in memory without writing private rows"),
        ("prepare-lab", "Create governed private modeling tables outside the repository"),
        ("train-lab", "Train scaffold-validated private endpoint regressors"),
    ):
        lab = subparsers.add_parser(name, help=help_text)
        lab.add_argument("--workbook", type=Path, required=True)
        lab.add_argument("--key-env", default="MENIN_EDIT_HMAC_KEY")
        lab.add_argument(
            "--cohort-role",
            choices=("train", "development", "locked_external", "prospective_blind"),
            default="development",
        )
        if name == "prepare-lab":
            lab.add_argument("--output-dir", type=Path, required=True)
        if name == "train-lab":
            lab.add_argument("--output-dir", type=Path, required=True)
            lab.add_argument(
                "--endpoint",
                action="append",
                dest="endpoints",
                help=(
                    "Endpoint to train; repeat for several. Defaults to Menin binding, "
                    "MV4;11, MOLM13, and numeric hERG."
                ),
            )
            lab.add_argument("--overwrite", action="store_true")

    return parser


def _request_from_yaml(path: Path, engine: MeninEditEngine) -> OptimizationRequest:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("Optimization request must be a YAML mapping")
    if "starting_smiles" not in payload:
        raise KeyError("Optimization request requires starting_smiles")
    defaults = engine.default_request(str(payload["starting_smiles"]))
    objectives = tuple(
        ObjectiveSpec(
            endpoint=str(item["endpoint"]),
            priority=float(item.get("priority", 1.0)),
            target=None if item.get("target") is None else float(item["target"]),
            minimum_meaningful_gain=float(item.get("minimum_meaningful_gain", 0.0)),
        )
        for item in payload.get("objectives", defaults.objectives)
    )
    constraints = tuple(
        item
        if isinstance(item, ConstraintSpec)
        else ConstraintSpec(
            endpoint=str(item["endpoint"]),
            operator=cast(ConstraintOperator, str(item["operator"])),
            value=float(item["value"]),
            confidence=float(item.get("confidence", 0.90)),
            relative_to=cast(ConstraintReference, str(item.get("relative_to", "absolute"))),
            apply_to=cast(ConstraintScope, str(item.get("apply_to", "each_step"))),
            missing_policy=cast(Literal["reject", "warn"], str(item.get("missing_policy", "reject"))),
            out_of_domain_policy=cast(
                Literal["reject", "warn"], str(item.get("out_of_domain_policy", "reject"))
            ),
        )
        for item in payload.get("constraints", defaults.constraints)
    )
    search_payload = payload.get("search")
    search = defaults.search if search_payload is None else SearchSpec(**dict(search_payload))
    return OptimizationRequest(
        starting_smiles=str(payload["starting_smiles"]),
        objectives=objectives,
        constraints=constraints,
        search=search,
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _runtime_secret(environment_variable: str) -> str:
    secret = os.environ.get(environment_variable, "")
    if len(secret.encode()) < 16:
        raise RuntimeError(
            f"Set {environment_variable} to a runtime-only secret containing at least 16 bytes"
        )
    return secret


def _validated_private_output_directory(
    output_directory: str | os.PathLike[str],
    *,
    repository_root: str | os.PathLike[str],
) -> Path:
    """Resolve a private output and reject the repository before private reads.

    ``Path.resolve`` follows every existing symlink in both paths, so an
    external-looking symlink that targets the repository is rejected as well.
    The caller deliberately invokes this guard before opening a workbook,
    deriving rows, or fitting a model.
    """

    root = Path(repository_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Configured repository root is not a directory: {root}")
    destination = Path(output_directory).expanduser().resolve()
    try:
        destination.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "Private derived tables and models must be written outside the Git repository; "
            "choose an access-controlled output directory"
        )
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"Private output path is not a directory: {destination}")
    return destination


def _lab_tables(args: argparse.Namespace):
    dataset = load_historical_lab_workbook(
        args.workbook.expanduser().resolve(),
        pseudonymization_key=_runtime_secret(args.key_env),
        cohort_role=args.cohort_role,
    )
    if args.cohort_role in {"train", "development"}:
        pairs = build_multi_endpoint_mmp_evidence(dataset.compounds, dataset.observations)
        pair_summary: dict[str, Any] = {
            "status": "available_for_edit_discovery",
            "directed_evidence_rows": int(len(pairs)),
            "directed_transformations": int(
                pairs[["source_fragment", "target_fragment"]].drop_duplicates().shape[0]
            ),
            "endpoint_counts": {
                str(endpoint): int(count)
                for endpoint, count in pairs["endpoint"].value_counts().sort_index().items()
            },
        }
    else:
        pairs = None
        pair_summary = {
            "status": "evaluation_only_role_excluded_from_edit_discovery",
            "directed_evidence_rows": 0,
            "directed_transformations": 0,
            "endpoint_counts": {},
        }
    summary = {
        **dict(dataset.summary),
        "mmp_evidence": pair_summary,
        "private_rows_written": False,
        "raw_identifiers_included": False,
    }
    return dataset, pairs, summary


def _prepare_lab_outputs(
    args: argparse.Namespace,
    *,
    repository_root: Path,
) -> tuple[dict[str, Any], Path]:
    destination = _validated_private_output_directory(
        args.output_dir,
        repository_root=repository_root,
    )
    dataset, pairs, summary = _lab_tables(args)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "compounds.csv": dataset.compounds,
        "observations.csv": dataset.observations,
        "issues.csv": dataset.issues,
    }
    if pairs is not None:
        outputs["multi_endpoint_mmp_evidence.csv"] = pairs
    hashes: dict[str, str] = {}
    for filename, frame in outputs.items():
        path = destination / filename
        frame.to_csv(path, index=False)
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "menin-edit-private-data-v1",
        "summary": {**summary, "private_rows_written": True},
        "files_sha256": hashes,
        "confidentiality": (
            "Derived tables remain confidential and must not be committed or sent externally"
        ),
    }
    _write_text(destination / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest, destination


def _train_lab_models(
    args: argparse.Namespace,
    *,
    repository_root: Path,
) -> tuple[dict[str, Any], Path]:
    destination = _validated_private_output_directory(
        args.output_dir,
        repository_root=repository_root,
    )
    dataset = load_historical_lab_workbook(
        args.workbook.expanduser().resolve(),
        pseudonymization_key=_runtime_secret(args.key_env),
        cohort_role=args.cohort_role,
    )
    if args.cohort_role not in {"train", "development"}:
        raise ValueError("Local model fitting accepts only train or development cohort roles")
    endpoints = tuple(
        args.endpoints
        or (
            "menin_biochemical_pIC50",
            "mv411_cellular_pIC50",
            "molm13_cellular_pIC50",
            "herg_pIC50",
        )
    )
    if len(endpoints) != len(set(endpoints)):
        raise ValueError("Training endpoints must be unique")
    records: dict[str, Any] = {}
    for endpoint in endpoints:
        artifact = train_local_regression(
            dataset.compounds,
            dataset.observations,
            endpoint=endpoint,
            output_dir=destination,
            config=LocalRegressionConfig(),
            overwrite=bool(args.overwrite),
        )
        manifest = dict(artifact.manifest)
        records[endpoint] = {
            "artifact": str(artifact.artifact_path),
            "manifest": str(artifact.manifest_path),
            "model_version": artifact.model_version,
            "eligible_structures": manifest["data"]["unique_structures"],
            "unique_scaffolds": manifest["data"]["unique_scaffolds"],
            "scaffold_oof_metrics": manifest["validation"]["oof_metrics"],
            "baseline_metrics": manifest["validation"]["fold_specific_median_baseline_metrics"],
            "baseline_mae_improvement_fraction": manifest["validation"]["baseline_mae_improvement_fraction"],
            "recommended_for_optimization": manifest["status"]["recommended_for_optimization"],
            "status_reason": manifest["status"]["reason"],
        }
    summary = {
        "schema_version": "menin-edit-local-training-v1",
        "workbook_sha256": dataset.summary["workbook_sha256"],
        "cohort_role": args.cohort_role,
        "models": records,
        "automatic_registration": False,
        "registration_policy": (
            "Only artifacts whose manifest recommends optimization are accepted by default"
        ),
    }
    directions = {
        "menin_biochemical_pIC50": "maximize",
        "mv411_cellular_pIC50": "maximize",
        "molm13_cellular_pIC50": "maximize",
        "herg_pIC50": "minimize",
    }
    accepted = {
        endpoint: record
        for endpoint, record in records.items()
        if record["recommended_for_optimization"] and endpoint in directions
    }
    registry_snippet = {
        "models": {
            endpoint: {
                "kind": "local_lab_regression",
                "enabled": True,
                "artifact": record["artifact"],
                "manifest": record["manifest"],
            }
            for endpoint, record in accepted.items()
        },
        "endpoints": {
            endpoint: {
                "direction": directions[endpoint],
                "task": "regression",
                "display_unit": "pIC50",
                "missing_policy": "reject",
                "out_of_domain_policy": "warn",
            }
            for endpoint in accepted
        },
    }
    snippet_path = destination / "accepted_registry_snippet.yaml"
    _write_text(snippet_path, yaml.safe_dump(registry_snippet, sort_keys=False))
    summary["accepted_registry_snippet"] = str(snippet_path)
    _write_text(
        destination / "training_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return summary, destination


def _audit(engine: MeninEditEngine, *, load_models: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "ready",
        "edit_library": engine.edit_library.manifest(),
        "registered_endpoints": sorted(engine.endpoints),
        "enabled_predictors": sorted(engine.predictors.predictors),
        "missing_active_predictors": sorted(set(engine.endpoints) - set(engine.predictors.predictors)),
        "toxicity_status": (
            "adapter-ready; explicit DILI/Ames artifacts are intentionally disabled until trained and validated"
        ),
        "private_data_policy": "referenced from approved storage; never copied by audit",
    }
    if load_models:
        payload["model_versions"] = engine.predictors.model_versions()
    return payload


def _optimize(
    engine: MeninEditEngine,
    request: OptimizationRequest,
    *,
    output_dir: Path | None,
    top: int,
) -> tuple[dict[str, Any], Path]:
    result = engine.optimize(request)
    destination = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else (PACKAGE_ROOT / "artifacts" / "sessions" / result.session_id).resolve()
    )
    paths = {
        candidate.node_id: engine.get_path(result.session_id, candidate.node_id)
        for candidate in result.candidates
    }
    _write_text(destination / "result.json", result.to_json() + "\n")
    _write_text(
        destination / "paths.json",
        json.dumps(
            {
                node_id: explain_path(path, endpoints=engine.endpoints, evidence=engine.evidence)
                for node_id, path in paths.items()
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _write_text(
        destination / "report.md",
        render_markdown_report(
            result,
            paths=paths,
            endpoints=engine.endpoints,
            evidence=engine.evidence,
            top_k=max(1, top),
        ),
    )
    manifest = {
        "schema_version": result.schema_version,
        "session_id": result.session_id,
        "request_sha256": hashlib.sha256(request.to_json(indent=None).encode()).hexdigest(),
        "result_sha256": hashlib.sha256((result.to_json() + "\n").encode()).hexdigest(),
        "candidate_count": len(result.candidates),
        "feasible_candidate_count": sum(ranking.eligible for ranking in result.rankings),
        "model_versions": dict(result.model_versions),
        "edit_library": engine.edit_library.manifest(),
        "interpretation": "computational hypotheses requiring expert and experimental review",
    }
    _write_text(destination / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest, destination


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "lab-audit":
        _dataset, _pairs, summary = _lab_tables(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "prepare-lab":
        config = load_config(args.config)
        manifest, destination = _prepare_lab_outputs(
            args,
            repository_root=config.repository_root,
        )
        print(json.dumps({**manifest, "output_dir": str(destination)}, indent=2, sort_keys=True))
        return 0
    if args.command == "train-lab":
        config = load_config(args.config)
        summary, destination = _train_lab_models(
            args,
            repository_root=config.repository_root,
        )
        print(json.dumps({**summary, "output_dir": str(destination)}, indent=2, sort_keys=True))
        return 0

    engine = MeninEditEngine.from_config(args.config)

    if args.command == "audit":
        print(json.dumps(_audit(engine, load_models=args.load_models), indent=2, sort_keys=True))
        return 0
    if args.command == "score":
        node = engine.score(args.smiles)
        text = node.to_json() + "\n"
        if args.output:
            _write_text(args.output.expanduser().resolve(), text)
        else:
            print(text, end="")
        return 0
    if args.command == "optimize":
        request = (
            _request_from_yaml(args.request.expanduser().resolve(), engine)
            if args.request
            else engine.default_request(args.smiles)
        )
        manifest, destination = _optimize(
            engine,
            request,
            output_dir=args.output_dir,
            top=args.top,
        )
        print(json.dumps({**manifest, "output_dir": str(destination)}, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
