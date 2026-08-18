#!/usr/bin/env python3
"""Run the six-core hERG fundamental-feature optimization campaign V6.

V6 deliberately builds on the validated V5 train-only data and scaffold-fold
contracts.  It performs a much wider, material search around the best V2
XGBoost region while comparing 2D/fingerprint foundations with QC-safe
electrostatic, conformational, shape, pharmacophore-proxy, and interaction
blocks.  Model selection is performed only on the three inner scaffold folds
inside each of five outer scaffold contexts.  Repository validation and test
labels are never loaded.

The campaign is resumable at every model unit.  Its outputs are internal
discovery evidence, not external validation, causal evidence, or proof of
predictive superiority.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
V5_PATH = HERE / "run_local_herg_feature_relationship_campaign_v5.py"
SPEC = importlib.util.spec_from_file_location("_herg_v5_engine_for_v6", V5_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise RuntimeError(f"unable to load validated V5 engine: {V5_PATH}")
V5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = V5
SPEC.loader.exec_module(V5)
_V5_INPUT_BINDINGS = V5._input_bindings


SCHEMA_VERSION = "platform-local-herg-fundamental-optimization-v6/1.0"
DEFAULT_FEATURE_ROOT = Path("research/local_runs/herg_quantitative_24conformer_v4")
DEFAULT_BASE_ROOT = Path("research/local_runs/herg_discovery_campaign_v1")
DEFAULT_OUTPUT = Path("research/local_runs/herg_fundamental_optimization_v6")
PERMUTATION_REPEATS = 30


def _xgb_profiles() -> list[tuple[str, dict[str, Any]]]:
    """Material, memory-bounded profiles spanning the proven V2 region."""

    return [
        (
            "v2_anchor",
            {
                "n_estimators": 1200,
                "max_depth": 8,
                "learning_rate": 0.02,
                "min_child_weight": 8.0,
                "subsample": 0.75,
                "colsample_bytree": 0.60,
                "reg_alpha": 0.50,
                "reg_lambda": 5.0,
                "max_bin": 96,
            },
        ),
        (
            "v2_shallow",
            {
                "n_estimators": 1500,
                "max_depth": 5,
                "learning_rate": 0.018,
                "min_child_weight": 8.0,
                "subsample": 0.75,
                "colsample_bytree": 0.60,
                "reg_alpha": 0.50,
                "reg_lambda": 5.0,
                "max_bin": 128,
            },
        ),
        (
            "medium_dense",
            {
                "n_estimators": 850,
                "max_depth": 6,
                "learning_rate": 0.03,
                "min_child_weight": 4.0,
                "subsample": 0.85,
                "colsample_bytree": 0.82,
                "reg_alpha": 0.10,
                "reg_lambda": 3.0,
                "max_bin": 128,
            },
        ),
        (
            "deep_sparse",
            {
                "n_estimators": 1050,
                "max_depth": 7,
                "learning_rate": 0.024,
                "min_child_weight": 12.0,
                "subsample": 0.72,
                "colsample_bytree": 0.50,
                "reg_alpha": 1.0,
                "reg_lambda": 8.0,
                "max_bin": 64,
            },
        ),
        (
            "strong_regularization",
            {
                "n_estimators": 1000,
                "max_depth": 6,
                "learning_rate": 0.025,
                "min_child_weight": 16.0,
                "subsample": 0.80,
                "colsample_bytree": 0.70,
                "reg_alpha": 2.0,
                "reg_lambda": 15.0,
                "max_bin": 96,
            },
        ),
        (
            "fast_local",
            {
                "n_estimators": 600,
                "max_depth": 5,
                "learning_rate": 0.05,
                "min_child_weight": 3.0,
                "subsample": 0.90,
                "colsample_bytree": 0.90,
                "reg_alpha": 0.0,
                "reg_lambda": 2.0,
                "max_bin": 128,
            },
        ),
    ]


def _lgb_profiles() -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            "balanced",
            {
                "n_estimators": 1000,
                "num_leaves": 31,
                "max_depth": -1,
                "learning_rate": 0.03,
                "min_child_samples": 25,
                "subsample": 0.82,
                "colsample_bytree": 0.75,
                "reg_alpha": 0.1,
                "reg_lambda": 3.0,
                "max_bin": 127,
            },
        ),
        (
            "small_leaves",
            {
                "n_estimators": 1300,
                "num_leaves": 15,
                "max_depth": -1,
                "learning_rate": 0.022,
                "min_child_samples": 35,
                "subsample": 0.78,
                "colsample_bytree": 0.68,
                "reg_alpha": 0.5,
                "reg_lambda": 6.0,
                "max_bin": 127,
            },
        ),
        (
            "large_leaves_regularized",
            {
                "n_estimators": 800,
                "num_leaves": 63,
                "max_depth": -1,
                "learning_rate": 0.035,
                "min_child_samples": 50,
                "subsample": 0.85,
                "colsample_bytree": 0.60,
                "reg_alpha": 1.0,
                "reg_lambda": 10.0,
                "max_bin": 127,
            },
        ),
    ]


def _candidate_plan() -> list[Any]:
    candidates: list[Any] = []
    # The full V2 region is tested on the two scientifically primary surfaces.
    for surface in ("2d_morgan", "qc_physics"):
        for profile, params in _xgb_profiles():
            candidates.append(V5.Candidate(f"xgb_{surface}_{profile}", "xgboost", surface, params))
    # Focused mechanistic surfaces determine which physical ideas add value.
    for surface in ("electrostatic", "pharmacophore", "conformer", "new3d"):
        for profile, params in _xgb_profiles()[1:5:2]:
            candidates.append(V5.Candidate(f"xgb_{surface}_{profile}", "xgboost", surface, params))
    # Independent boosting family for diversity and possible stacking later.
    for surface in ("2d_morgan", "qc_physics", "electrostatic", "pharmacophore"):
        for profile, params in _lgb_profiles():
            candidates.append(V5.Candidate(f"lgb_{surface}_{profile}", "lightgbm", surface, params))
    # Controls establish the value of fingerprints and fundamental-only physics.
    candidates.extend(
        [
            V5.Candidate("xgb_rdkit2d_control", "xgboost", "2d", _xgb_profiles()[2][1]),
            V5.Candidate("xgb_morgan_control", "xgboost", "morgan", _xgb_profiles()[2][1]),
            V5.Candidate("xgb_physics_only_control", "xgboost", "physics_only", _xgb_profiles()[1][1]),
            V5.Candidate(
                "xgb_full_relationship_reference",
                "xgboost",
                "full",
                {
                    **_xgb_profiles()[2][1],
                    "n_estimators": 400,
                    "max_depth": 5,
                    "min_child_weight": 5.0,
                    "max_bin": 128,
                },
            ),
            V5.Candidate(
                "lgb_full_relationship_reference",
                "lightgbm",
                "full",
                {
                    **_lgb_profiles()[0][1],
                    "n_estimators": 600,
                    "num_leaves": 31,
                    "min_child_samples": 35,
                    "max_bin": 127,
                },
            ),
        ]
    )
    return candidates


def _surfaces(families: dict[str, list[str]]) -> dict[str, list[str]]:
    def union(*names: str) -> list[str]:
        return sorted({column for name in names for column in families.get(name, [])})

    foundation = ("rdkit2d", "morgan")
    electrostatic = ("polarity_charge_internal_contacts",)
    conformer = ("energy_flexibility", "shape", "new3d_stable_misc")
    pharmacophore = ("polarity_charge_internal_contacts", "shape", "selected_interactions")
    qc_physics = (
        "energy_flexibility",
        "polarity_charge_internal_contacts",
        "shape",
        "autocorr3d",
        "new3d_stable_misc",
    )
    return {
        "2d": union("rdkit2d"),
        "morgan": union("morgan"),
        "2d_morgan": union(*foundation),
        "electrostatic": union(*foundation, *electrostatic),
        "pharmacophore": union(*foundation, *pharmacophore),
        "conformer": union(*foundation, *conformer),
        "qc_physics": union(*foundation, *qc_physics),
        "physics_only": union(*electrostatic, *conformer, "autocorr3d", "whim"),
        "old3d": union(*foundation, "old3d_stable"),
        "new3d": union(*foundation, *qc_physics, "whim", "selected_interactions"),
        "fundamental": union(*foundation, *electrostatic, *conformer, "selected_interactions"),
        "full": union(*tuple(families)),
    }


def _validate_candidate_plan(candidates: list[Any]) -> None:
    payloads = [V5._digest(candidate.payload()) for candidate in candidates]
    if len(candidates) < 35 or len(payloads) != len(set(payloads)):
        raise V5.CampaignError("V6 requires at least 35 unique material candidates")
    engines = {candidate.engine for candidate in candidates}
    if engines != {"xgboost", "lightgbm"}:
        raise V5.CampaignError("V6 requires independent XGBoost and LightGBM searches")
    required = {
        "2d",
        "morgan",
        "2d_morgan",
        "electrostatic",
        "pharmacophore",
        "conformer",
        "qc_physics",
        "physics_only",
        "new3d",
        "full",
    }
    surfaces = {candidate.surface for candidate in candidates}
    if not required <= surfaces:
        raise V5.CampaignError(f"V6 candidate surfaces missing: {sorted(required - surfaces)}")


def _input_bindings(feature_root: Path, base_root: Path) -> list[dict[str, Any]]:
    bindings = _V5_INPUT_BINDINGS(feature_root, base_root)
    bindings.append(V5._binding(V5_PATH, "validated_v5_campaign_engine"))
    return bindings


def _configure_engine() -> None:
    # Bind the new implementation, schema, plan, surfaces, and stronger analysis.
    V5.__dict__.update(
        {
            "__file__": str(Path(__file__).resolve()),
            "SCHEMA_VERSION": SCHEMA_VERSION,
            "DEFAULT_FEATURE_ROOT": DEFAULT_FEATURE_ROOT,
            "DEFAULT_BASE_ROOT": DEFAULT_BASE_ROOT,
            "DEFAULT_OUTPUT": DEFAULT_OUTPUT,
            "PERMUTATION_REPEATS": PERMUTATION_REPEATS,
            "_candidate_plan": _candidate_plan,
            "_validate_candidate_plan": _validate_candidate_plan,
            "_surfaces": _surfaces,
            "_input_bindings": _input_bindings,
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--repo-root", default=".")
    run.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT))
    run.add_argument("--base-root", default=str(DEFAULT_BASE_ROOT))
    run.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    run.add_argument("--workers", type=int, default=6)
    for name in ("status", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    return parser


def main() -> int:
    _configure_engine()
    args = _parser().parse_args()
    try:
        if args.command == "run":
            if not 1 <= int(args.workers) <= 6:
                raise V5.CampaignError("workers must be in [1,6]")
            result = V5._run(args)
        elif args.command == "status":
            result = V5._status(Path(args.output_root))
        else:
            result = V5._validate(Path(args.output_root).resolve())
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (V5.CampaignError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    raise SystemExit(main())
