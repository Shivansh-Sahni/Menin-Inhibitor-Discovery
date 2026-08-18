"""Scientific stage implementations behind the separate ``menin-research`` CLI."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .research_common import atomic_write_csv, atomic_write_json, atomic_write_parquet
from .research_decisions import build_optimizer_contract, expand_assay_requests, select_assay_panel
from .research_feature_ontology import selected_model_conformer_features
from .research_graph_models import grouped_conformer_mil_benchmark, grouped_dmpnn_herg_benchmark
from .research_herg import (
    HERG_RECEPTOR_ENSEMBLE,
    herg_process_observables,
    state_dependent_markov_architecture,
)
from .research_hierarchical import (
    compound_balanced_conventional_metrics,
    grouped_hierarchical_pk_benchmark,
)
from .research_modeling import (
    final_fit_censored_herg_predictions,
    final_fit_regression_predictions,
    grouped_censored_herg_benchmark,
    grouped_exact_herg_benchmark,
    grouped_joint_herg_benchmark,
    grouped_regression_benchmark,
    merge_feature_layers,
    model_ladder_registry,
    promotion_decision,
)
from .research_pk import derive_pk_prediction_views, pk_identifiability_contract
from .research_public_herg import run_sun_public_reproduction
from .research_regimes import apply_cross_outcome_cutoff_gate, bootstrap_mw_change_point


def load_canonical_tables(canonical_root: Path, source: str = "internal") -> dict[str, pd.DataFrame]:
    root = canonical_root / source
    if not root.exists():
        raise FileNotFoundError(f"Run normalize first; canonical directory is missing: {root}")
    return {path.stem: pd.read_parquet(path) for path in sorted(root.glob("*.parquet"))}


def compound_model_frame(compounds: pd.DataFrame, aliases: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = compounds.copy()
    frame = frame.rename(columns={"molecular_weight_g_mol": "mw"})
    required = {"compound_id", "standardized_smiles", "mw"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Compound table is missing model fields: {missing}")
    frame["compound_id"] = frame["compound_id"].astype(str)
    if (
        aliases is not None
        and not aliases.empty
        and {"compound_id", "source_compound_name"}.issubset(aliases)
    ):
        display = aliases.groupby("compound_id")["source_compound_name"].agg(
            lambda values: ";".join(sorted(set(map(str, values))))
        )
        frame["display_name"] = frame["compound_id"].map(display).fillna(frame["compound_id"])
    else:
        frame["display_name"] = frame.get("source_record_id", frame["compound_id"]).fillna(
            frame["compound_id"]
        )
    return frame


def nominal_physics_summary(path: Path, *, target_ph: float = 7.4) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if "pka_scenario" in frame:
        nominal = frame[frame["pka_scenario"].astype(str) == "nominal"]
        if not nominal.empty:
            frame = nominal
    if "ph" in frame and frame["ph"].notna().any():
        chosen = min(frame["ph"].dropna().unique(), key=lambda value: abs(float(value) - target_ph))
        frame = frame[np.isclose(frame["ph"].astype(float), float(chosen))]
    return frame.sort_values("compound_id").drop_duplicates("compound_id").reset_index(drop=True)


def _merge_conformer_state_weights(
    conformers: pd.DataFrame,
    populations: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """Join raw conformers to one state weight and one compound identity.

    Both raw physics tables carry ``structure_id``.  Joining only on
    ``state_id`` creates ``structure_id_x``/``structure_id_y`` and silently
    destroys the key needed for the compound registry.  The two-key join also
    fails closed if a state is ever associated with the wrong structure.
    """

    conformer_required = {"state_id", "structure_id", "conformer_weight"}
    population_required = {"state_id", "structure_id", "state_weight"}
    registry_required = {"compound_id", "structure_id"}
    for label, frame, required in (
        ("conformers", conformers, conformer_required),
        ("populations", populations, population_required),
        ("registry", registry, registry_required),
    ):
        if missing := sorted(required - set(frame.columns)):
            raise ValueError(f"Fast-physics {label} table is missing fields: {missing}")

    state_weights = populations[["state_id", "structure_id", "state_weight"]].drop_duplicates()
    if state_weights.duplicated(["state_id", "structure_id"], keep=False).any():
        raise ValueError("Fast-physics populations contain conflicting weights for one state/structure")

    structure_registry = registry[["compound_id", "structure_id"]].drop_duplicates()
    if structure_registry.duplicated("structure_id", keep=False).any():
        raise ValueError("Fast-physics registry maps one structure to multiple compound identities")

    merged = conformers.merge(
        state_weights,
        on=["state_id", "structure_id"],
        how="inner",
        validate="many_to_one",
    )
    merged["ensemble_weight"] = merged["conformer_weight"] * merged["state_weight"]
    return structure_registry.merge(
        merged,
        on="structure_id",
        how="inner",
        validate="one_to_many",
    )


def _censored_fit_status(metrics: Mapping[str, Any]) -> str:
    """Reject a censored rung when any held-out fold failed to converge."""

    fraction = float(metrics.get("fit_converged_fraction", 0.0))
    return "evaluated" if np.isfinite(fraction) and fraction >= 1.0 else "rejected-nonconverged"


def _preferred_auc(measurements: pd.DataFrame, route: str) -> pd.DataFrame:
    eligible = measurements[
        (measurements["endpoint"].isin(["auc_0_inf", "auc_0_t"]))
        & (measurements["route"].astype(str).str.upper() == route)
        & measurements["value"].notna()
        & measurements["relation"].isin(["=", "~"])
    ].copy()
    eligible["endpoint_priority"] = eligible["endpoint"].map({"auc_0_inf": 0, "auc_0_t": 1})
    key = "pk_study_id" if eligible["pk_study_id"].notna().any() else "measurement_id"
    return eligible.sort_values([key, "endpoint_priority"]).drop_duplicates(key)


def _validate_resolved_pk_links(measurements: pd.DataFrame, studies: pd.DataFrame) -> None:
    """Reject resolved PK rows whose declared study context is inconsistent.

    A route mismatch would silently divide an IV AUC by a PO dose (or vice
    versa), which is a direct target-definition error rather than ordinary
    missing data.  The canonical layer is expected to resolve these links; the
    modeling interface therefore fails closed if that contract is violated.
    """

    if studies["pk_study_id"].astype(str).duplicated().any():
        raise ValueError("PK studies contain duplicate pk_study_id values")
    if measurements["pk_study_id"].isna().any():
        ids = measurements.loc[measurements["pk_study_id"].isna(), "measurement_id"].astype(str).tolist()
        raise ValueError(f"Resolved PK measurements are missing pk_study_id: {ids[:5]}")
    study_context = studies[["pk_study_id", "route", "species"]].copy()
    study_context["pk_study_id"] = study_context["pk_study_id"].astype(str)
    linked = measurements[["measurement_id", "pk_study_id", "route", "species"]].copy()
    linked["pk_study_id"] = linked["pk_study_id"].astype(str)
    linked = linked.merge(
        study_context,
        on="pk_study_id",
        how="left",
        suffixes=("_measurement", "_study"),
        validate="many_to_one",
        indicator=True,
    )
    missing = linked[linked["_merge"] != "both"]
    if not missing.empty:
        raise ValueError(
            "Resolved PK measurements reference unknown studies: "
            f"{missing['measurement_id'].astype(str).tolist()[:5]}"
        )
    measurement_route = linked["route_measurement"].fillna("").astype(str).str.upper()
    study_route = linked["route_study"].fillna("").astype(str).str.upper()
    measurement_species = linked["species_measurement"].fillna("").astype(str).str.casefold()
    study_species = linked["species_study"].fillna("").astype(str).str.casefold()
    mismatch = linked[(measurement_route != study_route) | (measurement_species != study_species)]
    if not mismatch.empty:
        raise ValueError(
            "Resolved PK measurement/study route or species mismatch: "
            f"{mismatch['measurement_id'].astype(str).tolist()[:5]}"
        )


def _require_units(frame: pd.DataFrame, *, expected: str, context: str) -> None:
    if "unit" not in frame or frame.empty:
        return
    observed = set(frame["unit"].dropna().astype(str))
    if observed and observed != {expected}:
        raise ValueError(f"{context} requires unit {expected!r}; observed {sorted(observed)}")


def prepare_pk_tasks(
    compounds: pd.DataFrame,
    measurements: pd.DataFrame,
    pk_studies: pd.DataFrame,
    features: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Create study-level targets while excluding algebraic parent/child leakage."""

    studies = pk_studies[
        ["pk_study_id", "dose_value", "dose_unit", "route", "species", "pairing_status"]
    ].copy()
    studies["pk_study_id"] = studies["pk_study_id"].astype(str)
    measurements = measurements.copy()
    measurements["compound_id"] = measurements["compound_id"].astype(str)
    measurements = measurements[
        measurements["model_eligible"].fillna(False)
        & (measurements["species"].fillna("").str.casefold() == "rat")
        & measurements["pairing_status"].eq("resolved")
    ]
    _validate_resolved_pk_links(measurements, studies)
    tasks: dict[str, pd.DataFrame] = {}

    for route, task_name in (("IV", "iv_auc_dose_normalized"), ("PO", "po_auc_dose_normalized")):
        auc = _preferred_auc(measurements, route).merge(
            studies, on="pk_study_id", how="left", suffixes=("", "_study")
        )
        _require_units(auc, expected="ng*h/mL", context=f"Rat {route} AUC")
        dose_units = set(auc["dose_unit"].dropna().astype(str))
        if dose_units and dose_units != {"mg/kg"}:
            raise ValueError(f"Rat {route} dose requires unit 'mg/kg'; observed {sorted(dose_units)}")
        auc = auc[(auc["dose_value"] > 0) & auc["value"].notna()]
        auc["target_value"] = auc["value"] / auc["dose_value"]
        auc["target_definition"] = f"{route} AUC (AUC0-inf preferred) divided by route-specific dose"
        tasks[task_name] = auc

    endpoint_tasks = {
        "vdss": ("vdss", "IV"),
        "po_cmax_dose_normalized": ("cmax", "PO"),
        "po_tmax": ("tmax", "PO"),
    }
    for task_name, (endpoint, route) in endpoint_tasks.items():
        task = measurements[
            (measurements["endpoint"] == endpoint)
            & (measurements["route"].astype(str).str.upper() == route)
            & measurements["value"].notna()
            & measurements["relation"].isin(["=", "~"])
        ].copy()
        expected_unit = {"vdss": "L/kg", "cmax": "ng/mL", "tmax": "h"}[endpoint]
        _require_units(task, expected=expected_unit, context=f"Rat {route} {endpoint}")
        if endpoint == "cmax":
            task = task.merge(
                studies[["pk_study_id", "dose_value", "dose_unit"]],
                on="pk_study_id",
                how="left",
                validate="many_to_one",
            )
            dose_units = set(task["dose_unit"].dropna().astype(str))
            if dose_units and dose_units != {"mg/kg"}:
                raise ValueError(f"Rat PO Cmax dose requires unit 'mg/kg'; observed {sorted(dose_units)}")
            task = task[(task["dose_value"] > 0) & task["value"].notna()].copy()
            task["target_value"] = task["value"] / task["dose_value"]
            task["target_definition"] = (
                "reported study-level rat PO Cmax divided by route-specific dose; "
                "discovery-only sensitivity endpoint pending within-compound dose proportionality"
            )
        else:
            task["target_value"] = task["value"]
            task["target_definition"] = f"reported study-level rat {route} {endpoint}"
        tasks[task_name] = task

    prepared: dict[str, pd.DataFrame] = {}
    compound_columns = ["compound_id", "standardized_smiles", "mw", "display_name"]
    for task_name, task in tasks.items():
        frame = task.merge(compounds[compound_columns], on="compound_id", how="inner")
        frame = frame.merge(features, on="compound_id", how="inner", suffixes=("", "_feature"))
        frame = frame[(frame["target_value"] > 0) & np.isfinite(frame["target_value"])]
        frame["sample_id"] = frame["measurement_id"]
        prepared[task_name] = frame.reset_index(drop=True)
    return prepared


def _pic50_bounds(value_um: float, relation: str) -> tuple[float, float]:
    transformed = 6.0 - np.log10(float(value_um))
    if relation in {"<", "<="}:
        return transformed, np.inf
    if relation in {">", ">="}:
        return -np.inf, transformed
    return transformed, transformed


def prepare_herg_evidence(
    compounds: pd.DataFrame,
    measurements: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    measurements = measurements.copy()
    measurements["compound_id"] = measurements["compound_id"].astype(str)
    if "model_eligible" in measurements:
        measurements = measurements[measurements["model_eligible"].fillna(False)].copy()
    potency = measurements[
        (measurements["endpoint"] == "herg_ic50")
        & measurements["value"].notna()
        & (measurements["value"] > 0)
    ].copy()
    if "unit" in potency:
        accepted = {"uM", "µM", "μM"}
        observed = set(potency["unit"].dropna().astype(str))
        if not observed.issubset(accepted):
            raise ValueError(f"hERG IC50 modeling requires micromolar values; observed {sorted(observed)}")
    bounds = [
        _pic50_bounds(value, relation)
        for value, relation in zip(potency["value"], potency["relation"], strict=True)
    ]
    potency["pic50_lower"] = [item[0] for item in bounds]
    potency["pic50_upper"] = [item[1] for item in bounds]
    potency["observed_pic50"] = np.where(
        np.isclose(potency["pic50_lower"], potency["pic50_upper"]), potency["pic50_lower"], np.nan
    )
    inhibition = measurements[
        measurements["endpoint"].isin({"herg_percent_inhibition", "herg_inhibition"})
        & measurements["value"].notna()
        & measurements["test_concentration_value"].notna()
    ].copy()
    if "unit" in inhibition:
        observed_response_units = set(inhibition["unit"].dropna().astype(str))
        if observed_response_units and observed_response_units != {"%"}:
            raise ValueError(
                "hERG concentration-response modeling requires percent inhibition; "
                f"observed {sorted(observed_response_units)}"
            )
    if "test_concentration_unit" in inhibition:
        observed_concentration_units = set(inhibition["test_concentration_unit"].dropna().astype(str))
        if not observed_concentration_units.issubset({"uM", "µM", "μM"}):
            raise ValueError(
                "hERG test concentrations must be micromolar; "
                f"observed {sorted(observed_concentration_units)}"
            )
    inhibition = inhibition.rename(
        columns={
            "value": "inhibition_percent",
            "test_concentration_value": "test_concentration_um",
        }
    )
    feature_compounds = compounds[["compound_id", "standardized_smiles", "mw", "display_name"]].merge(
        features, on="compound_id", how="inner", suffixes=("", "_feature")
    )
    potency = potency.merge(feature_compounds, on="compound_id", how="inner")
    inhibition = inhibition.merge(feature_compounds, on="compound_id", how="inner")
    return feature_compounds, potency, inhibition


def baseline_inventory(project_root: Path, output_dir: Path) -> dict[str, Any]:
    """Read—not execute or rewrite—the frozen Menin/hERG baseline."""

    expected = {
        "public_herg_metrics": project_root / "research/reports/herg_classifier_metrics.json",
        "same_series_summary": project_root
        / "research/benchmarks/herg/strong_ml_indomain/results_summary.md",
        "same_series_model_card": project_root
        / "research/benchmarks/herg/strong_ml_indomain/indomain_model_card.md",
        "nested_scaffold_metrics": project_root
        / "research/benchmarks/herg/nested_strong_ml/nested_metrics.csv",
        "nested_scaffold_summary": project_root
        / "research/benchmarks/herg/nested_strong_ml/validation_summary.md",
        "menin_edit_package": project_root / "packages/menin-edit",
    }
    missing = [name for name, path in expected.items() if not path.exists()]
    public: dict[str, Any] = {}
    if expected["public_herg_metrics"].exists():
        payload = json.loads(expected["public_herg_metrics"].read_text(encoding="utf-8"))
        public = {
            "n_compounds": payload.get("n_compounds"),
            "model": payload.get("model"),
            "test_roc_auc": payload.get("test_roc_auc"),
            "test_balanced_accuracy": payload.get("test_balanced_accuracy"),
            "split_strategy": payload.get("split", {}).get("strategy"),
        }
    summary = {
        "status": "preserved_read_only" if not missing else "incomplete_baseline_inventory",
        "missing": missing,
        "artifacts": {name: str(path.relative_to(project_root)) for name, path in expected.items()},
        "public_herg": public,
        "execution_performed": False,
        "menin_edit_modified": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "frozen_baseline_inventory.json", summary)
    return summary


def _run_boosters_isolated(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    group_column: str,
    folds: int,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    available = [name for name in ("xgboost", "lightgbm") if importlib.util.find_spec(name) is not None]
    if not available:
        return pd.DataFrame(), pd.DataFrame()
    with tempfile.TemporaryDirectory(prefix="menin-research-boosters-") as temporary_name:
        temporary = Path(temporary_name)
        input_path = temporary / "input.parquet"
        metrics_path = temporary / "metrics.csv"
        predictions_path = temporary / "predictions.parquet"
        frame.to_parquet(input_path, index=False)
        command = [
            sys.executable,
            "-m",
            "menin_discovery.research_booster_worker",
            "--mode",
            mode,
            "--input",
            str(input_path),
            "--features",
            ",".join(feature_columns),
            "--target",
            target_column,
            "--group",
            group_column,
            "--folds",
            str(folds),
            "--models",
            ",".join(available),
            "--metrics",
            str(metrics_path),
            "--predictions",
            str(predictions_path),
        ]
        try:
            subprocess.run(command, check=True, timeout=1800)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return pd.DataFrame(), pd.DataFrame()
        return pd.read_csv(metrics_path), pd.read_parquet(predictions_path)


def run_pk_models(
    canonical_root: Path,
    physics_root: Path,
    output_dir: Path,
    *,
    folds: int,
    random_state: int,
    interval_level: float = 0.90,
) -> dict[str, Any]:
    tables = load_canonical_tables(canonical_root)
    alias_path = canonical_root / "internal" / "compound_aliases.parquet"
    aliases = pd.read_parquet(alias_path) if alias_path.exists() else None
    compounds = compound_model_frame(tables["compounds"], aliases)
    physics_path = physics_root / "fast_physics_summary.parquet"
    physics = nominal_physics_summary(physics_path)
    features, layers = merge_feature_layers(compounds, physics)
    scoring_features = compounds[["compound_id", "standardized_smiles"]].merge(
        features,
        on="compound_id",
        how="inner",
        validate="one_to_one",
    )
    tasks = prepare_pk_tasks(compounds, tables["measurements"], tables["pk_studies"], features)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    promotion_rows: list[dict[str, Any]] = []
    hierarchical_promotion_rows: list[dict[str, Any]] = []
    final_prediction_frames: list[pd.DataFrame] = []
    optimizer_endpoint_map = {
        "iv_auc_dose_normalized": ("rat_iv_auc_dose_normalized", "ng*h/mL per mg/kg"),
        "vdss": ("rat_iv_vdss_l_kg", "L/kg"),
        "po_auc_dose_normalized": ("rat_po_auc_dose_normalized", "ng*h/mL per mg/kg"),
        "po_cmax_dose_normalized": (
            "rat_po_cmax_dose_normalized",
            "ng/mL per mg/kg",
        ),
        "po_tmax": ("rat_po_tmax_h", "h"),
    }
    for task_name, task in tasks.items():
        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        layer_results: dict[str, pd.DataFrame] = {}
        for layer_name, columns in layers.items():
            optimizer_final_model: str | None = None
            usable = [column for column in columns if column in task and task[column].notna().any()]
            if len(task) < 8 or task["scaffold"].nunique() < 2 or not usable:
                summary_rows.append(
                    {
                        "domain": "pk",
                        "endpoint": task_name,
                        "feature_layer": layer_name,
                        "status": "insufficient_grouped_support",
                        "n": len(task),
                    }
                )
                continue
            core_metrics, core_predictions = grouped_regression_benchmark(
                task,
                feature_columns=usable,
                target_column="target_value",
                folds=folds,
                random_state=random_state,
                interval_level=interval_level,
            )
            metrics, predictions = core_metrics.copy(), core_predictions.copy()
            if layer_name == "structure_2d":
                booster_metrics, booster_predictions = _run_boosters_isolated(
                    task,
                    feature_columns=usable,
                    target_column="target_value",
                    group_column="scaffold",
                    folds=folds,
                    mode="pk",
                )
                if not booster_metrics.empty:
                    metrics = pd.concat([metrics, booster_metrics], ignore_index=True)
                    predictions = pd.concat([predictions, booster_predictions], ignore_index=True)
                hierarchical_metrics, hierarchical_predictions, variance_components, cluster_bootstrap = (
                    grouped_hierarchical_pk_benchmark(
                        task,
                        feature_columns=usable,
                        target_column="target_value",
                        folds=folds,
                        interval_level=interval_level,
                        random_state=random_state,
                    )
                )
                hierarchical_metrics["endpoint"] = task_name
                hierarchical_metrics["feature_layer"] = layer_name
                hierarchical_predictions["endpoint"] = task_name
                hierarchical_predictions["feature_layer"] = layer_name
                variance_components["endpoint"] = task_name
                variance_components["feature_layer"] = layer_name
                cluster_bootstrap["endpoint"] = task_name
                cluster_bootstrap["feature_layer"] = layer_name
                atomic_write_csv(task_dir / "hierarchical_metrics.csv", hierarchical_metrics)
                atomic_write_parquet(
                    task_dir / "structure_2d_hierarchical_predictions.parquet",
                    hierarchical_predictions,
                )
                atomic_write_csv(task_dir / "hierarchical_variance_components.csv", variance_components)
                atomic_write_parquet(task_dir / "hierarchical_scaffold_bootstrap.parquet", cluster_bootstrap)

                conventional_compound_metrics = compound_balanced_conventional_metrics(predictions)
                conventional_compound_metrics["comparison_role"] = "retained_conventional_candidate"
                hierarchical_primary = hierarchical_metrics[hierarchical_metrics["primary_evaluation"]].copy()
                hierarchical_primary["comparison_role"] = "hierarchical_discovery_candidate"
                comparison = pd.concat(
                    [conventional_compound_metrics, hierarchical_primary],
                    ignore_index=True,
                    sort=False,
                ).sort_values(["log_mae", "model"])
                comparison["endpoint"] = task_name
                comparison["feature_layer"] = layer_name
                atomic_write_csv(task_dir / "hierarchical_vs_conventional_compound_balanced.csv", comparison)
                best_conventional_compound = conventional_compound_metrics.sort_values(
                    ["log_mae", "model"]
                ).iloc[0]
                hierarchical_row = hierarchical_primary.iloc[0]
                noninferior = bool(hierarchical_row["log_mae"] <= best_conventional_compound["log_mae"])
                hierarchical_promotion_rows.append(
                    {
                        "endpoint": task_name,
                        "candidate": "compound_balanced_hierarchical_gaussian",
                        "promotion_status": "discovery-track",
                        "primary_metric": "compound_balanced_log_mae",
                        "baseline_model": best_conventional_compound["model"],
                        "baseline_value": float(best_conventional_compound["log_mae"]),
                        "candidate_value": float(hierarchical_row["log_mae"]),
                        "noninferior_to_baseline": noninferior,
                        "calibrated_on_untouched_set": False,
                        "reason": (
                            "Evaluation-only partial-pooling rung. It cannot replace optimizer final fits until "
                            "calibration and non-inferiority are confirmed on an untouched prospective set."
                        ),
                    }
                )
                summary_rows.append(
                    {
                        "domain": "pk",
                        "endpoint": task_name,
                        "feature_layer": "structure_2d_hierarchical",
                        "status": "evaluated_discovery_only",
                        "n": int(hierarchical_row["n"]),
                        "best_model": "compound_balanced_hierarchical_gaussian",
                        "optimizer_final_model": None,
                        "primary_metric": "compound_balanced_log_mae",
                        "primary_value": float(hierarchical_row["log_mae"]),
                        "interval_coverage": float(hierarchical_row["prediction_interval_coverage"]),
                    }
                )
                endpoint_spec = optimizer_endpoint_map.get(task_name)
                if endpoint_spec is not None:
                    selected = core_metrics.sort_values(["log_mae", "model"]).iloc[0]
                    optimizer_final_model = str(selected["model"])
                    final_predictions = final_fit_regression_predictions(
                        task,
                        scoring_features,
                        core_predictions,
                        feature_columns=usable,
                        target_column="target_value",
                        selected_model=str(selected["model"]),
                        endpoint=endpoint_spec[0],
                        unit=endpoint_spec[1],
                        interval_level=interval_level,
                        random_state=random_state,
                    )
                    final_prediction_frames.append(final_predictions)
                    atomic_write_parquet(task_dir / "final_fit_predictions.parquet", final_predictions)
            metrics["endpoint"] = task_name
            metrics["feature_layer"] = layer_name
            metrics["track"] = (
                "baseline-evaluation" if layer_name == "structure_2d" else "mechanistic-discovery"
            )
            predictions["endpoint"] = task_name
            predictions["feature_layer"] = layer_name
            atomic_write_csv(task_dir / f"{layer_name}_metrics.csv", metrics)
            atomic_write_parquet(task_dir / f"{layer_name}_predictions.parquet", predictions)
            layer_results[layer_name] = metrics
            best = metrics.sort_values("log_mae").iloc[0]
            summary_rows.append(
                {
                    "domain": "pk",
                    "endpoint": task_name,
                    "feature_layer": layer_name,
                    "status": "evaluated",
                    "n": int(best["n"]),
                    "best_model": best["model"],
                    "optimizer_final_model": optimizer_final_model,
                    "primary_metric": "log_mae",
                    "primary_value": float(best["log_mae"]),
                    "interval_coverage": float(best.get("prediction_interval_coverage", np.nan)),
                }
            )
        if "structure_2d" in layer_results and "state_conformer_physics" in layer_results:
            baseline = layer_results["structure_2d"].sort_values("log_mae").iloc[0].to_dict()
            candidate = layer_results["state_conformer_physics"].sort_values("log_mae").iloc[0].to_dict()
            promotion_rows.append(
                {
                    "endpoint": task_name,
                    **promotion_decision(
                        baseline, candidate, primary_metric="log_mae", calibrated=False, converged=False
                    ),
                }
            )
    summary = pd.DataFrame(summary_rows)
    atomic_write_csv(output_dir / "pk_model_ladder_summary.csv", summary)
    atomic_write_csv(output_dir / "pk_physics_promotion_gates.csv", pd.DataFrame(promotion_rows))
    atomic_write_csv(
        output_dir / "pk_hierarchical_promotion_gates.csv", pd.DataFrame(hierarchical_promotion_rows)
    )
    atomic_write_csv(output_dir / "pk_identifiability_contract.csv", pk_identifiability_contract())
    final_predictions = (
        pd.concat(final_prediction_frames, ignore_index=True) if final_prediction_frames else pd.DataFrame()
    )
    if not final_predictions.empty:
        final_predictions = derive_pk_prediction_views(final_predictions)
        atomic_write_parquet(output_dir / "optimizer_predictions_long.parquet", final_predictions)
        atomic_write_csv(output_dir / "optimizer_predictions_long.csv", final_predictions)
    return {
        "n_tasks": len(tasks),
        "summary_rows": len(summary),
        "optimizer_prediction_rows": len(final_predictions),
        "output_dir": str(output_dir),
    }


def run_herg_models(
    canonical_root: Path,
    physics_root: Path,
    output_dir: Path,
    *,
    folds: int,
    random_state: int,
    neural_epochs: int = 40,
    interval_level: float = 0.90,
) -> dict[str, Any]:
    tables = load_canonical_tables(canonical_root)
    alias_path = canonical_root / "internal" / "compound_aliases.parquet"
    aliases = pd.read_parquet(alias_path) if alias_path.exists() else None
    compounds = compound_model_frame(tables["compounds"], aliases)
    physics_path = physics_root / "fast_physics_summary.parquet"
    physics = nominal_physics_summary(physics_path)
    features, layers = merge_feature_layers(compounds, physics)
    feature_compounds, potency, inhibition = prepare_herg_evidence(
        compounds, tables["measurements"], features
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    structure_censored_predictions: pd.DataFrame | None = None
    structure_censored_columns: list[str] | None = None
    baseline_columns = [
        column for column in layers["structure_2d"] if column in potency and potency[column].notna().any()
    ]
    exact = potency[potency["observed_pic50"].notna()].copy()
    conventional_metrics, conventional_predictions = grouped_exact_herg_benchmark(
        exact,
        feature_columns=baseline_columns,
        folds=folds,
        random_state=random_state,
    )
    booster_metrics, booster_predictions = _run_boosters_isolated(
        exact,
        feature_columns=baseline_columns,
        target_column="observed_pic50",
        group_column="scaffold",
        folds=folds,
        mode="herg",
    )
    if not booster_metrics.empty:
        conventional_metrics = pd.concat([conventional_metrics, booster_metrics], ignore_index=True)
        conventional_predictions = pd.concat(
            [conventional_predictions, booster_predictions], ignore_index=True
        )
    atomic_write_csv(output_dir / "conventional_exact_pic50_metrics.csv", conventional_metrics)
    atomic_write_parquet(
        output_dir / "conventional_exact_pic50_predictions.parquet", conventional_predictions
    )
    best = conventional_metrics.sort_values("pic50_mae").iloc[0]
    rows.append(
        {
            "model": best["model"],
            "feature_layer": "structure_2d_exact",
            "status": "evaluated",
            "pic50_mae": best["pic50_mae"],
        }
    )

    for layer_name, columns in layers.items():
        usable = [column for column in columns if column in potency and potency[column].notna().any()]
        metrics, predictions = grouped_censored_herg_benchmark(potency, feature_columns=usable, folds=folds)
        metrics.update({"feature_layer": layer_name, "model": "censored_gaussian_ridge"})
        atomic_write_json(output_dir / f"{layer_name}_censored_metrics.json", metrics)
        atomic_write_parquet(output_dir / f"{layer_name}_censored_predictions.parquet", predictions)
        rows.append(
            {
                "model": "censored_gaussian_ridge",
                "feature_layer": layer_name,
                "status": _censored_fit_status(metrics),
                "pic50_mae": metrics.get("pic50_mae"),
            }
        )
        if layer_name == "structure_2d":
            structure_censored_predictions = predictions
            structure_censored_columns = usable

    joint_metrics, joint_predictions = grouped_joint_herg_benchmark(
        feature_compounds, potency, inhibition, feature_columns=baseline_columns, folds=folds
    )
    atomic_write_json(output_dir / "joint_pic50_inhibition_metrics.json", joint_metrics)
    atomic_write_parquet(output_dir / "joint_pic50_inhibition_predictions.parquet", joint_predictions)
    rows.append(
        {
            "model": joint_metrics["model"],
            "feature_layer": "structure_2d_joint_observations",
            "status": _censored_fit_status(joint_metrics),
            "pic50_mae": joint_metrics.get("pic50_mae"),
        }
    )

    dmpnn_input = potency[
        ["compound_id", "standardized_smiles", "scaffold", "pic50_lower", "pic50_upper"]
    ].copy()
    try:
        dmpnn_metrics, dmpnn_predictions = grouped_dmpnn_herg_benchmark(
            dmpnn_input, folds=folds, epochs=neural_epochs, random_state=random_state
        )
    except ImportError as exc:
        rows.append(
            {
                "model": "directed_message_passing_neural_network",
                "feature_layer": "molecular_graph",
                "status": f"unavailable: {exc}",
                "pic50_mae": np.nan,
            }
        )
    else:
        atomic_write_json(output_dir / "dmpnn_metrics.json", dmpnn_metrics)
        atomic_write_parquet(output_dir / "dmpnn_predictions.parquet", dmpnn_predictions)
        rows.append(
            {
                "model": dmpnn_metrics["model"],
                "feature_layer": "molecular_graph",
                "status": "discovery-track",
                "pic50_mae": dmpnn_metrics.get("pic50_mae"),
            }
        )

    conformer_path = physics_root / "fast_physics_conformers.parquet"
    population_path = physics_root / "fast_physics_state_populations.parquet"
    registry_path = physics_root / "fast_physics_structure_registry.parquet"
    if conformer_path.exists() and population_path.exists() and registry_path.exists():
        conformers = pd.read_parquet(conformer_path)
        populations = pd.read_parquet(population_path)
        if "pka_scenario" in populations:
            populations = populations[populations["pka_scenario"].astype(str) == "nominal"]
        if "ph" in populations and populations["ph"].notna().any():
            chosen_ph = min(populations["ph"].dropna().unique(), key=lambda value: abs(float(value) - 7.4))
            populations = populations[np.isclose(populations["ph"].astype(float), float(chosen_ph))]
        registry = pd.read_parquet(registry_path)[["compound_id", "structure_id"]]
        conformers = _merge_conformer_state_weights(conformers, populations, registry)
        # Fail closed: identifiers, minimization status/energy, aliases,
        # algebraic closures, and unreviewed future numeric columns cannot enter
        # the conformer model merely because they are numeric.
        mil_features = selected_model_conformer_features(conformers.columns)
        mil_compounds = potency[["compound_id", "scaffold", "pic50_lower", "pic50_upper"]]
        if mil_features and "ensemble_weight" in conformers:
            try:
                mil_metrics, mil_predictions = grouped_conformer_mil_benchmark(
                    mil_compounds,
                    conformers,
                    feature_columns=mil_features,
                    folds=folds,
                    epochs=neural_epochs,
                    random_state=random_state,
                )
            except ImportError as exc:
                rows.append(
                    {
                        "model": "conformer_attention_multiple_instance",
                        "feature_layer": "state_conformer_bags",
                        "status": f"unavailable: {exc}",
                        "pic50_mae": np.nan,
                    }
                )
            else:
                atomic_write_json(output_dir / "conformer_mil_metrics.json", mil_metrics)
                atomic_write_parquet(output_dir / "conformer_mil_predictions.parquet", mil_predictions)
                rows.append(
                    {
                        "model": mil_metrics["model"],
                        "feature_layer": "state_conformer_bags",
                        "status": "discovery-track",
                        "pic50_mae": mil_metrics.get("pic50_mae"),
                    }
                )

    public_root = canonical_root / "public_herg"
    if (public_root / "public_herg_normalized.parquet").exists():
        public_output = output_dir / "public_sun_reproduction"
        public_audit = run_sun_public_reproduction(public_root, public_output, random_state=random_state)
        public_classification = pd.read_csv(public_output / "classification_source_holdout_metrics.csv")
        public_regression = pd.read_csv(public_output / "regression_source_holdout_metrics.csv")
        best_public_classification = public_classification.sort_values("roc_auc", ascending=False).iloc[0]
        best_public_regression = public_regression.sort_values("pic50_mae").iloc[0]
        rows.extend(
            [
                {
                    "model": best_public_classification["model"],
                    "feature_layer": "sun_public_source_holdout_classification",
                    "status": public_audit["reproduction_status"],
                    "roc_auc": best_public_classification["roc_auc"],
                },
                {
                    "model": best_public_regression["model"],
                    "feature_layer": "sun_public_source_holdout_regression",
                    "status": public_audit["reproduction_status"],
                    "pic50_mae": best_public_regression["pic50_mae"],
                },
            ]
        )

    summary = pd.DataFrame(rows)
    atomic_write_csv(output_dir / "herg_model_ladder_summary.csv", summary)
    atomic_write_csv(output_dir / "receptor_ensemble.csv", HERG_RECEPTOR_ENSEMBLE)
    atomic_write_csv(output_dir / "herg_process_observables.csv", herg_process_observables())
    states, transitions = state_dependent_markov_architecture()
    atomic_write_csv(output_dir / "markov_states_architecture.csv", states)
    atomic_write_csv(output_dir / "markov_transitions_architecture.csv", transitions)
    final_predictions = pd.DataFrame()
    if structure_censored_predictions is not None and structure_censored_columns:
        final_predictions = final_fit_censored_herg_predictions(
            potency,
            feature_compounds,
            structure_censored_predictions,
            feature_columns=structure_censored_columns,
            interval_level=interval_level,
        )
        atomic_write_parquet(output_dir / "optimizer_predictions_long.parquet", final_predictions)
        atomic_write_csv(output_dir / "optimizer_predictions_long.csv", final_predictions)
    return {
        "n_potency_rows": len(potency),
        "n_inhibition_rows": len(inhibition),
        "summary_rows": len(summary),
        "optimizer_prediction_rows": len(final_predictions),
        "output_dir": str(output_dir),
    }


def build_regime_analysis(
    canonical_root: Path,
    physics_root: Path,
    output_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    tables = load_canonical_tables(canonical_root)
    compounds = compound_model_frame(tables["compounds"])
    physics_path = physics_root / "fast_physics_summary.parquet"
    physics = nominal_physics_summary(physics_path)
    features, _ = merge_feature_layers(compounds, physics)
    compound_features = compounds.merge(
        features,
        on="compound_id",
        how="inner",
        suffixes=("", "_feature"),
        validate="one_to_one",
    )
    measurements = tables["measurements"]
    outcomes: list[tuple[str, pd.DataFrame, str]] = []
    # Compound-level medians are used only for change-point discovery; source
    # rows and conflicts remain intact in the canonical measurement tables.
    exact_herg = measurements[
        (measurements["endpoint"] == "herg_ic50")
        & (measurements["relation"] == "=")
        & (measurements["value"] > 0)
    ].copy()
    if len(exact_herg):
        exact_herg["herg_pic50"] = 6.0 - np.log10(exact_herg["value"])
        outcome = (
            exact_herg.groupby("compound_id", as_index=False)["herg_pic50"]
            .median()
            .merge(compound_features, on="compound_id")
        )
        outcomes.append(("herg_pic50", outcome, "herg_pic50"))
    pk_tasks = prepare_pk_tasks(compounds, measurements, tables["pk_studies"], features)
    for name in ("iv_auc_dose_normalized", "po_auc_dose_normalized", "vdss"):
        task = pk_tasks.get(name, pd.DataFrame())
        if not task.empty:
            outcome = task.groupby("compound_id", as_index=False)["target_value"].median()
            outcome[f"log10_{name}"] = np.log10(outcome["target_value"])
            outcome = outcome.merge(compound_features, on="compound_id")
            outcomes.append((name, outcome, f"log10_{name}"))
    settings = config.get("mw_regime", {})
    summaries: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame, outcome_column in outcomes:
        covariates = [column for column in ("rotatable_bonds", "formal_charge", "tpsa") if column in frame]
        summary, bootstrap = bootstrap_mw_change_point(
            frame,
            outcome=outcome_column,
            mw_column="mw",
            group_column="scaffold",
            covariates=covariates,
            candidate_min=float(settings.get("candidate_min_da", 650)),
            candidate_max=float(settings.get("candidate_max_da", 780)),
            candidate_step=float(settings.get("candidate_step_da", 5)),
            minimum_per_side=int(settings.get("minimum_per_side", 15)),
            bootstrap_replicates=int(settings.get("bootstrap_replicates", 500)),
        )
        summary["analysis_name"] = name
        summaries.append(summary)
        atomic_write_parquet(output_dir / f"{name}_mw_bootstrap.parquet", bootstrap)
    summary_frame, gate = apply_cross_outcome_cutoff_gate(
        summaries,
        minimum_selection_frequency=float(settings.get("minimum_selection_frequency", 0.70)),
        maximum_location_width_da=float(settings.get("maximum_interval_width_da", 50)),
    )
    atomic_write_csv(output_dir / "mw_regime_summary.csv", summary_frame)
    atomic_write_json(output_dir / "mw_cutoff_decision.json", gate)
    return gate


def build_assay_and_optimizer_outputs(
    canonical_root: Path,
    physics_root: Path,
    output_dir: Path,
    optimizer_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    tables = load_canonical_tables(canonical_root)
    alias_path = canonical_root / "internal" / "compound_aliases.parquet"
    aliases = pd.read_parquet(alias_path) if alias_path.exists() else None
    compounds = compound_model_frame(tables["compounds"], aliases)
    physics_summary = nominal_physics_summary(physics_root / "fast_physics_summary.parquet")
    features, _ = merge_feature_layers(compounds, physics_summary)
    selection = compounds.merge(features, on="compound_id", how="left", suffixes=("", "_feature"))
    if physics_summary is not None and not physics_summary.empty:
        context_columns = [
            column
            for column in (
                "compound_id",
                "physics_quality_status",
                "physics_model_eligible",
                "physics_decision_track_eligible",
                "physics_convergence_claimed",
                "physics_smoke_mode",
                "physics_reason_flags",
            )
            if column in physics_summary and (column == "compound_id" or column not in selection.columns)
        ]
        if len(context_columns) > 1:
            selection = selection.merge(
                physics_summary[context_columns].drop_duplicates("compound_id"),
                on="compound_id",
                how="left",
                validate="one_to_one",
            )
    measurements = tables["measurements"]
    herg_evidence = measurements[
        (measurements["endpoint"] == "herg_ic50") & (measurements["value"] > 0)
    ].copy()
    exact_herg = herg_evidence[herg_evidence["relation"] == "="].copy()
    if not exact_herg.empty:
        exact_herg["herg_pic50"] = 6.0 - np.log10(exact_herg["value"])
        selection = selection.merge(
            exact_herg.groupby("compound_id", as_index=False)["herg_pic50"].median(),
            on="compound_id",
            how="left",
        )
    if not herg_evidence.empty:
        relation = herg_evidence["relation"].fillna("=").astype(str)
        value = pd.to_numeric(herg_evidence["value"], errors="coerce")
        herg_evidence["blocker_evidence"] = relation.isin({"=", "~", "<", "<="}) & value.le(10.0)
        herg_evidence["nonblocker_evidence"] = relation.isin({"=", "~", ">", ">="}) & value.ge(30.0)
        class_rows = []
        for compound_id, group in herg_evidence.groupby("compound_id", sort=True):
            blocker = bool(group["blocker_evidence"].any())
            nonblocker = bool(group["nonblocker_evidence"].any())
            label = (
                "blocker"
                if blocker and not nonblocker
                else "nonblocker"
                if nonblocker and not blocker
                else "intermediate_or_missing"
            )
            class_rows.append({"compound_id": compound_id, "herg_class": label})
        selection = selection.merge(pd.DataFrame(class_rows), on="compound_id", how="left")
    pk_tasks = prepare_pk_tasks(compounds, measurements, tables["pk_studies"], features)
    mapping = {
        "vdss": "rat_vdss_l_kg",
        "po_auc_dose_normalized": "rat_po_auc_dose_normalized",
    }
    for task_name, output_name in mapping.items():
        task = pk_tasks.get(task_name, pd.DataFrame())
        if not task.empty:
            selection = selection.merge(
                task.groupby("compound_id", as_index=False)["target_value"]
                .median()
                .rename(columns={"target_value": output_name}),
                on="compound_id",
                how="left",
            )
    # Reported CL is retained for assay stratification only, never as an independent model label.
    clearance = measurements[(measurements["endpoint"] == "clearance") & measurements["value"].notna()]
    if not clearance.empty:
        selection = selection.merge(
            clearance.groupby("compound_id", as_index=False)["value"]
            .median()
            .rename(columns={"value": "rat_cl_ml_kg_min"}),
            on="compound_id",
            how="left",
        )
    panel_config = config.get("assay_panel", {})
    quota_raw = panel_config.get("mw_bin_minimums", {})
    quotas = {
        "650-699": int(quota_raw.get("650_699", 3)),
        "700-749": int(quota_raw.get("700_749", 6)),
        "750+": int(quota_raw.get("750_plus", 3)),
    }
    panel, pk_profiles, herg_protocol, matched_pairs = select_assay_panel(
        selection,
        panel_size=int(panel_config.get("size", 16)),
        mw_bin_minimums=quotas,
        minimum_matched_pairs=int(panel_config.get("minimum_matched_pairs", 4)),
        complete_rat_profiles=int(panel_config.get("complete_rat_profiles", 8)),
        state_dependent_herg=int(panel_config.get("state_dependent_herg", 6)),
        herg_class_minimums={"blocker": 2, "nonblocker": 2, "intermediate_or_missing": 2},
    )
    assay_requests = expand_assay_requests(panel)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_dir / "assay_panel.csv", panel)
    atomic_write_csv(output_dir / "assay_requests.csv", assay_requests)
    atomic_write_csv(output_dir / "rat_profile_subset.csv", pk_profiles)
    atomic_write_csv(output_dir / "herg_kinetics_subset.csv", herg_protocol)
    atomic_write_csv(output_dir / "matched_pair_panel.csv", matched_pairs)
    optimizer_dir.mkdir(parents=True, exist_ok=True)
    prediction_frames: list[pd.DataFrame] = []
    model_root = Path(config["paths"]["models"])
    for domain in ("pk", "herg"):
        prediction_path = model_root / domain / "optimizer_predictions_long.parquet"
        if prediction_path.exists():
            prediction_frames.append(pd.read_parquet(prediction_path))
    optimizer_predictions = (
        pd.concat(prediction_frames, ignore_index=True, sort=False) if prediction_frames else pd.DataFrame()
    )
    if not optimizer_predictions.empty:
        duplicate = optimizer_predictions.duplicated(["compound_id", "endpoint"], keep=False)
        if duplicate.any():
            keys = optimizer_predictions.loc[duplicate, ["compound_id", "endpoint"]].drop_duplicates()
            raise ValueError(f"Duplicate final-fit optimizer predictions: {keys.to_dict('records')[:5]}")
        expected = set(compounds["compound_id"].astype(str))
        for endpoint, endpoint_rows in optimizer_predictions.groupby("endpoint", sort=True):
            observed = set(endpoint_rows["compound_id"].astype(str))
            if observed != expected:
                raise ValueError(
                    f"Final-fit endpoint {endpoint} does not cover the complete internal library: "
                    f"expected {len(expected)}, observed {len(observed)}"
                )
        atomic_write_parquet(optimizer_dir / "optimizer_predictions_long.parquet", optimizer_predictions)
        atomic_write_csv(optimizer_dir / "optimizer_predictions_long_review.csv", optimizer_predictions)
    optimizer = build_optimizer_contract(compounds, optimizer_predictions)
    atomic_write_parquet(optimizer_dir / "optimizer_contract.parquet", optimizer)
    atomic_write_csv(optimizer_dir / "optimizer_contract_review.csv", optimizer)
    return {
        "panel": panel,
        "assay_requests": assay_requests,
        "pk_profiles": pk_profiles,
        "herg_protocol": herg_protocol,
        "matched_pairs": matched_pairs,
        "quotas": quotas,
        "optimizer_rows": len(optimizer),
        "optimizer_prediction_rows": len(optimizer_predictions),
        "optimizer_prediction_endpoints": int(optimizer_predictions["endpoint"].nunique())
        if not optimizer_predictions.empty
        else 0,
    }


def write_model_ladder(path: Path) -> Path:
    return atomic_write_csv(path, model_ladder_registry())
