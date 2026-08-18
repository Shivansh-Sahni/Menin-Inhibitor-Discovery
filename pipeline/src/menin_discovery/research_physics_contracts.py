"""Validated contract projections for the detailed fast-physics artifacts.

The fast-physics engine intentionally writes wide, analysis-friendly tables.
This module keeps those tables untouched and adds a typed relational view for
the process-centred research interfaces.  The projection is explicit about an
important limitation: RDKit conformers are screening geometries without an
explicit solvent, not equilibrium environment-MD ensembles.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .research_common import atomic_write_csv, require_columns
from .research_contracts import write_contract_parquet
from .research_feature_ontology import (
    MODEL_PHYSICS_FEATURES,
    classify_physics_feature,
    feature_ontology_frame,
)

CONTRACT_PHYSICS_FILES = {
    "chemical_states": "chemical_states.parquet",
    "conformers": "conformers.parquet",
    "physics_runs": "physics_runs.parquet",
    "physics_observables": "physics_observables.parquet",
    "feature_lineage": "feature_lineage.parquet",
}

_SCREENING_ENVIRONMENT = "no_explicit_solvent_rdkit_screen"
_POPULATION_METHOD = "tautomer/protomer enumeration with approximate Henderson-Hasselbalch weighting"
_RUN_METHOD = "RDKit ETKDG conformer generation, force-field minimization, and approximate pKa weighting"
_KNOWN_CONFOUNDERS = (
    "approximate rather than experimental micro-pKa; finite conformer enumeration; "
    "MMFF94s/UFF energy-model sensitivity; no explicit solvent or membrane relaxation"
)


def _ontology_context(feature: str) -> tuple[str, str, bool]:
    """Return causal location, review text, and explicit model permission."""

    concept = classify_physics_feature(feature)
    if concept is None:
        return (
            "unclassified_fast_physics_output",
            "No ontology entry exists. This feature is retained for audit but fails closed for modeling.",
            False,
        )
    context = (
        f"Physical phenomenon: {concept.physical_phenomenon}. Biological event: "
        f"{concept.biological_event}. Permissible roles: {concept.permissible_model_roles}. "
        f"Status: {concept.status}. Redundancy group: {concept.redundancy_group}. "
        f"Hidden variables/confounders: {concept.hidden_variables_and_confounders}. "
        f"Falsification: {concept.falsification_test}"
    )
    return concept.causal_location, context, feature in MODEL_PHYSICS_FEATURES


def _text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _finite_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        return not bool(pd.isna(value)) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _ph_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _run_id(compound_id: str, ph: float, scenario: str) -> str:
    return f"FPR:{compound_id}:pH{_ph_token(ph)}:{scenario}"


def _state_id(compound_id: str, raw_state_id: str, ph: float) -> str:
    return f"FST:{compound_id}:{raw_state_id}:pH{_ph_token(ph)}"


def _conformer_id(compound_id: str, raw_state_id: str, raw_conformer_id: str, ph: float) -> str:
    return f"FCN:{compound_id}:{raw_state_id}:{raw_conformer_id}:pH{_ph_token(ph)}"


def _observable_id(run_id: str, observable: str) -> str:
    return f"FPO:{run_id}:{observable}"


def _lineage_id(run_id: str, feature: str) -> str:
    return f"FLN:{run_id}:{feature}"


def _state_type(transformation: Any) -> str:
    value = str(transformation or "").casefold()
    if "proton" in value or "deproton" in value:
        return "protomer"
    if "tautomer" in value:
        return "tautomer"
    return "microstate"


def _feature_unit(name: str) -> str:
    value = name.casefold()
    if "pmi" in value:
        return "amu*angstrom^2"
    if "ang2" in value or "sasa" in value or "polar_surface" in value:
        return "angstrom^2"
    if "angstrom" in value or "gyration" in value or "centroid_separation" in value:
        return "angstrom"
    if "debye" in value or "dipole" in value:
        return "debye"
    if "entropy_nats" in value:
        return "nat"
    if "count" in value and "fraction" not in value:
        return "count"
    return "dimensionless"


def _aggregation(name: str) -> str:
    if name.endswith("__mean"):
        return "microstate-by-conformer joint-weighted mean"
    if name.endswith("__sd"):
        return "microstate-by-conformer joint-weighted standard deviation"
    if "__q" in name:
        return "microstate-by-conformer joint-weighted quantile"
    if "entropy" in name:
        return "entropy of normalized microstate-by-conformer joint weights"
    if name == "effective_joint_state_conformer_count":
        return "inverse Simpson effective count of joint weights"
    if name == "joint_weight_sum":
        return "sum of normalized microstate-by-conformer joint weights"
    return "declared fast-physics ensemble aggregation"


def _population_projection(
    registry: pd.DataFrame,
    states: pd.DataFrame,
    populations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(registry, {"compound_id", "structure_id"}, label="fast-physics registry")
    require_columns(
        states,
        {"structure_id", "state_id", "state_smiles", "transformation", "formal_charge"},
        label="fast-physics states",
    )
    require_columns(
        populations,
        {"structure_id", "state_id", "ph", "pka_scenario", "state_weight"},
        label="fast-physics populations",
    )
    mapping = registry[["compound_id", "structure_id"]].dropna().drop_duplicates()
    joined = populations.merge(
        states,
        on=["structure_id", "state_id"],
        how="inner",
        validate="many_to_one",
    ).merge(mapping, on="structure_id", how="inner", validate="many_to_many")
    joined["state_weight"] = pd.to_numeric(joined["state_weight"], errors="coerce")
    nominal = joined[joined["pka_scenario"].astype(str) == "nominal"].copy()
    if nominal.empty:
        raise ValueError("Fast-physics populations contain no nominal pKa scenario")
    key = ["compound_id", "structure_id", "state_id", "ph"]
    bounds = joined.groupby(key, as_index=False)["state_weight"].agg(
        population_min="min",
        population_max="max",
    )
    nominal = nominal.merge(bounds, on=key, how="left", validate="one_to_one")
    nominal["population_uncertainty"] = np.maximum(
        (nominal["state_weight"] - nominal["population_min"]).abs(),
        (nominal["population_max"] - nominal["state_weight"]).abs(),
    )
    nominal["chemical_state_id"] = [
        _state_id(str(compound_id), str(state_id), float(ph))
        for compound_id, state_id, ph in zip(
            nominal["compound_id"], nominal["state_id"], nominal["ph"], strict=True
        )
    ]
    records: list[dict[str, Any]] = []
    for row in nominal.itertuples(index=False):
        pka_context = "; ".join(
            value
            for value in (
                f"raw_state_id={row.state_id}",
                f"pKa basis={_text(getattr(row, 'pka_basis', None))}"
                if _text(getattr(row, "pka_basis", None))
                else None,
                f"pKa source={_text(getattr(row, 'pka_source', None))}"
                if _text(getattr(row, "pka_source", None))
                else None,
                "population uncertainty is the maximum deviation across the explicit pKa -1/nominal/+1 scenarios",
            )
            if value
        )
        records.append(
            {
                "chemical_state_id": row.chemical_state_id,
                "compound_id": str(row.compound_id),
                "state_type": _state_type(row.transformation),
                "smiles": _text(row.state_smiles),
                "formal_charge": None if pd.isna(row.formal_charge) else int(row.formal_charge),
                "p_h": float(row.ph),
                "fractional_population": float(row.state_weight),
                "environment": "aqueous_pH_evidence_weighting_no_explicit_solvent",
                "method": _POPULATION_METHOD,
                "uncertainty": float(row.population_uncertainty),
                "source": "fast_physics_state_populations.parquet",
                "context_note": pka_context,
            }
        )
    return pd.DataFrame(records), nominal


def _conformer_projection(
    conformers: pd.DataFrame,
    nominal_states: pd.DataFrame,
    *,
    target_ph: float,
) -> pd.DataFrame:
    require_columns(
        conformers,
        {
            "structure_id",
            "state_id",
            "conformer_id",
            "conformer_rank",
            "relative_energy_kcal_mol",
            "conformer_weight",
            "minimization_method",
        },
        label="fast-physics conformers",
    )
    available_ph = sorted(pd.to_numeric(nominal_states["ph"], errors="coerce").dropna().unique())
    if not available_ph:
        raise ValueError("No pH values are available for the conformer contract projection")
    chosen_ph = float(min(available_ph, key=lambda value: abs(float(value) - target_ph)))
    anchors = nominal_states[np.isclose(nominal_states["ph"].astype(float), chosen_ph)][
        ["compound_id", "structure_id", "state_id", "chemical_state_id", "ph"]
    ].drop_duplicates()
    joined = conformers.merge(
        anchors,
        on=["structure_id", "state_id"],
        how="inner",
        validate="many_to_many",
    )
    records: list[dict[str, Any]] = []
    for row in joined.itertuples(index=False):
        records.append(
            {
                "conformer_id": _conformer_id(
                    str(row.compound_id), str(row.state_id), str(row.conformer_id), float(row.ph)
                ),
                "chemical_state_id": str(row.chemical_state_id),
                "rank": int(row.conformer_rank),
                "relative_energy_kcal_mol": float(row.relative_energy_kcal_mol),
                "population": float(row.conformer_weight),
                "environment": _SCREENING_ENVIRONMENT,
                "geometry_uri": (
                    f"structures/{row.structure_id}/conformers.sdf#conformer_id={row.conformer_id}"
                ),
                "method": _text(row.minimization_method) or "RDKit force-field minimization",
                "source_run_id": _run_id(str(row.compound_id), float(row.ph), "nominal"),
                "context_note": (
                    f"raw_state_id={row.state_id}; cluster_id={_text(getattr(row, 'cluster_id', None))}; "
                    "population is conditional within this chemical state. Geometry is a screening "
                    "conformer, not an explicit-solvent equilibrium sample."
                ),
            }
        )
    return pd.DataFrame(records)


def _summary_projection(
    summary: pd.DataFrame,
    composites: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    temperature_kelvin: float,
    random_seed: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_columns(
        summary,
        {"compound_id", "structure_id", "ph", "pka_scenario"},
        label="fast-physics summary",
    )
    summary = summary.drop_duplicates(["compound_id", "ph", "pka_scenario"]).copy()
    run_records: list[dict[str, Any]] = []
    observable_records: list[dict[str, Any]] = []
    lineage_records: list[dict[str, Any]] = []
    excluded = {"compound_id", "structure_id", "mw", "ph", "pka_scenario"}
    numeric_features = [
        column
        for column in summary.columns
        if column not in excluded
        and not column.startswith("physics_")
        and not column.startswith("composite__")
        and pd.api.types.is_numeric_dtype(summary[column])
    ]
    eligibility_by_condition: dict[tuple[str, float, str], bool] = {}
    for row in summary.itertuples(index=False):
        compound_id = str(row.compound_id)
        ph = float(row.ph)
        scenario = str(row.pka_scenario)
        feature_eligible = bool(getattr(row, "physics_model_eligible", True))
        eligibility_by_condition[(compound_id, ph, scenario)] = feature_eligible
        run_id = _run_id(compound_id, ph, scenario)
        run_records.append(
            {
                "physics_run_id": run_id,
                "compound_id": compound_id,
                "chemical_state_id": None,
                "conformer_id": None,
                "process": "fast_microstate_conformer_ensemble",
                "environment": _SCREENING_ENVIRONMENT,
                "method": _RUN_METHOD,
                "software": "RDKit",
                "software_version": None,
                "force_field": "MMFF94s with UFF fallback where required",
                "temperature_c": float(temperature_kelvin) - 273.15,
                "p_h": ph,
                "replicate": None,
                "random_seed": random_seed,
                "status": "complete",
                "configuration_uri": "fast_physics_run_summary.json",
                "source": "fast_physics_summary.parquet",
                "context_note": (
                    f"pKa sensitivity scenario={scenario}. Screening descriptors only; "
                    "this is not an equilibrium MD run."
                ),
            }
        )
        row_values = row._asdict()
        for feature in numeric_features:
            value = row_values.get(feature)
            if not _finite_number(value):
                continue
            uncertainty = None
            if feature.endswith("__mean"):
                sd_value = row_values.get(f"{feature[:-6]}__sd")
                if _finite_number(sd_value):
                    uncertainty = float(sd_value)
            observable_id = _observable_id(run_id, feature)
            aggregation = _aggregation(feature)
            unit = _feature_unit(feature)
            process_layer, ontology_context, ontology_model_eligible = _ontology_context(feature)
            observable_records.append(
                {
                    "physics_observable_id": observable_id,
                    "physics_run_id": run_id,
                    "compound_id": compound_id,
                    "chemical_state_id": None,
                    "conformer_id": None,
                    "observable": feature,
                    "value": float(value),
                    "unit": unit,
                    "relation": "=",
                    "censoring": "none",
                    "uncertainty": uncertainty,
                    "aggregation": aggregation,
                    "window_start": None,
                    "window_end": None,
                    "window_unit": None,
                    "source": "fast_physics_summary.parquet",
                    "context_note": f"pH={ph:g}; pKa sensitivity scenario={scenario}.",
                }
            )
            lineage_records.append(
                {
                    "feature_lineage_id": _lineage_id(run_id, feature),
                    "compound_id": compound_id,
                    "chemical_state_id": None,
                    "conformer_id": None,
                    "feature_name": feature,
                    "feature_value": float(value),
                    "feature_unit": unit,
                    "process_layer": process_layer,
                    "source_entity_type": "physics_observable",
                    "source_entity_ids": (observable_id,),
                    "transform": aggregation,
                    "formula": aggregation,
                    "leakage_role": "physics_feature",
                    "model_eligible": (
                        scenario == "nominal" and feature_eligible and ontology_model_eligible
                    ),
                    "version": "fast-physics-state-ensemble-v1",
                    "context_note": f"{ontology_context} Shared fast-layer limits: {_KNOWN_CONFOUNDERS}.",
                }
            )

    if not composites.empty:
        require_columns(
            composites,
            {"structure_id", "ph", "pka_scenario", "composite_name", "value", "definition"},
            label="fast-physics composites",
        )
        mapping = registry[["compound_id", "structure_id"]].dropna().drop_duplicates()
        composite_rows = composites.merge(
            mapping,
            on="structure_id",
            how="inner",
            validate="many_to_many",
        )
        for row in composite_rows.itertuples(index=False):
            if not _finite_number(row.value):
                continue
            compound_id = str(row.compound_id)
            ph = float(row.ph)
            scenario = str(row.pka_scenario)
            feature = str(row.composite_name)
            observable = f"composite::{feature}"
            run_id = _run_id(compound_id, ph, scenario)
            observable_id = _observable_id(run_id, observable)
            evidence_class = _text(getattr(row, "evidence_class", None)) or "unclassified"
            process_layer, ontology_context, ontology_model_eligible = _ontology_context(observable)
            observable_records.append(
                {
                    "physics_observable_id": observable_id,
                    "physics_run_id": run_id,
                    "compound_id": compound_id,
                    "chemical_state_id": None,
                    "conformer_id": None,
                    "observable": observable,
                    "value": float(row.value),
                    "unit": "dimensionless",
                    "relation": "=",
                    "censoring": "none",
                    "uncertainty": None,
                    "aggregation": str(row.definition),
                    "window_start": None,
                    "window_end": None,
                    "window_unit": None,
                    "source": "fast_physics_composites.parquet",
                    "context_note": (
                        f"pH={ph:g}; pKa sensitivity scenario={scenario}; evidence class={evidence_class}."
                    ),
                }
            )
            lineage_records.append(
                {
                    "feature_lineage_id": _lineage_id(run_id, observable),
                    "compound_id": compound_id,
                    "chemical_state_id": None,
                    "conformer_id": None,
                    "feature_name": feature,
                    "feature_value": float(row.value),
                    "feature_unit": "dimensionless",
                    "process_layer": process_layer,
                    "source_entity_type": "physics_observable",
                    "source_entity_ids": (observable_id,),
                    "transform": str(row.definition),
                    "formula": str(row.definition),
                    "leakage_role": "physics_feature",
                    "model_eligible": (
                        scenario == "nominal"
                        and eligibility_by_condition.get((compound_id, ph, scenario), False)
                        and ontology_model_eligible
                    ),
                    "version": "fast-physics-state-ensemble-v1",
                    "context_note": (
                        f"Evidence class={evidence_class}. {ontology_context} "
                        f"Shared fast-layer limits: {_KNOWN_CONFOUNDERS}."
                    ),
                }
            )
    return (
        pd.DataFrame(run_records),
        pd.DataFrame(observable_records),
        pd.DataFrame(lineage_records),
    )


def project_fast_physics_contracts(
    physics_dir: str | Path,
    *,
    target_ph: float = 7.4,
    temperature_kelvin: float = 298.15,
    random_seed: int | None = None,
) -> dict[str, int | Path]:
    """Create validated contract Parquets beside the detailed ``fast_*`` files."""

    root = Path(physics_dir)
    sources = {
        "registry": root / "fast_physics_structure_registry.parquet",
        "states": root / "fast_physics_states.parquet",
        "populations": root / "fast_physics_state_populations.parquet",
        "conformers": root / "fast_physics_conformers.parquet",
        "summary": root / "fast_physics_summary.parquet",
        "composites": root / "fast_physics_composites.parquet",
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Fast-physics contract projection is missing detailed inputs: {missing}")
    tables = {name: pd.read_parquet(path) for name, path in sources.items()}
    chemical_states, nominal_states = _population_projection(
        tables["registry"], tables["states"], tables["populations"]
    )
    conformers = _conformer_projection(tables["conformers"], nominal_states, target_ph=target_ph)
    runs, observables, lineage = _summary_projection(
        tables["summary"],
        tables["composites"],
        tables["registry"],
        temperature_kelvin=temperature_kelvin,
        random_seed=random_seed,
    )
    projected = {
        "chemical_states": chemical_states,
        "conformers": conformers,
        "physics_runs": runs,
        "physics_observables": observables,
        "feature_lineage": lineage,
    }
    for contract_name, frame in projected.items():
        if frame.empty:
            raise ValueError(f"Fast-physics contract projection produced no {contract_name} records")
        write_contract_parquet(contract_name, frame, root / CONTRACT_PHYSICS_FILES[contract_name])
    ontology_path = atomic_write_csv(root / "fast_physics_feature_ontology.csv", feature_ontology_frame())
    return {
        **{f"{name}_rows": int(len(frame)) for name, frame in projected.items()},
        **{f"{name}_path": root / CONTRACT_PHYSICS_FILES[name] for name in projected},
        "feature_ontology_rows": int(len(feature_ontology_frame())),
        "feature_ontology_path": ontology_path,
    }


__all__ = ["CONTRACT_PHYSICS_FILES", "project_fast_physics_contracts"]
