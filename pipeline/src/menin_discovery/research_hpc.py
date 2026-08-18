"""HPC-ready research bundles with hard guards against production launch.

The module writes reviewable OpenMM, PLUMED, and PyMBAR inputs.  It never
submits jobs or runs simulations.  The only executable template is a local
OpenMM smoke runner that refuses durations above two nanoseconds.
"""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .research_structures import validate_mmcif_coordinate

HPC_BUNDLE_VERSION = "physics-hpc-bundle-v2"

REQUIRED_ENVIRONMENTS = ("water", "chloroform_low_dielectric")
HERG_RECEPTOR_HYPOTHESES_PER_COMPOUND = 2
SMOKE_TOTAL_STEPS = 1_000_000
SMOKE_NVT_STEPS = 100_000

RECEPTOR_RAW_COORDINATE_PATHS = {
    "8ZYN": "research/literature/herg/structural_biology/2024_miyashita_inhibitor_bound/8ZYN.cif",
    "8ZYO": "research/literature/herg/structural_biology/2024_miyashita_inhibitor_bound/8ZYO.cif",
    "8ZYP": "research/literature/herg/structural_biology/2024_miyashita_inhibitor_bound/8ZYP.cif",
    "8ZYQ": "research/literature/herg/structural_biology/2024_miyashita_inhibitor_bound/8ZYQ.cif",
    "9CHP": "research/literature/herg/structural_biology/2024_lau_potassium_states/9CHP.cif",
    "9CHQ": "research/literature/herg/structural_biology/2024_lau_potassium_states/9CHQ.cif",
}


@dataclass(frozen=True)
class PilotSelectionConfig:
    """Constraints for deterministic physics-pilot selection."""

    n_select: int = 12
    matched_pair_slots: int = 4
    matched_pair_column: str = "matched_pair_id"
    endpoint_column: str = ""
    series_column: str = "series_id"
    max_per_series: int = 3
    strata_columns: tuple[str, ...] = ("herg_class", "pk_data_present")
    mandatory_ids: tuple[str, ...] = ()
    max_d_optimal_features: int = 32
    maximin_weight: float = 0.65
    d_optimal_weight: float = 0.35

    def __post_init__(self) -> None:
        if self.n_select < 1 or self.matched_pair_slots < 0:
            raise ValueError("Pilot counts must be non-negative and n_select must be positive")
        if self.max_per_series < 1 or self.max_d_optimal_features < 1:
            raise ValueError("Series and feature limits must be positive")
        if not math.isclose(self.maximin_weight + self.d_optimal_weight, 1.0, abs_tol=1e-12):
            raise ValueError("maximin_weight and d_optimal_weight must sum to one")


@dataclass(frozen=True)
class SimulationAcceptanceGates:
    """Predeclared gates; failure triggers extension or mechanistic review."""

    environment_min_effective_samples: int = 200
    environment_split_relative_delta_max: float = 0.15
    environment_replica_distribution_distance_max: float = 0.20
    pmf_last_half_drift_kcal_mol_max: float = 0.50
    pmf_analysis_tail_ns_min: float = 500.0
    pmf_leaflet_asymmetry_kcal_mol_max: float = 1.00
    pmf_replica_rmse_kcal_mol_max: float = 1.00
    pmf_forward_reverse_hysteresis_kcal_mol_max: float = 1.00
    pmf_min_window_effective_samples: int = 50
    pmf_adjacent_overlap_min: float = 0.03
    pmf_local_global_z_r2_min: float = 0.95
    pmf_patch_barrier_spread_kcal_mol_max: float = 1.50
    herg_backbone_rmsd_plateau_slope_angstrom_per_ns_max: float = 0.01
    herg_pose_retention_replica_fraction_min: float = 2.0 / 3.0
    herg_contact_occupancy_min: float = 0.30
    herg_state_classifier_confidence_min: float = 0.80
    rbfe_cycle_closure_kcal_mol_max: float = 1.00


def _default_receptor_ensemble() -> tuple[dict[str, str], ...]:
    receptors = (
        {
            "pdb_id": "8ZYN",
            "state": "digitonin_apo_C1",
            "ligand": "none",
            "selection_basis": "matched_recent_cavity_series_apo",
        },
        {
            "pdb_id": "8ZYO",
            "state": "inhibitor_bound_C1",
            "ligand": "astemizole",
            "selection_basis": "matched_recent_cavity_series_bound",
        },
        {
            "pdb_id": "8ZYP",
            "state": "inhibitor_bound_C1",
            "ligand": "E-4031",
            "selection_basis": "matched_recent_cavity_series_bound",
        },
        {
            "pdb_id": "8ZYQ",
            "state": "inhibitor_bound_C1",
            "ligand": "pimozide",
            "selection_basis": "matched_recent_cavity_series_bound",
        },
        {
            "pdb_id": "9CHP",
            "state": "high_K_C4",
            "ligand": "none",
            "selection_basis": "matched_C4_filter_condition_high_K",
        },
        {
            "pdb_id": "9CHQ",
            "state": "low_K_C4",
            "ligand": "none",
            "selection_basis": "matched_C4_filter_condition_low_K",
        },
    )
    return tuple(
        {
            **receptor,
            "canonical_raw_coordinate_path": RECEPTOR_RAW_COORDINATE_PATHS[receptor["pdb_id"]],
        }
        for receptor in receptors
    )


@dataclass(frozen=True)
class HPCBundleConfig:
    """Protocol metadata for review; none of these settings launches work."""

    project_id: str = "menin_pk_herg_physics"
    smoke_duration_ns: float = 2.0
    production_replicates: int = 3
    environment_initial_ns: float = 100.0
    environment_extension_ns: float = 100.0
    environment_max_ns: float = 500.0
    membrane_patch_sizes_popc: tuple[int, ...] = (64, 128, 256)
    membrane_primary_patch_popc: int = 128
    pmf_z_min_angstrom: float = -35.0
    pmf_z_max_angstrom: float = 35.0
    pmf_window_spacing_angstrom: float = 1.0
    pmf_equilibration_ns_per_window: float = 2.0
    pmf_production_ns_per_window: float = 20.0
    pmf_restraint_kj_mol_nm2: float = 1000.0
    herg_initial_ns: float = 200.0
    herg_extension_ns: float = 100.0
    herg_max_ns: float = 500.0
    ionic_strength_molar: float = 0.15
    temperature_kelvin: float = 310.0
    pressure_bar: float = 1.01325
    ligand_force_field: str = "CGenFF"
    ligand_charge_model: str = "CGenFF ParamChem with penalty review"
    protein_force_field: str = "CHARMM36m"
    lipid_force_field: str = "CHARMM36"
    water_model: str = "TIP3P"
    sensitivity_ligand_force_field: str = "GAFF2"
    sensitivity_ligand_charge_model: str = "AM1-BCC"
    sensitivity_protein_force_field: str = "Amber ff19SB"
    sensitivity_lipid_force_field: str = "Lipid21"
    sensitivity_water_model: str = "OPC"
    receptor_ensemble: tuple[dict[str, str], ...] = field(default_factory=_default_receptor_ensemble)
    acceptance: SimulationAcceptanceGates = field(default_factory=SimulationAcceptanceGates)

    def __post_init__(self) -> None:
        if not math.isclose(self.smoke_duration_ns, 2.0, abs_tol=1e-12):
            raise ValueError("The only local smoke definition is exactly 2 ns")
        if self.production_replicates < 3:
            raise ValueError("Production definitions require at least three independent replicas")
        if self.membrane_primary_patch_popc not in self.membrane_patch_sizes_popc:
            raise ValueError("Primary membrane patch must be included in patch-size sensitivity set")
        if min(self.membrane_patch_sizes_popc) < 32:
            raise ValueError("Membrane patches smaller than 32 POPC are unsupported")
        if self.pmf_z_max_angstrom <= self.pmf_z_min_angstrom or self.pmf_window_spacing_angstrom <= 0:
            raise ValueError("Invalid PMF window range")


def _standardized_feature_matrix(
    candidates: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    max_features: int,
) -> tuple[np.ndarray, list[str]]:
    numeric = candidates.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median()).fillna(0.0)
    varying = [column for column in numeric if float(numeric[column].std(ddof=0)) > 1e-12]
    if not varying:
        return np.zeros((len(candidates), 1), dtype=float), ["constant_fallback"]
    if len(varying) > max_features:
        ranked = sorted(varying, key=lambda column: (-float(numeric[column].var(ddof=0)), column))
        varying = ranked[:max_features]
    values = numeric[varying].to_numpy(dtype=float)
    values = (values - values.mean(axis=0)) / values.std(axis=0)
    return values, varying


def _farthest_pair(
    indices: list[int], matrix: np.ndarray, endpoint: np.ndarray | None
) -> tuple[int, int, float]:
    best: tuple[int, int, float] | None = None
    for left_position, left in enumerate(indices):
        for right in indices[left_position + 1 :]:
            distance = float(np.linalg.norm(matrix[left] - matrix[right]))
            if endpoint is not None and np.isfinite(endpoint[left]) and np.isfinite(endpoint[right]):
                distance += abs(float(endpoint[left] - endpoint[right]))
            candidate = (left, right, distance)
            if best is None or (candidate[2], -candidate[0], -candidate[1]) > (best[2], -best[0], -best[1]):
                best = candidate
    if best is None:
        raise ValueError("Matched-pair group requires at least two members")
    return best


def select_pilot_compounds(
    candidates: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    config: PilotSelectionConfig | None = None,
) -> pd.DataFrame:
    """Select deterministic matched-pair and maximin/D-optimal pilot archetypes."""

    config = config or PilotSelectionConfig()
    if "compound_id" not in candidates.columns:
        raise ValueError("candidates requires compound_id")
    table = candidates.drop_duplicates("compound_id", keep="first").copy().reset_index(drop=True)
    table["compound_id"] = table["compound_id"].astype(str)
    if table.empty:
        return table.assign(pilot_rank=pd.Series(dtype=int), selection_reason=pd.Series(dtype=str))
    target = min(config.n_select, len(table))
    matrix, used_features = _standardized_feature_matrix(
        table,
        feature_columns,
        max_features=config.max_d_optimal_features,
    )
    endpoint = None
    if config.endpoint_column and config.endpoint_column in table:
        endpoint = pd.to_numeric(table[config.endpoint_column], errors="coerce").to_numpy(dtype=float)

    selected: list[int] = []
    reasons: dict[int, list[str]] = {}
    scores: dict[int, float] = {}

    def add(index: int, reason: str, score: float) -> None:
        if index not in selected and len(selected) < target:
            selected.append(index)
            reasons[index] = [reason]
            scores[index] = float(score)
        elif index in selected and reason not in reasons[index]:
            reasons[index].append(reason)

    id_to_index = {compound_id: index for index, compound_id in enumerate(table["compound_id"])}
    for compound_id in config.mandatory_ids:
        if compound_id in id_to_index:
            add(id_to_index[compound_id], "mandatory", math.inf)

    if config.matched_pair_column in table and config.matched_pair_slots >= 2:
        pair_candidates: list[tuple[float, str, int, int]] = []
        grouped = table.groupby(config.matched_pair_column, sort=True, dropna=True).groups
        for pair_id, positions in grouped.items():
            indices = sorted(int(position) for position in positions)
            if len(indices) < 2:
                continue
            left, right, score = _farthest_pair(indices, matrix, endpoint)
            pair_candidates.append((score, str(pair_id), left, right))
        slots_used = 0
        for score, pair_id, left, right in sorted(pair_candidates, key=lambda item: (-item[0], item[1])):
            if slots_used + 2 > config.matched_pair_slots or len(selected) + 2 > target:
                continue
            add(left, f"matched_pair:{pair_id}", score)
            add(right, f"matched_pair:{pair_id}", score)
            slots_used += 2

    for column in config.strata_columns:
        if column not in table or len(selected) >= target:
            continue
        for value in sorted(table[column].dropna().astype(str).unique()):
            if selected and value in set(table.iloc[selected][column].dropna().astype(str)):
                continue
            options = [
                index
                for index in table.index
                if str(table.at[index, column]) == value and index not in selected
            ]
            if not options:
                continue
            if selected:
                best = max(
                    options,
                    key=lambda index: (
                        min(float(np.linalg.norm(matrix[index] - matrix[chosen])) for chosen in selected),
                        table.at[index, "compound_id"],
                    ),
                )
            else:
                best = min(options, key=lambda index: table.at[index, "compound_id"])
            add(best, f"stratum:{column}={value}", 0.0)

    def series_allowed(index: int) -> bool:
        if config.series_column not in table:
            return True
        value = table.at[index, config.series_column]
        if pd.isna(value):
            return True
        count = sum(table.at[chosen, config.series_column] == value for chosen in selected)
        return count < config.max_per_series

    while len(selected) < target:
        options = [index for index in table.index if index not in selected and series_allowed(index)]
        if not options:
            options = [index for index in table.index if index not in selected]
        if not options:
            break
        base_information = np.eye(matrix.shape[1])
        if selected:
            base_information += matrix[selected].T @ matrix[selected]
        _, base_logdet = np.linalg.slogdet(base_information)
        evaluated: list[tuple[float, str, int]] = []
        for index in options:
            if selected:
                min_distance = min(
                    float(np.linalg.norm(matrix[index] - matrix[chosen])) for chosen in selected
                )
            else:
                min_distance = float(np.linalg.norm(matrix[index]))
            candidate_information = base_information + np.outer(matrix[index], matrix[index])
            _, candidate_logdet = np.linalg.slogdet(candidate_information)
            d_gain = max(0.0, float(candidate_logdet - base_logdet))
            score = config.maximin_weight * min_distance + config.d_optimal_weight * d_gain
            evaluated.append((score, table.at[index, "compound_id"], index))
        score, _, chosen = max(evaluated, key=lambda item: (item[0], item[1]))
        add(chosen, "greedy_maximin_d_optimal", score)

    output = table.iloc[selected].copy()
    output["pilot_rank"] = np.arange(1, len(output) + 1, dtype=int)
    output["selection_reason"] = [";".join(reasons[index]) for index in selected]
    output["selection_score"] = [scores[index] for index in selected]
    output["selection_feature_columns"] = ";".join(used_features)
    return output.reset_index(drop=True)


def _herg_cross_class_pair_map(
    compounds: pd.DataFrame,
    *,
    minimum_similarity: float = 0.55,
    maximum_pairs: int = 3,
) -> pd.DataFrame:
    """Return deterministic, non-overlapping blocker/nonblocker matched pairs."""

    required = {"compound_id", "standardized_smiles", "herg_class"}
    missing = sorted(required - set(compounds.columns))
    if missing:
        raise ValueError(f"hERG matched-pair selection is missing columns: {missing}")
    pool = (
        compounds[compounds["herg_class"].astype(str).isin({"blocker", "nonblocker"})]
        .drop_duplicates("compound_id", keep="first")
        .copy()
    )
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints: dict[str, Any] = {}
    for row in pool.itertuples(index=False):
        molecule = Chem.MolFromSmiles(str(row.standardized_smiles or ""))
        if molecule is not None:
            fingerprints[str(row.compound_id)] = generator.GetFingerprint(molecule)
    blockers = pool[pool["herg_class"].astype(str) == "blocker"]
    nonblockers = pool[pool["herg_class"].astype(str) == "nonblocker"]
    candidates: list[dict[str, Any]] = []
    for blocker in blockers.itertuples(index=False):
        blocker_id = str(blocker.compound_id)
        if blocker_id not in fingerprints:
            continue
        for nonblocker in nonblockers.itertuples(index=False):
            nonblocker_id = str(nonblocker.compound_id)
            if nonblocker_id not in fingerprints:
                continue
            similarity = float(
                DataStructs.TanimotoSimilarity(
                    fingerprints[blocker_id],
                    fingerprints[nonblocker_id],
                )
            )
            if similarity < minimum_similarity:
                continue
            candidates.append(
                {
                    "blocker_compound_id": blocker_id,
                    "nonblocker_compound_id": nonblocker_id,
                    "tanimoto": similarity,
                }
            )
    candidates.sort(
        key=lambda row: (
            -float(row["tanimoto"]),
            str(row["blocker_compound_id"]),
            str(row["nonblocker_compound_id"]),
        )
    )
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for candidate in candidates:
        pair_ids = {
            str(candidate["blocker_compound_id"]),
            str(candidate["nonblocker_compound_id"]),
        }
        if pair_ids & used:
            continue
        selected.append(
            {
                "herg_matched_pair_id": f"HMP-{len(selected) + 1:02d}",
                **candidate,
            }
        )
        used.update(pair_ids)
        if len(selected) >= maximum_pairs:
            break
    return pd.DataFrame(
        selected,
        columns=(
            "herg_matched_pair_id",
            "blocker_compound_id",
            "nonblocker_compound_id",
            "tanimoto",
        ),
    )


def _select_herg_pilots(
    candidates: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    selection_config: PilotSelectionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select exactly 3 blockers and 3 nonblockers, prioritizing complete pairs."""

    decisive = (
        candidates[candidates["herg_class"].astype(str).isin({"blocker", "nonblocker"})]
        .drop_duplicates("compound_id", keep="first")
        .copy()
    )
    pair_map = _herg_cross_class_pair_map(decisive, maximum_pairs=3)
    selected_ids: list[str] = []
    pair_by_compound: dict[str, tuple[str, float]] = {}
    for pair in pair_map.itertuples(index=False):
        blocker_id = str(pair.blocker_compound_id)
        nonblocker_id = str(pair.nonblocker_compound_id)
        selected_ids.extend([blocker_id, nonblocker_id])
        pair_by_compound[blocker_id] = (str(pair.herg_matched_pair_id), float(pair.tanimoto))
        pair_by_compound[nonblocker_id] = (str(pair.herg_matched_pair_id), float(pair.tanimoto))

    for label in ("blocker", "nonblocker"):
        observed = (
            sum(
                decisive.set_index("compound_id").loc[compound_id, "herg_class"] == label
                for compound_id in selected_ids
            )
            if selected_ids
            else 0
        )
        needed = max(0, 3 - int(observed))
        if needed == 0:
            continue
        remaining = decisive[
            decisive["herg_class"].astype(str).eq(label)
            & ~decisive["compound_id"].astype(str).isin(selected_ids)
        ]
        if remaining.empty:
            continue
        fill = select_pilot_compounds(
            remaining,
            feature_columns=feature_columns,
            config=replace(
                selection_config,
                n_select=min(needed, len(remaining)),
                matched_pair_slots=0,
                strata_columns=(),
                mandatory_ids=(),
            ),
        )
        selected_ids.extend(fill["compound_id"].astype(str).tolist())

    selected_ids = list(dict.fromkeys(selected_ids))
    order = {compound_id: index for index, compound_id in enumerate(selected_ids)}
    output = decisive[decisive["compound_id"].astype(str).isin(selected_ids)].copy()
    output["_selection_order"] = output["compound_id"].astype(str).map(order)
    output = output.sort_values("_selection_order", kind="stable").drop(columns="_selection_order")
    output = output.head(6).reset_index(drop=True)
    output["herg_matched_pair_id"] = (
        output["compound_id"].astype(str).map(lambda value: pair_by_compound.get(value, (None, np.nan))[0])
    )
    output["herg_pair_similarity"] = (
        output["compound_id"].astype(str).map(lambda value: pair_by_compound.get(value, (None, np.nan))[1])
    )
    output["pilot_rank"] = np.arange(1, len(output) + 1, dtype=int)
    output["selection_reason"] = np.where(
        output["herg_matched_pair_id"].notna(),
        "hERG_cross_class_matched_pair",
        "hERG_class_quota_diversity_fill",
    )
    output["selection_score"] = output["herg_pair_similarity"].fillna(0.0)
    output["selection_feature_columns"] = ";".join(feature_columns)
    return output, pair_map


def _yaml_text(payload: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=False)


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return path


def _smoke_runner_source() -> str:
    return '''"""Run or restart one prepared OpenMM system for the guarded 2 ns smoke."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import yaml


TOTAL_STEPS = 1_000_000  # exactly 2 ns at 2 fs
NVT_STEPS = 100_000


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--system-xml", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint-in", type=Path)
    parser.add_argument("--pause-at-step", type=int)
    parser.add_argument("--execute-smoke", action="store_true")
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    smoke = protocol["smoke"]
    if protocol.get("production_launch_enabled", True):
        raise SystemExit("Refusing bundle with production_launch_enabled=true")
    if smoke.get("mode") != "local_smoke" or float(smoke.get("duration_ns", 0)) != 2.0:
        raise SystemExit("Only the exact local 2 ns smoke protocol is permitted")
    if int(smoke.get("total_steps", -1)) != TOTAL_STEPS:
        raise SystemExit("Smoke protocol must declare exactly 1,000,000 steps")
    if args.pause_at_step is not None and not (NVT_STEPS < args.pause_at_step < TOTAL_STEPS):
        raise SystemExit("Restart-test pause must lie after NVT and before the exact smoke endpoint")
    if not 1 <= args.seed < 2_147_483_647:
        raise SystemExit("Seed must be a positive 31-bit integer")
    missing = [path for path in (args.system_xml, args.coordinates) if not path.is_file()]
    if missing:
        raise SystemExit("Missing prepared smoke input(s): " + ", ".join(str(path) for path in missing))
    if args.checkpoint_in is not None and not args.checkpoint_in.is_file():
        raise SystemExit(f"Restart checkpoint does not exist: {args.checkpoint_in}")
    if args.checkpoint_in is not None:
        prior_record_path = args.output_dir / "smoke_run_record.json"
        if not prior_record_path.is_file():
            raise SystemExit("Restart requires the prior paused smoke_run_record.json")
        prior_record = json.loads(prior_record_path.read_text())
        if prior_record.get("status") != "paused_for_restart_validation":
            raise SystemExit("Restart requires a checkpoint from the declared pause boundary")
        if int(prior_record.get("seed", -1)) != args.seed:
            raise SystemExit("Restart seed does not match the paused smoke record")
    if not args.execute_smoke:
        print("Validated prepared inputs and exact 2 ns smoke definition; pass --execute-smoke to run")
        return

    from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform, XmlSerializer, unit
    from openmm.app import CheckpointReporter, DCDReporter, PDBFile, Simulation, StateDataReporter

    args.output_dir.mkdir(parents=True, exist_ok=True)
    system = XmlSerializer.deserialize(args.system_xml.read_text())
    coordinates = PDBFile(str(args.coordinates))
    if system.getNumParticles() != coordinates.topology.getNumAtoms():
        raise SystemExit("System particle count does not match coordinate topology")
    if coordinates.topology.getPeriodicBoxVectors() is None:
        raise SystemExit("Coordinate topology lacks periodic box vectors")
    if not system.usesPeriodicBoundaryConditions():
        raise SystemExit("Prepared smoke system must be periodic for the declared NVT/NPT stability test")
    barostats = [
        force
        for force in (system.getForce(index) for index in range(system.getNumForces()))
        if isinstance(force, MonteCarloBarostat)
    ]
    if len(barostats) > 1:
        raise SystemExit("Prepared system contains more than one MonteCarloBarostat")
    if barostats:
        barostat = barostats[0]
        barostat.setDefaultPressure(float(protocol["pressure_bar"]) * unit.bar)
        barostat.setDefaultTemperature(float(protocol["temperature_kelvin"]) * unit.kelvin)
    else:
        barostat = MonteCarloBarostat(
            float(protocol["pressure_bar"]) * unit.bar,
            float(protocol["temperature_kelvin"]) * unit.kelvin,
            0,
        )
        system.addForce(barostat)
    barostat.setRandomNumberSeed(args.seed)
    barostat.setFrequency(0)
    integrator = LangevinMiddleIntegrator(
        float(protocol["temperature_kelvin"]) * unit.kelvin,
        1.0 / unit.picosecond,
        2.0 * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(args.seed)
    simulation = Simulation(coordinates.topology, system, integrator, Platform.getPlatformByName("CPU"))
    resumed = args.checkpoint_in is not None
    if resumed:
        simulation.loadCheckpoint(str(args.checkpoint_in))
    else:
        simulation.context.setPositions(coordinates.positions)
        simulation.minimizeEnergy()
        minimized_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        minimized_value = minimized_energy.value_in_unit(unit.kilojoule_per_mole)
        if not math.isfinite(float(minimized_value)):
            raise SystemExit("Energy minimization produced a nonfinite potential energy")
        simulation.context.setVelocitiesToTemperature(
            float(protocol["temperature_kelvin"]) * unit.kelvin,
            args.seed,
        )
    current_step = int(simulation.currentStep)
    if current_step < 0 or current_step > TOTAL_STEPS:
        raise SystemExit(f"Checkpoint step {current_step} lies outside the exact smoke trajectory")

    append = resumed and (args.output_dir / "smoke.dcd").exists()
    simulation.reporters.append(DCDReporter(str(args.output_dir / "smoke.dcd"), 5000, append=append))
    simulation.reporters.append(
        StateDataReporter(
            str(args.output_dir / "smoke.csv"),
            5000,
            step=True,
            time=True,
            temperature=True,
            potentialEnergy=True,
            density=True,
            append=append,
        )
    )
    simulation.reporters.append(CheckpointReporter(str(args.output_dir / "restart.chk"), 50_000))

    if current_step < NVT_STEPS:
        simulation.step(NVT_STEPS - current_step)
        current_step = int(simulation.currentStep)
        simulation.saveCheckpoint(str(args.output_dir / "nvt_complete.chk"))
    barostat.setFrequency(25)
    simulation.context.reinitialize(preserveState=True)
    target_step = args.pause_at_step if args.pause_at_step is not None else TOTAL_STEPS
    if current_step > target_step:
        raise SystemExit(f"Checkpoint step {current_step} lies beyond requested pause step {target_step}")
    if current_step < target_step:
        simulation.step(target_step - current_step)
    if args.pause_at_step is not None:
        simulation.saveCheckpoint(str(args.output_dir / "restart.chk"))
        _atomic_json(
            args.output_dir / "smoke_run_record.json",
            {
                "completed_steps": int(simulation.currentStep),
                "duration_ns": int(simulation.currentStep) * 2.0e-6,
                "restart_capable": True,
                "resumed": resumed,
                "seed": args.seed,
                "status": "paused_for_restart_validation",
            },
        )
        print("Paused guarded smoke at the requested restart-validation boundary")
        return
    if int(simulation.currentStep) != TOTAL_STEPS:
        raise SystemExit("Smoke run did not terminate at the declared 2 ns step")
    final_checkpoint = args.output_dir / "smoke_complete.chk"
    simulation.saveCheckpoint(str(final_checkpoint))
    _atomic_json(
        args.output_dir / "smoke_run_record.json",
        {
            "completed_steps": int(simulation.currentStep),
            "duration_ns": 2.0,
            "npt_steps": TOTAL_STEPS - NVT_STEPS,
            "nvt_steps": NVT_STEPS,
            "restart_capable": True,
            "restart_validation_completed": resumed,
            "seed": args.seed,
            "resumed": resumed,
            "status": "completed_local_smoke_not_production",
        },
    )


if __name__ == "__main__":
    main()
'''


def _mbar_source() -> str:
    return '''"""Compute a guarded dimensionless MBAR matrix; this script runs no MD."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pymbar import MBAR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--u-kn", type=Path, required=True)
    parser.add_argument("--n-k", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    u_kn = np.load(args.u_kn)
    n_k = np.load(args.n_k)
    if u_kn.ndim != 2 or n_k.ndim != 1 or u_kn.shape[0] != len(n_k):
        raise SystemExit("Expected u_kn with shape (K, N) and n_k with shape (K,)")
    if int(np.sum(n_k)) != u_kn.shape[1] or np.any(n_k < 1):
        raise SystemExit("n_k must be positive and sum to the sampled-state count")
    if not np.isfinite(u_kn).all():
        raise SystemExit("Reduced potentials contain nonfinite values")
    mbar = MBAR(u_kn, n_k, initialize="BAR")
    result = mbar.compute_free_energy_differences()
    payload = {
        "Delta_f_dimensionless": result["Delta_f"].tolist(),
        "dDelta_f_dimensionless": result["dDelta_f"].tolist(),
        "effective_sample_number": mbar.compute_effective_sample_number().tolist(),
        "eligible_for_model_ingestion": False,
        "requires_declared_convergence_gate": True,
        "scope": "free-energy matrix only; not PMF, permeability, RBFE closure, or convergence evidence",
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\\n")


if __name__ == "__main__":
    main()
'''


def _smoke_validator_source() -> str:
    return '''"""Validate finite-energy, NVT/NPT, exact-duration, and restart smoke evidence."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

import yaml


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _column(columns: list[str], token: str) -> str:
    matches = [column for column in columns if token in _key(column)]
    if not matches:
        raise SystemExit(f"Smoke CSV lacks a {token!r} column")
    return matches[0]


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    failures: list[str] = []
    required_outputs = protocol["smoke"]["required_outputs"]
    for name in required_outputs:
        if not (args.smoke_dir / name).is_file():
            failures.append(f"missing_output:{name}")
    record_path = args.smoke_dir / "smoke_run_record.json"
    record = json.loads(record_path.read_text()) if record_path.is_file() else {}
    if record.get("completed_steps") != int(protocol["smoke"]["total_steps"]):
        failures.append("exact_step_count_not_completed")
    if record.get("status") != "completed_local_smoke_not_production":
        failures.append("smoke_completion_status_missing")
    if protocol["smoke"].get("restart_validation_required") and not record.get(
        "restart_validation_completed"
    ):
        failures.append("restart_path_not_exercised")

    csv_path = args.smoke_dir / "smoke.csv"
    rows: list[dict[str, str]] = []
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    minimum_rows = int(protocol["smoke"]["minimum_report_rows"])
    if len(rows) < minimum_rows:
        failures.append("insufficient_state_reports")
    if rows:
        columns = list(rows[0])
        step_column = _column(columns, "step")
        temperature_column = _column(columns, "temperature")
        energy_column = _column(columns, "potentialenergy")
        density_column = _column(columns, "density")
        steps = [int(float(row[step_column])) for row in rows]
        temperatures = [float(row[temperature_column]) for row in rows]
        energies = [float(row[energy_column]) for row in rows]
        densities = [float(row[density_column]) for row in rows]
        if any(right <= left for left, right in zip(steps, steps[1:])):
            failures.append("nonmonotonic_or_duplicate_steps")
        if steps[-1] != int(protocol["smoke"]["total_steps"]):
            failures.append("last_report_not_exact_endpoint")
        if not all(math.isfinite(value) for value in temperatures + energies + densities):
            failures.append("nonfinite_thermodynamic_value")
        tail_start = max(0, int(len(temperatures) * 0.8))
        tail_temperature = temperatures[tail_start:]
        target_temperature = float(protocol["temperature_kelvin"])
        relative_temperature_error = abs(sum(tail_temperature) / len(tail_temperature) - target_temperature) / target_temperature
        if relative_temperature_error > float(protocol["smoke"]["temperature_mean_relative_error_max"]):
            failures.append("temperature_mean_outside_smoke_tolerance")
        if any(value <= 0 for value in densities):
            failures.append("nonpositive_density")
    payload = {
        "eligible_for_production_preparation_review": not failures,
        "failures": sorted(set(failures)),
        "restart_validation_required": bool(protocol["smoke"].get("restart_validation_required")),
        "state_report_rows": len(rows),
        "status": "passed" if not failures else "blocked",
        "truth_boundary": "smoke stability is necessary but not production convergence",
    }
    _atomic_json(args.output, payload)
    if failures and not args.report_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
'''


def _plumed_template(config: HPCBundleConfig) -> str:
    return f"""
# Resolve the two atom placeholders during system preparation.  Production is disabled.
ligand: COM ATOMS=<LIGAND_HEAVY_ATOMS>
membrane: COM ATOMS=<LOCAL_POPC_PHOSPHORUS_ATOMS>
membrane_global: COM ATOMS=<ALL_POPC_PHOSPHORUS_ATOMS>
z_local: DISTANCE ATOMS=ligand,membrane COMPONENTS
z_global: DISTANCE ATOMS=ligand,membrane_global COMPONENTS
orientation: DISTANCE ATOMS=<LIGAND_ORIENTATION_ATOMS> COMPONENTS
restraint: RESTRAINT ARG=z_local.z AT=<WINDOW_CENTER_NM> KAPPA={config.pmf_restraint_kj_mol_nm2:.1f}
PRINT STRIDE=500 ARG=z_local.z,z_global.z,orientation.* FILE=COLVAR
"""


def _slurm_smoke_template() -> str:
    return """
#!/bin/bash
#SBATCH --job-name=menin-physics-smoke
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
set -euo pipefail
echo "This template validates inputs only. Production submission scripts are intentionally absent."
workflow_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$workflow_dir/run_openmm_smoke.py" "$@"
"""


def _convergence_gate_source() -> str:
    return '''"""Apply every declared convergence gate before exposing observables to models."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import yaml


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
    os.replace(temporary, path)


def _direction(name: str) -> str:
    minimum_tokens = ("_min", "minimum_", "_r2_min", "_fraction_min", "_confidence_min", "_overlap_min")
    return "minimum" if any(token in name for token in minimum_tokens) else "maximum"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text())
    metrics = json.loads(args.metrics.read_text())
    gates = protocol.get("acceptance_gates", {})
    results = []
    for name in sorted(gates):
        threshold = float(gates[name])
        raw_value = metrics.get(name)
        direction = _direction(name)
        if raw_value is None:
            passed = False
            value = None
            reason = "missing_metric"
        else:
            value = float(raw_value)
            finite = math.isfinite(value)
            passed = finite and (value >= threshold if direction == "minimum" else value <= threshold)
            reason = "passed" if passed else ("nonfinite_metric" if not finite else "threshold_failed")
        results.append(
            {
                "direction": direction,
                "gate": name,
                "reason": reason,
                "threshold": threshold,
                "value": value,
                "passed": passed,
            }
        )
    all_passed = bool(results) and all(item["passed"] for item in results)
    payload = {
        "all_declared_gates_passed": all_passed,
        "eligible_for_model_ingestion": all_passed,
        "gate_count": len(results),
        "results": results,
        "status": "passed" if all_passed else "blocked_failed_or_missing_convergence_evidence",
        "workflow": protocol.get("workflow"),
    }
    _atomic_json(args.output, payload)
    if not all_passed and not args.report_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
'''


def _rbfe_cycle_validator_source() -> str:
    return '''"""Reject charge-changing, nonlocal, pose-mismatched, or open RBFE cycles."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


EDGE_FIELDS = {
    "edge_id",
    "ligand_a",
    "ligand_b",
    "formal_charge_a",
    "formal_charge_b",
    "ligand_state_a",
    "ligand_state_b",
    "protonation_family_a",
    "protonation_family_b",
    "receptor_state",
    "pose_family",
    "local_perturbation_approved",
}
CYCLE_FIELDS = {"cycle_id", "ordered_ligands", "receptor_state", "pose_family"}
TRUE_VALUES = {"1", "true", "yes"}


def _read(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise SystemExit(f"{path}: missing columns {missing}")
        return list(reader)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--cycles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    edges = _read(args.edges, EDGE_FIELDS)
    cycles = _read(args.cycles, CYCLE_FIELDS)
    failures: list[dict[str, str]] = []
    usable_edges: dict[frozenset[str], list[dict[str, str]]] = {}
    for edge in edges:
        edge_id = edge["edge_id"]
        if not edge["ligand_a"] or not edge["ligand_b"] or edge["ligand_a"] == edge["ligand_b"]:
            failures.append({"code": "invalid_edge_endpoints", "subject": edge_id})
        if not edge["formal_charge_a"] or edge["formal_charge_a"] != edge["formal_charge_b"]:
            failures.append({"code": "charge_change", "subject": edge_id})
        if not edge["ligand_state_a"] or not edge["ligand_state_b"]:
            failures.append({"code": "missing_ligand_state_identity", "subject": edge_id})
        if (
            not edge["protonation_family_a"]
            or edge["protonation_family_a"] != edge["protonation_family_b"]
        ):
            failures.append({"code": "protonation_family_change", "subject": edge_id})
        if edge["local_perturbation_approved"].strip().casefold() not in TRUE_VALUES:
            failures.append({"code": "locality_not_approved", "subject": edge_id})
        if not edge["receptor_state"] or not edge["pose_family"]:
            failures.append({"code": "missing_receptor_or_pose_family", "subject": edge_id})
        usable_edges.setdefault(frozenset((edge["ligand_a"], edge["ligand_b"])), []).append(edge)

    if not cycles:
        failures.append({"code": "no_predeclared_closed_cycle", "subject": "rbfe_cycles.csv"})
    for cycle in cycles:
        cycle_id = cycle["cycle_id"]
        ligands = [item.strip() for item in cycle["ordered_ligands"].split(">") if item.strip()]
        if len(ligands) < 3 or len(set(ligands)) != len(ligands):
            failures.append({"code": "cycle_requires_three_distinct_ligands", "subject": cycle_id})
            continue
        cycle_assignments: dict[str, tuple[str, str, str]] = {}
        for left, right in zip(ligands, ligands[1:] + ligands[:1], strict=True):
            candidates = usable_edges.get(frozenset((left, right)), [])
            compatible = [
                edge
                for edge in candidates
                if edge["receptor_state"] == cycle["receptor_state"]
                and edge["pose_family"] == cycle["pose_family"]
            ]
            if not compatible:
                failures.append({"code": "missing_compatible_cycle_edge", "subject": f"{cycle_id}:{left}>{right}"})
                continue
            selected = sorted(compatible, key=lambda edge: edge["edge_id"])[0]
            for suffix in ("a", "b"):
                ligand = selected[f"ligand_{suffix}"]
                assignment = (
                    selected[f"formal_charge_{suffix}"],
                    selected[f"protonation_family_{suffix}"],
                    selected[f"ligand_state_{suffix}"],
                )
                previous = cycle_assignments.setdefault(ligand, assignment)
                if previous != assignment:
                    failures.append(
                        {"code": "inconsistent_ligand_state_across_cycle", "subject": f"{cycle_id}:{ligand}"}
                    )
    payload = {
        "closed_cycle_count": len(cycles),
        "edge_count": len(edges),
        "eligible_for_rbfe": not failures,
        "failures": sorted(failures, key=lambda item: (item["code"], item["subject"])),
        "status": "passed" if not failures else "blocked",
    }
    _atomic_json(args.output, payload)
    if failures and not args.report_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
'''


def _preflight_source() -> str:
    return '''"""Resolve bundle readiness without launching simulations."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REQUIRED_MATRIX_COLUMNS = {
    "run_id",
    "run_requirement",
    "ligand_state_id",
    "formal_charge",
    "ligand_parameter_file",
    "parameter_review_status",
    "force_field_role",
    "random_seed",
    "prepared_system_xml",
    "coordinates",
    "readiness_status",
}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
    os.replace(temporary, path)


def _portable_existing_file(root: Path, value: str) -> bool:
    if not value:
        return False
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        return False
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return False
    return resolved.is_file()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("preflight_report.json"))
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    root = args.bundle_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    blockers: list[dict[str, str]] = []
    conditional_blockers: list[dict[str, str]] = []
    _, receptor_rows = _read_csv(root / "herg_ensemble" / "receptors.csv")
    receptors = {row.get("pdb_id", ""): row for row in receptor_rows}
    if manifest.get("production_launch_enabled") is not False:
        blockers.append({"code": "production_guard_disabled", "subject": "manifest.json"})
    for relative in manifest.get("required_bundle_files", []):
        if not (root / relative).is_file():
            blockers.append({"code": "missing_bundle_file", "subject": relative})
    software_record = root / "software_environment.json"
    if not software_record.is_file():
        blockers.append({"code": "software_environment_not_recorded", "subject": "software_environment.json"})
    else:
        software = json.loads(software_record.read_text())
        for field in (
            "openmm_version",
            "pymbar_version",
            "plumed_version",
            "parameterization_tool_versions",
            "review_status",
        ):
            if not software.get(field):
                blockers.append({"code": "software_environment_field_missing", "subject": field})
        if software.get("review_status") != "approved_for_local_smoke":
            blockers.append({"code": "software_environment_not_approved", "subject": "review_status"})
        if software.get("plumed_openmm_integration_verified") is not True:
            blockers.append(
                {"code": "plumed_openmm_integration_not_verified", "subject": "software_environment.json"}
            )
    _, pilot_checks = _read_csv(root / "pilot_design_audit.csv")
    for check in pilot_checks:
        if check.get("passed", "").strip().casefold() not in {"1", "true", "yes"}:
            blockers.append({"code": "pilot_design_check_failed", "subject": check.get("check", "unknown")})
    required_run_count = 0
    seen_required_seeds: dict[int, str] = {}
    sensitivity_not_ready = 0
    for relative in manifest.get("run_matrices", []):
        path = root / relative
        if not path.is_file():
            blockers.append({"code": "missing_run_matrix", "subject": relative})
            continue
        columns, rows = _read_csv(path)
        missing_columns = sorted(REQUIRED_MATRIX_COLUMNS - set(columns))
        for column in missing_columns:
            blockers.append({"code": "run_matrix_missing_column", "subject": f"{relative}:{column}"})
        for row in rows:
            if (
                row.get("force_field_role") == "predeclared_sensitivity_subset"
                and row.get("readiness_status") != "ready"
            ):
                sensitivity_not_ready += 1
            if row.get("run_requirement") != "required":
                continue
            required_run_count += 1
            run_id = row.get("run_id", "unknown")
            if not row.get("ligand_state_id") or row.get("ligand_state_id", "").startswith("UNRESOLVED::"):
                blockers.append({"code": "unresolved_ligand_state", "subject": run_id})
            if not row.get("formal_charge"):
                blockers.append({"code": "missing_state_specific_formal_charge", "subject": run_id})
            if row.get("parameter_review_status") != "approved":
                blockers.append({"code": "ligand_parameter_review_not_approved", "subject": run_id})
            try:
                seed = int(row.get("random_seed", ""))
                if seed < 1 or seed >= 2_147_483_647:
                    raise ValueError
                if seed in seen_required_seeds:
                    blockers.append(
                        {
                            "code": "duplicate_required_replica_seed",
                            "subject": f"{seen_required_seeds[seed]}::{run_id}",
                        }
                    )
                else:
                    seen_required_seeds[seed] = run_id
            except ValueError:
                blockers.append({"code": "invalid_random_seed", "subject": run_id})
            if row.get("readiness_status") != "ready":
                blockers.append({"code": "run_not_explicitly_marked_ready", "subject": run_id})
            for field in ("ligand_parameter_file", "prepared_system_xml", "coordinates"):
                if not _portable_existing_file(root, row.get(field, "")):
                    blockers.append({"code": f"missing_or_nonportable_{field}", "subject": run_id})
            if "receptor_pdb_id" in row and not row.get("receptor_pdb_id"):
                blockers.append({"code": "unresolved_receptor_hypothesis", "subject": run_id})
            elif "receptor_pdb_id" in row:
                receptor = receptors.get(row.get("receptor_pdb_id", ""), {})
                if receptor.get("readiness_status") != "ready":
                    blockers.append({"code": "receptor_preparation_not_ready", "subject": run_id})
                for field in ("source_structure_path", "prepared_receptor_path"):
                    if not _portable_existing_file(root, receptor.get(field, "")):
                        blockers.append({"code": f"missing_or_nonportable_{field}", "subject": run_id})
                for field in ("ion_occupancy_assignment", "missing_residue_review", "protonation_review"):
                    if receptor.get(field, "") in {"", "pending"}:
                        blockers.append({"code": f"unresolved_{field}", "subject": run_id})
            if "pose_coordinates" in row and not _portable_existing_file(root, row.get("pose_coordinates", "")):
                blockers.append({"code": "missing_or_nonportable_pose_coordinates", "subject": run_id})
    if sensitivity_not_ready:
        conditional_blockers.append(
            {
                "code": "force_field_sensitivity_runs_not_ready",
                "subject": str(sensitivity_not_ready),
            }
        )
    rbfe_dir = root / "relative_free_energy"
    rbfe_result = subprocess.run(
        [
            sys.executable,
            str(rbfe_dir / "validate_rbfe_cycles.py"),
            "--edges",
            str(rbfe_dir / "rbfe_edges.csv"),
            "--cycles",
            str(rbfe_dir / "rbfe_cycles.csv"),
            "--output",
            str(rbfe_dir / "rbfe_cycle_preflight.json"),
            "--report-only",
        ],
        check=False,
    )
    if rbfe_result.returncode != 0:
        blockers.append({"code": "rbfe_validator_execution_failed", "subject": "relative_free_energy"})
        rbfe = {"eligible_for_rbfe": False}
    else:
        rbfe = json.loads((rbfe_dir / "rbfe_cycle_preflight.json").read_text())
    if not rbfe.get("eligible_for_rbfe", False):
        conditional_blockers.append(
            {"code": "rbfe_closed_cycle_not_ready", "subject": "relative_free_energy"}
        )
    else:
        _, declared_edges = _read_csv(rbfe_dir / "rbfe_edges.csv")
        _, rbfe_runs = _read_csv(rbfe_dir / "run_matrix.csv")
        required_replicates = int(manifest["production_replicates"])
        for edge in declared_edges:
            edge_id = edge["edge_id"]
            edge_runs = [row for row in rbfe_runs if row.get("edge_id") == edge_id]
            for leg in ("solvent", "receptor_complex"):
                leg_runs = [row for row in edge_runs if row.get("leg") == leg]
                replicas = {row.get("replicate") for row in leg_runs}
                if len(replicas) < required_replicates:
                    conditional_blockers.append(
                        {"code": "rbfe_missing_leg_replicates", "subject": f"{edge_id}:{leg}"}
                    )
                for row in leg_runs:
                    run_id = row.get("run_id", f"{edge_id}:{leg}")
                    if row.get("readiness_status") != "ready":
                        conditional_blockers.append(
                            {"code": "rbfe_run_not_ready", "subject": run_id}
                        )
                    for field in (
                        "prepared_system_xml",
                        "coordinates",
                        "ligand_parameter_file_a",
                        "ligand_parameter_file_b",
                    ):
                        if not _portable_existing_file(root, row.get(field, "")):
                            conditional_blockers.append(
                                {"code": f"rbfe_missing_or_nonportable_{field}", "subject": run_id}
                            )
                    for field in (
                        "ligand_state_a",
                        "ligand_state_b",
                        "formal_charge_a",
                        "formal_charge_b",
                        "protonation_family_a",
                        "protonation_family_b",
                    ):
                        if not row.get(field):
                            conditional_blockers.append(
                                {"code": f"rbfe_missing_{field}", "subject": run_id}
                            )
                    if leg == "receptor_complex":
                        if row.get("receptor_state") != edge["receptor_state"]:
                            conditional_blockers.append(
                                {"code": "rbfe_receptor_state_mismatch", "subject": run_id}
                            )
                        if row.get("pose_family") != edge["pose_family"]:
                            conditional_blockers.append(
                                {"code": "rbfe_pose_family_mismatch", "subject": run_id}
                            )
                        if not _portable_existing_file(root, row.get("pose_coordinates", "")):
                            conditional_blockers.append(
                                {"code": "rbfe_missing_or_nonportable_pose_coordinates", "subject": run_id}
                            )
    blockers = sorted(blockers, key=lambda item: (item["code"], item["subject"]))
    conditional_blockers = sorted(
        conditional_blockers,
        key=lambda item: (item["code"], item["subject"]),
    )
    if blockers:
        status = "blocked_input_preparation"
    elif conditional_blockers:
        status = "ready_for_required_local_smoke_with_conditional_workflows_disabled"
    else:
        status = "ready_for_local_smoke"
    payload = {
        "blocker_count": len(blockers),
        "blockers": blockers,
        "bundle_version": manifest.get("bundle_version"),
        "production_launch_enabled": False,
        "conditional_blocker_count": len(conditional_blockers),
        "conditional_blockers": conditional_blockers,
        "required_run_count": required_run_count,
        "status": status,
    }
    output = args.output if args.output.is_absolute() else root / args.output
    _atomic_json(output, payload)
    if blockers and not args.report_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
'''


def _safe_token(value: object) -> str:
    token = "".join(
        character if character.isalnum() or character in "-_" else "-" for character in str(value)
    )
    return token.strip("-") or "unknown"


def _stable_seed(value: str) -> int:
    modulus = 2_147_483_647
    accumulator = 17
    for index, character in enumerate(value, start=1):
        accumulator = (accumulator * 257 + index * ord(character)) % modulus
    return accumulator or 1


def _row_text(row: pd.Series, candidates: Sequence[str]) -> str:
    for column in candidates:
        if column not in row or pd.isna(row[column]):
            continue
        value = str(row[column]).strip()
        if value:
            return value
    return ""


def _ligand_state_matrix(pilots_by_workflow: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for workflow in sorted(pilots_by_workflow):
        selected = pilots_by_workflow[workflow].drop_duplicates("compound_id", keep="first")
        for _, row in selected.sort_values("compound_id", kind="stable").iterrows():
            compound_id = str(row["compound_id"])
            state_id = _row_text(row, ("chemical_state_id", "state_id", "dominant_state_id"))
            state_resolved = bool(state_id)
            records.append(
                {
                    "workflow": workflow,
                    "compound_id": compound_id,
                    "ligand_state_id": state_id or f"UNRESOLVED::{compound_id}",
                    "formal_charge": _row_text(row, ("net_charge", "formal_charge", "dominant_charge")),
                    "protonation_family": _row_text(row, ("protonation_family",)),
                    "state_pH": _row_text(row, ("ph", "pH")),
                    "state_population": _row_text(row, ("population_estimate", "state_population")),
                    "state_population_uncertainty": _row_text(
                        row,
                        ("population_uncertainty", "state_population_uncertainty"),
                    ),
                    "coordinates": "",
                    "primary_parameter_file": "",
                    "sensitivity_parameter_file": "",
                    "state_resolution_status": (
                        "resolved_identity_pending_coordinates_and_parameters"
                        if state_resolved
                        else "blocked_missing_retained_state_identity"
                    ),
                }
            )
    columns = [
        "workflow",
        "compound_id",
        "ligand_state_id",
        "formal_charge",
        "protonation_family",
        "state_pH",
        "state_population",
        "state_population_uncertainty",
        "coordinates",
        "primary_parameter_file",
        "sensitivity_parameter_file",
        "state_resolution_status",
    ]
    return pd.DataFrame(records, columns=columns).drop_duplicates(
        ["workflow", "compound_id", "ligand_state_id"],
        keep="first",
    )


def _common_run_record(
    *,
    workflow: str,
    compound_id: str,
    ligand_state_id: str,
    formal_charge: str,
    replicate: int,
    requirement: str,
    suffix: str,
) -> dict[str, object]:
    run_id = "__".join(
        (
            workflow,
            _safe_token(compound_id),
            _safe_token(ligand_state_id),
            suffix,
            f"rep{replicate:02d}",
        )
    )
    return {
        "run_id": run_id,
        "workflow": workflow,
        "compound_id": compound_id,
        "ligand_state_id": ligand_state_id,
        "formal_charge": formal_charge,
        "ligand_parameter_file": "",
        "parameter_review_status": "pending",
        "force_field_role": "primary_reproduction",
        "replicate": replicate,
        "random_seed": _stable_seed(run_id),
        "run_requirement": requirement,
        "prepared_system_xml": "",
        "coordinates": "",
        "checkpoint_in": "",
        "output_dir": f"{workflow}/runs/{run_id}",
        "readiness_status": "blocked_missing_prepared_inputs",
    }


def _run_matrices(
    config: HPCBundleConfig,
    ligand_states: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    matrices: dict[str, pd.DataFrame] = {}
    environment_records: list[dict[str, object]] = []
    environment_states = ligand_states[ligand_states["workflow"] == "environment_md"]
    for _, state in environment_states.iterrows():
        for environment in (*REQUIRED_ENVIRONMENTS, "POPC_water_interface"):
            requirement = "required" if environment in REQUIRED_ENVIRONMENTS else "conditional"
            for replicate in range(1, config.production_replicates + 1):
                record = _common_run_record(
                    workflow="environment_md",
                    compound_id=str(state["compound_id"]),
                    ligand_state_id=str(state["ligand_state_id"]),
                    formal_charge=str(state["formal_charge"]),
                    replicate=replicate,
                    requirement=requirement,
                    suffix=_safe_token(environment),
                )
                record.update(
                    {
                        "environment": environment,
                        "initial_ns": config.environment_initial_ns,
                        "maximum_ns": config.environment_max_ns,
                    }
                )
                environment_records.append(record)
    matrices["environment_md"] = pd.DataFrame(environment_records)

    membrane_records: list[dict[str, object]] = []
    membrane_states = ligand_states[ligand_states["workflow"] == "membrane_pmf"]
    for _, state in membrane_states.iterrows():
        for patch_size in config.membrane_patch_sizes_popc:
            requirement = "required" if patch_size in (64, 128) else "conditional"
            for direction in ("forward", "reverse"):
                for replicate in range(1, config.production_replicates + 1):
                    suffix = f"popc{patch_size}__{direction}"
                    record = _common_run_record(
                        workflow="membrane_pmf",
                        compound_id=str(state["compound_id"]),
                        ligand_state_id=str(state["ligand_state_id"]),
                        formal_charge=str(state["formal_charge"]),
                        replicate=replicate,
                        requirement=requirement,
                        suffix=suffix,
                    )
                    record.update(
                        {
                            "patch_popc": patch_size,
                            "sampling_direction": direction,
                            "window_manifest": "membrane_pmf/pmf_windows.csv",
                        }
                    )
                    membrane_records.append(record)
    matrices["membrane_pmf"] = pd.DataFrame(membrane_records)

    herg_records: list[dict[str, object]] = []
    herg_states = ligand_states[ligand_states["workflow"] == "herg_ensemble"]
    for _, state in herg_states.iterrows():
        for hypothesis in range(1, HERG_RECEPTOR_HYPOTHESES_PER_COMPOUND + 1):
            for replicate in range(1, config.production_replicates + 1):
                suffix = f"receptor-hypothesis-{hypothesis}"
                record = _common_run_record(
                    workflow="herg_ensemble",
                    compound_id=str(state["compound_id"]),
                    ligand_state_id=str(state["ligand_state_id"]),
                    formal_charge=str(state["formal_charge"]),
                    replicate=replicate,
                    requirement="required",
                    suffix=suffix,
                )
                record.update(
                    {
                        "receptor_hypothesis_rank": hypothesis,
                        "receptor_pdb_id": "",
                        "receptor_state": "",
                        "pose_coordinates": "",
                        "initial_ns": config.herg_initial_ns,
                        "maximum_ns": config.herg_max_ns,
                        "readiness_status": "blocked_pending_receptor_and_pose_selection",
                    }
                )
                herg_records.append(record)
    matrices["herg_ensemble"] = pd.DataFrame(herg_records)

    rbfe_columns = [
        "run_id",
        "workflow",
        "compound_id",
        "ligand_state_id",
        "formal_charge",
        "ligand_parameter_file",
        "parameter_review_status",
        "force_field_role",
        "replicate",
        "random_seed",
        "run_requirement",
        "prepared_system_xml",
        "coordinates",
        "checkpoint_in",
        "output_dir",
        "readiness_status",
        "edge_id",
        "leg",
        "receptor_state",
        "pose_family",
        "pose_coordinates",
    ]
    matrices["relative_free_energy"] = pd.DataFrame(columns=rbfe_columns)
    return matrices


def _pmf_window_matrix(config: HPCBundleConfig) -> pd.DataFrame:
    centers = np.arange(
        config.pmf_z_min_angstrom,
        config.pmf_z_max_angstrom + config.pmf_window_spacing_angstrom / 2.0,
        config.pmf_window_spacing_angstrom,
    )
    return pd.DataFrame(
        {
            "window_index": np.arange(len(centers), dtype=int),
            "center_angstrom": np.round(centers, 8),
            "center_nm": np.round(centers / 10.0, 8),
            "equilibration_ns": config.pmf_equilibration_ns_per_window,
            "production_ns": config.pmf_production_ns_per_window,
            "restraint_kj_mol_nm2": config.pmf_restraint_kj_mol_nm2,
        }
    )


def _smoke_command_matrix(workflow: str, run_matrix: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, row in run_matrix.iterrows():
        system_xml = str(row.get("prepared_system_xml", "")) or "REPLACE_WITH_PREPARED_SYSTEM_XML"
        coordinates = str(row.get("coordinates", "")) or "REPLACE_WITH_PREPARED_COORDINATES"
        output_dir = str(row.get("output_dir", ""))
        seed = int(row["random_seed"])
        base = (
            f"python {workflow}/run_openmm_smoke.py --protocol {workflow}/protocol.yaml "
            f"--system-xml {system_xml} --coordinates {coordinates} --output-dir {output_dir} "
            f"--seed {seed}"
        )
        records.append(
            {
                "run_id": row["run_id"],
                "fresh_smoke_command": f"{base} --execute-smoke",
                "restart_smoke_command": (f"{base} --checkpoint-in {output_dir}/restart.chk --execute-smoke"),
                "restart_test_part1_command": f"{base} --pause-at-step 500000 --execute-smoke",
                "restart_test_part2_command": (
                    f"{base} --checkpoint-in {output_dir}/restart.chk --execute-smoke"
                ),
                "executable_now": row.get("readiness_status") == "ready",
                "scope": "exact_local_2ns_smoke_only",
            }
        )
    return pd.DataFrame(
        records,
        columns=[
            "run_id",
            "fresh_smoke_command",
            "restart_smoke_command",
            "restart_test_part1_command",
            "restart_test_part2_command",
            "executable_now",
            "scope",
        ],
    )


def _validate_receptor_raw_coordinates(
    config: HPCBundleConfig,
    *,
    project_root: Path,
) -> HPCBundleConfig:
    """Attach validated deposited-coordinate facts without implying preparation."""

    resolved_project_root = project_root.resolve()
    validated: list[dict[str, str]] = []
    for receptor in config.receptor_ensemble:
        record = dict(receptor)
        pdb_id = str(record["pdb_id"]).upper()
        raw_value = str(record.get("canonical_raw_coordinate_path", ""))
        raw_path = Path(raw_value)
        if not raw_value or raw_path.is_absolute() or ".." in raw_path.parts:
            raise ValueError(f"Receptor {pdb_id} lacks a portable canonical raw-coordinate path")
        resolved = (resolved_project_root / raw_path).resolve()
        try:
            resolved.relative_to(resolved_project_root)
        except ValueError as exc:
            raise ValueError(f"Receptor coordinate escapes project root: {raw_value}") from exc
        qc = validate_mmcif_coordinate(resolved, expected_entry_id=pdb_id)
        record.update(
            {
                "raw_atom_count": str(qc.atom_count),
                "raw_auth_chain_count": str(qc.auth_chain_count),
                "raw_coordinate_format": "PDBx/mmCIF",
                "raw_coordinate_repository": "RCSB PDB",
                "raw_coordinate_validation_status": "validated_entry_and_atom_site",
                "raw_entry_id": qc.entry_id,
                "raw_model_count": str(qc.model_count),
            }
        )
        validated.append(record)
    return replace(config, receptor_ensemble=tuple(validated))


def _receptor_preparation_matrix(config: HPCBundleConfig) -> pd.DataFrame:
    records = []
    for receptor in config.receptor_ensemble:
        raw_status = receptor.get(
            "raw_coordinate_validation_status",
            "not_validated_in_direct_bundle_call",
        )
        records.append(
            {
                **receptor,
                "production_role": "canonical_raw_coordinate",
                "raw_coordinate_validation_status": raw_status,
                "raw_biological_assembly_review": "pending",
                "source_structure_path": "",
                "prepared_receptor_path": "",
                "ion_occupancy_assignment": "",
                "missing_residue_review": "pending",
                "protonation_review": "pending",
                "readiness_status": (
                    "blocked_raw_coordinate_only_requires_preparation"
                    if raw_status == "validated_entry_and_atom_site"
                    else "blocked_missing_local_structure_and_preparation"
                ),
            }
        )
    return pd.DataFrame(records).sort_values("pdb_id", kind="stable").reset_index(drop=True)


def _rbfe_candidate_edges(pilots: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    if "matched_pair_id" in pilots:
        groups = pilots.dropna(subset=["matched_pair_id"]).groupby("matched_pair_id", sort=True)
        for pair_id, group in groups:
            compound_ids = sorted(group["compound_id"].astype(str).unique())
            if len(compound_ids) < 2:
                continue
            records.append(
                {
                    "candidate_id": f"candidate-{_safe_token(pair_id)}",
                    "ligand_a": compound_ids[0],
                    "ligand_b": compound_ids[1],
                    "matched_pair_id": str(pair_id),
                    "eligibility_status": "blocked_pending_charge_state_locality_pose_and_cycle_review",
                }
            )
    return pd.DataFrame(
        records,
        columns=["candidate_id", "ligand_a", "ligand_b", "matched_pair_id", "eligibility_status"],
    )


def _rbfe_candidate_run_matrix(
    config: HPCBundleConfig,
    candidate_edges: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "run_id",
        "workflow",
        "compound_id",
        "ligand_state_id",
        "formal_charge",
        "ligand_parameter_file",
        "parameter_review_status",
        "force_field_role",
        "replicate",
        "random_seed",
        "run_requirement",
        "prepared_system_xml",
        "coordinates",
        "checkpoint_in",
        "output_dir",
        "readiness_status",
        "edge_id",
        "ligand_a",
        "ligand_b",
        "ligand_state_a",
        "ligand_state_b",
        "formal_charge_a",
        "formal_charge_b",
        "protonation_family_a",
        "protonation_family_b",
        "ligand_parameter_file_a",
        "ligand_parameter_file_b",
        "leg",
        "receptor_state",
        "pose_family",
    ]
    records: list[dict[str, object]] = []
    for _, edge in candidate_edges.sort_values("candidate_id", kind="stable").iterrows():
        ligand_a = str(edge["ligand_a"])
        ligand_b = str(edge["ligand_b"])
        edge_id = str(edge["candidate_id"])
        for leg in ("solvent", "receptor_complex"):
            for replicate in range(1, config.production_replicates + 1):
                record = _common_run_record(
                    workflow="relative_free_energy",
                    compound_id=f"{ligand_a}~{ligand_b}",
                    ligand_state_id=f"UNRESOLVED::{ligand_a}~{ligand_b}",
                    formal_charge="",
                    replicate=replicate,
                    requirement="conditional",
                    suffix=f"{_safe_token(edge_id)}__{leg}",
                )
                record.update(
                    {
                        "edge_id": edge_id,
                        "ligand_a": ligand_a,
                        "ligand_b": ligand_b,
                        "ligand_state_a": "",
                        "ligand_state_b": "",
                        "formal_charge_a": "",
                        "formal_charge_b": "",
                        "protonation_family_a": "",
                        "protonation_family_b": "",
                        "ligand_parameter_file_a": "",
                        "ligand_parameter_file_b": "",
                        "leg": leg,
                        "receptor_state": "" if leg == "receptor_complex" else "not_applicable",
                        "pose_family": "" if leg == "receptor_complex" else "not_applicable",
                        "pose_coordinates": "" if leg == "receptor_complex" else "not_applicable",
                        "readiness_status": "blocked_no_validated_closed_cycle",
                    }
                )
                records.append(record)
    return pd.DataFrame(records, columns=columns)


def _force_field_sensitivity_subset(pilots: pd.DataFrame, *, target: int = 4) -> pd.DataFrame:
    table = pilots.drop_duplicates("compound_id", keep="first").copy()
    if table.empty:
        return pd.DataFrame(
            columns=["compound_id", "sensitivity_rank", "selection_reason", "readiness_status"]
        )
    sort_columns = [column for column in ("pilot_rank", "compound_id") if column in table]
    table = table.sort_values(sort_columns, kind="stable") if sort_columns else table
    selected_indices: list[object] = []
    if "matched_pair_id" in table:
        groups = table.dropna(subset=["matched_pair_id"]).groupby("matched_pair_id", sort=True)
        for _, group in groups:
            if len(group) < 2 or len(selected_indices) + 2 > target:
                continue
            selected_indices.extend(group.index[:2].tolist())
            if len(selected_indices) >= target:
                break
    for index in table.index:
        if len(selected_indices) >= min(target, len(table)):
            break
        if index not in selected_indices:
            selected_indices.append(index)
    output = table.loc[selected_indices].copy()
    output["sensitivity_rank"] = np.arange(1, len(output) + 1, dtype=int)
    matched_pairs = output.get("matched_pair_id", pd.Series(index=output.index, dtype=object)).notna()
    output["selection_reason"] = np.where(
        matched_pairs,
        "matched_pair_priority",
        "pilot_space_fill",
    )
    output["readiness_status"] = "blocked_pending_amber_and_primary_parameter_review"
    return output.reset_index(drop=True)


def _append_force_field_sensitivity_runs(
    matrices: dict[str, pd.DataFrame],
    sensitivity_subset: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    sensitivity_ids = set(sensitivity_subset["compound_id"].astype(str))
    for workflow in ("environment_md", "membrane_pmf", "herg_ensemble"):
        matrix = matrices[workflow]
        selected = matrix.loc[
            matrix["compound_id"].astype(str).isin(sensitivity_ids)
            & (matrix["run_requirement"] == "required")
            & (matrix["force_field_role"] == "primary_reproduction")
        ].copy()
        if selected.empty:
            continue
        selected["run_id"] = selected["run_id"].astype(str) + "__ff-sensitivity"
        selected["output_dir"] = selected["output_dir"].astype(str) + "__ff-sensitivity"
        selected["random_seed"] = selected["run_id"].map(_stable_seed)
        selected["run_requirement"] = "conditional"
        selected["force_field_role"] = "predeclared_sensitivity_subset"
        selected["prepared_system_xml"] = ""
        selected["coordinates"] = ""
        selected["checkpoint_in"] = ""
        selected["ligand_parameter_file"] = ""
        selected["parameter_review_status"] = "pending"
        selected["readiness_status"] = "blocked_missing_sensitivity_force_field_preparation"
        matrices[workflow] = pd.concat([matrix, selected], ignore_index=True)
    return matrices


def _pilot_design_audit(selections: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    environment = selections["environment_md"]
    membrane = selections["membrane_pmf"]
    herg = selections["herg_ensemble"]
    pair_count = 0
    if "matched_pair_id" in membrane:
        pair_count = int(
            (membrane.dropna(subset=["matched_pair_id"]).groupby("matched_pair_id").size() >= 2).sum()
        )
    herg_counts = herg["herg_class"].astype(str).value_counts().to_dict() if "herg_class" in herg else {}
    herg_pair_count = 0
    if "herg_matched_pair_id" in herg:
        for _, group in herg.dropna(subset=["herg_matched_pair_id"]).groupby(
            "herg_matched_pair_id",
            sort=True,
        ):
            classes = set(group["herg_class"].astype(str)) if "herg_class" in group else set()
            if group["compound_id"].nunique() >= 2 and classes == {"blocker", "nonblocker"}:
                herg_pair_count += 1
    records = [
        {
            "check": "environment_compound_count",
            "observed": len(environment),
            "required": 12,
            "passed": len(environment) == 12,
            "failure_action": "resolve pilot availability or selection configuration before production",
        },
        {
            "check": "membrane_compound_count",
            "observed": len(membrane),
            "required": 4,
            "passed": len(membrane) == 4,
            "failure_action": "select exactly four membrane archetypes",
        },
        {
            "check": "membrane_complete_matched_pairs",
            "observed": pair_count,
            "required": 2,
            "passed": pair_count >= 2,
            "failure_action": "define two mechanistically discordant matched pairs",
        },
        {
            "check": "herg_compound_count",
            "observed": len(herg),
            "required": 6,
            "passed": len(herg) == 6,
            "failure_action": "select exactly six hERG compounds",
        },
        {
            "check": "herg_blocker_count",
            "observed": int(herg_counts.get("blocker", 0)),
            "required": 3,
            "passed": int(herg_counts.get("blocker", 0)) >= 3,
            "failure_action": "resolve three decisive same-series blockers",
        },
        {
            "check": "herg_nonblocker_count",
            "observed": int(herg_counts.get("nonblocker", 0)),
            "required": 3,
            "passed": int(herg_counts.get("nonblocker", 0)) >= 3,
            "failure_action": "resolve three decisive same-series nonblockers",
        },
        {
            "check": "herg_complete_cross_class_matched_pairs",
            "observed": herg_pair_count,
            "required": 1,
            "passed": herg_pair_count >= 1,
            "failure_action": "select at least one complete blocker/nonblocker matched pair",
        },
    ]
    return pd.DataFrame(records)


def _protocol_payloads(config: HPCBundleConfig) -> dict[str, dict[str, Any]]:
    common = {
        "bundle_version": HPC_BUNDLE_VERSION,
        "production_launch_enabled": False,
        "temperature_kelvin": config.temperature_kelvin,
        "pressure_bar": config.pressure_bar,
        "ionic_strength_molar": config.ionic_strength_molar,
        "replicates": config.production_replicates,
        "force_fields": {
            "primary_reproduction": {
                "ligand": config.ligand_force_field,
                "ligand_charges": config.ligand_charge_model,
                "protein": config.protein_force_field,
                "lipid": config.lipid_force_field,
                "water": config.water_model,
            },
            "predeclared_sensitivity_subset": {
                "ligand": config.sensitivity_ligand_force_field,
                "ligand_charges": config.sensitivity_ligand_charge_model,
                "protein": config.sensitivity_protein_force_field,
                "lipid": config.sensitivity_lipid_force_field,
                "water": config.sensitivity_water_model,
                "selection": "charge-preserving matched pairs spanning primary-system residuals",
            },
        },
        "smoke": {
            "mode": "local_smoke",
            "duration_ns": config.smoke_duration_ns,
            "time_step_fs": 2.0,
            "total_steps": SMOKE_TOTAL_STEPS,
            "nvt_steps": SMOKE_NVT_STEPS,
            "npt_steps": SMOKE_TOTAL_STEPS - SMOKE_NVT_STEPS,
            "replicates": 1,
            "restart_validation_required": True,
            "restart_test_pause_step": 500_000,
            "minimum_report_rows": 100,
            "temperature_mean_relative_error_max": 0.15,
            "required_outputs": [
                "smoke.csv",
                "smoke.dcd",
                "smoke_complete.chk",
                "smoke_run_record.json",
            ],
            "restart_checkpoint": "restart.chk",
        },
    }
    environment = {
        **common,
        "workflow": "environment_conditioned_md",
        "systems": [
            {"environment": "water", "box_padding_angstrom": 12.0},
            {
                "environment": "chloroform_low_dielectric",
                "box_padding_angstrom": 30.0,
                "interpretation": "membrane-interior proxy; not a permeability calculation",
            },
            {
                "environment": "POPC_water_interface",
                "patch_popc": config.membrane_primary_patch_popc,
                "interpretation": "environment-response ensemble; not a translocation PMF",
                "run_requirement": "conditional_discovery_extension",
            },
        ],
        "production_definition": {
            "initial_ns": config.environment_initial_ns,
            "extension_ns": config.environment_extension_ns,
            "maximum_ns": config.environment_max_ns,
            "extension_rule": "extend only when a declared convergence gate fails and the failure is sampling-limited",
        },
        "acceptance_gates": {
            "minimum_effective_samples": config.acceptance.environment_min_effective_samples,
            "split_relative_delta_max": config.acceptance.environment_split_relative_delta_max,
            "replica_distribution_distance_max": config.acceptance.environment_replica_distribution_distance_max,
        },
    }
    membrane = {
        **common,
        "workflow": "membrane_permeation_pmf",
        "lipid": "POPC",
        "primary_patch_popc": config.membrane_primary_patch_popc,
        "patch_size_sensitivity_popc": list(config.membrane_patch_sizes_popc),
        "patch_policy": {
            "required_popc": [64, 128],
            "conditional_popc": [256],
            "conditional_rule": "run 256 only when 64-versus-128 patch sensitivity remains unresolved",
        },
        "windows": {
            "z_min_angstrom": config.pmf_z_min_angstrom,
            "z_max_angstrom": config.pmf_z_max_angstrom,
            "spacing_angstrom": config.pmf_window_spacing_angstrom,
            "count": int(
                round(
                    (config.pmf_z_max_angstrom - config.pmf_z_min_angstrom)
                    / config.pmf_window_spacing_angstrom
                )
            )
            + 1,
            "equilibration_ns_each": config.pmf_equilibration_ns_per_window,
            "production_ns_each": config.pmf_production_ns_per_window,
            "restraint_kj_mol_nm2": config.pmf_restraint_kj_mol_nm2,
        },
        "collective_variables": [
            "ligand COM relative to local leaflet phosphorus COM",
            "ligand orientation",
            "local membrane thickness/deformation",
            "global membrane-relative z retained only as a hysteresis diagnostic",
        ],
        "analysis": "PyMBAR with block decorrelation, per-replica PMFs, forward/reverse and patch sensitivity",
        "acceptance_gates": {
            "last_half_drift_kcal_mol_max": config.acceptance.pmf_last_half_drift_kcal_mol_max,
            "analysis_tail_ns_min": config.acceptance.pmf_analysis_tail_ns_min,
            "leaflet_asymmetry_kcal_mol_max": config.acceptance.pmf_leaflet_asymmetry_kcal_mol_max,
            "replica_rmse_kcal_mol_max": config.acceptance.pmf_replica_rmse_kcal_mol_max,
            "forward_reverse_hysteresis_kcal_mol_max": config.acceptance.pmf_forward_reverse_hysteresis_kcal_mol_max,
            "minimum_window_effective_samples": config.acceptance.pmf_min_window_effective_samples,
            "adjacent_overlap_min": config.acceptance.pmf_adjacent_overlap_min,
            "local_global_z_r2_min": config.acceptance.pmf_local_global_z_r2_min,
            "patch_barrier_spread_kcal_mol_max": config.acceptance.pmf_patch_barrier_spread_kcal_mol_max,
        },
    }
    herg = {
        **common,
        "workflow": "herg_receptor_ensemble_md",
        "membrane": {"lipid": "POPC", "patch_popc": 256},
        "receptors_file": "receptors.csv",
        "receptor_hypotheses_per_compound": HERG_RECEPTOR_HYPOTHESES_PER_COMPOUND,
        "receptor_selection_rule": (
            "two predeclared receptor-state/pose hypotheses per compound; docking is pose generation only"
        ),
        "production_definition": {
            "initial_ns": config.herg_initial_ns,
            "extension_ns": config.herg_extension_ns,
            "maximum_ns": config.herg_max_ns,
        },
        "required_ligand_states": "all fast-physics states retained at configured hERG pH plus ±1 pKa sensitivity",
        "required_pose_sources": [
            "orthogonal docking poses",
            "cryo-EM interaction templates",
            "matched-pair analog transfer",
        ],
        "acceptance_gates": {
            "backbone_rmsd_plateau_slope_angstrom_per_ns_max": config.acceptance.herg_backbone_rmsd_plateau_slope_angstrom_per_ns_max,
            "pose_retention_replica_fraction_min": config.acceptance.herg_pose_retention_replica_fraction_min,
            "contact_occupancy_min": config.acceptance.herg_contact_occupancy_min,
            "receptor_state_classifier_confidence_min": config.acceptance.herg_state_classifier_confidence_min,
        },
    }
    relative_free_energy = {
        **common,
        "workflow": "relative_binding_free_energy",
        "eligibility": [
            "same formal charge and protonation family",
            "structurally local matched-pair perturbation",
            "closed perturbation cycle declared before execution",
            "same receptor-state hypothesis and comparable pose",
        ],
        "rejection_rules": [
            "charge-changing perturbation",
            "broad scaffold hop",
            "missing cycle edge",
            "unconverged receptor-state or ligand-state ensemble",
        ],
        "analysis": "PyMBAR per leg and replicate with cycle closure; docking scores are never labels",
        "required_replicates": config.production_replicates,
        "edge_declaration_file": "rbfe_edges.csv",
        "cycle_declaration_file": "rbfe_cycles.csv",
        "preflight_rule": "validate_rbfe_cycles.py must pass before any edge run matrix is populated",
        "acceptance_gates": {
            "cycle_closure_kcal_mol_max": config.acceptance.rbfe_cycle_closure_kcal_mol_max,
        },
    }
    return {
        "environment_md": environment,
        "membrane_pmf": membrane,
        "herg_ensemble": herg,
        "relative_free_energy": relative_free_energy,
    }


def create_hpc_bundle(
    output_dir: str | Path,
    *,
    pilots: pd.DataFrame,
    config: HPCBundleConfig | None = None,
    workflow_pilots: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, Path | int]:
    """Write one deterministic, guarded production-preparation bundle.

    The generated run matrices intentionally begin blocked: this function has no
    authority to invent retained chemical states, parameter files, receptor
    preparation, poses, or serialized OpenMM systems.
    """

    config = config or HPCBundleConfig()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    pilots_path = root / "pilots.csv"
    pilots.to_csv(pilots_path, index=False)
    pilot_selection_path = root / "pilot_selection.parquet"
    pilots.to_parquet(pilot_selection_path, index=False)
    protocols = _protocol_payloads(config)
    selections = {workflow: (workflow_pilots or {}).get(workflow, pilots).copy() for workflow in protocols}
    pilot_design_audit = _pilot_design_audit(selections)
    pilot_design_audit_path = root / "pilot_design_audit.csv"
    pilot_design_audit.to_csv(pilot_design_audit_path, index=False)
    sensitivity_subset = _force_field_sensitivity_subset(selections["environment_md"])
    sensitivity_subset_path = root / "force_field_sensitivity_subset.csv"
    sensitivity_subset.to_csv(sensitivity_subset_path, index=False)
    ligand_states = _ligand_state_matrix(selections)
    ligand_states_path = root / "ligand_state_matrix.csv"
    ligand_states.to_csv(ligand_states_path, index=False)
    run_matrices = _run_matrices(config, ligand_states)
    rbfe_candidates = _rbfe_candidate_edges(selections["relative_free_energy"])
    run_matrices["relative_free_energy"] = _rbfe_candidate_run_matrix(config, rbfe_candidates)
    run_matrices = _append_force_field_sensitivity_runs(run_matrices, sensitivity_subset)
    protocol_paths: dict[str, Path] = {}
    run_matrix_paths: dict[str, Path] = {}
    for workflow, payload in protocols.items():
        workflow_dir = root / workflow
        workflow_dir.mkdir(parents=True, exist_ok=True)
        payload["pilot_selection_file"] = f"{workflow}/pilots.csv"
        payload["ligand_state_matrix"] = f"{workflow}/ligand_states.csv"
        payload["run_matrix"] = f"{workflow}/run_matrix.csv"
        payload["force_field_sensitivity_subset_file"] = "force_field_sensitivity_subset.csv"
        payload["convergence_evidence_required"] = True
        protocol_path = _write_text(workflow_dir / "protocol.yaml", _yaml_text(payload))
        protocol_paths[f"{workflow}_protocol"] = protocol_path
        _write_text(workflow_dir / "run_openmm_smoke.py", _smoke_runner_source())
        _write_text(workflow_dir / "validate_smoke.py", _smoke_validator_source())
        _write_text(workflow_dir / "submit_smoke.slurm", _slurm_smoke_template())
        _write_text(workflow_dir / "evaluate_convergence.py", _convergence_gate_source())
        selected = selections[workflow]
        selected.to_csv(workflow_dir / "pilots.csv", index=False)
        ligand_states.loc[ligand_states["workflow"] == workflow].to_csv(
            workflow_dir / "ligand_states.csv",
            index=False,
        )
        run_matrix_path = workflow_dir / "run_matrix.csv"
        run_matrices[workflow].to_csv(run_matrix_path, index=False)
        run_matrix_paths[workflow] = run_matrix_path
        _smoke_command_matrix(workflow, run_matrices[workflow]).to_csv(
            workflow_dir / "smoke_commands.csv",
            index=False,
        )
        convergence_template = {gate: None for gate in sorted(payload.get("acceptance_gates", {}))}
        _write_text(
            workflow_dir / "convergence_metrics.template.json",
            json.dumps(convergence_template, indent=2, sort_keys=True),
        )
    _write_text(root / "membrane_pmf" / "plumed_umbrella.dat", _plumed_template(config))
    _write_text(root / "membrane_pmf" / "analyze_pymbar.py", _mbar_source())
    _pmf_window_matrix(config).to_csv(root / "membrane_pmf" / "pmf_windows.csv", index=False)
    _write_text(root / "relative_free_energy" / "analyze_pymbar.py", _mbar_source())
    _write_text(
        root / "relative_free_energy" / "validate_rbfe_cycles.py",
        _rbfe_cycle_validator_source(),
    )
    rbfe_edge_columns = [
        "edge_id",
        "ligand_a",
        "ligand_b",
        "formal_charge_a",
        "formal_charge_b",
        "ligand_state_a",
        "ligand_state_b",
        "protonation_family_a",
        "protonation_family_b",
        "receptor_state",
        "pose_family",
        "local_perturbation_approved",
    ]
    rbfe_cycle_columns = ["cycle_id", "ordered_ligands", "receptor_state", "pose_family"]
    pd.DataFrame(columns=rbfe_edge_columns).to_csv(
        root / "relative_free_energy" / "rbfe_edges.csv",
        index=False,
    )
    pd.DataFrame(columns=rbfe_cycle_columns).to_csv(
        root / "relative_free_energy" / "rbfe_cycles.csv",
        index=False,
    )
    rbfe_candidates.to_csv(
        root / "relative_free_energy" / "candidate_edges.csv",
        index=False,
    )
    _write_text(
        root / "relative_free_energy" / "rbfe_cycle_preflight.json",
        json.dumps(
            {
                "closed_cycle_count": 0,
                "edge_count": 0,
                "eligible_for_rbfe": False,
                "failures": [
                    {
                        "code": "no_predeclared_closed_cycle",
                        "subject": "rbfe_cycles.csv",
                    }
                ],
                "status": "blocked",
            },
            indent=2,
            sort_keys=True,
        ),
    )
    receptors_path = root / "herg_ensemble" / "receptors.csv"
    receptor_matrix = _receptor_preparation_matrix(config)
    receptor_matrix.to_csv(receptors_path, index=False)
    validated_raw_receptor_count = int(
        receptor_matrix["raw_coordinate_validation_status"].eq("validated_entry_and_atom_site").sum()
    )
    software_template_path = root / "software_environment.template.json"
    _write_text(
        software_template_path,
        json.dumps(
            {
                "execution_platform": "",
                "openmm_version": "",
                "parameterization_tool_versions": {},
                "plumed_openmm_integration_verified": False,
                "plumed_version": "",
                "pymbar_version": "",
                "python_version": "",
                "review_status": "pending",
                "truth_boundary": "copy to software_environment.json only after inspecting the actual runtime",
            },
            indent=2,
            sort_keys=True,
        ),
    )
    preflight_path = _write_text(root / "preflight.py", _preflight_source())

    required_run_count = sum(
        int((matrix["run_requirement"] == "required").sum())
        for matrix in run_matrices.values()
        if "run_requirement" in matrix
    )
    unresolved_state_count = int(
        ligand_states["ligand_state_id"].astype(str).str.startswith("UNRESOLVED::").sum()
    )
    sensitivity_run_count = sum(
        int((matrix["force_field_role"] == "predeclared_sensitivity_subset").sum())
        for matrix in run_matrices.values()
        if "force_field_role" in matrix
    )
    primary_herg_hypothesis_count = int(
        run_matrices["herg_ensemble"]
        .loc[
            run_matrices["herg_ensemble"]["force_field_role"] == "primary_reproduction",
            ["compound_id", "receptor_hypothesis_rank"],
        ]
        .drop_duplicates()
        .shape[0]
    )
    initial_blockers: list[dict[str, Any]] = [
        {
            "code": "missing_prepared_systems_and_coordinates",
            "count": required_run_count,
            "remediation": "populate portable paths in each required run_matrix.csv row",
        },
        {
            "code": "unresolved_ligand_states",
            "count": unresolved_state_count,
            "remediation": "map retained pH-specific states and state-specific parameters",
        },
        {
            "code": "unprepared_receptor_ensemble",
            "count": int(len(receptor_matrix)),
            "remediation": (
                "prepare the six selected deposited coordinates; review biological assemblies, "
                "ion occupancy, missing residues, and protonation"
            ),
        },
        {
            "code": "unselected_herg_receptor_pose_hypotheses",
            "count": primary_herg_hypothesis_count,
            "remediation": "record two receptor-state/pose hypotheses per hERG compound",
        },
        {
            "code": "software_environment_not_recorded",
            "count": 1,
            "remediation": "inspect the actual OpenMM/PLUMED/PyMBAR/parameterization runtime and approve it for smoke",
        },
    ]
    initial_conditional_blockers: list[dict[str, Any]] = [
        {
            "code": "rbfe_closed_cycle_not_declared",
            "count": 1,
            "remediation": "declare and validate a charge-preserving local cycle, or leave RBFE disabled",
        },
        {
            "code": "force_field_sensitivity_systems_not_prepared",
            "count": sensitivity_run_count,
            "remediation": "prepare the predeclared Amber/OPC sensitivity rows after primary-system review",
        },
    ]
    initial_blockers.extend(
        {
            "code": f"pilot_design::{row['check']}",
            "count": max(
                1,
                int(cast(Any, row["required"])) - int(cast(Any, row["observed"])),
            ),
            "remediation": str(row["failure_action"]),
        }
        for _, row in pilot_design_audit.loc[~pilot_design_audit["passed"]].iterrows()
    )
    initial_readiness_path = root / "readiness_initial.json"
    _write_text(
        initial_readiness_path,
        json.dumps(
            {
                "blocker_count": sum(int(item["count"]) for item in initial_blockers),
                "blockers": initial_blockers,
                "conditional_blocker_count": sum(int(item["count"]) for item in initial_conditional_blockers),
                "conditional_blockers": initial_conditional_blockers,
                "production_launch_enabled": False,
                "status": "blocked_input_preparation",
                "truth_boundary": (
                    "protocols and deterministic run intents exist; prepared systems and simulations do not"
                ),
            },
            indent=2,
            sort_keys=True,
        ),
    )

    required_bundle_files = sorted(
        [
            "README.md",
            "force_field_sensitivity_subset.csv",
            "herg_ensemble/receptors.csv",
            "ligand_state_matrix.csv",
            "pilot_design_audit.csv",
            "pilot_selection.parquet",
            "software_environment.template.json",
            "relative_free_energy/rbfe_cycle_preflight.json",
            "relative_free_energy/rbfe_cycles.csv",
            "relative_free_energy/rbfe_edges.csv",
            *[str(path.relative_to(root)) for path in protocol_paths.values()],
            *[str(path.relative_to(root)) for path in run_matrix_paths.values()],
        ]
    )

    manifest = {
        "bundle_version": HPC_BUNDLE_VERSION,
        "project_id": config.project_id,
        "production_launch_enabled": False,
        "production_submission_scripts_present": False,
        "prepared_systems_present": False,
        "validated_raw_receptor_coordinate_count": validated_raw_receptor_count,
        "completed_simulations_present": False,
        "software_environment_record_present": False,
        "readiness_status": "blocked_input_preparation",
        "local_smoke_duration_ns": config.smoke_duration_ns,
        "local_smoke_total_steps": SMOKE_TOTAL_STEPS,
        "production_replicates": config.production_replicates,
        "pilot_count": int(len(pilots)),
        "pilot_design_checks_passed": bool(pilot_design_audit["passed"].all()),
        "force_field_sensitivity_subset_count": int(len(sensitivity_subset)),
        "force_field_sensitivity_run_count": sensitivity_run_count,
        "workflow_pilot_counts": {workflow: int(len(selections[workflow])) for workflow in protocols},
        "workflow_required_run_counts": {
            workflow: int((matrix["run_requirement"] == "required").sum())
            if "run_requirement" in matrix
            else 0
            for workflow, matrix in run_matrices.items()
        },
        "workflow_conditional_run_counts": {
            workflow: int((matrix["run_requirement"] == "conditional").sum())
            if "run_requirement" in matrix
            else 0
            for workflow, matrix in run_matrices.items()
        },
        "protocols": {key: str(path.relative_to(root)) for key, path in protocol_paths.items()},
        "run_matrices": sorted(str(path.relative_to(root)) for path in run_matrix_paths.values()),
        "required_bundle_files": required_bundle_files,
        "force_fields": {
            "primary_reproduction": {
                "ligand": config.ligand_force_field,
                "ligand_charges": config.ligand_charge_model,
                "protein": config.protein_force_field,
                "lipid": config.lipid_force_field,
                "water": config.water_model,
            },
            "predeclared_sensitivity_subset": {
                "ligand": config.sensitivity_ligand_force_field,
                "ligand_charges": config.sensitivity_ligand_charge_model,
                "protein": config.sensitivity_protein_force_field,
                "lipid": config.sensitivity_lipid_force_field,
                "water": config.sensitivity_water_model,
            },
        },
        "acceptance": asdict(config.acceptance),
        "review_requirements": [
            "Run preflight.py and resolve every required-run blocker.",
            "Resolve every retained ligand state, atom-selection placeholder, parameter penalty, and net charge.",
            "Review receptor preparation, potassium occupancy, and two pose hypotheses per hERG compound.",
            "Pass the restart-capable exact 2 ns NVT/NPT local smoke for every system family.",
            "Run evaluate_convergence.py and exclude all observables until every declared gate passes.",
            "Permit RBFE only after validate_rbfe_cycles.py passes a charge-preserving closed cycle.",
            "Record scientific and compute review before separately creating any production launcher.",
        ],
    }
    manifest_path = root / "manifest.json"
    _write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
    readme_path = root / "README.md"
    _write_text(
        readme_path,
        """
# Guarded PK/hERG physics production-preparation bundle

This directory contains deterministic pilot, ligand-state, receptor-preparation, and
state/replicate run matrices; protocol metadata; restart-capable exact 2 ns local smoke
templates; and machine-enforced convergence and RBFE-cycle gates.

It does **not** contain prepared OpenMM systems, copied receptor-coordinate payloads,
approved ligand parameters, selected hERG poses, completed simulations, or a production
launcher. `herg_ensemble/receptors.csv` points to separately canonical deposited RCSB
PDBx/mmCIF records when they have passed entry/coordinate-table validation. Those raw
records are evidence, not simulation-ready receptors. `readiness_initial.json` therefore
reports blocked input preparation. Blank preparation paths and `UNRESOLVED::` state
identifiers are deliberate blockers, not missing-data defaults.

`run_requirement=required` rows gate the primary pilot. Conditional rows encode the
256-POPC escalation, POPC-interface discovery extension, Amber/OPC sensitivity subset,
and RBFE candidates; they remain explicitly disabled until their trigger and inputs are
resolved. Preflight reports these separately from blockers on the primary local smoke.

Preparation sequence:

1. Review `pilots.csv`, `ligand_state_matrix.csv`, and `herg_ensemble/receptors.csv`.
   Resolve each canonical raw-coordinate path from the project root, then review the
   deposited versus biological assembly before creating a bundle-local source structure.
   Resolve every failure in `pilot_design_audit.csv`; the four-row force-field
   sensitivity set is declared in `force_field_sensitivity_subset.csv`.
   Inspect the actual runtime, copy `software_environment.template.json` to
   `software_environment.json`, fill every field, and approve only the local smoke environment.
2. Populate each workflow's `run_matrix.csv` with bundle-relative prepared-system,
   coordinate, state, receptor, and pose paths. Do not use absolute or parent-relative paths.
3. Resolve `membrane_pmf/plumed_umbrella.dat` atom placeholders and retain both local
   and global membrane coordinates for the declared hysteresis checks.
4. Run `python preflight.py --bundle-dir . --output preflight_report.json`. A nonzero
   result is a hard block; `--report-only` records blockers without claiming readiness.
5. For a prepared row, execute the two `restart_test_part*` commands from
   `smoke_commands.csv`. The first pauses at 500,000 steps; the second loads that
   checkpoint and stops at exactly 1,000,000 2-fs steps. Run `validate_smoke.py`;
   an uninterrupted run alone cannot satisfy the required restart-path evidence.
6. Populate `convergence_metrics.template.json` from replica/block-bootstrap analyses,
   then run `evaluate_convergence.py`. Observables are model-ineligible until all gates pass.
7. Leave RBFE disabled unless `validate_rbfe_cycles.py` passes a predeclared, local,
   charge-preserving cycle in one receptor-state and pose family.

The `submit_smoke.slurm` files validate or execute only the same guarded smoke runner.
No generated script launches production work.
""",
    )

    for source in root.rglob("*.py"):
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    return {
        "bundle_dir": root,
        "manifest_path": manifest_path,
        "readme_path": readme_path,
        "pilots_path": pilots_path,
        "selection_path": pilot_selection_path,
        "ligand_states_path": ligand_states_path,
        "pilot_design_audit_path": pilot_design_audit_path,
        "sensitivity_subset_path": sensitivity_subset_path,
        "software_template_path": software_template_path,
        "initial_readiness_path": initial_readiness_path,
        "preflight_path": preflight_path,
        "receptors_path": receptors_path,
        **protocol_paths,
        "pilot_count": int(len(pilots)),
        "receptor_count": int(len(config.receptor_ensemble)),
    }


def _config_from_mapping(config: Mapping[str, Any] | HPCBundleConfig | None) -> HPCBundleConfig:
    if isinstance(config, HPCBundleConfig):
        return config
    raw: Mapping[str, Any] = config or {}
    if "hpc" in raw and isinstance(raw["hpc"], Mapping):
        raw = raw["hpc"]
    allowed = {item.name for item in fields(HPCBundleConfig)}
    kwargs = {key: raw[key] for key in raw if key in allowed and key != "acceptance"}
    for tuple_key in ("membrane_patch_sizes_popc",):
        if tuple_key in kwargs:
            kwargs[tuple_key] = tuple(int(item) for item in kwargs[tuple_key])
    if "receptor_ensemble" in kwargs:
        kwargs["receptor_ensemble"] = tuple(dict(item) for item in kwargs["receptor_ensemble"])
    acceptance_raw = raw.get("acceptance", {})
    if isinstance(acceptance_raw, Mapping):
        acceptance_allowed = {item.name for item in fields(SimulationAcceptanceGates)}
        kwargs["acceptance"] = SimulationAcceptanceGates(
            **{key: acceptance_raw[key] for key in acceptance_raw if key in acceptance_allowed}
        )
    return HPCBundleConfig(**kwargs)


def _selection_config_from_mapping(config: Mapping[str, Any] | None) -> PilotSelectionConfig:
    raw: Mapping[str, Any] = config or {}
    if "pilot_selection" in raw and isinstance(raw["pilot_selection"], Mapping):
        raw = raw["pilot_selection"]
    allowed = {item.name for item in fields(PilotSelectionConfig)}
    kwargs = {key: raw[key] for key in raw if key in allowed}
    for tuple_key in ("strata_columns", "mandatory_ids"):
        if tuple_key in kwargs:
            kwargs[tuple_key] = tuple(str(item) for item in kwargs[tuple_key])
    return PilotSelectionConfig(**kwargs)


def generate_hpc_bundles(
    compounds: pd.DataFrame,
    physics_summary: pd.DataFrame,
    output_dir: str | Path,
    config: Mapping[str, Any] | HPCBundleConfig | None = None,
) -> dict[str, Path | int]:
    """Select pilots and create guarded environment, PMF, and hERG bundles."""

    if "compound_id" not in compounds:
        raise ValueError("compounds requires compound_id")
    if "compound_id" not in physics_summary:
        raise ValueError("physics_summary requires compound_id")
    merged = compounds.merge(physics_summary, on="compound_id", how="inner", suffixes=("", "_physics"))
    if merged.empty:
        raise ValueError("No compounds have matching fast-physics summaries")
    if "pka_scenario" in merged:
        nominal = merged.loc[merged["pka_scenario"].astype(str) == "nominal"]
        if not nominal.empty:
            merged = nominal
    if "ph" in merged:
        available = sorted(pd.to_numeric(merged["ph"], errors="coerce").dropna().unique())
        if available:
            chosen_ph = min(available, key=lambda value: abs(float(value) - 7.4))
            merged = merged.loc[np.isclose(pd.to_numeric(merged["ph"], errors="coerce"), chosen_ph)]
    merged = merged.drop_duplicates("compound_id", keep="first").reset_index(drop=True)
    excluded = {
        "compound_id",
        "standardized_smiles",
        "smiles",
        "structure_id",
        "pka_scenario",
        "ph",
    }
    declared_features: list[str] = []
    if isinstance(config, Mapping):
        declared_features = [
            str(column) for column in config.get("pilot_selection", {}).get("feature_columns", [])
        ]
    if declared_features:
        missing = [column for column in declared_features if column not in merged]
        nonnumeric = [
            column
            for column in declared_features
            if column in merged and not pd.api.types.is_numeric_dtype(merged[column])
        ]
        if missing or nonnumeric:
            raise ValueError(
                "Declared pilot coverage features are unavailable or nonnumeric: "
                f"missing={missing}, nonnumeric={nonnumeric}"
            )
        feature_columns = declared_features
    else:
        feature_columns = [
            column
            for column in merged.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(merged[column])
        ]
    if not feature_columns:
        raise ValueError("No numeric features are available for pilot selection")
    raw_config = config if isinstance(config, Mapping) else None
    selection_config = _selection_config_from_mapping(raw_config)
    pilots = select_pilot_compounds(
        merged,
        feature_columns=feature_columns,
        config=selection_config,
    )
    membrane_pilots = select_pilot_compounds(
        merged,
        feature_columns=feature_columns,
        config=replace(
            selection_config,
            n_select=min(4, len(merged)),
            matched_pair_slots=min(4, max(0, len(merged) // 2 * 2)),
        ),
    )
    if "herg_class" in merged and {"blocker", "nonblocker"}.issubset(
        set(merged["herg_class"].dropna().astype(str))
    ):
        herg_pilots, herg_pair_map = _select_herg_pilots(
            merged,
            feature_columns=feature_columns,
            selection_config=selection_config,
        )
    else:
        herg_pair_map = pd.DataFrame(
            columns=(
                "herg_matched_pair_id",
                "blocker_compound_id",
                "nonblocker_compound_id",
                "tanimoto",
            )
        )
        herg_pilots = select_pilot_compounds(
            merged,
            feature_columns=feature_columns,
            config=replace(
                selection_config,
                n_select=min(6, len(merged)),
                matched_pair_slots=min(4, max(0, len(merged) // 2 * 2)),
            ),
        )
    bundle_config = _config_from_mapping(config)
    if isinstance(config, Mapping) and config.get("project_root"):
        bundle_config = _validate_receptor_raw_coordinates(
            bundle_config,
            project_root=Path(config["project_root"]),
        )
    outputs = create_hpc_bundle(
        output_dir,
        pilots=pilots,
        config=bundle_config,
        workflow_pilots={
            "environment_md": pilots,
            "membrane_pmf": membrane_pilots,
            "herg_ensemble": herg_pilots,
            "relative_free_energy": membrane_pilots,
        },
    )
    selection_path = Path(output_dir) / "pilot_selection.parquet"
    pilots.to_parquet(selection_path, index=False)
    herg_pair_map_path = Path(output_dir) / "herg_cross_class_matched_pairs.csv"
    herg_pair_map.to_csv(herg_pair_map_path, index=False)
    return {
        **outputs,
        "selection_path": selection_path,
        "herg_pair_map_path": herg_pair_map_path,
        "candidate_count": int(len(merged)),
        "pilot_count": int(len(pilots)),
        "environment_pilot_count": int(len(pilots)),
        "membrane_pilot_count": int(len(membrane_pilots)),
        "herg_pilot_count": int(len(herg_pilots)),
    }


__all__ = [
    "HPC_BUNDLE_VERSION",
    "RECEPTOR_RAW_COORDINATE_PATHS",
    "HPCBundleConfig",
    "PilotSelectionConfig",
    "SimulationAcceptanceGates",
    "create_hpc_bundle",
    "generate_hpc_bundles",
    "select_pilot_compounds",
]
