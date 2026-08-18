#!/usr/bin/env python3
"""Run the bounded, resumable M3 mechanistic-physics pilot.

This script is intentionally separate from ``menin-research --stage
physics-fast``.  It operates on a preregistered two-pair subset and never
promotes its outputs into PK/hERG predictors.  CREST/xTB execution is
resumable at the job-directory level and bounded by an explicit wall-clock
timeout.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser
from menin_discovery.research_local_physics import (
    LOCAL_PHYSICS_VERSION,
    acid_deprotonated_fraction,
    apply_transform,
    basin_selection_sensitivity,
    generate_diverse_basin_conformers,
    generate_independent_starting_conformers,
    geometry_observables,
    kabsch_transform,
    mol_to_xyz_text,
    polybase_macrostate_distribution,
    rare_state_flux_threshold,
    read_xyz_ensemble,
)
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "research/simulations/pk_herg/local_m3_pilot"
CANONICAL = ROOT / "research/data/pk_herg/canonical/internal"
RECEPTOR_PATHS = {
    "8ZYN": ROOT / "research/literature/herg/structural_biology/2024_miyashita_inhibitor_bound/8ZYN.cif",
    "8ZYO": ROOT / "research/literature/herg/structural_biology/2024_miyashita_inhibitor_bound/8ZYO.cif",
    "8ZYP": ROOT / "research/literature/herg/structural_biology/2024_miyashita_inhibitor_bound/8ZYP.cif",
    "8ZYQ": ROOT / "research/literature/herg/structural_biology/2024_miyashita_inhibitor_bound/8ZYQ.cif",
    "9CHP": ROOT / "research/literature/herg/structural_biology/2024_lau_potassium_states/9CHP.cif",
    "9CHQ": ROOT / "research/literature/herg/structural_biology/2024_lau_potassium_states/9CHQ.cif",
}
PAIR_DEFINITIONS = (
    {
        "pair_id": "LOCAL-MP-01",
        "compound_id_a": "CMP-D6B900FDA91C13513900",
        "compound_id_b": "CMP-642D79F70A93767590D0",
        "mechanistic_role": (
            "O-methylation pair with complete PK and decisive hERG contrast; "
            "tests rare neutral-state access and environment-conditioned exposure"
        ),
        "crest_execution": True,
        "xtb_execution": True,
    },
    {
        "pair_id": "LOCAL-MP-02",
        "compound_id_a": "CMP-47DADB26C12A7C3D5CB5",
        "compound_id_b": "CMP-593B478B10007352C89B",
        "mechanistic_role": (
            "near-isomass heteroatom-topology hERG cliff; tests site-resolved "
            "microstate and receptor-state hypotheses"
        ),
        "crest_execution": False,
        "xtb_execution": True,
    },
)
PH_GRID = (5.0, 6.5, 7.4, 8.0, 9.0)
PKA_OFFSETS = (-1.0, 0.0, 1.0)
START_SEEDS = (2026072401, 2026072402)
CREST_SOLVENTS = ("water", "chloroform")
XTB_SOLVENT_ARGUMENT = {"water": "water", "chloroform": "chcl3"}
HARTREE_TO_KCAL_MOL = 627.5094740631


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _selected_ids() -> list[str]:
    return sorted(
        {str(pair[column]) for pair in PAIR_DEFINITIONS for column in ("compound_id_a", "compound_id_b")}
    )


def _unique_measurements(
    measurements: pd.DataFrame,
    *,
    compound_id: str,
    endpoint: str,
) -> list[float]:
    selected = measurements[
        measurements["compound_id"].astype(str).eq(compound_id)
        & measurements["endpoint"].astype(str).eq(endpoint)
    ]
    values = pd.to_numeric(selected["value"], errors="coerce").dropna()
    return sorted(float(value) for value in values.drop_duplicates())


def _stepwise_pka_measurements(
    measurements: pd.DataFrame,
    *,
    compound_id: str,
    endpoint: str,
) -> list[float]:
    """Preserve distinct reported pKa steps even when values are identical."""

    selected = measurements[
        measurements["compound_id"].astype(str).eq(compound_id)
        & measurements["endpoint"].astype(str).eq(endpoint)
    ].copy()
    selected["numeric_value"] = pd.to_numeric(selected["value"], errors="coerce")
    selected = selected[selected["numeric_value"].notna()]
    if "source_record_id" in selected:
        with_source = selected[selected["source_record_id"].notna()].drop_duplicates(
            subset=["source_record_id"]
        )
        without_source = selected[selected["source_record_id"].isna()].drop_duplicates(
            subset=["numeric_value"]
        )
        selected = pd.concat([with_source, without_source], ignore_index=True)
    else:
        selected = selected.drop_duplicates(subset=["numeric_value"])
    return sorted(
        (float(value) for value in selected["numeric_value"]),
        reverse=True,
    )


def build_selection() -> pd.DataFrame:
    compounds = pd.read_parquet(CANONICAL / "compounds.parquet").set_index("compound_id")
    measurements = pd.read_parquet(CANONICAL / "measurements.parquet")
    rows: list[dict[str, Any]] = []
    for pair in PAIR_DEFINITIONS:
        for member, compound_id in (
            ("a", str(pair["compound_id_a"])),
            ("b", str(pair["compound_id_b"])),
        ):
            compound = compounds.loc[compound_id]
            basic_pkas = _stepwise_pka_measurements(
                measurements,
                compound_id=compound_id,
                endpoint="basic_pka",
            )
            acidic_pkas = _stepwise_pka_measurements(
                measurements, compound_id=compound_id, endpoint="acidic_pka_below_13"
            )
            herg = _unique_measurements(measurements, compound_id=compound_id, endpoint="herg_ic50")
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "pair_member": member,
                    "compound_id": compound_id,
                    "standardized_smiles": compound["standardized_smiles"],
                    "molecular_weight_g_mol": compound["molecular_weight_g_mol"],
                    "series_id": compound["series_id"],
                    "stereochemistry_status": compound["stereochemistry_status"],
                    "basic_macro_pkas": ";".join(f"{value:g}" for value in basic_pkas),
                    "acidic_macro_pkas": ";".join(f"{value:g}" for value in acidic_pkas),
                    "herg_ic50_um_values": ";".join(f"{value:g}" for value in herg),
                    "mechanistic_role": pair["mechanistic_role"],
                    "crest_execution": bool(pair["crest_execution"]),
                    "xtb_execution": bool(pair["xtb_execution"]),
                    "selection_semantics": (
                        "outcome-informed mechanistic falsification pair; not an unbiased "
                        "prospective model evaluation"
                    ),
                }
            )
    result = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT / "selected_compounds.csv", index=False)
    return result


def run_speciation() -> dict[str, int]:
    build_selection()
    measurements = pd.read_parquet(CANONICAL / "measurements.parquet")
    selected_evidence = measurements[
        measurements["compound_id"].astype(str).isin(_selected_ids())
    ].sort_values(["compound_id", "endpoint", "source_record_id"])
    selected_evidence.to_csv(OUTPUT / "selected_measurement_evidence.csv", index=False)
    macrostate_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    acid_rows: list[dict[str, Any]] = []
    for compound_id in _selected_ids():
        basic_pkas = _stepwise_pka_measurements(
            measurements,
            compound_id=compound_id,
            endpoint="basic_pka",
        )
        acidic_pkas = _stepwise_pka_measurements(
            measurements, compound_id=compound_id, endpoint="acidic_pka_below_13"
        )
        for ph in PH_GRID:
            for offset in PKA_OFFSETS:
                result = polybase_macrostate_distribution(
                    basic_pkas,
                    ph=ph,
                    pka_offset=offset,
                )
                for positive_charge, probability in enumerate(result.probabilities):
                    macrostate_rows.append(
                        {
                            "compound_id": compound_id,
                            "ph": ph,
                            "common_basic_pka_offset": offset,
                            "reported_basic_macrostep_count": len(basic_pkas),
                            "positive_charge_macrostate": positive_charge,
                            "probability": probability,
                            "mean_positive_charge": result.mean_positive_charge,
                            "population_semantics": (
                                "macroscopic charge-state sensitivity; protonation-site "
                                "identity and microstate partition are unresolved"
                            ),
                        }
                    )
                for target in (0.5, 0.9):
                    threshold = rare_state_flux_threshold(
                        result.neutral_fraction,
                        target_flux_fraction=target,
                        temperature_kelvin=310.0,
                    )
                    threshold_rows.append(
                        {
                            "compound_id": compound_id,
                            "ph": ph,
                            "common_basic_pka_offset": offset,
                            "reported_basic_macrostep_count": len(basic_pkas),
                            "neutral_macrostate_fraction": result.neutral_fraction,
                            "target_neutral_flux_fraction": target,
                            **threshold,
                            "interpretation": (
                                "minimum neutral/nonneutral state-specific permeability "
                                "ratio under a two-class flux identity; not a computed PMF"
                            ),
                        }
                    )
                for acid_step_index, acidic_pka in enumerate(acidic_pkas, start=1):
                    acid_rows.append(
                        {
                            "compound_id": compound_id,
                            "ph": ph,
                            "common_acidic_pka_offset": offset,
                            "acidic_macrostep_index": acid_step_index,
                            "acidic_pka": acidic_pka,
                            "deprotonated_fraction_single_step": acid_deprotonated_fraction(
                                acidic_pka,
                                ph=ph,
                                pka_offset=offset,
                            ),
                            "interpretation": (
                                "single-step acidic macrostate sensitivity; not combined "
                                "with polybasic macrostates because coupling is unidentified"
                            ),
                        }
                    )
    macrostates = pd.DataFrame(macrostate_rows)
    thresholds = pd.DataFrame(threshold_rows)
    acid = pd.DataFrame(acid_rows)
    macrostates.to_csv(OUTPUT / "macrostate_speciation.csv", index=False)
    thresholds.to_csv(OUTPUT / "rare_state_flux_thresholds.csv", index=False)
    acid.to_csv(OUTPUT / "acidic_state_sensitivity.csv", index=False)
    return {
        "macrostate_rows": len(macrostates),
        "flux_threshold_rows": len(thresholds),
        "acid_sensitivity_rows": len(acid),
        "selected_measurement_evidence_rows": len(selected_evidence),
    }


def prepare_crest_inputs() -> pd.DataFrame:
    selection = build_selection()
    input_root = OUTPUT / "crest_inputs"
    run_root = OUTPUT / "crest_runs"
    metadata_rows: list[dict[str, Any]] = []
    job_rows: list[dict[str, Any]] = []
    for row in selection.itertuples(index=False):
        starts = generate_independent_starting_conformers(
            row.standardized_smiles,
            seeds=START_SEEDS,
            pool_size=64,
            minimum_heavy_atom_rmsd_angstrom=1.5,
        )
        compound_dir = input_root / row.compound_id
        compound_dir.mkdir(parents=True, exist_ok=True)
        for start_index, (mol, metadata) in enumerate(starts, start=1):
            xyz_path = compound_dir / f"neutral_start_{start_index}.xyz"
            xyz_path.write_text(
                mol_to_xyz_text(
                    mol,
                    comment=(
                        f"{row.compound_id} neutral submitted-parent hypothesis; "
                        f"ETKDG/MMFF94s start {start_index}; not an equilibrium ensemble"
                    ),
                ),
                encoding="utf-8",
            )
            sdf_path = compound_dir / f"neutral_start_{start_index}.sdf"
            writer = Chem.SDWriter(str(sdf_path))
            writer.write(mol)
            writer.close()
            metadata_rows.append(
                {
                    "compound_id": row.compound_id,
                    "pair_id": row.pair_id,
                    "state_hypothesis": "submitted_parent_neutral_macrostate",
                    "formal_charge": 0,
                    "start_index": start_index,
                    "xyz_path": str(xyz_path.relative_to(ROOT)),
                    "sdf_path": str(sdf_path.relative_to(ROOT)),
                    **metadata,
                }
            )
            if bool(row.crest_execution):
                for solvent in CREST_SOLVENTS:
                    job_id = f"{row.compound_id}__neutral__{solvent}__start{start_index}"
                    job_dir = run_root / job_id
                    job_rows.append(
                        {
                            "job_id": job_id,
                            "compound_id": row.compound_id,
                            "pair_id": row.pair_id,
                            "state_hypothesis": "submitted_parent_neutral_macrostate",
                            "formal_charge": 0,
                            "solvent": solvent,
                            "start_index": start_index,
                            "input_xyz": str(xyz_path.relative_to(ROOT)),
                            "job_directory": str(job_dir.relative_to(ROOT)),
                            "method": "CREST 2.12 iMTD-GC quick initial gate",
                            "hamiltonian": "GFN2-xTB 6.7.1",
                            "solvation": f"ALPB({solvent})",
                            "threads": 4,
                            "sampling_tier": "initial_quick_gate_requires_cross_start_convergence",
                            "promotion_status": "hypothesis_only",
                        }
                    )
    metadata = pd.DataFrame(metadata_rows)
    jobs = pd.DataFrame(job_rows)
    metadata.to_csv(OUTPUT / "starting_conformer_audit.csv", index=False)
    jobs.to_csv(OUTPUT / "crest_jobs.csv", index=False)
    _write_json(
        OUTPUT / "software_environment.json",
        {
            "version": LOCAL_PHYSICS_VERSION,
            "crest_executable_expected": "/private/tmp/menin-crest212-env/bin/crest",
            "crest_version": "2.12",
            "xtb_version": "6.7.1",
            "runtime_environment_is_temporary": True,
            "reproduction": ("conda create -p <prefix> -c conda-forge crest=2.12 xtb=6.7.1"),
            "main_python_environment": sys.version,
        },
    )
    return jobs


def prepare_xtb_basin_inputs() -> pd.DataFrame:
    """Prepare paired-solvent xTB refinements for the primary matched pair.

    Four basins per compound are the maximum: two diverse basins are selected
    independently from each of two ETKDG/MMFF pools.  This is a local-minimum
    response experiment, not an attempted conformational ensemble.
    """

    selection = build_selection()
    selection = selection[selection["xtb_execution"].astype(bool)].copy()
    input_root = OUTPUT / "xtb_basin_inputs"
    run_root = OUTPUT / "xtb_basin_runs"
    audit_rows: list[dict[str, Any]] = []
    job_rows: list[dict[str, Any]] = []
    for row in selection.itertuples(index=False):
        basins = generate_diverse_basin_conformers(
            row.standardized_smiles,
            seeds=START_SEEDS,
            pool_size=64,
            maximum_per_seed=2,
            energy_window_kcal_mol=12.0,
            minimum_heavy_atom_rmsd_angstrom=1.25,
        )
        compound_dir = input_root / row.compound_id
        compound_dir.mkdir(parents=True, exist_ok=True)
        for basin_global_index, (mol, metadata) in enumerate(basins, start=1):
            seed = int(metadata["seed"])
            basin_index = int(metadata["basin_index_within_seed"])
            stem = f"seed{seed}_basin{basin_index}"
            xyz_path = compound_dir / f"{stem}.xyz"
            sdf_path = compound_dir / f"{stem}.sdf"
            xyz_path.write_text(
                mol_to_xyz_text(
                    mol,
                    comment=(
                        f"{row.compound_id}; seed={seed}; basin={basin_index}; "
                        "ETKDGv3/MMFF94s diverse local-minimum hypothesis"
                    ),
                ),
                encoding="utf-8",
            )
            writer = Chem.SDWriter(str(sdf_path))
            writer.write(mol)
            writer.close()
            audit_rows.append(
                {
                    "compound_id": row.compound_id,
                    "pair_id": row.pair_id,
                    "basin_global_index": basin_global_index,
                    "state_hypothesis": "submitted_parent_neutral_macrostate",
                    "formal_charge": 0,
                    "xyz_path": str(xyz_path.relative_to(ROOT)),
                    "sdf_path": str(sdf_path.relative_to(ROOT)),
                    **metadata,
                }
            )
            for solvent in CREST_SOLVENTS:
                job_id = f"{row.compound_id}__neutral__{solvent}__seed{seed}__basin{basin_index}"
                job_dir = run_root / job_id
                job_rows.append(
                    {
                        "job_id": job_id,
                        "compound_id": row.compound_id,
                        "pair_id": row.pair_id,
                        "state_hypothesis": "submitted_parent_neutral_macrostate",
                        "formal_charge": 0,
                        "solvent": solvent,
                        "xtb_solvent_argument": XTB_SOLVENT_ARGUMENT[solvent],
                        "seed": seed,
                        "basin_index_within_seed": basin_index,
                        "input_xyz": str(xyz_path.relative_to(ROOT)),
                        "input_sdf": str(sdf_path.relative_to(ROOT)),
                        "job_directory": str(job_dir.relative_to(ROOT)),
                        "method": "GFN2-xTB 6.7.1 tight geometry optimization",
                        "solvation": f"ALPB({solvent})",
                        "threads": 4,
                        "design": (
                            "same starting basin paired across solvents; two independent ETKDG/MMFF pools"
                        ),
                        "truth_boundary": (
                            "environment-conditioned local minimum; not an "
                            "equilibrium population, transition rate, PMF, or permeability"
                        ),
                    }
                )
    audit = pd.DataFrame(audit_rows)
    jobs = pd.DataFrame(job_rows)
    audit.to_csv(OUTPUT / "xtb_basin_selection_audit.csv", index=False)
    jobs.to_csv(OUTPUT / "xtb_basin_jobs.csv", index=False)
    return jobs


def run_basin_selection_sensitivity() -> dict[str, int]:
    """Audit the nonphysical local basin-selection settings."""

    selection = build_selection()
    rows: list[dict[str, Any]] = []
    for compound in selection.itertuples(index=False):
        compound_rows = basin_selection_sensitivity(
            compound.standardized_smiles,
            seeds=START_SEEDS,
            maximum_pool_size=64,
            pool_size_prefixes=(32, 64),
            energy_windows_kcal_mol=(8.0, 12.0, 16.0),
            rmsd_gates_angstrom=(1.0, 1.25, 1.5),
            maximum_per_seed=2,
        )
        rows.extend(
            {
                "compound_id": compound.compound_id,
                "pair_id": compound.pair_id,
                **row,
            }
            for row in compound_rows
        )
    detail = pd.DataFrame(rows)
    setting_counts = (
        detail.groupby(["compound_id", "seed", "setting_id"])["selected_basin_index"]
        .max()
        .rename("selected_basin_count")
        .reset_index()
    )
    summary_rows: list[dict[str, Any]] = []
    for (compound_id, seed), group in detail.groupby(
        ["compound_id", "seed"],
        sort=True,
    ):
        counts = setting_counts[
            setting_counts["compound_id"].eq(compound_id) & setting_counts["seed"].eq(seed)
        ]["selected_basin_count"]
        first_ids = group[group["selected_basin_index"].eq(1)]["selected_pool_conformer_id"]
        second_ids = group[group["selected_basin_index"].eq(2)]["selected_pool_conformer_id"]
        first_mode_fraction = float(first_ids.value_counts(normalize=True).iloc[0])
        second_mode_fraction = (
            float(second_ids.value_counts(normalize=True).iloc[0]) if not second_ids.empty else float("nan")
        )
        nominal = group[
            group["pool_size_prefix"].eq(64)
            & group["energy_window_kcal_mol"].eq(12.0)
            & group["rmsd_gate_angstrom"].eq(1.25)
        ].sort_values("selected_basin_index")
        summary_rows.append(
            {
                "compound_id": compound_id,
                "seed": int(seed),
                "setting_count": counts.size,
                "minimum_selected_basin_count": int(counts.min()),
                "maximum_selected_basin_count": int(counts.max()),
                "unique_first_basin_ids_across_settings": ";".join(
                    str(value) for value in sorted(set(first_ids))
                ),
                "first_basin_modal_selection_fraction": first_mode_fraction,
                "unique_second_basin_ids_across_settings": ";".join(
                    str(value) for value in sorted(set(second_ids))
                ),
                "second_basin_modal_selection_fraction": second_mode_fraction,
                "nominal_pool64_window12_rmsd1p25_ids": ";".join(
                    str(value) for value in nominal["selected_pool_conformer_id"]
                ),
                "interpretation": (
                    "selection-rule sensitivity only; changes restrict local "
                    "claims but do not estimate missing conformer probability mass"
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    detail.to_csv(OUTPUT / "basin_selection_sensitivity.csv", index=False)
    summary.to_csv(OUTPUT / "basin_selection_sensitivity_summary.csv", index=False)
    return {
        "detail_rows": len(detail),
        "summary_rows": len(summary),
        "compounds": detail["compound_id"].nunique(),
        "settings_per_seed": setting_counts["setting_id"].nunique(),
    }


def _xtb_json_properties(path: Path) -> dict[str, float | None]:
    """Extract the small set of xTB outputs with direct physical roles."""

    if not path.exists():
        return {
            "total_energy_hartree": None,
            "dipole_magnitude_debye": None,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates: list[float] = []

    def visit(value: Any, key_path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                visit(nested, (*key_path, str(key).lower()))
        elif isinstance(value, list):
            for nested in value:
                visit(nested, key_path)
        elif isinstance(value, (int, float)):
            joined = " ".join(key_path)
            if "energy" in joined and ("total" in joined or joined.endswith("energy")):
                candidates.append(float(value))

    visit(payload)
    finite = [value for value in candidates if math.isfinite(value)]
    dipole = payload.get("dipole / a.u.")
    dipole_magnitude_debye: float | None = None
    if isinstance(dipole, list) and len(dipole) == 3:
        dipole_array = np.asarray(dipole, dtype=float)
        if np.isfinite(dipole_array).all():
            dipole_magnitude_debye = float(np.linalg.norm(dipole_array) * 2.541746473)
    return {
        "total_energy_hartree": finite[0] if finite else None,
        "dipole_magnitude_debye": dipole_magnitude_debye,
    }


def _run_one_xtb_job(
    row: pd.Series,
    *,
    xtb_executable: Path,
    timeout_seconds: float,
    force: bool,
) -> dict[str, Any]:
    job_dir = ROOT / str(row["job_directory"])
    job_dir.mkdir(parents=True, exist_ok=True)
    input_source = ROOT / str(row["input_xyz"])
    input_destination = job_dir / "input.xyz"
    if not input_destination.exists() or force:
        shutil.copy2(input_source, input_destination)
    record_path = job_dir / "run_record.json"
    optimized_path = job_dir / "xtbopt.xyz"
    if record_path.exists() and optimized_path.exists() and not force:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") == "completed":
            return record

    command = [
        str(xtb_executable),
        "input.xyz",
        "--gfn",
        str(int(row.get("gfn_level", 2))),
        "--alpb",
        str(row["xtb_solvent_argument"]),
        "--chrg",
        str(int(row["formal_charge"])),
        "--uhf",
        "0",
        "-P",
        str(int(row["threads"])),
        "--opt",
        "tight",
        "--json",
    ]
    started = time.time()
    log_path = job_dir / "xtb.log"
    timeout_hit = False
    exit_code: int | None = None
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=job_dir,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
            env={
                **os.environ,
                "PATH": f"{xtb_executable.parent}:{os.environ.get('PATH', '')}",
                "OMP_NUM_THREADS": str(int(row["threads"])),
            },
        )
        try:
            exit_code = process.wait(timeout=float(timeout_seconds))
        except subprocess.TimeoutExpired:
            timeout_hit = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            exit_code = process.returncode
    elapsed = time.time() - started
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    normal_termination = "normal termination of xtb" in log_text.lower()
    if exit_code == 0 and optimized_path.exists() and normal_termination:
        status = "completed"
    elif timeout_hit:
        status = "stopped_at_wallclock_gate"
    else:
        status = "failed"
    xtb_properties = _xtb_json_properties(job_dir / "xtbout.json")
    record = {
        "job_id": row["job_id"],
        "status": status,
        "command": command,
        "started_epoch_seconds": started,
        "elapsed_seconds": elapsed,
        "timeout_seconds": float(timeout_seconds),
        "timeout_hit": timeout_hit,
        "exit_code": exit_code,
        "normal_termination": normal_termination,
        "optimized_geometry_exists": optimized_path.exists(),
        **xtb_properties,
        "truth_boundary": row["truth_boundary"],
    }
    _write_json(record_path, record)
    return record


def run_xtb_basin_jobs(
    *,
    xtb_executable: Path,
    timeout_minutes: float,
    max_jobs: int,
    force: bool,
) -> list[dict[str, Any]]:
    jobs_path = OUTPUT / "xtb_basin_jobs.csv"
    jobs = pd.read_csv(jobs_path) if jobs_path.exists() else prepare_xtb_basin_inputs()
    records: list[dict[str, Any]] = []
    executed = 0
    for _, row in jobs.iterrows():
        record_path = ROOT / str(row["job_directory"]) / "run_record.json"
        complete = False
        if record_path.exists():
            complete = json.loads(record_path.read_text(encoding="utf-8")).get("status") == "completed"
        if not complete and executed >= int(max_jobs):
            continue
        record = _run_one_xtb_job(
            row,
            xtb_executable=xtb_executable,
            timeout_seconds=float(timeout_minutes) * 60.0,
            force=force,
        )
        records.append(record)
        if not complete:
            executed += 1
    _write_json(OUTPUT / "xtb_basin_execution_summary.json", records)
    return records


def _heavy_atom_aligned_rmsd(
    template: Chem.Mol,
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    heavy = [atom.GetIdx() for atom in template.GetAtoms() if atom.GetAtomicNum() > 1]
    _rotation, _translation, rmsd = kabsch_transform(
        np.asarray(left)[heavy],
        np.asarray(right)[heavy],
    )
    return float(rmsd)


def analyze_xtb_basin_results() -> dict[str, int]:
    """Analyze completed paired-solvent local minima without assigning weights."""

    jobs = pd.read_csv(OUTPUT / "xtb_basin_jobs.csv")
    geometry_rows: list[dict[str, Any]] = []
    coordinates_by_job: dict[str, np.ndarray] = {}
    templates_by_compound: dict[str, Chem.Mol] = {}
    for row in jobs.itertuples(index=False):
        job_dir = ROOT / row.job_directory
        record_path = job_dir / "run_record.json"
        optimized_path = job_dir / "xtbopt.xyz"
        if not record_path.exists() or not optimized_path.exists():
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "completed":
            continue
        supplier = Chem.SDMolSupplier(str(ROOT / row.input_sdf), removeHs=False)
        template = supplier[0]
        if template is None:
            raise ValueError(f"Could not read hydrogenated SDF template for {row.job_id}")
        symbols_in, input_frames, _input_comments = read_xyz_ensemble(ROOT / row.input_xyz)
        symbols_out, output_frames, _output_comments = read_xyz_ensemble(optimized_path)
        expected_symbols = [atom.GetSymbol() for atom in template.GetAtoms()]
        if symbols_in != expected_symbols or symbols_out != expected_symbols:
            raise ValueError(f"xTB changed atom identity/order for {row.job_id}")
        input_coordinates = input_frames[-1]
        output_coordinates = output_frames[-1]
        input_observables = geometry_observables(template, input_coordinates)
        output_observables = geometry_observables(template, output_coordinates)
        xtb_properties = _xtb_json_properties(job_dir / "xtbout.json")
        coordinates_by_job[row.job_id] = output_coordinates
        templates_by_compound[row.compound_id] = template
        geometry_rows.append(
            {
                "job_id": row.job_id,
                "compound_id": row.compound_id,
                "pair_id": row.pair_id,
                "solvent": row.solvent,
                "seed": int(row.seed),
                "basin_index_within_seed": int(row.basin_index_within_seed),
                "elapsed_seconds": float(record["elapsed_seconds"]),
                "total_energy_hartree": xtb_properties["total_energy_hartree"],
                "optimized_dipole_magnitude_debye": xtb_properties["dipole_magnitude_debye"],
                "input_to_optimized_heavy_atom_rmsd_angstrom": _heavy_atom_aligned_rmsd(
                    template,
                    input_coordinates,
                    output_coordinates,
                ),
                **{f"input_{key}": value for key, value in input_observables.items()},
                **{f"optimized_{key}": value for key, value in output_observables.items()},
                "truth_boundary": (
                    "one environment-conditioned local minimum from a selected "
                    "basin; no equilibrium weight or transition rate"
                ),
            }
        )
    geometry = pd.DataFrame(geometry_rows)
    if geometry.empty:
        raise ValueError("No completed xTB basin jobs are available for analysis")
    geometry["relative_energy_within_compound_solvent_kcal_mol"] = geometry.groupby(
        ["compound_id", "solvent"]
    )["total_energy_hartree"].transform(lambda values: (values - values.min()) * HARTREE_TO_KCAL_MOL)

    observable_names = (
        "radius_of_gyration_angstrom",
        "polar_heavy_atom_sasa_angstrom2",
        "carbon_halogen_heavy_atom_sasa_angstrom2",
        "imhb_heavy_atom_distance_pair_count",
        "dipole_magnitude_debye",
    )
    paired_rows: list[dict[str, Any]] = []
    for keys, group in geometry.groupby(
        ["compound_id", "seed", "basin_index_within_seed"],
        sort=True,
    ):
        by_solvent = group.set_index("solvent")
        if not {"water", "chloroform"}.issubset(by_solvent.index):
            continue
        for observable in observable_names:
            water = float(by_solvent.loc["water", f"optimized_{observable}"])
            chloroform = float(by_solvent.loc["chloroform", f"optimized_{observable}"])
            paired_rows.append(
                {
                    "compound_id": keys[0],
                    "seed": int(keys[1]),
                    "basin_index_within_seed": int(keys[2]),
                    "observable": observable,
                    "water_value": water,
                    "chloroform_value": chloroform,
                    "water_minus_chloroform": water - chloroform,
                    "contrast_semantics": (
                        "paired local-minimum response from identical starting "
                        "coordinates; not an equilibrium ensemble shift"
                    ),
                }
            )
    paired = pd.DataFrame(paired_rows)
    stability_rows: list[dict[str, Any]] = []
    if not paired.empty:
        for (compound_id, observable), group in paired.groupby(
            ["compound_id", "observable"],
            sort=True,
        ):
            seed_medians = group.groupby("seed")["water_minus_chloroform"].median()
            nonzero_signs = set(np.sign(seed_medians[seed_medians.ne(0.0)]))
            overall_median = float(group["water_minus_chloroform"].median())
            if overall_median == 0.0:
                sign_agreement_fraction = float(group["water_minus_chloroform"].eq(0.0).mean())
            else:
                sign_agreement_fraction = float(
                    (np.sign(group["water_minus_chloroform"]) == np.sign(overall_median)).mean()
                )
            stability_rows.append(
                {
                    "compound_id": compound_id,
                    "observable": observable,
                    "paired_basin_count": len(group),
                    "independent_seed_count": seed_medians.size,
                    "median_water_minus_chloroform": overall_median,
                    "minimum_water_minus_chloroform": float(group["water_minus_chloroform"].min()),
                    "maximum_water_minus_chloroform": float(group["water_minus_chloroform"].max()),
                    "seed_medians": ";".join(
                        f"{int(seed)}:{value:.8g}" for seed, value in seed_medians.items()
                    ),
                    "direction_replicated_across_seed_pools": bool(
                        seed_medians.size == len(START_SEEDS)
                        and len(nonzero_signs) == 1
                        and not seed_medians.eq(0.0).any()
                    ),
                    "all_basin_contrasts_same_nonzero_direction": bool(
                        (
                            group["water_minus_chloroform"].gt(0.0).all()
                            or group["water_minus_chloroform"].lt(0.0).all()
                        )
                        and group["water_minus_chloroform"].ne(0.0).all()
                    ),
                    "basin_sign_agreement_fraction": sign_agreement_fraction,
                    "promotion_status": "mechanistic_hypothesis_only",
                }
            )
    stability = pd.DataFrame(stability_rows)

    pairwise_rows: list[dict[str, Any]] = []
    for (compound_id, solvent), group in geometry.groupby(
        ["compound_id", "solvent"],
        sort=True,
    ):
        group = group.sort_values(["seed", "basin_index_within_seed"])
        template = templates_by_compound[compound_id]
        records = list(group.itertuples(index=False))
        for left_index, left in enumerate(records):
            for right in records[left_index + 1 :]:
                pairwise_rows.append(
                    {
                        "compound_id": compound_id,
                        "solvent": solvent,
                        "left_job_id": left.job_id,
                        "right_job_id": right.job_id,
                        "different_seed_pools": bool(left.seed != right.seed),
                        "optimized_heavy_atom_rmsd_angstrom": _heavy_atom_aligned_rmsd(
                            template,
                            coordinates_by_job[left.job_id],
                            coordinates_by_job[right.job_id],
                        ),
                        "interpretation": (
                            "continuous local-minimum separation; no arbitrary "
                            "cluster cutoff or kinetic connectivity assigned"
                        ),
                    }
                )
    pairwise = pd.DataFrame(pairwise_rows)

    convergence_rows: list[dict[str, Any]] = []
    for (compound_id, solvent), group in geometry.groupby(
        ["compound_id", "solvent"],
        sort=True,
    ):
        seed_minima = (
            group.loc[group.groupby("seed")["total_energy_hartree"].idxmin()]
            .sort_values("seed")
            .reset_index(drop=True)
        )
        if len(seed_minima) != len(START_SEEDS):
            continue
        left = seed_minima.iloc[0]
        right = seed_minima.iloc[1]
        rmsd_match = pairwise[
            pairwise["compound_id"].eq(compound_id)
            & pairwise["solvent"].eq(solvent)
            & (
                (pairwise["left_job_id"].eq(left["job_id"]) & pairwise["right_job_id"].eq(right["job_id"]))
                | (pairwise["left_job_id"].eq(right["job_id"]) & pairwise["right_job_id"].eq(left["job_id"]))
            )
        ]
        if len(rmsd_match) != 1:
            raise ValueError(f"Could not resolve one cross-seed RMSD for {compound_id}/{solvent}")
        convergence_rows.append(
            {
                "compound_id": compound_id,
                "solvent": solvent,
                "left_seed": int(left["seed"]),
                "right_seed": int(right["seed"]),
                "left_lowest_discovered_job_id": left["job_id"],
                "right_lowest_discovered_job_id": right["job_id"],
                "absolute_energy_gap_kcal_mol": float(
                    abs(float(left["total_energy_hartree"]) - float(right["total_energy_hartree"]))
                    * HARTREE_TO_KCAL_MOL
                ),
                "heavy_atom_rmsd_angstrom": float(rmsd_match.iloc[0]["optimized_heavy_atom_rmsd_angstrom"]),
                "absolute_radius_of_gyration_difference_angstrom": float(
                    abs(
                        float(left["optimized_radius_of_gyration_angstrom"])
                        - float(right["optimized_radius_of_gyration_angstrom"])
                    )
                ),
                "absolute_polar_sasa_difference_angstrom2": float(
                    abs(
                        float(left["optimized_polar_heavy_atom_sasa_angstrom2"])
                        - float(right["optimized_polar_heavy_atom_sasa_angstrom2"])
                    )
                ),
                "absolute_dipole_difference_debye": float(
                    abs(
                        float(left["optimized_dipole_magnitude_debye"])
                        - float(right["optimized_dipole_magnitude_debye"])
                    )
                ),
                "interpretation": (
                    "continuous agreement between the lowest local minima found "
                    "independently from each seed pool; no binary convergence cutoff"
                ),
            }
        )
    convergence = pd.DataFrame(convergence_rows)

    analysis_dir = OUTPUT / "xtb_basin_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    geometry.to_csv(analysis_dir / "optimized_local_minima.csv", index=False)
    paired.to_csv(analysis_dir / "paired_environment_responses.csv", index=False)
    stability.to_csv(
        analysis_dir / "cross_seed_direction_stability.csv",
        index=False,
    )
    pairwise.to_csv(analysis_dir / "optimized_basin_pairwise_rmsd.csv", index=False)
    convergence.to_csv(
        analysis_dir / "lowest_discovered_cross_seed_convergence.csv",
        index=False,
    )
    return {
        "completed_local_minima": len(geometry),
        "paired_environment_contrasts": len(paired),
        "cross_seed_stability_rows": len(stability),
        "pairwise_geometry_rows": len(pairwise),
        "cross_seed_lowest_minimum_comparisons": len(convergence),
    }


def assess_crest_runtime_gate() -> pd.DataFrame:
    """Convert the measured CREST estimate into an explicit stop decision."""

    rows: list[dict[str, Any]] = []
    run_root = OUTPUT / "crest_runs"
    pattern = re.compile(
        r"Estimated runtime for (one MTD|a batch of 6 MTDs).*?: "
        r"(?:(\d+) h )?(?:(\d+) min )?([0-9.]+) sec"
    )
    for log_path in sorted(run_root.glob("*/crest.log")):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        estimates: dict[str, float] = {}
        for label, hours, minutes, seconds in pattern.findall(text):
            estimates[label] = float(hours or 0) + float(minutes or 0) / 60.0 + float(seconds) / 3600.0
        if not estimates:
            continue
        record_path = log_path.parent / "run_record.json"
        record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
        one_mtd_hours = estimates.get("one MTD")
        batch_hours = estimates.get("a batch of 6 MTDs")
        decision = (
            "defer_full_crest_to_hpc"
            if batch_hours is not None and batch_hours > 2.0
            else "local_runtime_not_rejected"
        )
        if decision == "defer_full_crest_to_hpc" and not (log_path.parent / "crest_conformers.xyz").exists():
            record["status"] = "stopped_at_runtime_estimate_gate"
            record["estimated_one_mtd_hours_single_thread"] = one_mtd_hours
            record["estimated_six_mtd_batch_hours_four_threads"] = batch_hours
            record["stop_reason"] = (
                "CREST's measured trial-MTD estimate exceeds the bounded local "
                "compute budget before an ensemble was produced."
            )
            _write_json(record_path, record)
        rows.append(
            {
                "job_directory": str(log_path.parent.relative_to(ROOT)),
                "estimated_one_mtd_hours_single_thread": one_mtd_hours,
                "estimated_six_mtd_batch_hours_four_threads": batch_hours,
                "local_decision": decision,
                "ensemble_produced": (log_path.parent / "crest_conformers.xyz").exists(),
                "evidence_source": "CREST 2.12 trial-MTD runtime estimate in crest.log",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "crest_runtime_feasibility.csv", index=False)
    return result


def _run_one_crest_job(
    row: pd.Series,
    *,
    crest_executable: Path,
    timeout_seconds: float,
    force: bool,
) -> dict[str, Any]:
    job_dir = ROOT / str(row["job_directory"])
    job_dir.mkdir(parents=True, exist_ok=True)
    input_source = ROOT / str(row["input_xyz"])
    input_destination = job_dir / "input.xyz"
    if not input_destination.exists() or force:
        shutil.copy2(input_source, input_destination)
    record_path = job_dir / "run_record.json"
    ensemble_path = job_dir / "crest_conformers.xyz"
    if record_path.exists() and ensemble_path.exists() and not force:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") == "completed":
            return record
    command = [
        str(crest_executable),
        "input.xyz",
        "--gfn2",
        "--alpb",
        str(row["solvent"]),
        "--chrg",
        str(int(row["formal_charge"])),
        "--uhf",
        "0",
        "-T",
        str(int(row["threads"])),
        "--quick",
        "--opt",
        "tight",
        "--origin",
    ]
    started = time.time()
    log_path = job_dir / "crest.log"
    status = "failed"
    exit_code: int | None = None
    timeout_hit = False
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=job_dir,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
            env={
                **os.environ,
                "PATH": f"{crest_executable.parent}:{os.environ.get('PATH', '')}",
            },
        )
        try:
            exit_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timeout_hit = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            exit_code = process.returncode
    elapsed = time.time() - started
    if exit_code == 0 and ensemble_path.exists():
        status = "completed"
    elif timeout_hit:
        status = "stopped_at_wallclock_gate"
    record = {
        "job_id": row["job_id"],
        "status": status,
        "command": command,
        "started_epoch_seconds": started,
        "elapsed_seconds": elapsed,
        "timeout_seconds": timeout_seconds,
        "timeout_hit": timeout_hit,
        "exit_code": exit_code,
        "ensemble_exists": ensemble_path.exists(),
        "truth_boundary": (
            "Quick CREST is an initial cross-start convergence gate. It is not a "
            "production conformational free-energy calculation."
        ),
    }
    _write_json(record_path, record)
    return record


def run_crest_jobs(
    *,
    crest_executable: Path,
    timeout_minutes: float,
    max_jobs: int,
    force: bool,
) -> list[dict[str, Any]]:
    jobs_path = OUTPUT / "crest_jobs.csv"
    jobs = pd.read_csv(jobs_path) if jobs_path.exists() else prepare_crest_inputs()
    records: list[dict[str, Any]] = []
    executed = 0
    for _, row in jobs.iterrows():
        record_path = ROOT / str(row["job_directory"]) / "run_record.json"
        complete = False
        if record_path.exists():
            complete = json.loads(record_path.read_text(encoding="utf-8")).get("status") == "completed"
        if not complete and executed >= int(max_jobs):
            continue
        record = _run_one_crest_job(
            row,
            crest_executable=crest_executable,
            timeout_seconds=float(timeout_minutes) * 60.0,
            force=force,
        )
        records.append(record)
        if not complete:
            executed += 1
    _write_json(OUTPUT / "crest_execution_summary.json", records)
    return records


def _resolution_from_cif(path: Path) -> float:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"_em_3d_reconstruction\.resolution\s+([0-9]+(?:\.[0-9]+)?)",
        text,
    )
    return float(match.group(1)) if match else float("nan")


def _protein_atoms(structure: Any) -> dict[tuple[str, int, str], np.ndarray]:
    model = next(structure.get_models())
    atoms: dict[tuple[str, int, str], np.ndarray] = {}
    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            for atom in residue:
                atoms[(str(chain.id), int(residue.id[1]), str(atom.name))] = np.asarray(
                    atom.coord, dtype=float
                )
    return atoms


def _keys_in_range(
    atoms: dict[tuple[str, int, str], np.ndarray],
    *,
    residues: range,
    atom_names: tuple[str, ...],
) -> set[tuple[str, int, str]]:
    return {
        key for key in atoms if key[1] in residues and key[2] in atom_names and key[0] in {"A", "B", "C", "D"}
    }


def _best_chain_permutation_alignment(
    reference_atoms: dict[tuple[str, int, str], np.ndarray],
    mobile_atoms: dict[tuple[str, int, str], np.ndarray],
    *,
    residues: range = range(545, 620),
    atom_names: tuple[str, ...] = ("N", "CA", "C"),
) -> tuple[
    dict[str, str],
    np.ndarray,
    np.ndarray,
    float,
    int,
]:
    """Resolve arbitrary tetramer chain labels before structural comparison.

    The returned mapping is ``reference_chain -> mobile_chain``.  All 24
    permutations are evaluated because C4/C1 coordinate deposits do not
    guarantee homologous author-chain labels.
    """

    reference_chains = ("A", "B", "C", "D")
    mobile_chains = ("A", "B", "C", "D")
    best: tuple[float, int, tuple[str, ...], np.ndarray, np.ndarray] | None = None
    for permutation in itertools.permutations(mobile_chains):
        mapping = dict(zip(reference_chains, permutation, strict=True))
        reference_coordinates: list[np.ndarray] = []
        mobile_coordinates: list[np.ndarray] = []
        for reference_chain in reference_chains:
            mobile_chain = mapping[reference_chain]
            for residue in residues:
                for atom_name in atom_names:
                    reference_key = (reference_chain, residue, atom_name)
                    mobile_key = (mobile_chain, residue, atom_name)
                    if reference_key in reference_atoms and mobile_key in mobile_atoms:
                        reference_coordinates.append(reference_atoms[reference_key])
                        mobile_coordinates.append(mobile_atoms[mobile_key])
        if len(reference_coordinates) < 12:
            continue
        rotation, translation, rmsd = kabsch_transform(
            np.asarray(mobile_coordinates),
            np.asarray(reference_coordinates),
        )
        candidate = (rmsd, -len(reference_coordinates), permutation, rotation, translation)
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    if best is None:
        raise ValueError("No tetramer chain permutation supplied enough common alignment atoms")
    rmsd, negative_count, permutation, rotation, translation = best
    return (
        dict(zip(reference_chains, permutation, strict=True)),
        rotation,
        translation,
        float(rmsd),
        int(-negative_count),
    )


def _align_atom_map(
    reference_atoms: dict[tuple[str, int, str], np.ndarray],
    mobile_atoms: dict[tuple[str, int, str], np.ndarray],
) -> tuple[
    dict[tuple[str, int, str], np.ndarray],
    dict[str, str],
    float,
    int,
]:
    mapping, rotation, translation, rmsd, atom_count = _best_chain_permutation_alignment(
        reference_atoms,
        mobile_atoms,
    )
    mobile_to_reference = {mobile: reference for reference, mobile in mapping.items()}
    transformed: dict[tuple[str, int, str], np.ndarray] = {}
    for (mobile_chain, residue, atom_name), coordinates in mobile_atoms.items():
        reference_chain = mobile_to_reference.get(mobile_chain, mobile_chain)
        transformed[(reference_chain, residue, atom_name)] = apply_transform(
            coordinates[None, :],
            rotation,
            translation,
        )[0]
    return transformed, mapping, rmsd, atom_count


def _ring_centroid(
    atoms: dict[tuple[str, int, str], np.ndarray],
    *,
    chain: str,
    residue: int,
) -> np.ndarray | None:
    names = (
        ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")
        if residue == 652
        else ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")
    )
    points = [atoms[(chain, residue, name)] for name in names if (chain, residue, name) in atoms]
    return np.mean(points, axis=0) if len(points) >= 5 else None


def _receptor_selection_table(
    alignment: pd.DataFrame,
    regions: pd.DataFrame,
    completeness: pd.DataFrame,
    contacts: pd.DataFrame,
) -> pd.DataFrame:
    """Define a purpose-specific four-core/two-sensitivity receptor set."""

    roles = {
        "8ZYN": {
            "production_tier": "core",
            "state_family": "cavity",
            "state_role": "unliganded cavity reference",
            "selection_rationale": (
                "Only matched apo model in the recent inhibitor-bound series; "
                "anchors receptor reorganization without assuming a ligand pose."
            ),
        },
        "8ZYP": {
            "production_tier": "core",
            "state_family": "cavity",
            "state_role": "E-4031-conditioned canonical blocker cavity",
            "selection_rationale": (
                "Best compact canonical cationic-blocker hypothesis in the matched "
                "series, 3.19 A reported resolution, and recurrent T623/S624/Y652 contacts."
            ),
        },
        "8ZYO": {
            "production_tier": "sensitivity",
            "state_family": "cavity",
            "state_role": "astemizole-conditioned alternate cavity",
            "selection_rationale": (
                "Preserves ligand-pose degeneracy and strong Y652 asymmetry as an "
                "induced-fit sensitivity, but is not needed in every production calculation."
            ),
        },
        "8ZYQ": {
            "production_tier": "sensitivity",
            "state_family": "cavity",
            "state_role": "pimozide-conditioned alternate cavity",
            "selection_rationale": (
                "Adds a chemically distinct bulky blocker-conditioned pocket and F656 "
                "proximity; reserved for sensitivity to avoid redundant cavity multiplication."
            ),
        },
        "9CHP": {
            "production_tier": "core",
            "state_family": "filter",
            "state_role": "high-potassium C4 filter",
            "selection_rationale": (
                "One member of the best-resolution matched high-/low-potassium C4 pair; "
                "required to test filter-state rather than ligand-induced cavity effects."
            ),
        },
        "9CHQ": {
            "production_tier": "core",
            "state_family": "filter",
            "state_role": "low-potassium C4 filter",
            "selection_rationale": (
                "Matched low-potassium contrast to 9CHP and highest reported resolution "
                "of the retained filter models; required for state-difference attribution."
            ),
        },
    }
    alignment_index = alignment.set_index("pdb_id")
    region_index = regions.pivot(
        index="pdb_id",
        columns="region",
        values="post_scaffold_alignment_rmsd_angstrom",
    )
    missing = completeness.groupby("pdb_id")["missing_residue_count_407_665"].agg(["min", "max"])
    contact_summary = (
        contacts.groupby("pdb_id")
        .agg(
            deposited_ligand_contact_residue_count=("protein_residue", "nunique"),
            deposited_ligand_contact_residues=(
                "protein_residue",
                lambda values: ";".join(str(value) for value in sorted(set(values))),
            ),
        )
        .to_dict(orient="index")
        if not contacts.empty
        else {}
    )
    rows: list[dict[str, Any]] = []
    for pdb_id, role in roles.items():
        rows.append(
            {
                "pdb_id": pdb_id,
                **role,
                "reported_resolution_angstrom": float(
                    alignment_index.loc[pdb_id, "reported_resolution_angstrom"]
                ),
                "scaffold_rmsd_vs_8ZYN_angstrom": float(
                    alignment_index.loc[pdb_id, "alignment_backbone_rmsd_angstrom"]
                ),
                "filter_rmsd_vs_8ZYN_angstrom": float(region_index.loc[pdb_id, "selectivity_filter"]),
                "cavity_rmsd_vs_8ZYN_angstrom": float(region_index.loc[pdb_id, "cavity_s6"]),
                "missing_residue_count_407_665_min_per_chain": int(missing.loc[pdb_id, "min"]),
                "missing_residue_count_407_665_max_per_chain": int(missing.loc[pdb_id, "max"]),
                **contact_summary.get(
                    pdb_id,
                    {
                        "deposited_ligand_contact_residue_count": 0,
                        "deposited_ligand_contact_residues": "",
                    },
                ),
                "coordinate_truth_boundary": (
                    "raw deposited coordinate hypothesis; map review, protonation, "
                    "missing-region handling, ions, membrane, and equilibration remain required"
                ),
                "execution_rule": (
                    "prepare four core systems first; use the two sensitivity cavities "
                    "only for preregistered receptor-state robustness; never concatenate "
                    "all six scores or construct unvalidated coordinate chimeras"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["production_tier", "state_family", "pdb_id"])


def analyze_receptors() -> dict[str, int]:
    parser = MMCIFParser(QUIET=True)
    structures = {pdb_id: parser.get_structure(pdb_id, path) for pdb_id, path in RECEPTOR_PATHS.items()}
    atom_maps = {pdb_id: _protein_atoms(structure) for pdb_id, structure in structures.items()}
    reference_id = "8ZYN"
    reference_atoms = atom_maps[reference_id]
    alignment_rows: list[dict[str, Any]] = []
    transformed_maps: dict[str, dict[tuple[str, int, str], np.ndarray]] = {}
    for pdb_id, atoms in atom_maps.items():
        # Residues 545-619 provide a shared transmembrane/pore-helix frame while
        # excluding the 623-631 selectivity filter and Y652/F656 cavity readouts.
        transformed, mapping, fit_rmsd, alignment_atom_count = _align_atom_map(
            reference_atoms,
            atoms,
        )
        transformed_maps[pdb_id] = transformed
        alignment_rows.append(
            {
                "pdb_id": pdb_id,
                "reference_id": reference_id,
                "alignment_atom_count": alignment_atom_count,
                "alignment_residue_range": "545-619",
                "reference_to_mobile_chain_mapping": ";".join(
                    f"{reference}:{mobile}" for reference, mobile in mapping.items()
                ),
                "alignment_backbone_rmsd_angstrom": fit_rmsd,
                "reported_resolution_angstrom": _resolution_from_cif(RECEPTOR_PATHS[pdb_id]),
            }
        )

    region_rows: list[dict[str, Any]] = []
    region_definitions = {
        "selectivity_filter": range(623, 632),
        "cavity_s6": range(640, 666),
    }
    for pdb_id, transformed in transformed_maps.items():
        for region, residues in region_definitions.items():
            common = sorted(
                _keys_in_range(
                    reference_atoms,
                    residues=residues,
                    atom_names=("N", "CA", "C", "O"),
                )
                & _keys_in_range(
                    transformed,
                    residues=residues,
                    atom_names=("N", "CA", "C", "O"),
                )
            )
            delta = np.asarray([transformed[key] - reference_atoms[key] for key in common])
            rmsd = float(np.sqrt(np.mean(np.sum(np.square(delta), axis=1))))
            region_rows.append(
                {
                    "pdb_id": pdb_id,
                    "reference_id": reference_id,
                    "region": region,
                    "atom_count": len(common),
                    "post_scaffold_alignment_rmsd_angstrom": rmsd,
                    "interpretation": ("coordinate-model contrast; not a dynamic population or free energy"),
                }
            )

    pairwise_rows: list[dict[str, Any]] = []
    for reference_pair_id, mobile_pair_id in itertools.combinations(sorted(atom_maps), 2):
        pairwise_mobile, mapping, fit_rmsd, alignment_atom_count = _align_atom_map(
            atom_maps[reference_pair_id],
            atom_maps[mobile_pair_id],
        )
        for region, residues in region_definitions.items():
            reference_pair_atoms = atom_maps[reference_pair_id]
            common = sorted(
                _keys_in_range(
                    reference_pair_atoms,
                    residues=residues,
                    atom_names=("N", "CA", "C", "O"),
                )
                & _keys_in_range(
                    pairwise_mobile,
                    residues=residues,
                    atom_names=("N", "CA", "C", "O"),
                )
            )
            delta = np.asarray([pairwise_mobile[key] - reference_pair_atoms[key] for key in common])
            pairwise_rows.append(
                {
                    "reference_pdb_id": reference_pair_id,
                    "mobile_pdb_id": mobile_pair_id,
                    "region": region,
                    "alignment_atom_count": alignment_atom_count,
                    "reference_to_mobile_chain_mapping": ";".join(
                        f"{reference}:{mobile}" for reference, mobile in mapping.items()
                    ),
                    "alignment_backbone_rmsd_angstrom": fit_rmsd,
                    "region_atom_count": len(common),
                    "post_scaffold_alignment_rmsd_angstrom": float(
                        np.sqrt(np.mean(np.sum(np.square(delta), axis=1)))
                    ),
                }
            )

    pocket_rows: list[dict[str, Any]] = []
    for pdb_id, transformed in transformed_maps.items():
        for residue in (652, 656):
            centroids = {
                chain: _ring_centroid(transformed, chain=chain, residue=residue)
                for chain in ("A", "B", "C", "D")
            }
            valid = {chain: value for chain, value in centroids.items() if value is not None}
            pair_distances: list[float] = []
            for left_index, left in enumerate(sorted(valid)):
                for right in sorted(valid)[left_index + 1 :]:
                    pair_distances.append(float(np.linalg.norm(valid[left] - valid[right])))
            pore_center = np.mean(list(valid.values()), axis=0)
            radial = [float(np.linalg.norm(value - pore_center)) for value in valid.values()]
            pocket_rows.append(
                {
                    "pdb_id": pdb_id,
                    "residue": residue,
                    "chain_count": len(valid),
                    "ring_centroid_pair_distance_mean_angstrom": float(np.mean(pair_distances)),
                    "ring_centroid_pair_distance_sd_angstrom": float(np.std(pair_distances)),
                    "ring_centroid_pair_distance_min_angstrom": float(np.min(pair_distances)),
                    "ring_centroid_pair_distance_max_angstrom": float(np.max(pair_distances)),
                    "ring_radial_distance_mean_angstrom": float(np.mean(radial)),
                    "ring_radial_distance_sd_angstrom": float(np.std(radial)),
                    "symmetry_interpretation": (
                        "within-model coordinate asymmetry; limited by reconstruction "
                        "resolution, refinement symmetry, and side-chain uncertainty"
                    ),
                }
            )

    completeness_rows: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []
    for pdb_id, structure in structures.items():
        model = next(structure.get_models())
        for chain in model:
            protein_residues = [residue for residue in chain if residue.id[0] == " "]
            present = {int(residue.id[1]) for residue in protein_residues}
            completeness_rows.append(
                {
                    "pdb_id": pdb_id,
                    "chain": chain.id,
                    "protein_residue_count": len(protein_residues),
                    "missing_residues_407_665": ";".join(
                        str(value) for value in range(407, 666) if value not in present
                    ),
                    "missing_residue_count_407_665": sum(value not in present for value in range(407, 666)),
                }
            )
        hetero_atoms = [
            atom
            for chain in model
            for residue in chain
            if residue.id[0] != " "
            for atom in residue
            if atom.element != "H"
        ]
        protein_heavy = [
            (chain.id, residue.id[1], residue.resname, atom)
            for chain in model
            for residue in chain
            if residue.id[0] == " "
            for atom in residue
            if atom.element != "H"
        ]
        if hetero_atoms:
            by_residue: dict[tuple[str, int, str], float] = {}
            for chain_id, residue_id, residue_name, atom in protein_heavy:
                distance = min(
                    float(np.linalg.norm(np.asarray(atom.coord) - np.asarray(ligand_atom.coord)))
                    for ligand_atom in hetero_atoms
                )
                key = (str(chain_id), int(residue_id), str(residue_name))
                by_residue[key] = min(distance, by_residue.get(key, math.inf))
            for (chain_id, residue_id, residue_name), distance in sorted(by_residue.items()):
                if distance <= 4.5:
                    contact_rows.append(
                        {
                            "pdb_id": pdb_id,
                            "protein_chain": chain_id,
                            "protein_residue": residue_id,
                            "protein_residue_name": residue_name,
                            "minimum_ligand_heavy_atom_distance_angstrom": distance,
                            "contact_cutoff_angstrom": 4.5,
                        }
                    )

    alignment = pd.DataFrame(alignment_rows)
    regions = pd.DataFrame(region_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    pockets = pd.DataFrame(pocket_rows)
    completeness = pd.DataFrame(completeness_rows)
    contacts = pd.DataFrame(contact_rows)
    receptor_dir = OUTPUT / "receptor_state_analysis"
    receptor_dir.mkdir(parents=True, exist_ok=True)
    alignment.to_csv(receptor_dir / "scaffold_alignment.csv", index=False)
    regions.to_csv(receptor_dir / "region_coordinate_contrasts.csv", index=False)
    pairwise.to_csv(receptor_dir / "pairwise_coordinate_contrasts.csv", index=False)
    pockets.to_csv(receptor_dir / "pocket_symmetry_metrics.csv", index=False)
    completeness.to_csv(receptor_dir / "coordinate_completeness.csv", index=False)
    contacts.to_csv(receptor_dir / "deposited_ligand_contacts.csv", index=False)
    selection = _receptor_selection_table(
        alignment,
        regions,
        completeness,
        contacts,
    )
    selection.to_csv(receptor_dir / "receptor_state_selection.csv", index=False)
    return {
        "structures": len(structures),
        "core_structures": int(selection["production_tier"].eq("core").sum()),
        "sensitivity_structures": int(selection["production_tier"].eq("sensitivity").sum()),
        "region_comparisons": len(regions),
        "pairwise_region_comparisons": len(pairwise),
        "pocket_rows": len(pockets),
        "ligand_contact_rows": len(contacts),
    }


def write_protocol() -> None:
    protocol = f"""# Local M3 mechanistic-physics pilot

version: {LOCAL_PHYSICS_VERSION}
compute_boundary:
  memory_gb: 16
  crest_threads_per_job: 4
  concurrent_crest_jobs: 1
  quick_job_wallclock_minutes: 30
  xtb_threads_per_job: 4
  concurrent_xtb_jobs: 1
  xtb_job_wallclock_minutes: 10
  production_md_launched: false

selection:
  pairs:
    - LOCAL-MP-01: O-methylation pair with PK and hERG evidence
    - LOCAL-MP-02: near-isomass heteroatom-topology hERG cliff
  selection_role: outcome-informed mechanistic falsification
  prohibited_role: unbiased prospective model evaluation

rare_state_thermodynamics:
  source: measured macroscopic pKa rows
  pH_grid: [5.0, 6.5, 7.4, 8.0, 9.0]
  common_pKa_offsets: [-1.0, 0.0, 1.0]
  site_assignment: unresolved
  output: charge-macrostate sensitivity and flux-dominance thresholds

environment_folding:
  state: submitted-parent neutral macrostate only
  rationale: directly tests the rare neutral transport hypothesis without
    inventing a protonation site
  environments: [ALPB-water, ALPB-chloroform]
  starts: 2 independent ETKDG/MMFF94s basins
  initial_method: CREST 2.12 quick iMTD-GC with GFN2-xTB 6.7.1
  measured_crest_gate: trial runtime estimate decides local feasibility before
    full sampling
  local_fallback: paired GFN2-xTB/ALPB tight optimization of at most two diverse
    basins from each of two independent ETKDG/MMFF94s pools
  local_fallback_role: test whether the sign of an environment-conditioned
    local-minimum response replicates across independent seed pools
  prohibited_local_fallback_role: equilibrium populations, entropy, kinetics,
    PMF, permeability, or production model features
  escalation: full conformer sampling and explicit-solvent dynamics on HPC
  truth_boundary: implicit-solvent semiempirical conformer evidence, not a
    membrane PMF, permeability, explicit-solvent population, or kinetic rate

transition_dynamics:
  local_status: not inferred from biased CREST trajectories
  requirement: independent unbiased trajectories with converged state
    decomposition, implied-timescale/Chapman-Kolmogorov tests, and replica
    agreement

herg_structure:
  coordinates: [8ZYN, 8ZYO, 8ZYP, 8ZYQ, 9CHP, 9CHQ]
  allowed_local_analysis: deposited-coordinate completeness, alignment,
    selectivity-filter/cavity contrast, pocket symmetry, deposited-ligand contacts
  prohibited_local_analysis: affinity ranking, docking score, or MD readiness
"""
    (OUTPUT / "protocol.yaml").write_text(protocol, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "prepare",
            "prepare-xtb",
            "speciation",
            "receptors",
            "run-crest",
            "assess-compute",
            "run-xtb",
            "analyze-xtb",
            "selection-sensitivity",
            "all-preflight",
        ),
        required=True,
    )
    parser.add_argument(
        "--crest-executable",
        default="/private/tmp/menin-crest212-env/bin/crest",
    )
    parser.add_argument(
        "--xtb-executable",
        default="/private/tmp/menin-crest212-env/bin/xtb",
    )
    parser.add_argument("--timeout-minutes", type=float, default=30.0)
    parser.add_argument("--max-jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_protocol()
    result: dict[str, Any] = {"version": LOCAL_PHYSICS_VERSION, "stage": args.stage}
    if args.stage in {"prepare", "all-preflight"}:
        jobs = prepare_crest_inputs()
        result["prepared_crest_jobs"] = len(jobs)
    if args.stage == "prepare-xtb":
        jobs = prepare_xtb_basin_inputs()
        result["prepared_xtb_jobs"] = len(jobs)
    if args.stage in {"speciation", "all-preflight"}:
        result["speciation"] = run_speciation()
    if args.stage in {"receptors", "all-preflight"}:
        result["receptors"] = analyze_receptors()
    if args.stage == "run-crest":
        result["crest_records"] = run_crest_jobs(
            crest_executable=Path(args.crest_executable),
            timeout_minutes=args.timeout_minutes,
            max_jobs=args.max_jobs,
            force=args.force,
        )
    if args.stage == "assess-compute":
        feasibility = assess_crest_runtime_gate()
        result["crest_runtime_assessments"] = len(feasibility)
        result["deferred_crest_jobs"] = int(feasibility["local_decision"].eq("defer_full_crest_to_hpc").sum())
    if args.stage == "run-xtb":
        result["xtb_records"] = run_xtb_basin_jobs(
            xtb_executable=Path(args.xtb_executable),
            timeout_minutes=args.timeout_minutes,
            max_jobs=args.max_jobs,
            force=args.force,
        )
    if args.stage == "analyze-xtb":
        result["xtb_analysis"] = analyze_xtb_basin_results()
    if args.stage == "selection-sensitivity":
        result["basin_selection_sensitivity"] = run_basin_selection_sensitivity()
    _write_json(OUTPUT / f"stage_{args.stage}_summary.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
