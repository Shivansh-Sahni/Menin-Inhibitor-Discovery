"""hERG mechanism interfaces beyond static binary classification."""

from __future__ import annotations

import numpy as np
import pandas as pd

HERG_RECEPTOR_ENSEMBLE = pd.DataFrame(
    [
        {
            "pdb": "8ZYN",
            "state": "apo_inhibitor_study",
            "potassium": "reported",
            "symmetry": "C1/C4 source maps",
            "role": "canonical_raw_coordinate",
            "selection_basis": "matched_recent_cavity_series_apo",
        },
        {
            "pdb": "8ZYO",
            "state": "astemizole_bound",
            "potassium": "reported",
            "symmetry": "C1",
            "role": "canonical_raw_coordinate",
            "selection_basis": "matched_recent_cavity_series_bound",
        },
        {
            "pdb": "8ZYP",
            "state": "E4031_bound",
            "potassium": "reported",
            "symmetry": "C1",
            "role": "canonical_raw_coordinate",
            "selection_basis": "matched_recent_cavity_series_bound",
        },
        {
            "pdb": "8ZYQ",
            "state": "pimozide_bound",
            "potassium": "reported",
            "symmetry": "C1",
            "role": "canonical_raw_coordinate",
            "selection_basis": "matched_recent_cavity_series_bound",
        },
        {
            "pdb": "9CHP",
            "state": "conductive_high_K",
            "potassium": "high",
            "symmetry": "C4",
            "role": "canonical_raw_coordinate",
            "selection_basis": "matched_C4_filter_condition_high_K",
        },
        {
            "pdb": "9CHQ",
            "state": "nonconductive_low_K",
            "potassium": "low",
            "symmetry": "C4",
            "role": "canonical_raw_coordinate",
            "selection_basis": "matched_C4_filter_condition_low_K",
        },
    ]
)


def state_dependent_markov_architecture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a protocol-aware architecture that cannot yet be fitted."""

    states = pd.DataFrame(
        [
            {"state": "C", "description": "closed unbound", "observable_current": 0, "ligand_bound": False},
            {
                "state": "O",
                "description": "open conducting unbound",
                "observable_current": 1,
                "ligand_bound": False,
            },
            {
                "state": "I",
                "description": "inactivated/nonconducting unbound",
                "observable_current": 0,
                "ligand_bound": False,
            },
            {"state": "OB", "description": "open-state bound", "observable_current": 0, "ligand_bound": True},
            {
                "state": "IB",
                "description": "inactivated-state bound",
                "observable_current": 0,
                "ligand_bound": True,
            },
            {
                "state": "TB",
                "description": "trapped ligand after channel closure",
                "observable_current": 0,
                "ligand_bound": True,
            },
        ]
    )
    transitions = pd.DataFrame(
        [
            {"source": "C", "target": "O", "rate_role": "voltage_dependent_activation"},
            {"source": "O", "target": "C", "rate_role": "voltage_dependent_deactivation"},
            {"source": "O", "target": "I", "rate_role": "voltage_dependent_inactivation"},
            {"source": "I", "target": "O", "rate_role": "recovery_from_inactivation"},
            {"source": "O", "target": "OB", "rate_role": "concentration_dependent_on_rate"},
            {"source": "OB", "target": "O", "rate_role": "off_rate"},
            {"source": "I", "target": "IB", "rate_role": "state_dependent_on_rate"},
            {"source": "IB", "target": "I", "rate_role": "off_rate"},
            {"source": "OB", "target": "TB", "rate_role": "closure_and_trapping"},
            {"source": "TB", "target": "OB", "rate_role": "reopening_and_untrapping"},
        ]
    )
    states["fit_status"] = "architecture_only_missing_voltage_onset_recovery_data"
    transitions["fit_status"] = "architecture_only_missing_voltage_onset_recovery_data"
    return states, transitions


def calculate_free_exposure_margin(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate margin only where Cmax, fraction unbound, and hERG IC50 coexist."""

    frame = data.copy()
    required = {"compound_id", "herg_ic50_um", "cmax_total_ng_ml", "molecular_weight", "fraction_unbound"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Free exposure margin requires columns: {missing}")
    complete = frame[list(required - {"compound_id"})].notna().all(axis=1)
    frame["free_cmax_um"] = np.nan
    # ng/mL is ug/L; uM = ug/L / (g/mol) because the scale factors cancel.
    frame.loc[complete, "free_cmax_um"] = (
        frame.loc[complete, "cmax_total_ng_ml"]
        * frame.loc[complete, "fraction_unbound"]
        / frame.loc[complete, "molecular_weight"]
    )
    frame["free_exposure_margin"] = frame["herg_ic50_um"] / frame["free_cmax_um"]
    frame["margin_status"] = np.where(
        complete,
        "computable_from_reported_total_cmax_and_fu",
        "required_data_missing_no_margin_reported",
    )
    return frame


def herg_process_observables() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "layer": "aqueous_access",
                "observable": "microstate_population",
                "required_context": "pH and micro-pKa uncertainty",
            },
            {
                "layer": "membrane_access",
                "observable": "partition_free_energy",
                "required_context": "lipid composition and charge state",
            },
            {
                "layer": "cavity_entry",
                "observable": "access_path_work",
                "required_context": "channel state and ligand conformation",
            },
            {
                "layer": "binding",
                "observable": "state_specific_binding_free_energy",
                "required_context": "receptor ensemble and ion occupancy",
            },
            {
                "layer": "adaptation",
                "observable": "Y652_symmetry_breaking",
                "required_context": "replicate and rotamer ensemble",
            },
            {
                "layer": "kinetics",
                "observable": "on_off_trapping_rates",
                "required_context": "voltage protocol and free concentration",
            },
            {
                "layer": "assay",
                "observable": "current_inhibition",
                "required_context": "temperature, time, cell line, voltage and free concentration",
            },
        ]
    )
