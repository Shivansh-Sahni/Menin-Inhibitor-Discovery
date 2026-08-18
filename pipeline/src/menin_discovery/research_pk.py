"""Gray-box rat IV/PO equations and structural-uncertainty diagnostics.

This is intentionally not presented as a calibrated PBPK model.  With summary
PK alone, Fa, Fg, and Fh are not separately identifiable.  The code preserves
that fact and uses mechanistic equations only for closure and sensitivity.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd

DISTRIBUTION_HYPOTHESES = {
    "berezhkovskiy": 1.00,
    "rodgers_rowland": 1.18,
    "poulin_theil": 0.82,
    "pk_sim_style": 1.10,
    "schmitt": 0.92,
}


def recompute_pk_closure(studies: pd.DataFrame) -> pd.DataFrame:
    """Recompute CL and F solely as diagnostics from their declared parents."""

    frame = studies.copy()
    required = {"iv_dose_mg_kg", "po_dose_mg_kg", "iv_auc0_inf_ng_h_ml", "po_auc0_inf_ng_h_ml"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"PK closure requires columns: {missing}")
    # mg/kg divided by ng*h/mL gives L/kg/h after the 1e6 ng/mg and 1e3 mL/L
    # conversions: dose*1e6 / AUC / 1e3 = dose*1000/AUC L/kg/h.
    frame["cl_recomputed_ml_kg_min"] = (
        frame["iv_dose_mg_kg"] * 1_000_000.0 / frame["iv_auc0_inf_ng_h_ml"] / 60.0
    )
    frame["f_recomputed_percent"] = (
        frame["po_auc0_inf_ng_h_ml"]
        / frame["po_dose_mg_kg"]
        / (frame["iv_auc0_inf_ng_h_ml"] / frame["iv_dose_mg_kg"])
        * 100.0
    )
    if "reported_cl_ml_kg_min" in frame:
        frame["cl_relative_closure_error"] = (
            frame["reported_cl_ml_kg_min"] - frame["cl_recomputed_ml_kg_min"]
        ).abs() / frame["cl_recomputed_ml_kg_min"].abs().clip(lower=1e-12)
    if "reported_f_percent" in frame:
        frame["f_absolute_closure_error_percent"] = (
            frame["reported_f_percent"] - frame["f_recomputed_percent"]
        ).abs()
    frame["reported_cl_model_role"] = "closure_only_derived_not_independent_label"
    frame["reported_f_model_role"] = "closure_only_derived_not_independent_label"
    return frame


def one_compartment_iv_profile(
    time_h: np.ndarray,
    *,
    dose_mg_kg: float,
    clearance_ml_kg_min: float,
    vdss_l_kg: float,
) -> np.ndarray:
    time = np.asarray(time_h, dtype=float)
    clearance_l_kg_h = clearance_ml_kg_min * 0.06
    elimination_h = clearance_l_kg_h / vdss_l_kg
    initial_ng_ml = dose_mg_kg * 1_000.0 / vdss_l_kg
    return initial_ng_ml * np.exp(-elimination_h * time)


def one_compartment_po_profile(
    time_h: np.ndarray,
    *,
    dose_mg_kg: float,
    clearance_ml_kg_min: float,
    vdss_l_kg: float,
    ka_h: float,
    bioavailability_fraction: float,
) -> np.ndarray:
    time = np.asarray(time_h, dtype=float)
    clearance_l_kg_h = clearance_ml_kg_min * 0.06
    ke_h = clearance_l_kg_h / vdss_l_kg
    if abs(ka_h - ke_h) < 1e-9:
        return (
            bioavailability_fraction * dose_mg_kg * 1_000.0 / vdss_l_kg * ka_h * time * np.exp(-ke_h * time)
        )
    scale = bioavailability_fraction * dose_mg_kg * 1_000.0 / vdss_l_kg * ka_h / (ka_h - ke_h)
    return scale * (np.exp(-ke_h * time) - np.exp(-ka_h * time))


def nonidentifiable_oral_factor_scenarios(
    total_f: float,
    *,
    grid: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
    tolerance: float = 0.05,
) -> pd.DataFrame:
    """Enumerate equally compatible Fa/Fg/Fh decompositions, not estimates."""

    rows: list[dict[str, Any]] = []
    for fa, fg, fh in product(grid, repeat=3):
        product_f = fa * fg * fh
        if abs(product_f - total_f) <= tolerance:
            rows.append(
                {
                    "fa_scenario": fa,
                    "fg_scenario": fg,
                    "fh_scenario": fh,
                    "product_f": product_f,
                    "identifiability_status": "observationally_equivalent_summary_pk_scenario",
                }
            )
    return pd.DataFrame(rows)


def distribution_structural_uncertainty(
    parameters: pd.DataFrame,
    *,
    time_h: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compare named distribution hypotheses as sensitivity scenarios."""

    required = {"compound_id", "iv_dose_mg_kg", "clearance_ml_kg_min", "vdss_l_kg"}
    missing = sorted(required - set(parameters.columns))
    if missing:
        raise ValueError(f"Distribution sensitivity requires columns: {missing}")
    times = np.asarray(
        time_h if time_h is not None else np.array([0.083, 0.25, 0.5, 1, 2, 4, 8, 12, 24]), dtype=float
    )
    rows: list[dict[str, Any]] = []
    for compound in parameters.itertuples(index=False):
        for hypothesis, multiplier in DISTRIBUTION_HYPOTHESES.items():
            vd = float(compound.vdss_l_kg) * multiplier
            concentration = one_compartment_iv_profile(
                times,
                dose_mg_kg=float(compound.iv_dose_mg_kg),
                clearance_ml_kg_min=float(compound.clearance_ml_kg_min),
                vdss_l_kg=vd,
            )
            for time, value in zip(times, concentration, strict=True):
                rows.append(
                    {
                        "compound_id": compound.compound_id,
                        "distribution_hypothesis": hypothesis,
                        "vdss_multiplier": multiplier,
                        "time_h": float(time),
                        "predicted_concentration_ng_ml": float(value),
                        "calibration_status": "sensitivity_only_no_raw_profiles",
                    }
                )
    return pd.DataFrame(rows)


def pk_identifiability_contract() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "quantity": "systemic_clearance_or_dose_normalized_iv_auc",
                "current_status": "one_independent_model_target_only",
                "reason": "CL is algebraically determined by IV dose/AUC within reported precision",
                "required_new_data": "raw IV profiles plus dose/formulation/animal metadata",
            },
            {
                "quantity": "Fa_Fg_Fh",
                "current_status": "not_separately_identifiable",
                "reason": "summary oral/IV exposure identifies only their product",
                "required_new_data": "permeability, solubility, gut/liver metabolism and profiles",
            },
            {
                "quantity": "distribution_model",
                "current_status": "structural_uncertainty_ensemble",
                "reason": "summary Vdss cannot distinguish tissue-partition hypotheses",
                "required_new_data": "serial plasma plus tissue or blood partition evidence",
            },
            {
                "quantity": "neural_ode_profile",
                "current_status": "not_fit",
                "reason": "no concentration-time observations are available",
                "required_new_data": "per-animal concentration-time rows with LLOQ/censoring",
            },
        ]
    )


def derive_pk_prediction_views(predictions: pd.DataFrame) -> pd.DataFrame:
    """Append CL and F as algebraic views of independently modeled exposure.

    No reported clearance or bioavailability value is fitted here.  Interval
    bounds use conservative endpoint-wise combinations and therefore do not
    imply that the covariance between the IV and PO models is identified.
    """

    if predictions.empty:
        return predictions.copy()
    required = {"compound_id", "endpoint", "mean", "lower", "upper"}
    if missing := sorted(required - set(predictions.columns)):
        raise ValueError(f"PK prediction views require columns: {missing}")
    result = predictions.copy()
    views: list[pd.DataFrame] = []
    iv = result[result["endpoint"] == "rat_iv_auc_dose_normalized"].copy()
    if not iv.empty:
        conversion = 1_000_000.0 / 60.0
        clearance = iv.copy()
        clearance["endpoint"] = "rat_iv_clearance_ml_kg_min"
        clearance["mean"] = conversion / iv["mean"]
        clearance["lower"] = conversion / iv["upper"]
        clearance["upper"] = conversion / iv["lower"]
        clearance["uncertainty"] = (clearance["upper"] - clearance["lower"]) / 2.0
        clearance["unit"] = "mL/kg/min"
        clearance["model_name"] = clearance["model_name"].astype(str) + "+algebraic_closure"
        clearance["estimate_semantics"] = "dose divided by predicted IV AUC; not an independent label"
        clearance["lineage_role"] = "derived_from_rat_iv_auc_dose_normalized"
        views.append(clearance)

    po = result[result["endpoint"] == "rat_po_auc_dose_normalized"].copy()
    if not iv.empty and not po.empty:
        columns = ["compound_id", "mean", "lower", "upper"]
        joined = po.merge(iv[columns], on="compound_id", suffixes=("_po", "_iv"), validate="one_to_one")
        bioavailability = po.set_index("compound_id").loc[joined["compound_id"]].reset_index()
        bioavailability["endpoint"] = "rat_bioavailability_closure_percent"
        bioavailability["mean"] = 100.0 * joined["mean_po"].to_numpy() / joined["mean_iv"].to_numpy()
        bioavailability["lower"] = 100.0 * joined["lower_po"].to_numpy() / joined["upper_iv"].to_numpy()
        bioavailability["upper"] = 100.0 * joined["upper_po"].to_numpy() / joined["lower_iv"].to_numpy()
        bioavailability["uncertainty"] = (bioavailability["upper"] - bioavailability["lower"]) / 2.0
        bioavailability["unit"] = "%"
        bioavailability["model_name"] = (
            bioavailability["model_name"].astype(str) + "+iv_auc_model+algebraic_closure"
        )
        bioavailability["estimate_semantics"] = (
            "100 * predicted dose-normalized PO AUC / predicted dose-normalized IV AUC; "
            "not an independent Fa, Fg, or Fh estimate"
        )
        bioavailability["lineage_role"] = (
            "derived_from_rat_po_and_iv_auc_dose_normalized_without_covariance_model"
        )
        if "domain_status" in bioavailability and "domain_status" in iv:
            iv_domain = iv.set_index("compound_id")["domain_status"].astype(str)
            bioavailability["domain_status"] = np.where(
                bioavailability["domain_status"].astype(str).eq("inside")
                & bioavailability["compound_id"].map(iv_domain).eq("inside"),
                "inside",
                "outside",
            )
        views.append(bioavailability)
    return pd.concat([result, *views], ignore_index=True, sort=False) if views else result
