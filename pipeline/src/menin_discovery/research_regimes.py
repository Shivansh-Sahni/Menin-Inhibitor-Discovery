"""Mechanism-aware molecular-weight interaction and change-point diagnostics.

The analysis deliberately treats molecular weight as a continuous variable first.
A segmented model is allowed to nominate a boundary only when it improves on a
continuous model that already contains MW-by-mechanism interactions.  A boundary
is never promoted from one endpoint alone.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

# One usable variable is selected from every available physical family.  Explicit
# covariates supplied by a caller are retained, then gaps are filled from this
# ordered list.  Rich ensemble observables precede their 2D proxies.
_MECHANISTIC_COVARIATE_FAMILIES: dict[str, tuple[str, ...]] = {
    "flexibility": (
        "joint_conformational_entropy_normalized",
        "effective_joint_state_conformer_count",
        "rotatable_bonds",
    ),
    "charge_state_behavior": (
        "charge_separation_per_gyration_candidate",
        "charge_centroid_separation_angstrom__mean",
        "gasteiger_dipole_proxy_debye__mean",
        "formal_charge",
    ),
    "exposed_polarity": (
        "exposure_adjusted_hbond_burden",
        "polar_sasa_ang2__mean",
        "sa_3d_psa_ang2__mean",
        "exposed_hbd_sasa_ang2__mean",
        "exposed_hba_sasa_ang2__mean",
        "tpsa",
    ),
    "folding": (
        "folded_low_polarity_fraction",
        "intramolecular_shielding_candidate",
        "radius_of_gyration_angstrom__mean",
        "imhb_count_proxy__mean",
        "npr1__mean",
        "npr2__mean",
    ),
}
_LOCKED_MINIMUM_PER_SIDE = 15


def _fit_linear(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    design = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    rss = max(float(np.sum(residual**2)), 1e-12)
    bic = len(y) * np.log(rss / len(y)) + design.shape[1] * np.log(len(y))
    return beta, rss, float(bic)


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.exp(values - maximum).sum()))


def _numeric_nonconstant(data: pd.DataFrame, column: str, *, minimum_coverage: float) -> bool:
    values = pd.to_numeric(data[column], errors="coerce")
    if float(values.notna().mean()) < minimum_coverage:
        return False
    return bool(values.nunique(dropna=True) > 1)


def _mechanistic_family(column: str) -> str | None:
    for family, candidates in _MECHANISTIC_COVARIATE_FAMILIES.items():
        if column in candidates:
            return family
    return None


def _resolve_mechanistic_covariates(
    data: pd.DataFrame,
    requested: Iterable[str],
    *,
    minimum_coverage: float,
) -> tuple[list[str], dict[str, str]]:
    """Choose one best available variable per physical family plus custom terms.

    Standard 2D variables requested by the caller remain fallbacks.  When an
    ensemble observable from the same family is available, it supersedes the 2D
    proxy for this mechanistic analysis instead of adding a collinear duplicate.
    """

    resolved: list[str] = []
    families: dict[str, str] = {}
    for column in requested:
        if (
            _mechanistic_family(column) is None
            and column in data
            and column not in resolved
            and _numeric_nonconstant(data, column, minimum_coverage=minimum_coverage)
        ):
            resolved.append(column)
            families[column] = "caller_supplied"

    for family, candidates in _MECHANISTIC_COVARIATE_FAMILIES.items():
        for column in candidates:
            if column in data and _numeric_nonconstant(data, column, minimum_coverage=minimum_coverage):
                resolved.append(column)
                families[column] = family
                break
    return resolved, families


def _standardized_covariates(
    frame: pd.DataFrame, covariates: list[str]
) -> tuple[np.ndarray, dict[str, float]]:
    if not covariates:
        return np.empty((len(frame), 0)), {}
    columns: list[np.ndarray] = []
    missing_fractions: dict[str, float] = {}
    for column in covariates:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        missing_fractions[column] = float(np.mean(~np.isfinite(values)))
        finite = values[np.isfinite(values)]
        median = float(np.median(finite)) if finite.size else 0.0
        values = np.where(np.isfinite(values), values, median)
        scale = max(float(np.std(values)), 1e-9)
        columns.append((values - float(np.mean(values))) / scale)
    return np.column_stack(columns), missing_fractions


def _continuous_design(mw: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """MW, physical main effects, and every available MW-by-physics interaction."""

    centered_mw = (mw - np.mean(mw)) / max(float(np.std(mw)), 1e-9)
    if not covariates.size:
        return centered_mw[:, None]
    return np.column_stack([centered_mw, covariates, centered_mw[:, None] * covariates])


def _candidate_design(
    mw: np.ndarray,
    threshold: float,
    covariates: np.ndarray,
) -> np.ndarray:
    continuous = _continuous_design(mw, covariates)
    hinge = np.maximum(mw - threshold, 0.0) / max(float(np.std(mw)), 1e-9)
    return np.column_stack([continuous, hinge])


def _best_breakpoint(
    frame: pd.DataFrame,
    *,
    outcome: str,
    mw_column: str,
    covariates: list[str],
    candidates: np.ndarray,
    minimum_per_side: int,
) -> dict[str, Any]:
    y = frame[outcome].to_numpy(dtype=float)
    mw = frame[mw_column].to_numpy(dtype=float)
    if len(frame) < 2 * minimum_per_side or np.std(mw) <= 1e-9:
        return {
            "selected": False,
            "reason": "insufficient_observations_or_mw_variation",
            "baseline_bic": float("nan"),
            "candidate_count": 0,
        }
    cov, missing_fractions = _standardized_covariates(frame, covariates)
    _, _, baseline_bic = _fit_linear(_continuous_design(mw, cov), y)
    rows: list[dict[str, Any]] = []
    for threshold in candidates:
        n_left = int(np.sum(mw < threshold))
        n_right = int(np.sum(mw >= threshold))
        if n_left < minimum_per_side or n_right < minimum_per_side:
            continue
        beta, rss, bic = _fit_linear(_candidate_design(mw, float(threshold), cov), y)
        rows.append(
            {
                "threshold": float(threshold),
                "bic": bic,
                "rss": rss,
                "delta_bic_vs_continuous": float(baseline_bic - bic),
                "hinge_effect": float(beta[-1]),
                "n_left": n_left,
                "n_right": n_right,
            }
        )
    if not rows:
        return {
            "selected": False,
            "reason": "no_candidate_has_minimum_per_side",
            "baseline_bic": baseline_bic,
            "candidate_count": 0,
            "covariate_missing_fractions": missing_fractions,
        }

    # Equal prior mass is assigned to the continuous and segmented families;
    # candidate thresholds share the segmented-family mass.  This BIC-derived
    # probability is an explicitly approximate Bayesian diagnostic, not a full
    # posterior change-point model.
    breakpoint_log_evidence = _logsumexp(-0.5 * np.asarray([row["bic"] for row in rows])) - np.log(len(rows))
    continuous_log_evidence = -0.5 * baseline_bic
    normalizer = _logsumexp(np.asarray([continuous_log_evidence, breakpoint_log_evidence]))
    breakpoint_probability = float(np.exp(breakpoint_log_evidence - normalizer))
    threshold_normalizer = _logsumexp(-0.5 * np.asarray([row["bic"] for row in rows]))
    for row in rows:
        row["threshold_weight_within_segmented_family"] = float(
            np.exp(-0.5 * row["bic"] - threshold_normalizer)
        )

    best = min(rows, key=lambda row: (row["bic"], row["threshold"]))
    best["breakpoint_model_probability"] = breakpoint_probability
    best["selected"] = bool(best["delta_bic_vs_continuous"] >= 2.0 and breakpoint_probability >= 0.50)
    best["reason"] = "segmented_evidence" if best["selected"] else "continuous_interaction_model_not_worse"
    best["baseline_bic"] = baseline_bic
    best["candidate_count"] = len(rows)
    best["covariate_missing_fractions"] = missing_fractions
    return best


def bootstrap_mw_change_point(
    data: pd.DataFrame,
    *,
    outcome: str,
    mw_column: str = "mw",
    group_column: str = "scaffold",
    covariates: list[str] | None = None,
    candidate_min: float = 650.0,
    candidate_max: float = 780.0,
    candidate_step: float = 5.0,
    minimum_per_side: int = 15,
    bootstrap_replicates: int = 500,
    random_state: int = 20260721,
    minimum_covariate_coverage: float = 0.60,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Test a breakpoint against a continuous interaction model under scaffold resampling."""

    if candidate_step <= 0 or candidate_max < candidate_min:
        raise ValueError("MW candidates require candidate_step > 0 and candidate_max >= candidate_min")
    if minimum_per_side < 1 or bootstrap_replicates < 1:
        raise ValueError("minimum_per_side and bootstrap_replicates must both be positive")
    requested_minimum_per_side = int(minimum_per_side)
    minimum_per_side = max(_LOCKED_MINIMUM_PER_SIDE, requested_minimum_per_side)
    required = {outcome, mw_column, group_column}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"MW regime data are missing columns: {missing}")

    frame = data.copy()
    frame[outcome] = pd.to_numeric(frame[outcome], errors="coerce")
    frame[mw_column] = pd.to_numeric(frame[mw_column], errors="coerce")
    frame = frame.dropna(subset=[outcome, mw_column, group_column]).copy()
    resolved_covariates, covariate_families = _resolve_mechanistic_covariates(
        frame,
        covariates or [],
        minimum_coverage=minimum_covariate_coverage,
    )
    candidates = np.arange(candidate_min, candidate_max + candidate_step / 2.0, candidate_step)
    observed = _best_breakpoint(
        frame,
        outcome=outcome,
        mw_column=mw_column,
        covariates=resolved_covariates,
        candidates=candidates,
        minimum_per_side=minimum_per_side,
    )

    rng = np.random.default_rng(random_state)
    group_values = frame[group_column].astype(str)
    groups = group_values.unique()
    rows: list[dict[str, Any]] = []
    if len(groups):
        for replicate in range(bootstrap_replicates):
            sampled_groups = rng.choice(groups, size=len(groups), replace=True)
            parts: list[pd.DataFrame] = []
            for draw_index, group in enumerate(sampled_groups):
                part = frame.loc[group_values == group].copy()
                part[group_column] = f"{group}__bootstrap_{draw_index}"
                parts.append(part)
            sampled = pd.concat(parts, ignore_index=True)
            result = _best_breakpoint(
                sampled,
                outcome=outcome,
                mw_column=mw_column,
                covariates=resolved_covariates,
                candidates=candidates,
                minimum_per_side=minimum_per_side,
            )
            rows.append({"replicate": replicate, **result})
    bootstrap = pd.DataFrame(rows)
    if bootstrap.empty or "selected" not in bootstrap:
        selected = pd.DataFrame()
        valid_candidate_frequency = 0.0
    else:
        selected = bootstrap[bootstrap["selected"].fillna(False).astype(bool)]
        valid_candidate_frequency = float(np.mean(bootstrap.get("candidate_count", 0) > 0))
    selection_frequency = float(len(selected) / bootstrap_replicates)
    if selected.empty:
        interval = (float("nan"), float("nan"))
        direction_stability = 0.0
        positive_effect_frequency = float("nan")
    else:
        quantiles = np.quantile(selected["threshold"], [0.025, 0.975])
        interval = (float(quantiles[0]), float(quantiles[1]))
        observed_effect = float(observed.get("hinge_effect", float("nan")))
        observed_sign = np.sign(observed_effect) if np.isfinite(observed_effect) else 0.0
        selected_signs = np.sign(selected["hinge_effect"].to_numpy(dtype=float))
        direction_stability = float(np.mean(selected_signs == observed_sign)) if observed_sign else 0.0
        positive_effect_frequency = float(np.mean(selected_signs > 0))

    summary = {
        "outcome": outcome,
        "n": int(len(frame)),
        "n_scaffolds": int(len(groups)),
        "observed_selected": bool(observed.get("selected", False)),
        "observed_reason": observed.get("reason"),
        "observed_threshold_da": observed.get("threshold"),
        "observed_delta_bic": observed.get("delta_bic_vs_continuous"),
        "observed_breakpoint_model_probability": observed.get("breakpoint_model_probability"),
        "observed_hinge_effect": observed.get("hinge_effect"),
        "observed_n_left": observed.get("n_left", 0),
        "observed_n_right": observed.get("n_right", 0),
        "continuous_covariates": resolved_covariates,
        "mw_interaction_terms": [f"mw_x_{column}" for column in resolved_covariates],
        "covariate_families": covariate_families,
        "covariate_missing_fractions": observed.get("covariate_missing_fractions", {}),
        "bootstrap_breakpoint_selection_frequency": selection_frequency,
        "bootstrap_valid_candidate_frequency": valid_candidate_frequency,
        "location_interval_95_low_da": interval[0],
        "location_interval_95_high_da": interval[1],
        "location_interval_width_da": interval[1] - interval[0]
        if np.all(np.isfinite(interval))
        else float("nan"),
        "effect_direction_stability": direction_stability,
        "positive_hinge_effect_frequency": positive_effect_frequency,
        "candidate_range_da": [float(candidate_min), float(candidate_max)],
        "minimum_per_side": int(minimum_per_side),
        "requested_minimum_per_side": requested_minimum_per_side,
        "interpretation": "candidate_only_pending_cross_outcome_gate",
    }
    return summary, bootstrap


def _outcome_process(outcome: str) -> str:
    normalized = outcome.casefold().replace("log10_", "")
    if "herg" in normalized and ("inhibition" in normalized or "current" in normalized):
        return "herg_inhibition"
    if "herg" in normalized and any(
        token in normalized for token in ("onset", "recovery", "trapping", "kinetic")
    ):
        return "herg_kinetics"
    if "herg" in normalized or "pic50" in normalized:
        return "herg_potency"
    if "vdss" in normalized or "volume" in normalized:
        return "distribution"
    if any(token in normalized for token in ("po_auc", "cmax", "tmax", "oral")):
        return "oral_exposure"
    if any(token in normalized for token in ("iv_auc", "clearance", "_cl", "cl_")):
        return "systemic_disposition"
    return "unclassified"


def _mechanistically_adjacent(left: pd.Series, right: pd.Series) -> bool:
    left_value = left.get("mechanistic_family", "")
    right_value = right.get("mechanistic_family", "")
    left_family = "" if pd.isna(left_value) else str(left_value).strip()
    right_family = "" if pd.isna(right_value) else str(right_value).strip()
    if left_family and right_family:
        return left_family == right_family
    left_process = _outcome_process(str(left["outcome"]))
    right_process = _outcome_process(str(right["outcome"]))
    if left_process == right_process and left_process != "unclassified":
        return True
    adjacent = {
        frozenset(("systemic_disposition", "oral_exposure")),
        frozenset(("systemic_disposition", "distribution")),
        frozenset(("oral_exposure", "distribution")),
        frozenset(("herg_potency", "herg_inhibition")),
        frozenset(("herg_potency", "herg_kinetics")),
        frozenset(("herg_inhibition", "herg_kinetics")),
    }
    return frozenset((left_process, right_process)) in adjacent


def apply_cross_outcome_cutoff_gate(
    summaries: list[dict[str, Any]],
    *,
    minimum_selection_frequency: float = 0.70,
    maximum_location_width_da: float = 50.0,
    minimum_direction_stability: float = 0.70,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Require all locked stability gates in at least two adjacent outcomes."""

    minimum_selection_frequency = max(0.70, float(minimum_selection_frequency))
    maximum_location_width_da = min(50.0, float(maximum_location_width_da))
    minimum_direction_stability = max(0.70, float(minimum_direction_stability))

    frame = pd.DataFrame(summaries)
    if frame.empty:
        return frame, {
            "supported_cutoff": False,
            "supported_cutoff_da": None,
            "reason": "No outcomes were available; no defensible single MW cutoff exists.",
        }

    numeric_defaults = {
        "bootstrap_breakpoint_selection_frequency": 0.0,
        "location_interval_width_da": float("nan"),
        "effect_direction_stability": 0.0,
        "observed_n_left": 0,
        "observed_n_right": 0,
        "minimum_per_side": 15,
        "location_interval_95_low_da": float("nan"),
        "location_interval_95_high_da": float("nan"),
    }
    for column, default in numeric_defaults.items():
        if column not in frame:
            frame[column] = default
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if np.isfinite(default):
            frame[column] = frame[column].fillna(default)
    if "observed_selected" not in frame:
        frame["observed_selected"] = False
    frame["observed_selected"] = frame["observed_selected"].fillna(False).astype(bool)
    finite_interval = np.isfinite(pd.to_numeric(frame["location_interval_width_da"], errors="coerce"))
    finite_bounds = np.isfinite(frame["location_interval_95_low_da"]) & np.isfinite(
        frame["location_interval_95_high_da"]
    )
    side_requirement = pd.to_numeric(frame["minimum_per_side"], errors="coerce").fillna(
        _LOCKED_MINIMUM_PER_SIDE
    )
    side_requirement = side_requirement.clip(lower=_LOCKED_MINIMUM_PER_SIDE)
    frame["passes_within_outcome_gate"] = (
        frame["observed_selected"]
        & (frame["bootstrap_breakpoint_selection_frequency"] >= minimum_selection_frequency)
        & finite_interval
        & finite_bounds
        & (frame["location_interval_width_da"] <= maximum_location_width_da)
        & (frame["effect_direction_stability"] >= minimum_direction_stability)
        & (frame["observed_n_left"] >= side_requirement)
        & (frame["observed_n_right"] >= side_requirement)
    )

    failure_rows: list[str] = []
    for _, row in frame.iterrows():
        failures: list[str] = []
        if not bool(row["observed_selected"]):
            failures.append("observed_segmented_model_not_selected")
        if float(row["bootstrap_breakpoint_selection_frequency"]) < minimum_selection_frequency:
            failures.append("selection_frequency_below_70_percent")
        width = float(row["location_interval_width_da"])
        if not np.isfinite(width) or width > maximum_location_width_da:
            failures.append("location_interval_too_wide_or_unidentified")
        if float(row["effect_direction_stability"]) < minimum_direction_stability:
            failures.append("effect_direction_unstable")
        row_side_requirement = max(_LOCKED_MINIMUM_PER_SIDE, int(row["minimum_per_side"]))
        if (
            int(row["observed_n_left"]) < row_side_requirement
            or int(row["observed_n_right"]) < row_side_requirement
        ):
            failures.append("fewer_than_15_observations_on_one_side")
        failure_rows.append(";".join(failures))
    frame["gate_failures"] = failure_rows

    passing = frame[frame["passes_within_outcome_gate"]]
    pairs: list[dict[str, Any]] = []
    passing_indices = list(passing.index)
    for position, left_index in enumerate(passing_indices):
        left = passing.loc[left_index]
        for right_index in passing_indices[position + 1 :]:
            right = passing.loc[right_index]
            low = max(float(left["location_interval_95_low_da"]), float(right["location_interval_95_low_da"]))
            high = min(
                float(left["location_interval_95_high_da"]), float(right["location_interval_95_high_da"])
            )
            if low <= high and _mechanistically_adjacent(left, right):
                pairs.append(
                    {
                        "outcomes": [str(left["outcome"]), str(right["outcome"])],
                        "overlap_interval_da": [low, high],
                        "minimum_selection_frequency": float(
                            min(
                                left["bootstrap_breakpoint_selection_frequency"],
                                right["bootstrap_breakpoint_selection_frequency"],
                            )
                        ),
                    }
                )

    pairs.sort(
        key=lambda pair: (
            -pair["minimum_selection_frequency"],
            pair["overlap_interval_da"][1] - pair["overlap_interval_da"][0],
            pair["outcomes"],
        )
    )
    supported = bool(pairs)
    result: dict[str, Any] = {
        "supported_cutoff": supported,
        "supported_cutoff_da": None,
        "supporting_outcome_pairs": [pair["outcomes"] for pair in pairs],
        "locked_gates": {
            "minimum_selection_frequency": minimum_selection_frequency,
            "maximum_location_interval_width_da": maximum_location_width_da,
            "minimum_direction_stability": minimum_direction_stability,
            "minimum_observations_per_side": _LOCKED_MINIMUM_PER_SIDE,
            "minimum_mechanistically_adjacent_outcomes": 2,
        },
        "reason": (
            "At least two mechanistically adjacent outcomes passed every stability gate with overlapping breakpoint intervals."
            if supported
            else "No defensible single MW cutoff exists; retain MW continuously and define regimes from physical-state variables."
        ),
    }
    if supported:
        result["supported_interval_da"] = pairs[0]["overlap_interval_da"]
        result["supported_cutoff_da"] = float(np.mean(pairs[0]["overlap_interval_da"]))
    return frame, result
