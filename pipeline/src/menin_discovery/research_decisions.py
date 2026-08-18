"""Mechanistic assay-panel selection and optimizer-facing endpoint contracts.

The panel score is intentionally described as a heuristic acquisition priority,
not expected information gain: the present data do not identify an observation
noise model or posterior utility function from which information gain could be
calculated.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def mw_bin(value: float) -> str:
    if value < 700:
        return "650-699"
    if value < 750:
        return "700-749"
    return "750+"


def _fingerprints(smiles: pd.Series):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    result = []
    for value in smiles.fillna("").astype(str):
        mol = Chem.MolFromSmiles(value)
        result.append(generator.GetFingerprint(mol) if mol is not None else None)
    return result


def _pair_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    fps = _fingerprints(frame["standardized_smiles"])
    rows: list[dict[str, Any]] = []
    for i in range(len(frame)):
        if fps[i] is None:
            continue
        for j in range(i + 1, len(frame)):
            if fps[j] is None:
                continue
            similarity = float(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            if similarity < 0.55:
                continue
            herg_a, herg_b = frame.iloc[i].get("herg_class"), frame.iloc[j].get("herg_class")
            pk_a, pk_b = frame.iloc[i].get("pk_extreme"), frame.iloc[j].get("pk_extreme")
            discordance = float(pd.notna(herg_a) and pd.notna(herg_b) and herg_a != herg_b)
            discordance += float(pd.notna(pk_a) and pd.notna(pk_b) and pk_a != pk_b)
            rows.append(
                {
                    "compound_id_a": frame.iloc[i]["compound_id"],
                    "compound_id_b": frame.iloc[j]["compound_id"],
                    "tanimoto": similarity,
                    "observed_discordance": discordance,
                    "pair_priority": similarity + 0.35 * discordance,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["compound_id_a", "compound_id_b", "tanimoto", "observed_discordance", "pair_priority"]
        )
    return (
        pd.DataFrame(rows).sort_values(["pair_priority", "tanimoto"], ascending=False).reset_index(drop=True)
    )


def _diversity_matrix(frame: pd.DataFrame) -> np.ndarray:
    numeric_candidates = [
        column
        for column in (
            "mw",
            "tpsa",
            "logp",
            "rotatable_bonds",
            "formal_charge",
            "herg_pic50",
            "rat_cl_ml_kg_min",
            "rat_vdss_l_kg",
            "rat_po_auc_dose_normalized",
            "physics_folded_fraction",
            "physics_exposed_polarity",
            "physics_conformer_entropy",
        )
        if column in frame
    ]
    if not numeric_candidates:
        return np.arange(len(frame), dtype=float).reshape(-1, 1)
    values = SimpleImputer().fit_transform(frame[numeric_candidates])
    return StandardScaler().fit_transform(values)


def _greedy_maximin(values: np.ndarray, candidates: list[int], selected: list[int]) -> int:
    if not selected:
        return max(candidates, key=lambda index: float(np.linalg.norm(values[index])))
    return max(
        candidates,
        key=lambda index: float(min(np.linalg.norm(values[index] - values[chosen]) for chosen in selected)),
    )


def select_assay_panel(
    compounds: pd.DataFrame,
    *,
    panel_size: int = 16,
    mw_bin_minimums: dict[str, int] | None = None,
    minimum_matched_pairs: int = 4,
    complete_rat_profiles: int = 8,
    state_dependent_herg: int = 6,
    herg_class_minimums: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select a quota-constrained, mechanism-diverse panel without ranking molecules for optimization."""

    required = {"compound_id", "standardized_smiles", "mw"}
    missing = sorted(required - set(compounds.columns))
    if missing:
        raise ValueError(f"Assay selection is missing columns: {missing}")
    frame = compounds.drop_duplicates("compound_id").copy().reset_index(drop=True)
    frame = frame[frame["mw"].astype(float) >= 650].reset_index(drop=True)
    frame["mw_bin"] = frame["mw"].astype(float).map(mw_bin)
    supplied_herg_class = frame.get("herg_class", pd.Series(pd.NA, index=frame.index)).astype("string")
    if "herg_pic50" in frame:
        pic50 = pd.to_numeric(frame["herg_pic50"], errors="coerce")
        calculated_class = np.select(
            [
                pic50.ge(5.0).fillna(False).to_numpy(dtype=bool),
                pic50.le(6.0 - np.log10(30.0)).fillna(False).to_numpy(dtype=bool),
            ],
            ["blocker", "nonblocker"],
            default="intermediate_or_missing",
        )
        frame["herg_class"] = supplied_herg_class.fillna(pd.Series(calculated_class, index=frame.index))
    else:
        frame["herg_class"] = supplied_herg_class.fillna("intermediate_or_missing")
    if "rat_cl_ml_kg_min" in frame:
        clearance = pd.to_numeric(frame["rat_cl_ml_kg_min"], errors="coerce")
        low, high = clearance.quantile([0.2, 0.8])
        frame["pk_extreme"] = np.select(
            [
                clearance.le(low).fillna(False).to_numpy(dtype=bool),
                clearance.ge(high).fillna(False).to_numpy(dtype=bool),
            ],
            ["low_clearance", "high_clearance"],
            default="middle_or_missing",
        )
    else:
        frame["pk_extreme"] = "missing"

    measurement_columns = [
        column
        for column in (
            "solubility_ph2",
            "solubility_ph5",
            "solubility_ph6p5",
            "solubility_ph7p4",
            "pampa",
            "caco2_ab",
            "caco2_ba",
            "rat_clint",
            "rat_fu",
            "blood_plasma_ratio",
            "herg_pic50",
            "rat_cl_ml_kg_min",
            "rat_po_auc_dose_normalized",
        )
        if column in frame
    ]
    missing_fraction = (
        frame[measurement_columns].isna().mean(axis=1)
        if measurement_columns
        else pd.Series(1.0, index=frame.index)
    )
    uncertainty = frame.get("model_uncertainty", pd.Series(0.5, index=frame.index)).fillna(0.5).astype(float)
    intermediate = frame["herg_class"].isin(["intermediate_or_missing", "missing"]).astype(float)
    frame["missing_process_data_fraction"] = missing_fraction
    frame["uncertainty_percentile"] = uncertainty.rank(pct=True)
    frame["intermediate_or_missing_herg_indicator"] = intermediate
    frame["acquisition_priority_score"] = (
        0.45 * frame["missing_process_data_fraction"]
        + 0.35 * frame["uncertainty_percentile"]
        + 0.20 * frame["intermediate_or_missing_herg_indicator"]
    )
    frame["acquisition_score_definition"] = (
        "heuristic_0.45_missingness_plus_0.35_uncertainty_rank_plus_0.20_herg_gap"
    )
    frame["evaluation_role"] = "mechanistic_assay_design_not_unbiased_prospective_model_test"

    pairs = _pair_candidates(frame)
    selected: list[int] = []
    id_to_index = {str(value): index for index, value in enumerate(frame["compound_id"])}
    pair_ids: dict[str, list[str]] = defaultdict(list)
    used: set[str] = set()
    accepted_pairs = 0
    for row in pairs.itertuples(index=False):
        left, right = str(row.compound_id_a), str(row.compound_id_b)
        if left in used or right in used or len(selected) + 2 > panel_size:
            continue
        selected.extend([id_to_index[left], id_to_index[right]])
        used.update([left, right])
        pair_name = f"MP-{accepted_pairs + 1:02d}"
        pair_ids[left].append(pair_name)
        pair_ids[right].append(pair_name)
        accepted_pairs += 1
        if accepted_pairs >= minimum_matched_pairs:
            break

    values = _diversity_matrix(frame)
    quotas = mw_bin_minimums or {"650-699": 3, "700-749": 6, "750+": 3}
    for bin_name, minimum in quotas.items():
        while sum(frame.iloc[index]["mw_bin"] == bin_name for index in selected) < minimum:
            candidates = [
                index
                for index in frame.index
                if index not in selected and frame.loc[index, "mw_bin"] == bin_name
            ]
            if not candidates or len(selected) >= panel_size:
                break
            chosen = _greedy_maximin(values, candidates, selected)
            selected.append(chosen)
    requested_herg_quotas = herg_class_minimums or {}
    for class_name, minimum in requested_herg_quotas.items():
        while sum(frame.iloc[index]["herg_class"] == class_name for index in selected) < minimum:
            candidates = [
                index
                for index in frame.index
                if index not in selected and frame.loc[index, "herg_class"] == class_name
            ]
            if not candidates or len(selected) >= panel_size:
                break
            selected.append(_greedy_maximin(values, candidates, selected))
    while len(selected) < min(panel_size, len(frame)):
        candidates = [index for index in frame.index if index not in selected]
        if not candidates:
            break
        # Information gain and distance both matter; distance is normalized by
        # comparing ranks to avoid domination by a single descriptor scale.
        distance_choice = _greedy_maximin(values, candidates, selected)
        score_choice = max(
            candidates,
            key=lambda index: float(frame.loc[index, "acquisition_priority_score"]),
        )
        chosen = (
            score_choice
            if frame.loc[score_choice, "acquisition_priority_score"]
            >= frame.loc[distance_choice, "acquisition_priority_score"]
            else distance_choice
        )
        selected.append(chosen)

    panel = frame.iloc[selected].copy()
    panel["matched_pair_ids"] = (
        panel["compound_id"].astype(str).map(lambda value: ";".join(pair_ids.get(value, [])))
    )
    panel["panel_priority"] = (
        panel["acquisition_priority_score"].rank(method="first", ascending=False).astype(int)
    )
    panel = panel.sort_values("panel_priority").reset_index(drop=True)

    pk_score = panel["acquisition_priority_score"].copy()
    pk_score += panel["pk_extreme"].isin(["low_clearance", "high_clearance"]).astype(float) * 0.5
    pk_profiles = panel.assign(selection_score=pk_score).nlargest(
        min(complete_rat_profiles, len(panel)), "selection_score"
    )
    pk_profiles = pk_profiles[
        ["compound_id", "selection_score", "mw_bin", "pk_extreme", "matched_pair_ids"]
    ].copy()
    pk_profiles["requested_profile"] = "complete rat IV/PO concentration-time profile"

    herg_parts: list[pd.DataFrame] = []
    quotas_herg = herg_class_minimums or {
        "blocker": 0,
        "nonblocker": 0,
        "intermediate_or_missing": state_dependent_herg,
    }
    for class_name, count in quotas_herg.items():
        candidates = panel[panel["herg_class"] == class_name].nlargest(
            count,
            "acquisition_priority_score",
        )
        herg_parts.append(candidates)
    herg_protocol = (
        pd.concat(herg_parts, ignore_index=True).drop_duplicates("compound_id")
        if herg_parts
        else panel.head(0)
    )
    if len(herg_protocol) < min(state_dependent_herg, len(panel)):
        remaining = panel[~panel["compound_id"].isin(herg_protocol["compound_id"])].nlargest(
            min(state_dependent_herg, len(panel)) - len(herg_protocol),
            "acquisition_priority_score",
        )
        herg_protocol = pd.concat([herg_protocol, remaining], ignore_index=True)
    herg_protocol = herg_protocol.head(state_dependent_herg)[
        [
            "compound_id",
            "acquisition_priority_score",
            "mw_bin",
            "herg_class",
            "matched_pair_ids",
            "evaluation_role",
        ]
    ].copy()
    observed_herg_counts = herg_protocol["herg_class"].value_counts()
    missing_herg_quotas = {
        class_name: count - int(observed_herg_counts.get(class_name, 0))
        for class_name, count in quotas_herg.items()
        if int(observed_herg_counts.get(class_name, 0)) < count
    }
    if herg_class_minimums and missing_herg_quotas:
        raise ValueError(f"Cannot satisfy state-dependent hERG class quotas: {missing_herg_quotas}")
    herg_protocol["requested_protocol"] = "state-dependent hERG onset/recovery/trapping"

    selected_ids = set(panel["compound_id"].astype(str))
    selected_pairs = (
        pairs[
            pairs["compound_id_a"].astype(str).isin(selected_ids)
            & pairs["compound_id_b"].astype(str).isin(selected_ids)
        ]
        .head(max(minimum_matched_pairs, 1))
        .copy()
    )
    return panel, pk_profiles, herg_protocol, selected_pairs


ASSAY_REQUESTS = (
    (
        "thermodynamic_solubility",
        "pH 2.0; pH 5.0; pH 6.5; pH 7.4",
        "kinetic and thermodynamic with equilibration time",
    ),
    (
        "passive_permeability",
        "PAMPA or matched liposome",
        "report membrane composition and free concentration",
    ),
    ("bidirectional_permeability", "Caco-2 or MDCK A-to-B/B-to-A", "include transporter-inhibitor controls"),
    ("microsomal_clearance", "rat plus human", "CLint with incubation/protein concentration"),
    ("hepatocyte_clearance", "rat plus human", "CLint and extraction conditions"),
    ("plasma_protein_binding", "rat plus human", "fraction unbound; mass-balance acceptance"),
    ("blood_plasma_ratio", "rat", "concentration and equilibration-time series"),
    (
        "herg_patch_clamp",
        "full concentration-response",
        "temperature, pH, voltage protocol, exposure time, nominal and free concentration",
    ),
)


def expand_assay_requests(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for compound in panel.itertuples(index=False):
        for assay, conditions, reporting in ASSAY_REQUESTS:
            rows.append(
                {
                    "compound_id": compound.compound_id,
                    "panel_priority": compound.panel_priority,
                    "mw_bin": compound.mw_bin,
                    "matched_pair_ids": compound.matched_pair_ids,
                    "assay": assay,
                    "conditions": conditions,
                    "required_reporting": reporting,
                    "rationale": "reduce process-level non-identifiability and model uncertainty",
                }
            )
    return pd.DataFrame(rows)


def build_optimizer_contract(
    compounds: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Expose continuous endpoints and flags; deliberately omit scores and ranks."""

    base = compounds[["compound_id", "standardized_smiles"]].drop_duplicates("compound_id").copy()
    if predictions is not None and not predictions.empty:
        duplicate = predictions.duplicated(["compound_id", "endpoint"], keep=False)
        if duplicate.any():
            raise ValueError("Optimizer predictions must have one row per compound and endpoint")
        wide = predictions.pivot(index="compound_id", columns="endpoint")
        wide.columns = [f"{metric}__{endpoint}" for metric, endpoint in wide.columns]
        base = base.merge(wide.reset_index(), on="compound_id", how="left")
    required_endpoints = (
        "rat_iv_auc_dose_normalized",
        "rat_iv_clearance_ml_kg_min",
        "rat_iv_vdss_l_kg",
        "rat_po_auc_dose_normalized",
        "rat_po_cmax_dose_normalized",
        "rat_po_tmax_h",
        "rat_bioavailability_closure_percent",
        "herg_pic50",
        "herg_blocker_probability",
        "free_exposure_margin",
    )
    observed_endpoints = {column.removeprefix("mean__") for column in base if column.startswith("mean__")}
    for endpoint in sorted(set(required_endpoints) | observed_endpoints):
        for field in ("mean", "lower", "upper", "uncertainty", "domain_status", "promotion_status"):
            column = f"{field}__{endpoint}"
            if column not in base:
                base[column] = (
                    np.nan if field in {"mean", "lower", "upper", "uncertainty"} else "required_data"
                )
        required_column = f"required_data__{endpoint}"
        base[required_column] = base[f"mean__{endpoint}"].isna()
        status_column = f"target_definition_status__{endpoint}"
        eligible_column = f"optimization_eligible__{endpoint}"
        if endpoint == "rat_po_cmax_ng_ml":
            base[status_column] = "retired_mixed_dose_raw_target"
            base[eligible_column] = False
        elif endpoint == "rat_po_cmax_dose_normalized":
            base[status_column] = "discovery_only_pending_dose_proportionality"
            base[eligible_column] = False
        else:
            base[status_column] = "provisional_pending_prospective_validation"
            base[eligible_column] = False
    base["scalar_objective"] = "NOT_DEFINED"
    base["molecule_rank"] = "NOT_COMPUTED"
    base["generation_allowed"] = False
    base["contract_status"] = "BLOCKED_FOR_OPTIMIZATION_PENDING_PROSPECTIVE_VALIDATION"
    return base
