#!/usr/bin/env python3
"""Score new same-series structures with the retained hERG model pair.

Input CSV columns:
  compound_id,smiles

The command refits both retained conventional models on the original internal
hERG evidence plus the 46 measured, non-overlapping Ascentage extension
structures. It reports continuous pIC50, derived blocker probability,
group-held-out uncertainty, nearest-training similarity, and model-form
disagreement. It never treats predictions as experimental measurements.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/menin-prediction-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/menin-prediction-cache")

# isort: off
import numpy as np
import pandas as pd
from menin_discovery.chemistry import standardize_smiles
from menin_discovery.features import scaffold_key
from menin_discovery.research_common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
)
from menin_discovery.research_modeling import (
    final_fit_censored_herg_predictions,
    grouped_censored_herg_benchmark,
    structure_feature_frame,
)
from rdkit import Chem
from scipy.stats import norm
from run_ascentage_complete_feature_model import (
    _fingerprints,
    _fit_score,
    _oof,
)
from run_ascentage_complete_feature_model import (
    OUTPUT as AUDIT_OUTPUT,
)
from run_combined_internal_angelo_herg import _prepare_data as _prepare_clean_combined_data
# isort: on

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "research/reports/pk_herg/new_compound_predictions"
REQUIRED_COLUMNS = {"compound_id", "smiles"}
OPTIONAL_COLUMNS = {
    "series_id",
    "same_series_confirmed",
    "synthesis_status",
    "assay_protocol_id",
}
PROHIBITED_OUTCOME_COLUMNS = {
    "activity",
    "herg_activity",
    "herg_ic50",
    "herg_ic50_relation",
    "herg_ic50_value_um",
    "herg_ic50_um",
    "herg_inhibition",
    "herg_pic50",
    "herg_pic50_relation",
    "herg_pic50_value",
    "herg_result",
    "label",
    "outcome",
}


def _normalized_confirmation(value: object) -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        return "not_provided"
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return "true"
    if normalized in {"0", "false", "no", "n"}:
        return "false"
    raise ValueError("same_series_confirmed must be true/false, yes/no, 1/0, or blank")


def _stereochemistry_audit(smiles: str) -> tuple[int, int, str]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return 0, 0, "invalid"
    potential = list(Chem.FindPotentialStereo(molecule))
    unspecified = sum(item.specified == Chem.StereoSpecified.Unspecified for item in potential)
    specified = sum(item.specified == Chem.StereoSpecified.Specified for item in potential)
    status = (
        "unresolved_potential_stereochemistry"
        if unspecified
        else "fully_specified_or_no_rdkit_detectable_stereochemistry"
    )
    return int(unspecified), int(specified), status


def _standardize_input(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    normalized_columns = {str(column).strip().lower() for column in raw.columns}
    prohibited = sorted(normalized_columns & PROHIBITED_OUTCOME_COLUMNS)
    if prohibited:
        raise ValueError(
            "Prediction input contains outcome columns and is not blind: "
            f"{prohibited}. Supply outcomes only to the later evaluation command."
        )
    missing = sorted(REQUIRED_COLUMNS - set(raw.columns))
    if missing:
        raise ValueError(f"Prediction input is missing columns: {missing}")
    if raw.empty:
        raise ValueError("Prediction input has no compound rows")
    if raw["compound_id"].isna().any() or raw["smiles"].isna().any():
        raise ValueError("compound_id and smiles must be present for every row")
    identifiers = raw["compound_id"].astype(str).str.strip()
    if identifiers.eq("").any():
        raise ValueError("compound_id values must not be blank")
    if identifiers.duplicated().any():
        raise ValueError("compound_id values must be unique")

    rows: list[dict[str, object]] = []
    for record in raw.itertuples(index=False):
        result = standardize_smiles(str(record.smiles), require_rdkit=True)
        if result.structure_valid is not True:
            raise ValueError(f"Invalid structure for {record.compound_id}: {result.structure_error}")
        unspecified_stereo, specified_stereo, stereo_status = _stereochemistry_audit(str(record.smiles))
        optional = {
            column: getattr(record, column)
            if column in raw.columns and not pd.isna(getattr(record, column))
            else ""
            for column in OPTIONAL_COLUMNS - {"same_series_confirmed"}
        }
        rows.append(
            {
                "compound_id": str(record.compound_id).strip(),
                "submitted_smiles": str(record.smiles),
                "standardized_smiles": result.standardized_smiles,
                "structure_id": result.structure_id,
                "standard_inchi_key": result.standard_inchi_key,
                "scaffold": scaffold_key(result.standardized_smiles)[0],
                "standardization_version": result.structure_standardization_version,
                "rdkit_version": result.rdkit_version,
                "unresolved_stereoelement_count": unspecified_stereo,
                "specified_stereoelement_count": specified_stereo,
                "stereochemistry_status": stereo_status,
                "same_series_confirmed": _normalized_confirmation(
                    record.same_series_confirmed if "same_series_confirmed" in raw.columns else None
                ),
                **optional,
            }
        )
    frame = pd.DataFrame(rows)
    duplicates = frame[frame["structure_id"].duplicated(False)]
    if not duplicates.empty:
        identifiers = duplicates["compound_id"].tolist()
        raise ValueError(f"Input contains duplicate standardized structures: {identifiers}")
    return frame


def _pic50_relation(lower: float, upper: float) -> str:
    if np.isfinite(lower) and np.isfinite(upper) and np.isclose(lower, upper):
        return "="
    if not np.isfinite(lower) and np.isfinite(upper):
        return "<"
    if np.isfinite(lower) and not np.isfinite(upper):
        return ">"
    return "interval_or_unknown"


def _neighbor_support(
    submitted: pd.DataFrame,
    training: pd.DataFrame,
    training_bits: np.ndarray,
    domain_thresholds: pd.Series,
) -> pd.DataFrame:
    unique_mask = ~training["compound_id"].astype(str).duplicated()
    unique_training = training.loc[unique_mask].reset_index(drop=True)
    unique_bits = training_bits[np.flatnonzero(unique_mask.to_numpy())]
    query_bits = _fingerprints(submitted["standardized_smiles"])
    rows: list[dict[str, object]] = []
    for query_index, query in enumerate(query_bits):
        intersections = np.logical_and(unique_bits, query).sum(axis=1)
        unions = np.logical_or(unique_bits, query).sum(axis=1)
        similarities = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=float),
            where=unions > 0,
        )
        nearest_index = int(np.argmax(similarities))
        nearest = unique_training.iloc[nearest_index]
        threshold = float(domain_thresholds.iloc[query_index])
        local_mask = similarities >= 0.80
        threshold_mask = similarities >= threshold
        local = unique_training.loc[local_mask]
        exact_mask = (
            np.isfinite(local["pic50_lower"])
            & np.isfinite(local["pic50_upper"])
            & np.isclose(local["pic50_lower"], local["pic50_upper"])
        )
        local_exact = local.loc[exact_mask, "pic50_lower"].astype(float)
        lower = float(nearest["pic50_lower"])
        upper = float(nearest["pic50_upper"])
        rows.append(
            {
                "nearest_training_structure_id": str(nearest["compound_id"]),
                "nearest_training_tanimoto": float(similarities[nearest_index]),
                "nearest_training_pic50_relation": _pic50_relation(lower, upper),
                "nearest_training_pic50_lower": lower,
                "nearest_training_pic50_upper": upper,
                "neighbor_structures_ge_domain_threshold": int(threshold_mask.sum()),
                "neighbor_structures_ge_0p80": int(local_mask.sum()),
                "neighbor_exact_measurements_ge_0p80": int(exact_mask.sum()),
                "neighbor_exact_pic50_min_ge_0p80": (
                    float(local_exact.min()) if not local_exact.empty else np.nan
                ),
                "neighbor_exact_pic50_max_ge_0p80": (
                    float(local_exact.max()) if not local_exact.empty else np.nan
                ),
                "scaffold_seen_in_training": bool(
                    submitted.iloc[query_index]["scaffold"] in set(unique_training["scaffold"].astype(str))
                ),
            }
        )
    return pd.DataFrame(rows)


def _training_context() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    np.ndarray,
    np.ndarray,
]:
    (
        compounds,
        _internal,
        _extension,
        combined,
        _overlap,
        _internal_audit,
        _ambiguous,
        controls,
    ) = _prepare_clean_combined_data()
    combined_controls = combined[controls].fillna(combined[controls].median()).to_numpy(dtype=float)
    combined_bits = _fingerprints(combined["standardized_smiles"])
    return compounds, combined, controls, combined_controls, combined_bits


def _global_control_predictions(
    training: pd.DataFrame,
    scoring_features: pd.DataFrame,
    controls: list[str],
) -> pd.DataFrame:
    _, oof = grouped_censored_herg_benchmark(
        training,
        feature_columns=controls,
        folds=5,
    )
    predictions = final_fit_censored_herg_predictions(
        training,
        scoring_features,
        oof,
        feature_columns=controls,
        interval_level=0.90,
        promotion_status="discovery_only_same_series_analog",
    )
    pic50 = predictions[predictions["endpoint"].eq("herg_pic50")].rename(
        columns={
            "compound_id": "structure_id",
            "mean": "global_control_predicted_pic50",
            "lower": "global_control_pic50_lower",
            "upper": "global_control_pic50_upper",
            "domain_status": "applicability_domain",
            "max_train_tanimoto": "max_train_tanimoto",
        }
    )
    blocker = predictions[predictions["endpoint"].eq("herg_blocker_probability")][
        ["compound_id", "mean", "lower", "upper"]
    ].rename(
        columns={
            "compound_id": "structure_id",
            "mean": "global_control_blocker_probability",
            "lower": "global_control_blocker_probability_lower",
            "upper": "global_control_blocker_probability_upper",
        }
    )
    keep = [
        "structure_id",
        "global_control_predicted_pic50",
        "global_control_pic50_lower",
        "global_control_pic50_upper",
        "applicability_domain",
        "max_train_tanimoto",
        "domain_threshold",
        "oof_mae_pic50",
        "oof_sigma_pic50",
        "fit_converged",
        "promotion_status",
    ]
    return pic50[keep].merge(blocker, on="structure_id", validate="one_to_one")


def _prediction_report(result: pd.DataFrame) -> str:
    rows = [
        "| Compound | Ensemble pIC50 | Ensemble IC50 (uM) | Global pIC50 | Complete pIC50 | Conservative envelope | "
        "Nearest similarity | Neighbors ≥0.80 | Decision status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in result.itertuples(index=False):
        rows.append(
            f"| {record.compound_id} | {record.ensemble_predicted_pic50:.2f} | "
            f"{record.ensemble_predicted_herg_ic50_um:.2f} | "
            f"{record.global_control_predicted_pic50:.2f} | "
            f"{record.complete_feature_predicted_pic50:.2f} | "
            f"{record.conservative_model_envelope_pic50_lower:.2f}–"
            f"{record.conservative_model_envelope_pic50_upper:.2f} | "
            f"{record.max_train_tanimoto:.3f} | "
            f"{record.neighbor_structures_ge_0p80} | {record.decision_status} |"
        )
    table = "\n".join(rows)
    return f"""# New-compound hERG prediction readout

{table}

The primary readout is the untuned equal-weight ensemble of the compact-control and
structure-sensitive models. The scientific endpoint is continuous pIC50. The secondary
blocker boundary is pIC50 5,
equivalent to IC50 10 µM. The conservative envelope spans both retained point-model
intervals; it is not a calibrated guarantee and excludes assay/source uncertainty.

`inside` domain indicates structural support only. Same-series confirmation,
stereochemistry, close measured-neighbor counts, model disagreement, interval crossing,
and exact-training overlap must all be read before using a result. Every row remains a
discovery hypothesis until protocol-matched experimental confirmation. No Menin potency
or physics feature contributes to this readout.
"""


def predict(input_path: Path, output: Path) -> pd.DataFrame:
    submitted = _standardize_input(input_path)
    compounds, training, controls, training_controls, training_bits = _training_context()
    scoring_compounds = pd.DataFrame(
        {
            "compound_id": submitted["structure_id"],
            "standardized_smiles": submitted["standardized_smiles"],
        }
    )
    scoring_features = scoring_compounds.merge(
        structure_feature_frame(scoring_compounds),
        on="compound_id",
        validate="one_to_one",
    )
    global_predictions = _global_control_predictions(
        training,
        scoring_features,
        controls,
    )
    neighbor_support = _neighbor_support(
        submitted,
        training,
        training_bits,
        global_predictions["domain_threshold"],
    )

    audit_metrics = json.loads((AUDIT_OUTPUT / "extension_metrics.json").read_text())
    components = int(audit_metrics["selected_components"])
    _, complete_oof = _oof(
        training,
        training_bits,
        training_controls,
        components=components,
    )
    complete_scoring = submitted[["structure_id", "compound_id", "scaffold"]].rename(
        columns={"compound_id": "internal_id"}
    )
    complete_scoring["herg_pic50_relation"] = ""
    complete_scoring["herg_pic50_value"] = np.nan
    complete_scoring["herg_pic50_upper_bound"] = np.nan
    scoring_controls = (
        scoring_features.set_index("compound_id")
        .loc[submitted["structure_id"], controls]
        .fillna(training[controls].median())
        .to_numpy(dtype=float)
    )
    scoring_bits = _fingerprints(submitted["standardized_smiles"])
    complete_predictions = _fit_score(
        training,
        training_bits,
        training_controls,
        complete_scoring,
        scoring_bits,
        scoring_controls,
        complete_oof,
        components=components,
    )

    result = (
        submitted.merge(
            global_predictions,
            on="structure_id",
            validate="one_to_one",
        )
        .join(
            neighbor_support,
            validate="one_to_one",
        )
        .merge(
            complete_predictions[
                [
                    "structure_id",
                    "complete_feature_predicted_pic50",
                    "complete_feature_pic50_lower",
                    "complete_feature_pic50_upper",
                    "complete_feature_blocker_probability",
                    "complete_feature_oof_sigma",
                    "model_fit_converged",
                ]
            ],
            on="structure_id",
            validate="one_to_one",
        )
    )
    retained_points = result[["global_control_predicted_pic50", "complete_feature_predicted_pic50"]]
    result["ensemble_predicted_pic50"] = retained_points.mean(axis=1)
    ensemble_mean = result["ensemble_predicted_pic50"].to_numpy(dtype=float)
    global_mean = result["global_control_predicted_pic50"].to_numpy(dtype=float)
    complete_mean = result["complete_feature_predicted_pic50"].to_numpy(dtype=float)
    ensemble_variance = 0.5 * (
        result["oof_sigma_pic50"].to_numpy(dtype=float) ** 2 + (global_mean - ensemble_mean) ** 2
    ) + 0.5 * (
        result["complete_feature_oof_sigma"].to_numpy(dtype=float) ** 2 + (complete_mean - ensemble_mean) ** 2
    )
    result["ensemble_predictive_sigma"] = np.sqrt(np.maximum(ensemble_variance, 1e-12))
    result["ensemble_pic50_lower_90"] = (
        result["ensemble_predicted_pic50"] - norm.ppf(0.95) * result["ensemble_predictive_sigma"]
    )
    result["ensemble_pic50_upper_90"] = (
        result["ensemble_predicted_pic50"] + norm.ppf(0.95) * result["ensemble_predictive_sigma"]
    )
    result["ensemble_blocker_probability"] = norm.sf(
        (5.0 - result["ensemble_predicted_pic50"]) / result["ensemble_predictive_sigma"].clip(lower=1e-6)
    )
    result["ensemble_predicted_herg_ic50_um"] = 10.0 ** (6.0 - result["ensemble_predicted_pic50"])
    result["ensemble_herg_ic50_lower_90_um"] = 10.0 ** (6.0 - result["ensemble_pic50_upper_90"])
    result["ensemble_herg_ic50_upper_90_um"] = 10.0 ** (6.0 - result["ensemble_pic50_lower_90"])
    result["retained_model_pic50_min"] = retained_points.min(axis=1)
    result["retained_model_pic50_max"] = retained_points.max(axis=1)
    result["retained_model_spread_pic50"] = (
        result["retained_model_pic50_max"] - result["retained_model_pic50_min"]
    )
    result["conservative_model_envelope_pic50_lower"] = result[
        ["global_control_pic50_lower", "complete_feature_pic50_lower"]
    ].min(axis=1)
    result["conservative_model_envelope_pic50_upper"] = result[
        ["global_control_pic50_upper", "complete_feature_pic50_upper"]
    ].max(axis=1)
    result["conservative_model_envelope_width"] = (
        result["conservative_model_envelope_pic50_upper"] - result["conservative_model_envelope_pic50_lower"]
    )
    result["retained_blocker_probability_min"] = result[
        ["global_control_blocker_probability", "complete_feature_blocker_probability"]
    ].min(axis=1)
    result["retained_blocker_probability_max"] = result[
        ["global_control_blocker_probability", "complete_feature_blocker_probability"]
    ].max(axis=1)
    global_blocker = result["global_control_predicted_pic50"].ge(5.0)
    complete_blocker = result["complete_feature_predicted_pic50"].ge(5.0)
    result["threshold_consensus"] = np.select(
        [
            global_blocker & complete_blocker,
            ~global_blocker & ~complete_blocker,
        ],
        ["predicted_blocker", "predicted_nonblocker"],
        default="model_disagreement",
    )
    result["threshold_interval_status"] = np.select(
        [
            result["conservative_model_envelope_pic50_lower"].ge(5.0),
            result["conservative_model_envelope_pic50_upper"].lt(5.0),
        ],
        ["envelope_entirely_blocker", "envelope_entirely_nonblocker"],
        default="envelope_crosses_threshold",
    )
    baseline_structures = set(compounds["structure_id"].astype(str))
    training_structures = set(training["compound_id"].astype(str))
    result["exact_original_training_overlap"] = result["structure_id"].isin(baseline_structures)
    result["exact_augmented_training_overlap"] = result["structure_id"].isin(training_structures)
    result["prediction_scope"] = np.where(
        result["applicability_domain"].eq("inside"),
        "same_series_analog_discovery_only",
        "outside_domain_extrapolation_do_not_use_for_decisions",
    )
    result["prediction_eligibility"] = np.select(
        [
            result["exact_augmented_training_overlap"],
            result["unresolved_stereoelement_count"].gt(0),
            result["same_series_confirmed"].ne("true"),
            result["applicability_domain"].ne("inside"),
        ],
        [
            "training_overlap_not_a_novel_prediction",
            "resolve_structure_stereochemistry_before_use",
            "confirm_same_series_before_use",
            "outside_domain_do_not_use",
        ],
        default="eligible_same_series_discovery_hypothesis",
    )
    result["decision_status"] = np.select(
        [
            result["prediction_eligibility"].ne("eligible_same_series_discovery_hypothesis"),
            result["threshold_consensus"].eq("model_disagreement"),
            result["retained_model_spread_pic50"].ge(0.5),
            result["threshold_interval_status"].eq("envelope_crosses_threshold"),
        ],
        [
            result["prediction_eligibility"],
            "model_threshold_disagreement_requires_measurement",
            "material_model_form_uncertainty_requires_measurement",
            "uncertainty_envelope_crosses_threshold_requires_measurement",
        ],
        default="models_and_envelope_agree_discovery_hypothesis_only",
    )
    result["required_warning"] = (
        "No Menin potency is modeled; hERG prediction is not an efficacy/safety "
        "optimization score. Domain membership is not proof of calibration. "
        "Protocol-matched experimental confirmation is required."
    )

    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output / "predictions.csv", result)
    atomic_write_parquet(output / "predictions.parquet", result)
    atomic_write_text(output / "prediction_report.md", _prediction_report(result))
    summary = {
        "status": "pass",
        "input_rows": len(result),
        "inside_applicability_domain": int(result["applicability_domain"].eq("inside").sum()),
        "outside_applicability_domain": int(result["applicability_domain"].eq("outside").sum()),
        "threshold_disagreements": int(result["threshold_consensus"].eq("model_disagreement").sum()),
        "unresolved_stereochemistry": int(result["unresolved_stereoelement_count"].gt(0).sum()),
        "same_series_confirmed": int(result["same_series_confirmed"].eq("true").sum()),
        "eligible_same_series_discovery_hypotheses": int(
            result["prediction_eligibility"].eq("eligible_same_series_discovery_hypothesis").sum()
        ),
        "conservative_envelopes_crossing_pic50_5": int(
            result["threshold_interval_status"].eq("envelope_crosses_threshold").sum()
        ),
        "training_rows": len(training),
        "training_unique_compounds": int(training["compound_id"].nunique()),
        "same_series_only": True,
        "menin_potency_available": False,
        "models": [
            "censored ridge with nine global physicochemical controls",
            f"censored ridge with {components} fold-selected ECFP4 latent components plus controls",
            "untuned equal-weight moment-matched ensemble of the two retained models",
        ],
        "training_sources": (
            "70 concordant unique old-internal structures plus 46 measured non-overlapping "
            "Angelo structures; 22 duplicate overlaps are not double-weighted and five "
            "discordant old labels are quarantined"
        ),
        "physics_features_used": False,
        "promotion_status": "discovery_only_same_series_analog",
        "blocker_definition": "pIC50 >= 5, equivalent to IC50 <= 10 micromolar",
        "domain_warning": (
            "The domain flag measures structural support only and is not proof of "
            "prospective calibration or series identity."
        ),
    }
    atomic_write_json(output / "prediction_summary.json", summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = predict(args.input, args.output)
    print(
        json.dumps(
            {
                "predictions": len(result),
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
