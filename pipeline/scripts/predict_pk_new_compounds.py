#!/usr/bin/env python3
"""Generate guarded rat-PK hypotheses for new same-series Menin inhibitors.

The models are refit on the 46 internal structures with eligible rat IV/PO
summary parameters.  Predictions are discovery-stage analog hypotheses, not
PBPK simulations and not substitutes for concentration-time profiles.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/menin-prediction-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/menin-prediction-cache")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline/src"))

from menin_discovery.research_common import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
)
from menin_discovery.research_modeling import structure_feature_frame  # noqa: E402
from menin_discovery.research_reviewer_audit import (  # noqa: E402
    _source_collapsed_pk_frames,
)
from menin_discovery.research_workflows import (  # noqa: E402
    compound_model_frame,
    load_canonical_tables,
)
from predict_herg_new_compounds import _standardize_input  # noqa: E402
from run_herg_pk_mix_match import (  # noqa: E402
    CANONICAL,
    PROXY_COLUMNS,
    _aggregate_internal_pka,
    _fingerprints,
    _fit_predict,
    _tanimoto_predict,
)

DEFAULT_OUTPUT = ROOT / "research/reports/pk_herg/new_compound_predictions/pk"
MIX_MATCH = ROOT / "research/reports/pk_herg/mix_match"
INTERVAL_LEVEL = 0.90

OPTIONAL_NUMERIC_COLUMNS = {
    "basic_pka",
    "planned_iv_dose_mg_kg",
    "planned_po_dose_mg_kg",
}
PROHIBITED_PK_OUTCOME_COLUMNS = {
    "auc",
    "clearance",
    "cl",
    "cmax",
    "f",
    "fraction_absorbed",
    "pk_result",
    "po_auc",
    "iv_auc",
    "tmax",
    "vdss",
}


@dataclass(frozen=True)
class ModelSpec:
    endpoint: str
    role: str
    feature_layer: str
    model: str
    unit: str
    evidence_status: str


MODEL_SPECS = (
    ModelSpec(
        "iv_auc_dose_normalized",
        "primary_analogue",
        "morgan_tanimoto",
        "tanimoto_3nn",
        "ng*h/mL per mg/kg",
        "retrospective_internal_scaffold_cv",
    ),
    ModelSpec(
        "iv_auc_dose_normalized",
        "representation_challenger",
        "morgan_latent",
        "ridge",
        "ng*h/mL per mg/kg",
        "retrospective_internal_scaffold_cv",
    ),
    ModelSpec(
        "po_auc_dose_normalized",
        "primary_analogue",
        "morgan_tanimoto",
        "tanimoto_3nn",
        "ng*h/mL per mg/kg",
        "retrospective_internal_scaffold_cv",
    ),
    ModelSpec(
        "po_auc_dose_normalized",
        "representation_challenger",
        "morgan_latent",
        "ridge",
        "ng*h/mL per mg/kg",
        "retrospective_internal_scaffold_cv",
    ),
    ModelSpec(
        "po_cmax_dose_normalized",
        "primary_analogue",
        "morgan_tanimoto",
        "tanimoto_3nn",
        "ng/mL per mg/kg",
        "retrospective_internal_scaffold_cv; dose_proportionality_unverified",
    ),
    ModelSpec(
        "po_cmax_dose_normalized",
        "representation_challenger",
        "morgan_latent",
        "ridge",
        "ng/mL per mg/kg",
        "retrospective_internal_scaffold_cv; dose_proportionality_unverified",
    ),
    ModelSpec(
        "vdss",
        "primary_no_pka",
        "hybrid",
        "extra_trees",
        "L/kg",
        "retrospective_internal_scaffold_cv",
    ),
    ModelSpec(
        "vdss",
        "property_proxy_challenger",
        "compact_proxies",
        "svr",
        "L/kg",
        "retrospective_internal_scaffold_cv",
    ),
    ModelSpec(
        "po_tmax",
        "weak_exploratory_only",
        "compact_proxies",
        "random_forest",
        "h",
        "withheld_from_ranking; held_scaffold_spearman_near_zero",
    ),
)


def _standardize_pk_input(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    lowered = {str(column).strip().lower() for column in raw.columns}
    prohibited = sorted(lowered & PROHIBITED_PK_OUTCOME_COLUMNS)
    if prohibited:
        raise ValueError(f"Prediction input contains PK outcome columns and is not blind: {prohibited}")
    submitted = _standardize_input(path)
    extras = raw[["compound_id", *sorted(OPTIONAL_NUMERIC_COLUMNS & set(raw.columns))]].copy()
    for column in OPTIONAL_NUMERIC_COLUMNS:
        if column not in extras:
            extras[column] = np.nan
        extras[column] = pd.to_numeric(extras[column], errors="coerce")
    for dose_column in ("planned_iv_dose_mg_kg", "planned_po_dose_mg_kg"):
        invalid = extras[dose_column].notna() & extras[dose_column].le(0)
        if invalid.any():
            raise ValueError(f"{dose_column} must be positive when supplied")
    invalid_pka = extras["basic_pka"].notna() & ~extras["basic_pka"].between(0, 14)
    if invalid_pka.any():
        raise ValueError("basic_pka must lie between 0 and 14 when supplied")
    return submitted.merge(extras, on="compound_id", how="left", validate="one_to_one")


def _training_frames() -> dict[str, pd.DataFrame]:
    tables = load_canonical_tables(CANONICAL)
    compounds = compound_model_frame(tables["compounds"], tables.get("compound_aliases"))
    frames, _ = _source_collapsed_pk_frames(
        compounds,
        tables["measurements"],
        tables["pk_studies"],
    )
    descriptors = structure_feature_frame(compounds[["compound_id", "standardized_smiles"]])
    pka = _aggregate_internal_pka(tables["measurements"])
    result: dict[str, pd.DataFrame] = {}
    for endpoint in {spec.endpoint for spec in MODEL_SPECS}:
        frame = (
            frames[endpoint]
            .merge(
                descriptors[["compound_id", *PROXY_COLUMNS]],
                on="compound_id",
                how="inner",
                validate="one_to_one",
            )
            .merge(pka, on="compound_id", how="left", validate="one_to_one")
        )
        frame["target_pic50"] = frame["target_log10"]
        frame["fingerprint"] = list(_fingerprints(frame["standardized_smiles"]))
        frame["sample_weight"] = 1.0
        result[endpoint] = frame
    return result


def _query_features(submitted: pd.DataFrame) -> pd.DataFrame:
    descriptors = structure_feature_frame(submitted[["compound_id", "standardized_smiles"]])
    result = submitted.merge(
        descriptors[["compound_id", *PROXY_COLUMNS]],
        on="compound_id",
        how="inner",
        validate="one_to_one",
    )
    result["maximum_basic_pka"] = result["basic_pka"]
    result["reported_basic_pka_count"] = result["basic_pka"].notna().astype(int)
    result["single_site_cation_fraction_pH7p4_proxy"] = np.where(
        result["basic_pka"].notna(),
        1.0 / (1.0 + np.power(10.0, 7.4 - result["basic_pka"])),
        np.nan,
    )
    result["fingerprint"] = list(_fingerprints(result["standardized_smiles"]))
    return result


def _nearest_support(
    train: pd.DataFrame,
    query: pd.DataFrame,
) -> pd.DataFrame:
    train_bits = np.vstack(train["fingerprint"].to_numpy()).astype(bool)
    query_bits = np.vstack(query["fingerprint"].to_numpy()).astype(bool)
    train_integer = train_bits.astype(np.int16)
    intersections = train_integer @ train_integer.T
    train_counts = train_bits.sum(axis=1)
    unions = train_counts[:, None] + train_counts[None, :] - intersections
    train_similarity = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=float),
        where=unions > 0,
    )
    np.fill_diagonal(train_similarity, -np.inf)
    nearest_train = np.max(train_similarity, axis=1)
    threshold = float(np.quantile(nearest_train[np.isfinite(nearest_train)], 0.05))

    rows: list[dict[str, Any]] = []
    targets = train["target_log10"].to_numpy(dtype=float)
    for index, bits in enumerate(query_bits):
        query_intersection = np.logical_and(train_bits, bits).sum(axis=1)
        query_union = np.logical_or(train_bits, bits).sum(axis=1)
        similarities = np.divide(
            query_intersection,
            query_union,
            out=np.zeros_like(query_intersection, dtype=float),
            where=query_union > 0,
        )
        nearest_index = int(np.argmax(similarities))
        rows.append(
            {
                "max_train_tanimoto": float(similarities[nearest_index]),
                "domain_threshold": threshold,
                "applicability_domain": ("inside" if similarities[nearest_index] >= threshold else "outside"),
                "nearest_training_compound_id": str(train.iloc[nearest_index]["compound_id"]),
                "nearest_training_observed": float(np.power(10.0, targets[nearest_index])),
                "neighbor_structures_ge_0p80": int(np.sum(similarities >= 0.80)),
                "scaffold_seen_in_training": bool(
                    str(query.iloc[index]["scaffold"]) in set(train["scaffold"].astype(str))
                ),
            }
        )
    return pd.DataFrame(rows)


def _empirical_radius(endpoint: str, layer: str, model: str) -> tuple[float, float, float]:
    predictions = pd.read_parquet(MIX_MATCH / "pk_feature_model_predictions.parquet")
    selected = predictions[
        predictions["endpoint"].eq(endpoint)
        & predictions["feature_layer"].eq(layer)
        & predictions["model"].eq(model)
    ]
    if selected.empty:
        raise ValueError(f"No held-scaffold residuals for {endpoint}/{layer}/{model}")
    residual = np.abs(
        selected["predicted_log10"].to_numpy(dtype=float) - selected["observed_log10"].to_numpy(dtype=float)
    )
    level = min(1.0, math.ceil((len(residual) + 1) * INTERVAL_LEVEL) / len(residual))
    radius = float(np.quantile(residual, level, method="higher"))
    return radius, float(np.mean(residual)), float(np.median(np.power(10.0, residual)))


def _predict_spec(
    spec: ModelSpec,
    train: pd.DataFrame,
    query: pd.DataFrame,
) -> np.ndarray:
    if spec.model == "tanimoto_3nn":
        return _tanimoto_predict(train, query, neighbors=3)
    return _fit_predict(
        train,
        query,
        feature_layer=spec.feature_layer,
        model_name=spec.model,
    )


def _decision_status(endpoint: str, domain: str, same_series: str) -> str:
    if same_series != "true":
        return "withheld_same_series_not_confirmed"
    if domain != "inside":
        return "withheld_outside_structural_support"
    if endpoint == "po_tmax":
        return "withheld_model_has_no_reliable_rank_signal"
    return "eligible_same_series_discovery_hypothesis"


def _long_predictions(
    submitted: pd.DataFrame,
    query: pd.DataFrame,
    training: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        train = training[spec.endpoint]
        support = _nearest_support(train, query)
        predicted_log = _predict_spec(spec, train, query)
        radius, oof_mae, median_fold_error = _empirical_radius(
            spec.endpoint,
            spec.feature_layer,
            spec.model,
        )
        for index, record in enumerate(submitted.itertuples(index=False)):
            point = float(np.power(10.0, predicted_log[index]))
            lower = float(np.power(10.0, predicted_log[index] - radius))
            upper = float(np.power(10.0, predicted_log[index] + radius))
            domain = str(support.iloc[index]["applicability_domain"])
            rows.append(
                {
                    "compound_id": record.compound_id,
                    "endpoint": spec.endpoint,
                    "model_role": spec.role,
                    "feature_layer": spec.feature_layer,
                    "model": spec.model,
                    "predicted_log10": float(predicted_log[index]),
                    "predicted_value": point,
                    "empirical_90pct_lower": lower,
                    "empirical_90pct_upper": upper,
                    "unit": spec.unit,
                    "held_scaffold_oof_mae_log10": oof_mae,
                    "held_scaffold_median_fold_error": median_fold_error,
                    "max_train_tanimoto": support.iloc[index]["max_train_tanimoto"],
                    "domain_threshold": support.iloc[index]["domain_threshold"],
                    "applicability_domain": domain,
                    "nearest_training_compound_id": support.iloc[index]["nearest_training_compound_id"],
                    "nearest_training_observed": support.iloc[index]["nearest_training_observed"],
                    "neighbor_structures_ge_0p80": support.iloc[index]["neighbor_structures_ge_0p80"],
                    "scaffold_seen_in_training": support.iloc[index]["scaffold_seen_in_training"],
                    "evidence_status": spec.evidence_status,
                    "prediction_status": _decision_status(
                        spec.endpoint,
                        domain,
                        record.same_series_confirmed,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _endpoint_summary(long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (compound_id, endpoint), group in long.groupby(["compound_id", "endpoint"], sort=True):
        primary = group[~group["model_role"].str.contains("challenger")].iloc[0]
        rows.append(
            {
                "compound_id": compound_id,
                "endpoint": endpoint,
                "primary_model_role": primary["model_role"],
                "primary_predicted_value": primary["predicted_value"],
                "model_point_min": float(group["predicted_value"].min()),
                "model_point_max": float(group["predicted_value"].max()),
                "conservative_90pct_lower": float(group["empirical_90pct_lower"].min()),
                "conservative_90pct_upper": float(group["empirical_90pct_upper"].max()),
                "unit": primary["unit"],
                "max_train_tanimoto": primary["max_train_tanimoto"],
                "applicability_domain": primary["applicability_domain"],
                "neighbor_structures_ge_0p80": primary["neighbor_structures_ge_0p80"],
                "prediction_status": primary["prediction_status"],
                "model_disagreement_fold": float(
                    group["predicted_value"].max() / max(group["predicted_value"].min(), 1e-12)
                ),
            }
        )
    return pd.DataFrame(rows)


def _derived_predictions(
    submitted: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in submitted.itertuples(index=False):
        compound = summary[summary["compound_id"].eq(record.compound_id)].set_index("endpoint")
        iv = compound.loc["iv_auc_dose_normalized"]
        po = compound.loc["po_auc_dose_normalized"]
        cmax = compound.loc["po_cmax_dose_normalized"]
        cl_point = 1000.0 / iv["primary_predicted_value"]
        cl_lower = 1000.0 / iv["conservative_90pct_upper"]
        cl_upper = 1000.0 / iv["conservative_90pct_lower"]
        f_point = 100.0 * po["primary_predicted_value"] / iv["primary_predicted_value"]
        f_lower = 100.0 * po["conservative_90pct_lower"] / iv["conservative_90pct_upper"]
        f_upper = 100.0 * po["conservative_90pct_upper"] / iv["conservative_90pct_lower"]
        rows.extend(
            [
                {
                    "compound_id": record.compound_id,
                    "derived_endpoint": "systemic_clearance_from_iv_auc",
                    "predicted_value": cl_point,
                    "conservative_90pct_lower": cl_lower,
                    "conservative_90pct_upper": cl_upper,
                    "unit": "L/h/kg",
                    "lineage": "1000 / predicted IV dose-normalized AUC",
                    "status": "derived_closure_not_independent_model",
                },
                {
                    "compound_id": record.compound_id,
                    "derived_endpoint": "apparent_oral_bioavailability",
                    "predicted_value": f_point,
                    "conservative_90pct_lower": f_lower,
                    "conservative_90pct_upper": f_upper,
                    "unit": "%",
                    "lineage": "100 * predicted PO/IV dose-normalized AUC ratio",
                    "status": (
                        "model_closure_conflict_if_above_100pct; not_Fa_Fg_Fh_decomposition"
                        if f_point > 100 or f_upper > 100
                        else "derived_closure_not_Fa_Fg_Fh_decomposition"
                    ),
                },
            ]
        )
        if pd.notna(record.planned_iv_dose_mg_kg):
            rows.append(
                {
                    "compound_id": record.compound_id,
                    "derived_endpoint": "iv_auc_at_planned_dose",
                    "predicted_value": iv["primary_predicted_value"] * record.planned_iv_dose_mg_kg,
                    "conservative_90pct_lower": iv["conservative_90pct_lower"] * record.planned_iv_dose_mg_kg,
                    "conservative_90pct_upper": iv["conservative_90pct_upper"] * record.planned_iv_dose_mg_kg,
                    "unit": "ng*h/mL",
                    "lineage": "dose-normalized IV AUC * planned IV dose",
                    "status": "assumes_linear_PK_at_planned_dose",
                }
            )
        if pd.notna(record.planned_po_dose_mg_kg):
            for endpoint, name, source in (
                ("po_auc_dose_normalized", "po_auc_at_planned_dose", po),
                ("po_cmax_dose_normalized", "po_cmax_at_planned_dose", cmax),
            ):
                rows.append(
                    {
                        "compound_id": record.compound_id,
                        "derived_endpoint": name,
                        "predicted_value": source["primary_predicted_value"] * record.planned_po_dose_mg_kg,
                        "conservative_90pct_lower": source["conservative_90pct_lower"]
                        * record.planned_po_dose_mg_kg,
                        "conservative_90pct_upper": source["conservative_90pct_upper"]
                        * record.planned_po_dose_mg_kg,
                        "unit": "ng*h/mL" if "auc" in endpoint else "ng/mL",
                        "lineage": f"{endpoint} * planned PO dose",
                        "status": "assumes_linear_PK_at_planned_dose",
                    }
                )
    return pd.DataFrame(rows)


def _report(summary: pd.DataFrame, derived: pd.DataFrame) -> str:
    lines = [
        "# New-compound rat-PK hypothesis readout",
        "",
        "These are same-series, internal-summary-parameter hypotheses. They are not "
        "PBPK predictions and have no concentration-time-profile calibration.",
        "",
    ]
    for compound_id, group in summary.groupby("compound_id", sort=True):
        lines.extend(
            [
                f"## {compound_id}",
                "",
                "| Endpoint | Primary | Model range | Empirical 90% envelope | Unit | Similarity | Status |",
                "|---|---:|---:|---:|---|---:|---|",
            ]
        )
        for row in group.itertuples(index=False):
            lines.append(
                f"| {row.endpoint} | {row.primary_predicted_value:.3g} | "
                f"{row.model_point_min:.3g}–{row.model_point_max:.3g} | "
                f"{row.conservative_90pct_lower:.3g}–{row.conservative_90pct_upper:.3g} | "
                f"{row.unit} | {row.max_train_tanimoto:.3f} | {row.prediction_status} |"
            )
        lines.extend(
            [
                "",
                "Derived closure checks:",
                "",
                "| Quantity | Point | Envelope | Unit | Status |",
                "|---|---:|---:|---|---|",
            ]
        )
        for row in derived[derived["compound_id"].eq(compound_id)].itertuples(index=False):
            lines.append(
                f"| {row.derived_endpoint} | {row.predicted_value:.3g} | "
                f"{row.conservative_90pct_lower:.3g}–{row.conservative_90pct_upper:.3g} | "
                f"{row.unit} | {row.status} |"
            )
        lines.append("")
    lines.extend(
        [
            "Tmax is deliberately withheld from ranking because held-scaffold Spearman "
            "is approximately zero. PO AUC and Cmax dose conversions assume linear PK. "
            "Apparent F and CL are algebraic descendants, not independent labels or "
            "mechanistic decompositions. Protocol-matched rat profiles remain required.",
            "",
        ]
    )
    return "\n".join(lines)


def predict(input_path: Path, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    submitted = _standardize_pk_input(input_path)
    query = _query_features(submitted)
    training = _training_frames()
    long = _long_predictions(submitted, query, training)
    summary = _endpoint_summary(long)
    derived = _derived_predictions(submitted, summary)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output / "pk_predictions_long.csv", long)
    atomic_write_parquet(output / "pk_predictions_long.parquet", long)
    atomic_write_csv(output / "pk_endpoint_summary.csv", summary)
    atomic_write_csv(output / "pk_derived_closure.csv", derived)
    atomic_write_text(output / "pk_prediction_report.md", _report(summary, derived))
    atomic_write_json(
        output / "pk_prediction_summary.json",
        {
            "compounds": len(submitted),
            "endpoint_predictions": len(summary),
            "inside_domain_endpoint_predictions": int(summary["applicability_domain"].eq("inside").sum()),
            "decision_eligible_endpoint_predictions": int(
                summary["prediction_status"].eq("eligible_same_series_discovery_hypothesis").sum()
            ),
            "evidence_boundary": (
                "internal rat summary parameters; retrospective same-series hypothesis generation only"
            ),
            "models_refit": True,
            "hpc_physics_used": False,
        },
    )
    return summary, derived


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, derived = predict(args.input, args.output)
    print(
        json.dumps(
            {
                "compounds": int(summary["compound_id"].nunique()),
                "endpoint_predictions": len(summary),
                "derived_predictions": len(derived),
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
