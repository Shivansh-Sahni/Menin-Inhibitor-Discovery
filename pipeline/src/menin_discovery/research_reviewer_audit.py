"""Skeptical-review controls for the local mechanistic PK/hERG program.

The analyses in this module are deliberately adversarial. They test target
definitions, split topology, trivial and fingerprint baselines, label
availability bias, source repetition, optimizer overlap, and apparent
activity cliffs. They do not generate physics features or promote models.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from menin_discovery.features import (
    fingerprint_matrix,
    rdkit_descriptors,
)
from menin_discovery.research_common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
)
from menin_discovery.research_feature_ontology import (
    CONVENTIONAL_DESCRIPTOR_COLUMNS,
    conventional_feature_ontology_frame,
    feature_ontology_frame,
)
from menin_discovery.research_local_analysis import (
    _benjamini_hochberg,
    grouped_bootstrap_spearman,
)
from menin_discovery.research_modeling import merge_feature_layers
from menin_discovery.research_parameter_ontology import (
    parameter_ontology_frame,
    validate_parameter_ontology,
)
from menin_discovery.research_workflows import (
    compound_model_frame,
    load_canonical_tables,
    prepare_pk_tasks,
)

SIMILARITY_THRESHOLDS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
PRIMARY_SIMILARITY_COMPONENT_THRESHOLD = 0.85
DESCRIPTOR_COLUMNS = CONVENTIONAL_DESCRIPTOR_COLUMNS
PK_ENDPOINTS = (
    "iv_auc_dose_normalized",
    "po_auc_dose_normalized",
    "vdss",
    "po_cmax_raw",
    "po_cmax_dose_normalized",
    "po_tmax",
)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3 or np.unique(x[finite]).size < 2 or np.unique(y[finite]).size < 2:
        return float("nan")
    return float(spearmanr(x[finite], y[finite]).statistic)


def _effective_count(values: pd.Series) -> float:
    fractions = values.value_counts(normalize=True).to_numpy(dtype=float)
    return float(1.0 / np.sum(fractions**2))


def _full_tanimoto(smiles: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    fingerprints, _ = fingerprint_matrix(
        smiles,
        backend="rdkit",
        n_bits=2048,
        radius=2,
    )
    counts = np.asarray(fingerprints.sum(axis=1)).ravel()
    intersections = (fingerprints @ fingerprints.T).toarray()
    denominators = counts[:, None] + counts[None, :] - intersections
    similarity = np.divide(
        intersections,
        denominators,
        out=np.zeros_like(intersections, dtype=float),
        where=denominators > 0,
    )
    np.fill_diagonal(similarity, 1.0)
    return fingerprints.toarray().astype(np.float32), similarity


def _component_labels(similarity: np.ndarray, threshold: float) -> np.ndarray:
    adjacency = (similarity >= threshold).astype(np.int8)
    _, labels = connected_components(adjacency, directed=False)
    return labels.astype(int)


def _split_topology(
    compounds: pd.DataFrame,
    similarity: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    assignments = compounds[["compound_id", "scaffold"]].copy()
    group_sets: list[tuple[str, np.ndarray, float | None]] = [
        ("bemis_murcko", compounds["scaffold"].astype(str).to_numpy(), None)
    ]
    for similarity_threshold in SIMILARITY_THRESHOLDS:
        labels = _component_labels(similarity, similarity_threshold)
        column = f"similarity_component_{similarity_threshold:.2f}".replace(".", "p")
        assignments[column] = [f"SIM{similarity_threshold:.2f}-{value:03d}" for value in labels]
        group_sets.append((column, assignments[column].to_numpy(), similarity_threshold))

    upper = np.triu(np.ones_like(similarity, dtype=bool), k=1)
    for name, groups, group_threshold in group_sets:
        counts = pd.Series(groups).value_counts()
        different = groups[:, None] != groups[None, :]
        cross = upper & different
        cross_values = similarity[cross]
        rows.append(
            {
                "split_definition": name,
                "similarity_threshold": group_threshold,
                "n_groups": int(len(counts)),
                "largest_group_n": int(counts.max()),
                "largest_group_fraction": float(counts.max() / len(groups)),
                "singleton_group_count": int(counts.eq(1).sum()),
                "effective_group_count": _effective_count(pd.Series(groups)),
                "maximum_cross_group_tanimoto": (
                    float(cross_values.max()) if len(cross_values) else float("nan")
                ),
                "cross_group_pair_fraction_ge_0p80": (
                    float(np.mean(cross_values >= 0.80)) if len(cross_values) else float("nan")
                ),
                "cross_group_pair_fraction_ge_0p85": (
                    float(np.mean(cross_values >= 0.85)) if len(cross_values) else float("nan")
                ),
                "interpretation": (
                    "Outcome-blind graph components prohibit any cross-group pair at or "
                    "above the named threshold; Bemis-Murcko does not."
                ),
            }
        )
    return pd.DataFrame(rows), assignments


def _source_collapsed_pk_frames(
    compounds: pd.DataFrame,
    measurements: pd.DataFrame,
    studies: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    features, _ = merge_feature_layers(compounds, None)
    tasks = prepare_pk_tasks(compounds, measurements, studies, features)
    study_context = studies[["pk_study_id", "dose_value", "dose_unit", "route"]].drop_duplicates(
        "pk_study_id"
    )
    frames: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, Any]] = []

    for endpoint, task in tasks.items():
        work = task.copy()
        if "dose_value" not in work or work["dose_value"].isna().all():
            work = work.merge(
                study_context,
                on="pk_study_id",
                how="left",
                suffixes=("", "_study"),
                validate="many_to_one",
            )
        work["source_target_value"] = pd.to_numeric(work["target_value"], errors="coerce")
        endpoint_names = [endpoint]
        if endpoint == "po_cmax_dose_normalized":
            endpoint_names = ["po_cmax_raw", "po_cmax_dose_normalized"]
        for output_endpoint in endpoint_names:
            candidate = work.copy()
            if output_endpoint == "po_cmax_raw":
                candidate["analysis_target"] = pd.to_numeric(candidate["value"], errors="coerce")
            else:
                candidate["analysis_target"] = candidate["source_target_value"]
            candidate = candidate[
                candidate["analysis_target"].gt(0) & np.isfinite(candidate["analysis_target"])
            ].copy()
            signature_columns = [
                "compound_id",
                "analysis_target",
                "dose_value",
                "dose_unit",
                "route",
            ]
            unique = candidate.drop_duplicates(signature_columns).copy()
            unique["target_log10"] = np.log10(unique["analysis_target"])
            collapsed = (
                unique.groupby("compound_id", as_index=False)
                .agg(
                    target_log10=("target_log10", "mean"),
                    n_unique_source_signatures=("analysis_target", "size"),
                    dose_value=("dose_value", "first"),
                    dose_count=("dose_value", "nunique"),
                )
                .merge(
                    compounds[
                        [
                            "compound_id",
                            "standardized_smiles",
                            "scaffold",
                            "series_id",
                        ]
                    ],
                    on="compound_id",
                    how="inner",
                    validate="one_to_one",
                )
            )
            collapsed["endpoint"] = output_endpoint
            frames[output_endpoint] = collapsed
            audit_rows.append(
                {
                    "endpoint": output_endpoint,
                    "raw_evidence_rows": int(len(candidate)),
                    "unique_source_signatures": int(len(unique)),
                    "compound_count": int(len(collapsed)),
                    "compounds_with_multiple_unique_signatures": int(
                        collapsed["n_unique_source_signatures"].gt(1).sum()
                    ),
                    "dose_values": "|".join(
                        str(value)
                        for value in sorted(
                            pd.to_numeric(candidate["dose_value"], errors="coerce").dropna().unique()
                        )
                    ),
                    "compounds_observed_at_multiple_doses": int(collapsed["dose_count"].gt(1).sum()),
                    "aggregation": (
                        "geometric mean after collapsing identical compound/value/dose/"
                        "route source signatures"
                    ),
                }
            )
    return frames, pd.DataFrame(audit_rows)


def _source_collapsed_herg_frame(
    compounds: pd.DataFrame,
    measurements: pd.DataFrame,
) -> pd.DataFrame:
    exact = measurements[
        measurements["endpoint"].eq("herg_ic50")
        & measurements["model_eligible"].fillna(False)
        & measurements["relation"].isin(["=", "~"])
        & measurements["value"].gt(0)
    ].copy()
    exact["target_log10"] = 6.0 - np.log10(exact["value"].astype(float))
    exact = exact.drop_duplicates(["compound_id", "target_log10", "unit", "relation"])
    collapsed = (
        exact.groupby("compound_id", as_index=False)
        .agg(
            target_log10=("target_log10", "mean"),
            n_unique_source_signatures=("target_log10", "size"),
        )
        .merge(
            compounds[["compound_id", "standardized_smiles", "scaffold", "series_id"]],
            on="compound_id",
            how="inner",
            validate="one_to_one",
        )
    )
    collapsed["endpoint"] = "herg_pic50"
    return collapsed


def _dose_target_audit(
    frames: dict[str, pd.DataFrame],
    studies: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    definitions = {
        "iv_auc_dose_normalized": (
            True,
            "AUC is dose dependent; current target is normalized by IV dose.",
            "retain_with_linearity_caveat",
        ),
        "po_auc_dose_normalized": (
            True,
            "AUC is dose dependent; current target is normalized by PO dose.",
            "retain_with_linearity_caveat",
        ),
        "vdss": (
            False,
            "Vdss should be dose independent under linear PK.",
            "retain_but_require_protocol_and_nonlinearity_checks",
        ),
        "po_cmax_raw": (
            True,
            "Cmax is dose dependent, but the current production target is raw ng/mL.",
            "retract_raw_optimizer_endpoint",
        ),
        "po_cmax_dose_normalized": (
            True,
            "Dose-normalized Cmax is the defensible sensitivity target only if PK is linear.",
            "use_as_replacement_discovery_target",
        ),
        "po_tmax": (
            False,
            "Tmax is not algebraically normalized by dose, but can change with formulation or nonlinear absorption.",
            "retain_as_protocol_confounded_discovery_target",
        ),
    }
    for endpoint, frame in frames.items():
        route = "IV" if endpoint.startswith("iv_") or endpoint == "vdss" else "PO"
        route_studies = studies[studies["route"].astype(str).str.upper().eq(route)]
        dose_values = sorted(pd.to_numeric(route_studies["dose_value"], errors="coerce").dropna().unique())
        dose_dependent, rationale, action = definitions[endpoint]
        rows.append(
            {
                "endpoint": endpoint,
                "route": route,
                "dose_dependent_quantity": dose_dependent,
                "observed_doses_mg_kg": "|".join(str(value) for value in dose_values),
                "n_observed_doses": int(len(dose_values)),
                "n_compounds": int(len(frame)),
                "compounds_at_multiple_doses": int(frame["dose_count"].gt(1).sum()),
                "rationale": rationale,
                "reviewer_action": action,
            }
        )
    return pd.DataFrame(rows)


def _attach_analysis_features(
    frame: pd.DataFrame,
    compounds: pd.DataFrame,
    descriptor_frame: pd.DataFrame,
    fingerprint_array: np.ndarray,
    similarity: np.ndarray,
    component_assignments: pd.DataFrame,
) -> pd.DataFrame:
    index_map = {compound_id: index for index, compound_id in enumerate(compounds["compound_id"].astype(str))}
    result = frame.merge(
        pd.concat(
            [
                compounds[["compound_id"]].reset_index(drop=True),
                descriptor_frame.reset_index(drop=True),
            ],
            axis=1,
        ),
        on="compound_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        component_assignments.drop(columns=["scaffold"]),
        on="compound_id",
        how="left",
        validate="one_to_one",
    )
    result["_full_index"] = result["compound_id"].map(index_map).astype(int)
    result.attrs["fingerprint_array"] = fingerprint_array
    result.attrs["similarity"] = similarity
    return result


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    return {
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
        "r2": float(r2_score(observed, predicted)),
        "spearman": _safe_spearman(observed, predicted),
        "fraction_within_0p3_log": float(np.mean(np.abs(error) <= 0.3)),
        "fraction_within_0p5_log": float(np.mean(np.abs(error) <= 0.5)),
    }


def _leave_group_out_benchmarks(
    frame: pd.DataFrame,
    *,
    group_column: str,
    fingerprint_array: np.ndarray,
    similarity: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = frame[group_column].astype(str).to_numpy()
    y = frame["target_log10"].to_numpy(dtype=float)
    descriptors = frame[list(DESCRIPTOR_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    full_indices = frame["_full_index"].to_numpy(dtype=int)
    unique_groups = np.unique(groups)
    for held_group in unique_groups:
        test = np.flatnonzero(groups == held_group)
        train = np.flatnonzero(groups != held_group)
        if len(train) < 5:
            continue
        train_full, test_full = full_indices[train], full_indices[test]
        train_mean = float(np.mean(y[train]))
        predictions: dict[str, np.ndarray] = {"train_mean": np.full(len(test), train_mean, dtype=float)}
        descriptor_model = Pipeline(
            [
                ("impute", SimpleImputer()),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        )
        descriptor_model.fit(descriptors.iloc[train], y[train])
        predictions["descriptor_ridge"] = descriptor_model.predict(descriptors.iloc[test])
        morgan_model = Ridge(alpha=10.0)
        morgan_model.fit(fingerprint_array[train_full], y[train])
        predictions["morgan_ridge"] = morgan_model.predict(fingerprint_array[test_full])
        train_test_similarity = similarity[np.ix_(test_full, train_full)]
        for neighbors in (1, 3, 5):
            k = min(neighbors, len(train))
            nearest = np.argpartition(
                -train_test_similarity,
                kth=k - 1,
                axis=1,
            )[:, :k]
            neighbor_similarity = np.take_along_axis(
                train_test_similarity,
                nearest,
                axis=1,
            )
            weights = np.maximum(neighbor_similarity, 1e-6)
            neighbor_values = y[train][nearest]
            predictions[f"morgan_{neighbors}nn"] = np.sum(
                weights * neighbor_values,
                axis=1,
            ) / np.sum(weights, axis=1)
        max_similarity = train_test_similarity.max(axis=1)
        for model, predicted in predictions.items():
            for position, row_index in enumerate(test):
                rows.append(
                    {
                        "compound_id": frame.iloc[row_index]["compound_id"],
                        "endpoint": frame.iloc[row_index]["endpoint"],
                        "group_definition": group_column,
                        "held_out_group": held_group,
                        "model": model,
                        "observed": y[row_index],
                        "predicted": float(predicted[position]),
                        "residual": float(y[row_index] - predicted[position]),
                        "absolute_error": float(abs(y[row_index] - predicted[position])),
                        "max_train_tanimoto": float(max_similarity[position]),
                    }
                )
    return pd.DataFrame(rows)


def _benchmark_summary(
    predictions: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_state)
    for (endpoint, definition, model), group in predictions.groupby(
        ["endpoint", "group_definition", "model"],
        sort=True,
    ):
        observed = group["observed"].to_numpy(dtype=float)
        predicted = group["predicted"].to_numpy(dtype=float)
        metrics = _metrics(observed, predicted)
        mean_errors = predictions[
            predictions["endpoint"].eq(endpoint)
            & predictions["group_definition"].eq(definition)
            & predictions["model"].eq("train_mean")
        ][["compound_id", "absolute_error"]].rename(
            columns={"absolute_error": "mean_baseline_absolute_error"}
        )
        paired = group.merge(
            mean_errors,
            on="compound_id",
            how="inner",
            validate="one_to_one",
        )
        paired["delta_mae_vs_mean"] = paired["absolute_error"] - paired["mean_baseline_absolute_error"]
        held_groups = paired["held_out_group"].unique()
        group_frames = {value: paired[paired["held_out_group"].eq(value)] for value in held_groups}
        deltas = np.empty(bootstrap_replicates, dtype=float)
        for replicate in range(bootstrap_replicates):
            sampled = rng.choice(held_groups, size=len(held_groups), replace=True)
            sample = pd.concat(
                [group_frames[value] for value in sampled],
                ignore_index=True,
            )
            deltas[replicate] = sample["delta_mae_vs_mean"].mean()
        rows.append(
            {
                "endpoint": endpoint,
                "group_definition": definition,
                "model": model,
                "n_compounds": int(len(group)),
                "n_held_out_groups": int(group["held_out_group"].nunique()),
                **metrics,
                "mean_delta_mae_vs_train_mean": float(paired["delta_mae_vs_mean"].mean()),
                "delta_mae_lower_95": float(np.quantile(deltas, 0.025)),
                "delta_mae_upper_95": float(np.quantile(deltas, 0.975)),
                "bootstrap_probability_improves_on_mean": float(np.mean(deltas < 0)),
                "selection_status": "fixed_baseline_no_model_selection",
            }
        )
    return pd.DataFrame(rows)


def _groupkfold_descriptor_predictions(
    frame: pd.DataFrame,
    target: np.ndarray,
) -> np.ndarray:
    groups = frame["scaffold"].astype(str).to_numpy()
    splits = GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(frame, groups=groups)
    descriptors = frame[list(DESCRIPTOR_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    result = np.full(len(frame), np.nan, dtype=float)
    for train, test in splits:
        model = Pipeline(
            [
                ("impute", SimpleImputer()),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        )
        model.fit(descriptors.iloc[train], target[train])
        result[test] = model.predict(descriptors.iloc[test])
    return result


def _y_scrambling_controls(
    frames: dict[str, pd.DataFrame],
    *,
    permutations: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    for endpoint_index, (endpoint, frame) in enumerate(frames.items()):
        y = frame["target_log10"].to_numpy(dtype=float)
        groups = frame["scaffold"].astype(str).to_numpy()
        observed_prediction = _groupkfold_descriptor_predictions(frame, y)
        observed_spearman = _safe_spearman(y, observed_prediction)
        observed_r2 = float(r2_score(y, observed_prediction))
        rng = np.random.default_rng(random_state + endpoint_index)
        informative_groups = [
            np.flatnonzero(groups == value) for value in np.unique(groups) if np.sum(groups == value) >= 2
        ]
        permutable_indices = (
            np.concatenate(informative_groups) if informative_groups else np.asarray([], dtype=int)
        )
        for mode in ("global", "within_scaffold"):
            null_spearman: list[float] = []
            null_r2: list[float] = []
            for replicate in range(permutations):
                permuted = y.copy()
                if mode == "global":
                    permuted = rng.permutation(permuted)
                else:
                    for indices in informative_groups:
                        permuted[indices] = rng.permutation(permuted[indices])
                prediction = _groupkfold_descriptor_predictions(frame, permuted)
                value_spearman = _safe_spearman(permuted, prediction)
                value_r2 = float(r2_score(permuted, prediction))
                null_spearman.append(value_spearman)
                null_r2.append(value_r2)
                null_rows.append(
                    {
                        "endpoint": endpoint,
                        "permutation_mode": mode,
                        "replicate": replicate,
                        "spearman": value_spearman,
                        "r2": value_r2,
                    }
                )
            finite_spearman = np.asarray([value for value in null_spearman if np.isfinite(value)])
            summary_rows.append(
                {
                    "endpoint": endpoint,
                    "model": "fixed_descriptor_ridge",
                    "split": "five_fold_bemis_murcko",
                    "permutation_mode": mode,
                    "permutations": permutations,
                    "observed_spearman": observed_spearman,
                    "observed_r2": observed_r2,
                    "spearman_monte_carlo_p": float(
                        (1 + np.sum(finite_spearman >= observed_spearman)) / (len(finite_spearman) + 1)
                    ),
                    "r2_monte_carlo_p": float(
                        (1 + np.sum(np.asarray(null_r2, dtype=float) >= observed_r2)) / (len(null_r2) + 1)
                    ),
                    "within_scaffold_permutable_compounds": int(len(permutable_indices)),
                    "within_scaffold_permutable_fraction": float(len(permutable_indices) / len(frame)),
                    "interpretation": (
                        "Negative control only; the within-scaffold null is weak when "
                        "many scaffolds are singletons."
                    ),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(null_rows)


def _measurement_dispersion(
    measurements: pd.DataFrame,
    studies: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    study_dose = studies[["pk_study_id", "dose_value", "dose_unit"]].drop_duplicates("pk_study_id")
    records: list[pd.DataFrame] = []
    for endpoint, route in (
        ("auc_0_inf", "IV"),
        ("auc_0_inf", "PO"),
        ("cmax", "PO"),
        ("tmax", "PO"),
        ("vdss", "IV"),
        ("herg_ic50", ""),
    ):
        work = measurements[
            measurements["endpoint"].eq(endpoint)
            & measurements["value"].gt(0)
            & measurements["relation"].isin(["=", "~"])
        ].copy()
        if route:
            work = work[work["route"].astype(str).str.upper().eq(route)]
        work = work.merge(
            study_dose,
            on="pk_study_id",
            how="left",
            suffixes=("", "_study"),
        )
        work["analysis_endpoint"] = f"{route.lower()}_{endpoint}" if route else endpoint
        work["analysis_value_log10"] = np.log10(work["value"].astype(float))
        if endpoint == "herg_ic50":
            work["analysis_value_log10"] = 6.0 - work["analysis_value_log10"]
        signature = [
            "compound_id",
            "analysis_endpoint",
            "analysis_value_log10",
            "dose_value",
            "dose_unit",
            "relation",
        ]
        records.append(work.drop_duplicates(signature))
    combined = pd.concat(records, ignore_index=True)
    compound = combined.groupby(["analysis_endpoint", "compound_id"], as_index=False).agg(
        n_unique_values=("analysis_value_log10", "nunique"),
        minimum=("analysis_value_log10", "min"),
        maximum=("analysis_value_log10", "max"),
        standard_deviation=("analysis_value_log10", "std"),
    )
    compound["range_log10"] = compound["maximum"] - compound["minimum"]
    rows: list[dict[str, Any]] = []
    for endpoint, group in compound.groupby("analysis_endpoint"):
        repeated = group[group["n_unique_values"].gt(1)]
        rows.append(
            {
                "endpoint": endpoint,
                "n_compounds": int(len(group)),
                "compounds_with_multiple_distinct_values": int(len(repeated)),
                "median_range_log10_among_repeated": (
                    float(repeated["range_log10"].median()) if len(repeated) else 0.0
                ),
                "maximum_range_log10": float(group["range_log10"].max()),
                "noise_ceiling_estimable": False,
                "reason": (
                    "No replicate identity or protocol covariates; observed spread mixes "
                    "source conflict, study context, and biological variability."
                ),
            }
        )
    return pd.DataFrame(rows), compound


def _measurement_selection_bias(
    compounds: pd.DataFrame,
    measurements: pd.DataFrame,
    descriptors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.concat(
        [
            compounds[["compound_id", "scaffold"]].reset_index(drop=True),
            descriptors.reset_index(drop=True),
        ],
        axis=1,
    )
    pk_ids = set(
        measurements.loc[
            measurements["species"].fillna("").str.casefold().eq("rat")
            & measurements["route"].fillna("").str.upper().isin(["IV", "PO"])
            & measurements["model_eligible"].fillna(False),
            "compound_id",
        ].astype(str)
    )
    herg_ids = set(
        measurements.loc[
            measurements["endpoint"].astype(str).str.startswith("herg")
            & measurements["model_eligible"].fillna(False),
            "compound_id",
        ].astype(str)
    )
    frame["pk_available"] = frame["compound_id"].astype(str).isin(pk_ids)
    frame["herg_available"] = frame["compound_id"].astype(str).isin(herg_ids)
    rows: list[dict[str, Any]] = []
    for endpoint in ("pk_available", "herg_available"):
        for descriptor in DESCRIPTOR_COLUMNS:
            available = frame.loc[frame[endpoint], descriptor].to_numpy(dtype=float)
            missing = frame.loc[~frame[endpoint], descriptor].to_numpy(dtype=float)
            pooled = math.sqrt((np.var(available, ddof=1) + np.var(missing, ddof=1)) / 2.0)
            smd = float((np.mean(available) - np.mean(missing)) / pooled) if pooled > 0 else float("nan")
            p_value = (
                float(mannwhitneyu(available, missing, alternative="two-sided").pvalue)
                if len(np.unique(np.concatenate([available, missing]))) > 1
                else float("nan")
            )
            rows.append(
                {
                    "availability_endpoint": endpoint,
                    "descriptor": descriptor,
                    "available_n": int(len(available)),
                    "missing_n": int(len(missing)),
                    "available_mean": float(np.mean(available)),
                    "missing_mean": float(np.mean(missing)),
                    "standardized_mean_difference": smd,
                    "mann_whitney_p": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["fdr_q"] = result.groupby("availability_endpoint")["mann_whitney_p"].transform(_benjamini_hochberg)
    by_scaffold = (
        frame.groupby("scaffold", as_index=False)
        .agg(
            scaffold_n=("compound_id", "size"),
            pk_available_fraction=("pk_available", "mean"),
            herg_available_fraction=("herg_available", "mean"),
        )
        .sort_values("scaffold_n", ascending=False)
    )
    return result, by_scaffold


def _structure_representation_audit(
    compounds: pd.DataFrame,
    measurements: pd.DataFrame,
    descriptors: pd.DataFrame,
) -> dict[str, Any]:
    correlations = descriptors[list(DESCRIPTOR_COLUMNS)].corr(method="spearman")
    redundant_pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(DESCRIPTOR_COLUMNS):
        for right in DESCRIPTOR_COLUMNS[left_index + 1 :]:
            value = float(correlations.loc[left, right])
            if np.isfinite(value) and abs(value) >= 0.95:
                redundant_pairs.append({"left": left, "right": right, "spearman": value})
    basic = measurements[measurements["endpoint"].eq("basic_pka")]
    strongest_basic = basic.groupby("compound_id")["value"].max()
    return {
        "n_compounds": int(len(compounds)),
        "partially_specified_stereochemistry": int(
            compounds["stereochemistry_status"].eq("partially_specified").sum()
        ),
        "neutral_standardized_form_count": int(compounds["formal_charge"].eq(0).sum()),
        "formal_charge_descriptor_unique_values": sorted(
            float(value) for value in descriptors["formal_charge"].unique()
        ),
        "compounds_with_basic_pka_evidence": int(strongest_basic.index.nunique()),
        "median_strongest_reported_basic_pka": float(strongest_basic.median()),
        "compounds_predicted_or_reported_predominantly_protonated_at_ph7p4": int(
            strongest_basic.gt(7.4).sum()
        ),
        "highly_redundant_descriptor_pairs": redundant_pairs,
        "reviewer_interpretation": (
            "The neutralized 2D parent representation cannot encode assay-pH charge "
            "state even though basic-pKa evidence suggests protonation is widespread. "
            "Reported pKas lack microscopic site/state assignment and cannot be treated "
            "as exact mechanistic features."
        ),
    }


def _protocol_completeness_audit(
    assay_protocols: pd.DataFrame,
    studies: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify whether nominal protocol records contain causal covariates."""

    protocol_fields = (
        "cell_system",
        "method",
        "temperature_c",
        "p_h",
        "duration_value",
        "strain",
        "matrix",
    )
    protocol_rows: list[dict[str, Any]] = []
    for family, group in assay_protocols.groupby("assay_family", dropna=False, sort=True):
        for field in protocol_fields:
            present = group[field].notna()
            if field in {"cell_system", "method", "strain", "matrix"}:
                present &= group[field].astype("string").str.strip().ne("")
            protocol_rows.append(
                {
                    "record_type": "assay_protocol",
                    "stratum": str(family),
                    "field": field,
                    "records": int(len(group)),
                    "nonmissing": int(present.sum()),
                    "fraction_nonmissing": float(present.mean()),
                    "mechanistic_role": {
                        "cell_system": "expression/background system and access",
                        "method": "assay transfer function",
                        "temperature_c": "channel kinetics, transport, and enzyme kinetics",
                        "p_h": "ligand microstate and protein state",
                        "duration_value": "equilibration, slow binding, and trapping",
                        "strain": "physiological system variability",
                        "matrix": "binding, free concentration, and distribution context",
                    }[field],
                }
            )

    study_fields = ("strain", "sex", "formulation", "vehicle", "matrix")
    study_rows: list[dict[str, Any]] = []
    for (species, route), group in studies.groupby(["species", "route"], dropna=False, sort=True):
        for field in study_fields:
            present = group[field].notna() & group[field].astype("string").str.strip().ne("")
            study_rows.append(
                {
                    "record_type": "pk_study",
                    "stratum": f"{species}_{route}",
                    "field": field,
                    "records": int(len(group)),
                    "nonmissing": int(present.sum()),
                    "fraction_nonmissing": float(present.mean()),
                    "mechanistic_role": {
                        "strain": "system physiology and clearance background",
                        "sex": "physiological and enzyme-expression context",
                        "formulation": "dissolution, supersaturation, and absorption",
                        "vehicle": "delivered state and precipitation risk",
                        "matrix": "measured compartment and binding context",
                    }[field],
                }
            )
    return pd.DataFrame(protocol_rows), pd.DataFrame(study_rows)


def _validation_issue_audit(validation_issues: pd.DataFrame) -> pd.DataFrame:
    """Separate unique normalization problems from exact duplicate issue rows."""

    parsed = validation_issues.copy()

    def compound_from_context(value: object) -> str | None:
        try:
            payload = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        compound = payload.get("compound") or payload.get("compound_id")
        return str(compound) if compound is not None else None

    parsed["affected_compound"] = parsed["context_json"].map(compound_from_context)
    rows: list[dict[str, Any]] = []
    for (severity, code), group in parsed.groupby(["severity", "code"], sort=True):
        unique = group.drop_duplicates(["source", "record_type", "severity", "code", "context_json"])
        rows.append(
            {
                "severity": severity,
                "code": code,
                "raw_issue_rows": int(len(group)),
                "unique_issue_rows": int(len(unique)),
                "exact_duplicate_issue_rows": int(len(group) - len(unique)),
                "affected_compounds": int(unique["affected_compound"].nunique()),
                "publication_status": (
                    "unresolved_source_or_pairing_limit" if severity == "error" else "manual_review_required"
                ),
                "reviewer_interpretation": (
                    "Raw issue-row counts must not be presented as independent errors; "
                    "unique unresolved issues still limit study-level reconstruction."
                ),
            }
        )
    return pd.DataFrame(rows)


def _pk_closure_audit(
    derived: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit algebraic closure without treating derived quantities as labels."""

    signature = [
        "endpoint",
        "compound_id",
        "reported_value",
        "recomputed_value",
        "closure_relative_error",
        "closure_status",
    ]
    unique = derived.drop_duplicates(signature).copy()
    summary_rows: list[dict[str, Any]] = []
    for endpoint, group in derived.groupby("endpoint", sort=True):
        unique_group = unique[unique["endpoint"].eq(endpoint)]
        failures = unique_group[unique_group["closure_status"].eq("fail")]
        errors = pd.to_numeric(
            unique_group["closure_relative_error"],
            errors="coerce",
        ).dropna()
        summary_rows.append(
            {
                "endpoint": endpoint,
                "raw_derived_rows": int(len(group)),
                "unique_derived_signatures": int(len(unique_group)),
                "unique_compounds": int(unique_group["compound_id"].nunique()),
                "failed_unique_signatures": int(len(failures)),
                "failed_compounds": int(failures["compound_id"].nunique()),
                "median_relative_error": float(errors.median()) if len(errors) else float("nan"),
                "maximum_relative_error": float(errors.max()) if len(errors) else float("nan"),
                "closure_tolerance": 0.15,
                "tolerance_status": "operational_qc_threshold_not_biological_constant",
                "model_role": "closure_diagnostic_only_never_independent_target",
            }
        )
    detail_columns = [
        "endpoint",
        "compound_id",
        "reported_value",
        "recomputed_value",
        "closure_relative_error",
        "closure_status",
        "source_locator",
        "context_note",
    ]
    failed_detail = (
        derived[derived["closure_status"].eq("fail")][detail_columns]
        .sort_values(["endpoint", "closure_relative_error"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return pd.DataFrame(summary_rows), failed_detail


def _herg_observation_consistency(
    measurements: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare concentration-specific inhibition with exact IC50 evidence.

    The Hill-1 calculation is a source-consistency diagnostic, not an assumption
    about the biological Hill coefficient.
    """

    exact = measurements[
        measurements["endpoint"].eq("herg_ic50")
        & measurements["relation"].isin(["=", "~"])
        & measurements["value"].gt(0)
    ].copy()
    exact_by_compound = exact.groupby("compound_id")["value"].median().rename("exact_ic50_um")
    inhibition = measurements[
        measurements["endpoint"].eq("herg_percent_inhibition")
        & measurements["relation"].isin(["=", "~"])
        & measurements["value"].between(0, 100)
        & measurements["test_concentration_value"].gt(0)
    ][
        [
            "compound_id",
            "value",
            "test_concentration_value",
            "test_concentration_unit",
            "source_locator",
        ]
    ].drop_duplicates()
    detail = inhibition.merge(
        exact_by_compound,
        on="compound_id",
        how="inner",
        validate="many_to_one",
    )
    concentration = detail["test_concentration_value"].to_numpy(dtype=float)
    ic50 = detail["exact_ic50_um"].to_numpy(dtype=float)
    detail["hill1_expected_inhibition_percent"] = 100.0 * concentration / (concentration + ic50)
    detail["observed_minus_hill1_percent"] = detail["value"] - detail["hill1_expected_inhibition_percent"]
    detail["absolute_hill1_discrepancy_percent"] = detail["observed_minus_hill1_percent"].abs()
    detail["interpretation"] = (
        "Diagnostic only: deviations can reflect Hill slope, protocol, timing, "
        "free concentration, source conflict, or measurement error."
    )
    summary = {
        "paired_rows": int(len(detail)),
        "paired_compounds": int(detail["compound_id"].nunique()),
        "spearman_observed_vs_hill1_expected": _safe_spearman(
            detail["value"].to_numpy(dtype=float),
            detail["hill1_expected_inhibition_percent"].to_numpy(dtype=float),
        ),
        "median_absolute_hill1_discrepancy_percent": float(
            detail["absolute_hill1_discrepancy_percent"].median()
        ),
        "mean_absolute_hill1_discrepancy_percent": float(detail["absolute_hill1_discrepancy_percent"].mean()),
        "maximum_absolute_hill1_discrepancy_percent": float(
            detail["absolute_hill1_discrepancy_percent"].max()
        ),
        "protocols_with_cell_method_temperature_ph_duration_complete": 0,
        "mechanistic_conclusion_allowed": False,
    }
    return detail, summary


def _interval_efficiency_audit(project_root: Path) -> pd.DataFrame:
    """Report interval width as well as coverage for existing OOF models."""

    rows: list[dict[str, Any]] = []
    pk_root = project_root / "research/models/pk_herg/pk"
    for endpoint in (
        "iv_auc_dose_normalized",
        "po_auc_dose_normalized",
        "vdss",
        "po_cmax_dose_normalized",
        "po_tmax",
    ):
        path = pk_root / endpoint / "structure_2d_predictions.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        for model, group in frame.groupby("model", sort=True):
            observed = group["observed_log10"].to_numpy(dtype=float)
            lower = group["interval_lower_log10"].to_numpy(dtype=float)
            upper = group["interval_upper_log10"].to_numpy(dtype=float)
            width = upper - lower
            observed_range = float(np.max(observed) - np.min(observed))
            rows.append(
                {
                    "endpoint": endpoint,
                    "model": model,
                    "evaluation_unit": "evidence_row",
                    "rows": int(len(group)),
                    "unique_compounds": int(group["compound_id"].nunique()),
                    "nominal_interval_level": 0.90,
                    "empirical_coverage": float(np.mean((observed >= lower) & (observed <= upper))),
                    "mean_width_log10": float(np.mean(width)),
                    "median_width_log10": float(np.median(width)),
                    "observed_range_log10": observed_range,
                    "mean_width_over_observed_range": (
                        float(np.mean(width) / observed_range) if observed_range > 0 else float("nan")
                    ),
                    "minimum_calibration_rows": (
                        int(group["calibration_rows"].min()) if "calibration_rows" in group else None
                    ),
                    "interval_status": (
                        "discovery_only_pending_dose_proportionality"
                        if endpoint == "po_cmax_dose_normalized"
                        else "cross_conformal_heuristic_not_prospectively_calibrated"
                    ),
                }
            )

    herg_path = project_root / "research/models/pk_herg/herg/conventional_exact_pic50_predictions.parquet"
    if herg_path.exists():
        frame = pd.read_parquet(herg_path)
        for model, group in frame.groupby("model", sort=True):
            observed = group["observed_pic50"].to_numpy(dtype=float)
            lower = group["interval_lower_pic50"].to_numpy(dtype=float)
            upper = group["interval_upper_pic50"].to_numpy(dtype=float)
            width = upper - lower
            observed_range = float(np.max(observed) - np.min(observed))
            rows.append(
                {
                    "endpoint": "herg_pic50_exact",
                    "model": model,
                    "evaluation_unit": "measurement_row",
                    "rows": int(len(group)),
                    "unique_compounds": int(group["compound_id"].nunique()),
                    "nominal_interval_level": 0.90,
                    "empirical_coverage": float(np.mean((observed >= lower) & (observed <= upper))),
                    "mean_width_log10": float(np.mean(width)),
                    "median_width_log10": float(np.median(width)),
                    "observed_range_log10": observed_range,
                    "mean_width_over_observed_range": (
                        float(np.mean(width) / observed_range) if observed_range > 0 else float("nan")
                    ),
                    "minimum_calibration_rows": (
                        int(group["calibration_rows"].min()) if "calibration_rows" in group else None
                    ),
                    "interval_status": ("cross_conformal_heuristic_not_prospectively_calibrated"),
                }
            )
    return pd.DataFrame(rows)


def _assay_panel_design_audit(project_root: Path) -> pd.DataFrame:
    path = project_root / "research/reports/pk_herg/final/assay_panel.csv"
    panel = pd.read_csv(path) if path.exists() else pd.DataFrame()
    columns = set(panel.columns)
    return pd.DataFrame(
        [
            {
                "design_element": "acquisition_score",
                "implemented": bool({"acquisition_priority_score", "information_gain_score"} & columns),
                "reviewer_status": "heuristic_not_expected_information_gain",
                "why": (
                    "No posterior, assay-noise model, utility, or expected entropy reduction is calculated."
                ),
                "allowed_use": "mechanistic assay prioritization",
                "prohibited_use": "unbiased prospective performance test",
            },
            {
                "design_element": "outcome_blind_selection",
                "implemented": False,
                "reviewer_status": "observed_endpoint_informed",
                "why": ("hERG class, PK extremes, and matched-pair discordance influence selection."),
                "allowed_use": "falsification-rich matched-pair design",
                "prohibited_use": "final generalization estimate",
            },
            {
                "design_element": "matched_pair_definition",
                "implemented": True,
                "reviewer_status": "operational_morgan_similarity_rule",
                "why": (
                    "Tanimoto >=0.55 plus endpoint discordance is useful for discovery "
                    "but does not prove a single causal transformation."
                ),
                "allowed_use": "candidate pair nomination followed by chemist review",
                "prohibited_use": "causal matched-molecular-pair claim without atom mapping",
            },
            {
                "design_element": "mw_quotas",
                "implemented": True,
                "reviewer_status": "coverage_quota_not_regime_evidence",
                "why": "650/700/750 bins are design strata, not biological thresholds.",
                "allowed_use": "coverage balancing",
                "prohibited_use": "MW cutoff validation",
            },
        ]
    )


def _methodological_gap_register() -> pd.DataFrame:
    """Machine-readable claim blockers found by the skeptical review."""

    rows = [
        (
            1,
            "generalization",
            "No independent medicinal-chemistry series or prospective holdout",
            "optimistic",
            "new_series",
            "analogue-space retrospective baseline only",
        ),
        (
            2,
            "target_definition",
            "Raw oral Cmax combines 3 and 5 mg/kg with no within-compound dose crossover",
            "uncontrolled",
            "local_plus_experimental",
            "retire raw Cmax; normalized sensitivity only until proportionality data",
        ),
        (
            3,
            "split",
            "Bemis-Murcko groups permit cross-fold analogues up to Tanimoto 0.937",
            "optimistic",
            "local_and_new_series",
            "similarity-component stress test plus locked new-series test",
        ),
        (
            4,
            "series_metadata",
            "Computed scaffolds are mislabeled as medicinal series",
            "unknown",
            "metadata_or_chemist_review",
            "do not claim leave-series-out performance",
        ),
        (
            5,
            "model_selection",
            "Best model and reported OOF performance use the same grouped CV",
            "optimistic",
            "local_then_prospective",
            "fixed baselines now; nested selection and untouched final set later",
        ),
        (
            6,
            "uncertainty",
            "OOF residuals are reused after family selection and intervals lack prospective calibration",
            "optimistic_or_uninformative",
            "prospective_data",
            "report width and coverage; calibrate only on locked external data",
        ),
        (
            7,
            "applicability_domain",
            "5th-percentile nearest-neighbor threshold is heuristic and training overlap counts as inside",
            "optimistic",
            "local_then_prospective",
            "separate train overlap and calibrate error-versus-distance prospectively",
        ),
        (
            8,
            "representation",
            "All standardized parents are neutral while basic pKa evidence implies widespread protonation",
            "uncontrolled",
            "hpc_or_validated_state_tool",
            "state-aware representation with pKa uncertainty",
        ),
        (
            9,
            "stereochemistry",
            "Eight compounds have partially specified stereochemistry",
            "unknown",
            "source_or_experimental",
            "resolve identity or carry explicit stereochemical uncertainty",
        ),
        (
            10,
            "protocol",
            "PK lacks strain, sex, formulation, vehicle, and matrix metadata",
            "confounded",
            "additional_data",
            "protocol-complete study records and raw profiles",
        ),
        (
            11,
            "protocol",
            "hERG lacks cell system, method, temperature, pH, duration, and free concentration",
            "confounded",
            "additional_data",
            "harmonized best-practice patch-clamp protocol",
        ),
        (
            12,
            "pseudoreplication",
            "Repeated source rows are not biological replicates and original metrics are evidence-row weighted",
            "optimistic_or_distorted",
            "local_and_additional_data",
            "compound-balanced analysis; collect replicate identities",
        ),
        (
            13,
            "selection_bias",
            "PK availability differs systematically in chemistry space",
            "unknown",
            "new_data",
            "measure outcome-blind coverage panel and model missingness",
        ),
        (
            14,
            "pk_identifiability",
            "Summary AUC/Cmax/Tmax/Vdss cannot identify dissolution, Fa, Fg, Fh, or rate constants",
            "nonidentifiable",
            "additional_data",
            "raw IV/PO profiles plus orthogonal process assays",
        ),
        (
            15,
            "derived_endpoints",
            "CL and F are algebraically derived and closure failures remain",
            "leakage_if_misused",
            "local_source_resolution",
            "closure diagnostic only; never independent labels",
        ),
        (
            16,
            "herg_observation_model",
            "IC50 and 10/30 uM inhibition can disagree under a simple equilibrium curve",
            "confounded",
            "additional_data",
            "full curves, kinetics, actual/free concentration, protocol metadata",
        ),
        (
            17,
            "herg_endpoint",
            "Static IC50 does not identify state dependence, onset, recovery, or trapping",
            "nonidentifiable",
            "additional_data",
            "voltage- and time-resolved electrophysiology",
        ),
        (
            18,
            "public_validation",
            "Attached public hERG data are far outside the internal large-molecule domain",
            "nontransportable",
            "new_compatible_series",
            "method reproduction only; no internal calibration claim",
        ),
        (
            19,
            "activity_cliffs",
            "Operational cliffs may be protocol or measurement cliffs",
            "unknown",
            "additional_data",
            "same-run replicate matched pairs and atom-mapped transformations",
        ),
        (
            20,
            "causal_inference",
            "Residual correlations cannot distinguish molecular mechanism from shared study context",
            "confounded",
            "experimental_perturbation",
            "preregistered intervention and mediation tests",
        ),
        (
            21,
            "multiple_testing",
            "Numerous endpoints, models, features, cutoffs, and residual hypotheses create researcher degrees of freedom",
            "optimistic",
            "local_preregistration",
            "lock primary hypotheses, controls, and multiplicity families before new outcomes",
        ),
        (
            22,
            "mw_regime",
            "Current sample cannot support a universal MW threshold",
            "unstable",
            "new_series",
            "continuous state variables; test cutoffs only as secondary hypotheses",
        ),
        (
            23,
            "assay_panel",
            "Panel score is a heuristic and outcome-informed",
            "selected_test_bias",
            "local_redesign",
            "mechanistic panel only; create a separate outcome-blind validation panel",
        ),
        (
            24,
            "physics",
            "No microstate, conformer, membrane, or receptor observable has been executed",
            "unsupported",
            "hpc",
            "all physics claims remain hypotheses until convergence and perturbational validation",
        ),
        (
            25,
            "physics_novelty",
            "Composite feature novelty does not establish a new physical mechanism",
            "overclaim",
            "literature_plus_experiment",
            "claim novelty only for validated coupling or prediction under intervention",
        ),
        (
            26,
            "receptor_ensemble",
            "Six raw coordinates are hypotheses, not six equally probable biological states",
            "model_uncertainty",
            "hpc_and_experimental",
            "prepare and validate separately; infer or bound weights under protocol",
        ),
        (
            27,
            "deep_learning",
            "The D-MPNN is one fixed 40-epoch, single-seed-per-fold comparator on repeated rows",
            "unstable",
            "local_but_low_priority",
            "do not interpret architecture ranking; repeat seeds only after adequate data",
        ),
        (
            28,
            "clinical_interpretation",
            "hERG block alone is not clinical proarrhythmic risk",
            "overclaim",
            "additional_data",
            "limit claims to hERG liability; broader cardiac assessment is separate",
        ),
        (
            29,
            "optimization",
            "Final-fit predictions include training compounds and no prospective calibration",
            "optimistic",
            "prospective_data",
            "optimizer remains disabled",
        ),
        (
            30,
            "reproducibility",
            "No immutable publication analysis snapshot is currently locked",
            "auditability",
            "publication_governance",
            "create a release-level frozen artifact only when publication scope is final",
        ),
        (
            31,
            "feature_semantics",
            "Conventional controls, proxies, derived functionals, and fundamental parameters can be conflated",
            "causal_double_counting_and_overclaim",
            "local_ontology_plus_future_direct_evidence",
            "enforce role and parent graphs; admit mechanisms only as free energies/rates with direct falsification",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "priority",
            "area",
            "gap",
            "likely_bias",
            "required_resource",
            "allowed_claim_or_remedy",
        ],
    )


def _optimizer_overlap_audit(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = project_root / "research/data/pk_herg/optimizer/optimizer_predictions_long.parquet"
    predictions = pd.read_parquet(path)
    reviewed = predictions.copy()
    exact_overlap = np.isclose(
        pd.to_numeric(reviewed["max_train_tanimoto"], errors="coerce"),
        1.0,
    )
    reviewed["evaluation_role"] = np.where(
        exact_overlap,
        "training_structure_in_sample_final_fit",
        "unmeasured_internal_analog",
    )
    reviewed["reviewer_domain_status"] = np.where(
        exact_overlap,
        "training_overlap_not_domain_evidence",
        np.where(
            reviewed["domain_status"].eq("inside"),
            "inside_unvalidated_similarity_rule",
            "outside_unvalidated_similarity_rule",
        ),
    )
    reviewed["qualified_for_optimization"] = False
    reviewed["reviewer_blocker"] = np.where(
        reviewed["endpoint"].eq("rat_po_cmax_dose_normalized"),
        "dose_normalization_assumes_unverified_within_compound_proportionality",
        "no_prospective_calibration_or_final_set",
    )
    rows: list[dict[str, Any]] = []
    for endpoint, group in reviewed.groupby("endpoint"):
        overlap = group["evaluation_role"].eq("training_structure_in_sample_final_fit")
        rows.append(
            {
                "endpoint": endpoint,
                "prediction_rows": int(len(group)),
                "training_structure_overlap_n": int(overlap.sum()),
                "unmeasured_internal_analog_n": int((~overlap).sum()),
                "reported_inside_domain_n": int(group["domain_status"].eq("inside").sum()),
                "inside_domain_training_overlap_n": int(
                    (overlap & group["domain_status"].eq("inside")).sum()
                ),
                "qualified_for_optimization_n": 0,
                "primary_issue": (
                    "Dose-normalized Cmax assumes unverified within-compound dose proportionality."
                    if endpoint == "rat_po_cmax_dose_normalized"
                    else "Training-overlap predictions and an unvalidated similarity "
                    "threshold inflate apparent coverage."
                ),
            }
        )
    return pd.DataFrame(rows), reviewed


def _activity_cliff_candidates(
    frames: dict[str, pd.DataFrame],
    similarity: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for endpoint, frame in frames.items():
        full_indices = frame["_full_index"].to_numpy(dtype=int)
        pair_similarity = similarity[np.ix_(full_indices, full_indices)]
        values = frame["target_log10"].to_numpy(dtype=float)
        endpoint_rows: list[dict[str, Any]] = []
        for left in range(len(frame)):
            for right in range(left + 1, len(frame)):
                sim = float(pair_similarity[left, right])
                if sim < 0.85:
                    continue
                delta = float(values[left] - values[right])
                endpoint_rows.append(
                    {
                        "endpoint": endpoint,
                        "compound_id_left": frame.iloc[left]["compound_id"],
                        "compound_id_right": frame.iloc[right]["compound_id"],
                        "morgan_tanimoto": sim,
                        "absolute_outcome_delta_log10": abs(delta),
                        "signed_left_minus_right_log10": delta,
                        "same_bemis_murcko_scaffold": bool(
                            frame.iloc[left]["scaffold"] == frame.iloc[right]["scaffold"]
                        ),
                        "operational_cliff_0p3_log": bool(abs(delta) >= 0.3),
                        "operational_cliff_0p5_log": bool(abs(delta) >= 0.5),
                        "interpretation": (
                            "Experiment-prioritization candidate only; protocol and "
                            "measurement uncertainty can mimic a molecular cliff."
                        ),
                    }
                )
        rows.extend(endpoint_rows)
        cliff = pd.DataFrame(endpoint_rows)
        summary_rows.append(
            {
                "endpoint": endpoint,
                "pairs_ge_0p85_similarity": int(len(cliff)),
                "pairs_ge_0p3_log_difference": (
                    int(cliff["operational_cliff_0p3_log"].sum()) if len(cliff) else 0
                ),
                "pairs_ge_0p5_log_difference": (
                    int(cliff["operational_cliff_0p5_log"].sum()) if len(cliff) else 0
                ),
                "maximum_observed_delta_log10": (
                    float(cliff["absolute_outcome_delta_log10"].max()) if len(cliff) else float("nan")
                ),
                "threshold_status": ("operational screening thresholds, not biological constants"),
            }
        )
    detail = pd.DataFrame(rows)
    if len(detail):
        detail = detail.sort_values(
            ["absolute_outcome_delta_log10", "morgan_tanimoto"],
            ascending=[False, False],
        )
    return detail, pd.DataFrame(summary_rows)


def _residual_target_definition_sensitivity(
    predictions: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    random_state: int,
) -> pd.DataFrame:
    fixed_model = "descriptor_ridge"
    fixed_split = "scaffold"
    selected = predictions[
        predictions["model"].eq(fixed_model) & predictions["group_definition"].eq(fixed_split)
    ].copy()
    endpoint_frames = {
        endpoint: group[["compound_id", "observed", "residual"]].drop_duplicates("compound_id")
        for endpoint, group in selected.groupby("endpoint")
    }
    comparisons = (
        ("po_auc_dose_normalized", "po_cmax_raw"),
        ("po_auc_dose_normalized", "po_cmax_dose_normalized"),
        ("iv_auc_dose_normalized", "po_cmax_raw"),
        ("iv_auc_dose_normalized", "po_cmax_dose_normalized"),
    )
    rows: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(comparisons):
        merged = endpoint_frames[left].merge(
            endpoint_frames[right],
            on="compound_id",
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        scaffold_lookup = selected[["compound_id", "held_out_group"]].drop_duplicates("compound_id")
        merged = merged.merge(
            scaffold_lookup,
            on="compound_id",
            how="left",
            validate="one_to_one",
        ).rename(columns={"held_out_group": "scaffold"})
        result, _ = grouped_bootstrap_spearman(
            merged,
            x="residual_left",
            y="residual_right",
            group_column="scaffold",
            bootstrap_replicates=bootstrap_replicates,
            random_state=random_state + index,
        )
        result.update(
            {
                "left_endpoint": left,
                "right_endpoint": right,
                "model": fixed_model,
                "split": "leave_one_bemis_murcko_group_out",
                "interpretation": (
                    "Target-definition sensitivity; residual covariance is not a causal mechanism."
                ),
            }
        )
        rows.append(result)
    result_frame = pd.DataFrame(rows)
    result_frame["within_scaffold_fdr_q"] = _benjamini_hochberg(result_frame["within_scaffold_permutation_p"])
    return result_frame


def run_reviewer_audit(
    project_root: Path,
    *,
    bootstrap_replicates: int = 2000,
    permutations: int = 500,
    random_state: int = 20260724,
) -> dict[str, Any]:
    canonical_root = project_root / "research/data/pk_herg/canonical"
    output = project_root / "research/reports/pk_herg/reviewer_audit"
    output.mkdir(parents=True, exist_ok=True)
    tables = load_canonical_tables(canonical_root)
    compounds = compound_model_frame(
        tables["compounds"],
        tables.get("compound_aliases"),
    )
    measurements = tables["measurements"]
    studies = tables["pk_studies"]
    descriptors = rdkit_descriptors(compounds["standardized_smiles"])
    fingerprints, similarity = _full_tanimoto(compounds["standardized_smiles"])

    topology, assignments = _split_topology(compounds, similarity)
    atomic_write_csv(output / "split_topology.csv", topology)
    atomic_write_csv(output / "similarity_component_assignments.csv", assignments)

    pk_frames, target_rows = _source_collapsed_pk_frames(
        compounds,
        measurements,
        studies,
    )
    herg_frame = _source_collapsed_herg_frame(compounds, measurements)
    target_audit = _dose_target_audit(pk_frames, studies).merge(
        target_rows,
        on="endpoint",
        how="left",
        validate="one_to_one",
    )
    atomic_write_csv(output / "target_definition_audit.csv", target_audit)

    all_frames: dict[str, pd.DataFrame] = {
        **pk_frames,
        "herg_pic50": herg_frame,
    }
    enriched: dict[str, pd.DataFrame] = {}
    for endpoint, frame in all_frames.items():
        enriched[endpoint] = _attach_analysis_features(
            frame,
            compounds,
            descriptors,
            fingerprints,
            similarity,
            assignments,
        )

    prediction_frames: list[pd.DataFrame] = []
    group_definitions = (
        "scaffold",
        f"similarity_component_{PRIMARY_SIMILARITY_COMPONENT_THRESHOLD:.2f}".replace(
            ".",
            "p",
        ),
    )
    for frame in enriched.values():
        for group_definition in group_definitions:
            group_count = frame[group_definition].nunique()
            largest_fraction = frame[group_definition].value_counts().max() / len(frame)
            if group_count < 3 or largest_fraction > 0.80:
                continue
            predictions = _leave_group_out_benchmarks(
                frame,
                group_column=group_definition,
                fingerprint_array=fingerprints,
                similarity=similarity,
            )
            prediction_frames.append(predictions)
    benchmark_predictions = pd.concat(prediction_frames, ignore_index=True)
    benchmark_summary = _benchmark_summary(
        benchmark_predictions,
        bootstrap_replicates=bootstrap_replicates,
        random_state=random_state,
    )
    atomic_write_parquet(
        output / "fixed_baseline_loco_predictions.parquet",
        benchmark_predictions,
    )
    atomic_write_csv(
        output / "fixed_baseline_loco_metrics.csv",
        benchmark_summary,
    )

    scramble_summary, scramble_null = _y_scrambling_controls(
        enriched,
        permutations=permutations,
        random_state=random_state,
    )
    atomic_write_csv(output / "y_scrambling_summary.csv", scramble_summary)
    atomic_write_parquet(output / "y_scrambling_null.parquet", scramble_null)

    dispersion_summary, dispersion_compound = _measurement_dispersion(
        measurements,
        studies,
    )
    atomic_write_csv(output / "measurement_dispersion_summary.csv", dispersion_summary)
    atomic_write_csv(
        output / "measurement_dispersion_by_compound.csv",
        dispersion_compound,
    )

    missingness, availability_by_scaffold = _measurement_selection_bias(
        compounds,
        measurements,
        descriptors,
    )
    atomic_write_csv(output / "measurement_selection_bias.csv", missingness)
    atomic_write_csv(
        output / "measurement_availability_by_scaffold.csv",
        availability_by_scaffold,
    )

    representation = _structure_representation_audit(
        compounds,
        measurements,
        descriptors,
    )
    atomic_write_json(output / "structure_representation_audit.json", representation)

    protocol_completeness, study_completeness = _protocol_completeness_audit(
        tables["assay_protocols"],
        studies,
    )
    atomic_write_csv(
        output / "assay_protocol_completeness_reviewer.csv",
        protocol_completeness,
    )
    atomic_write_csv(
        output / "pk_study_protocol_completeness_reviewer.csv",
        study_completeness,
    )

    validation_audit = _validation_issue_audit(tables["validation_issues"])
    atomic_write_csv(output / "validation_issue_audit.csv", validation_audit)

    closure_summary, closure_failures = _pk_closure_audit(tables["derived_pk_parameters"])
    atomic_write_csv(output / "pk_closure_summary.csv", closure_summary)
    atomic_write_csv(output / "pk_closure_failures.csv", closure_failures)

    herg_consistency, herg_consistency_summary = _herg_observation_consistency(measurements)
    atomic_write_csv(
        output / "herg_observation_consistency.csv",
        herg_consistency,
    )
    atomic_write_json(
        output / "herg_observation_consistency_summary.json",
        herg_consistency_summary,
    )

    interval_efficiency = _interval_efficiency_audit(project_root)
    atomic_write_csv(
        output / "prediction_interval_efficiency.csv",
        interval_efficiency,
    )

    assay_design = _assay_panel_design_audit(project_root)
    atomic_write_csv(output / "assay_panel_design_audit.csv", assay_design)

    gap_register = _methodological_gap_register()
    atomic_write_csv(output / "methodological_gap_register.csv", gap_register)

    conventional_ontology = conventional_feature_ontology_frame()
    fast_physics_ontology = feature_ontology_frame()
    parameter_ontology = parameter_ontology_frame()
    validate_parameter_ontology(parameter_ontology)
    atomic_write_csv(
        output / "conventional_feature_role_audit.csv",
        conventional_ontology,
    )
    atomic_write_csv(
        output / "fast_physics_feature_role_audit.csv",
        fast_physics_ontology,
    )
    atomic_write_csv(
        output / "pk_parameter_ontology.csv",
        parameter_ontology[parameter_ontology["domain"].eq("pk")],
    )
    atomic_write_csv(
        output / "herg_parameter_ontology.csv",
        parameter_ontology[parameter_ontology["domain"].eq("herg")],
    )

    optimizer_audit, reviewed_optimizer = _optimizer_overlap_audit(project_root)
    atomic_write_csv(output / "optimizer_overlap_audit.csv", optimizer_audit)
    atomic_write_parquet(
        output / "optimizer_reviewer_safe_contract.parquet",
        reviewed_optimizer,
    )
    atomic_write_csv(
        output / "optimizer_reviewer_safe_contract.csv",
        reviewed_optimizer,
    )

    cliffs, cliff_summary = _activity_cliff_candidates(enriched, similarity)
    atomic_write_csv(output / "activity_cliff_candidates.csv", cliffs)
    atomic_write_csv(output / "activity_cliff_summary.csv", cliff_summary)

    residual_sensitivity = _residual_target_definition_sensitivity(
        benchmark_predictions,
        bootstrap_replicates=bootstrap_replicates,
        random_state=random_state,
    )
    atomic_write_csv(
        output / "cmax_target_residual_sensitivity.csv",
        residual_sensitivity,
    )

    summary = {
        "status": "reviewer_audit_complete",
        "n_internal_compounds": int(len(compounds)),
        "nominal_bemis_murcko_scaffolds": int(compounds["scaffold"].nunique()),
        "medicinal_series_metadata_available": False,
        "similarity_component_threshold_primary": (PRIMARY_SIMILARITY_COMPONENT_THRESHOLD),
        "all_compounds_one_component_at_tanimoto_0p70": bool(
            topology.loc[
                topology["split_definition"].eq("similarity_component_0p70"),
                "n_groups",
            ].iloc[0]
            == 1
        ),
        "raw_cmax_target_valid_for_optimizer": False,
        "unique_validation_error_rows": int(
            validation_audit.loc[
                validation_audit["severity"].eq("error"),
                "unique_issue_rows",
            ].sum()
        ),
        "bioavailability_closure_failure_compounds": int(
            closure_summary.loc[
                closure_summary["endpoint"].eq("bioavailability"),
                "failed_compounds",
            ].iloc[0]
        ),
        "herg_ic50_inhibition_consistency_compounds": int(herg_consistency_summary["paired_compounds"]),
        "fixed_baseline_model_selection_performed": False,
        "conventional_internal_descriptor_count": int(
            conventional_ontology["selected_internal_descriptor"].sum()
        ),
        "fast_physics_model_proxy_count": int(
            fast_physics_ontology["selected_model_columns"].astype(str).ne("").sum()
        ),
        "pk_parameter_modules": int(parameter_ontology["domain"].eq("pk").sum()),
        "herg_parameter_modules": int(parameter_ontology["domain"].eq("herg").sum()),
        "bootstrap_replicates": bootstrap_replicates,
        "y_scrambling_permutations": permutations,
        "physics_executed": False,
        "decision_track_promotion": False,
    }
    atomic_write_json(output / "reviewer_audit_summary.json", summary)
    report = f"""# Skeptical reviewer audit: quantitative local controls

## Scope

This audit attempts to falsify the current retrospective evidence. It uses no HPC
approximation and performs no model promotion. Full interpretation and the redesigned
research plan are in the companion reviewer report.

## Critical quantitative findings

- The 110 compounds have {summary["nominal_bemis_murcko_scaffolds"]} nominal
  Bemis-Murcko scaffolds, but all compounds form one Morgan-similarity-connected
  component at Tanimoto 0.70. The scaffold labels are not independent medicinal-series
  metadata.
- Raw PO Cmax combines 3 and 5 mg/kg studies; no compound was observed at both doses.
  It has therefore been withdrawn from production modeling and the optimizer; the
  dose-normalized replacement remains discovery-only pending proportionality data.
- The fixed-baseline tables compare a training-mean control, descriptor ridge, Morgan
  ridge, and 1/3/5-nearest-analogue predictors without selecting among them.
- Y-scrambling, measurement-selection, source-dispersion, optimizer-overlap, and
  operational activity-cliff results are written as separate machine-readable tables.
- Interval efficiency is reported with coverage; high coverage from intervals as wide
  as the observed target range is not evidence of useful calibration.
- Protocol completeness, unique-versus-duplicate validation issues, PK algebraic
  closure, hERG IC50/inhibition consistency, and assay-panel semantics are audited.
- Neutralized parent structures give no formal-charge variation even though strongest
  reported basic pKas exceed 7.4 for nearly the complete series. Static parent charge
  is not an assay-state representation.
- The feature-role audit separates {summary["conventional_internal_descriptor_count"]} conventional
  empirical controls from fast-physics diagnostics/proxies. None is called
  a fundamental parameter. Fundamental free energies/rates, derived functionals,
  boundary conditions, and observation parameters are separated in the PK and hERG
  parameter ontologies.

## Truth boundary

Similarity components are outcome-blind split stress tests, not biological series.
Operational cliffs are experiment-selection candidates, not mechanisms. Source
signature collapse is a sensitivity analysis, not proof that repeated rows are
duplicates. No retrospective control replaces a new-series prospective evaluation.
"""
    atomic_write_text(output / "quantitative_reviewer_audit.md", report)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=20260724)
    args = parser.parse_args()
    summary = run_reviewer_audit(
        args.project_root.resolve(),
        bootstrap_replicates=args.bootstrap_replicates,
        permutations=args.permutations,
        random_state=args.random_state,
    )
    print(summary)


if __name__ == "__main__":
    main()
