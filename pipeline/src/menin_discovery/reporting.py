"""Publication-oriented tables, figures, diagnostics, and narrative reports."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_curve

from .config import HERG_TARGET, MENIN_TARGET
from .features import fingerprint_matrix

COLORS = {
    "navy": "#264653",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "red": "#E76F51",
    "gray": "#7A7A7A",
}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna("").astype(str).str.casefold().isin({"true", "1", "yes"})


def _identity_column(frame: pd.DataFrame) -> str:
    return next(
        (
            column
            for column in (
                "structure_id",
                "standard_inchi_key",
                "standardized_smiles",
                "canonical_smiles",
                "smiles",
            )
            if column in frame.columns
        ),
        "smiles",
    )


def _primary_menin_prefix(settings: dict[str, Any] | None) -> str:
    model = (settings or {}).get("modeling", {})

    def slug(value: object) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")

    endpoint = slug(model.get("primary_menin_endpoint", "IC50"))
    family = slug(model.get("primary_menin_assay_family", "biochemical_binding"))
    return f"menin_activity_{endpoint}_{family}"


def _markdown_table(frame: pd.DataFrame, *, limit: int = 20) -> str:
    if frame.empty:
        return "_No rows available._"
    data = frame.head(limit).copy()
    for column in data.columns:
        if pd.api.types.is_float_dtype(data[column]):
            data[column] = data[column].map(lambda value: "" if pd.isna(value) else f"{value:.3g}")
    lines = [
        "| " + " | ".join(map(str, data.columns)) + " |",
        "| " + " | ".join(["---"] * len(data.columns)) + " |",
    ]

    def render_cell(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("[", "\\[").replace("]", "\\]").replace("\n", " ")

    for row in data.itertuples(index=False, name=None):
        values = [render_cell(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _write_analysis_tables(
    processed_dir: Path,
    reports_dir: Path,
    settings: dict[str, Any] | None = None,
    analysis_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    menin = _read_csv(processed_dir / "menin_activity_measurements.csv")
    menin_tasks = _read_csv(processed_dir / "menin_compounds_curated.csv")
    herg = _read_csv(processed_dir / "herg_activity_measurements.csv")
    herg_tasks = _read_csv(processed_dir / "herg_compounds_curated.csv")
    pk = _read_csv(processed_dir / "pk_admet_observations.csv")

    inventory_rows: list[dict[str, Any]] = []
    for name, frame in (
        ("menin_measurements", menin),
        ("menin_endpoint_assay_tasks", menin_tasks),
        ("herg_measurements", herg),
        ("herg_endpoint_assay_tasks", herg_tasks),
        ("pk_admet_observations", pk),
    ):
        identity = _identity_column(frame) if not frame.empty and "smiles" in frame else None
        eligible = (
            int(_as_bool(frame["is_modeling_eligible"]).sum()) if "is_modeling_eligible" in frame else None
        )
        inventory_rows.append(
            {
                "dataset": name,
                "rows": int(len(frame)),
                "unique_structures": int(frame[identity].nunique()) if identity else None,
                "modeling_eligible_rows": eligible,
            }
        )
    inventory = pd.DataFrame(inventory_rows)

    endpoint_assay = pd.DataFrame()
    if not menin.empty:
        identity = _identity_column(menin)
        work = menin.copy()
        work["eligible"] = _as_bool(work.get("is_modeling_eligible", pd.Series(False, index=work.index)))
        work["exact"] = _as_bool(work.get("is_exact", pd.Series(False, index=work.index)))
        work["eligible_p_activity"] = pd.to_numeric(work.get("p_value"), errors="coerce").where(
            work["eligible"]
        )
        endpoint_assay = (
            work.groupby(["endpoint", "assay_family"], dropna=False)
            .agg(
                measurements=("endpoint", "size"),
                unique_structures=(identity, "nunique"),
                exact_measurements=("exact", "sum"),
                modeling_eligible=("eligible", "sum"),
                median_p_activity=("eligible_p_activity", "median"),
            )
            .reset_index()
            .sort_values("measurements", ascending=False)
        )

    source_endpoint = pd.DataFrame()
    if not menin.empty:
        identity = _identity_column(menin)
        source_endpoint = (
            menin.groupby(["source", "endpoint"], dropna=False)
            .agg(measurements=("source", "size"), unique_structures=(identity, "nunique"))
            .reset_index()
            .sort_values("measurements", ascending=False)
        )

    replicate = pd.DataFrame()
    if not menin.empty:
        identity = _identity_column(menin)
        exact = menin[
            _as_bool(menin.get("is_exact", pd.Series(False, index=menin.index)))
            & _as_bool(menin.get("is_modeling_eligible", pd.Series(False, index=menin.index)))
        ].copy()
        exact["p_value"] = pd.to_numeric(exact.get("p_value"), errors="coerce")
        exact = exact.dropna(subset=["p_value"])
        replicate = (
            exact.groupby([identity, "endpoint", "assay_family"], dropna=False)
            .agg(
                measurements=("p_value", "size"),
                sources=("source", "nunique"),
                median_p_activity=("p_value", "median"),
                min_p_activity=("p_value", "min"),
                max_p_activity=("p_value", "max"),
                sd_p_activity=("p_value", "std"),
            )
            .reset_index()
        )
        replicate["range_log10"] = replicate["max_p_activity"] - replicate["min_p_activity"]
        replicate = replicate[replicate["measurements"] >= 2].sort_values("range_log10", ascending=False)

    settings = settings or {}
    herg_cfg = settings.get("herg", {})
    primary_herg_endpoint = str(herg_cfg.get("primary_endpoint", "IC50"))
    primary_herg_family = str(herg_cfg.get("primary_assay_family", "electrophysiology_functional"))

    def hERG_label_summary(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
        if frame.empty or "herg_blocker_label" not in frame:
            return pd.DataFrame()
        labels = pd.to_numeric(frame["herg_blocker_label"], errors="coerce")
        return pd.DataFrame(
            {
                "scope": [scope] * 3,
                "endpoint": [primary_herg_endpoint if scope == "primary" else "pooled"] * 3,
                "assay_family": [primary_herg_family if scope == "primary" else "pooled"] * 3,
                "label": ["blocker", "non_blocker", "ambiguous_10_to_30_uM"],
                "task_rows": [
                    int((labels == 1).sum()),
                    int((labels == 0).sum()),
                    int(labels.isna().sum()),
                ],
            }
        )

    primary_herg_tasks = herg_tasks.copy()
    if not primary_herg_tasks.empty and "endpoint" in primary_herg_tasks:
        primary_herg_tasks = primary_herg_tasks[
            primary_herg_tasks["endpoint"].astype(str).str.casefold() == primary_herg_endpoint.casefold()
        ]
    if not primary_herg_tasks.empty and "assay_family" in primary_herg_tasks:
        primary_herg_tasks = primary_herg_tasks[
            primary_herg_tasks["assay_family"].astype(str).str.casefold() == primary_herg_family.casefold()
        ]
    primary_herg_labels = hERG_label_summary(primary_herg_tasks, "primary")
    pooled_herg_labels = hERG_label_summary(herg_tasks, "pooled_sensitivity")
    herg_labels = primary_herg_labels

    pk_coverage = pd.DataFrame()
    if not pk.empty:
        endpoint_col = "admet_endpoint" if "admet_endpoint" in pk else "standard_type"
        group_columns = [endpoint_col]
        for context_column in (
            "species",
            "matrix",
            "administration_route",
            "experimental_context",
        ):
            if context_column in pk:
                group_columns.append(context_column)
        pk_coverage = (
            pk.groupby(group_columns, dropna=False)
            .agg(
                observations=(endpoint_col, "size"),
                compounds=("molecule_chembl_id", "nunique")
                if "molecule_chembl_id" in pk
                else (endpoint_col, "size"),
                units=("standard_units", lambda values: ";".join(sorted(set(values.dropna().astype(str)))))
                if "standard_units" in pk
                else (endpoint_col, "size"),
            )
            .reset_index()
            .sort_values("observations", ascending=False)
        )

    top = pd.DataFrame()
    if not menin_tasks.empty and "p_activity_median" in menin_tasks:
        top = menin_tasks.sort_values(["p_activity_median", "n_measurements"], ascending=[False, False]).head(
            100
        )
        preferred = [
            "structure_id",
            "standard_inchi_key",
            "smiles",
            "compound_ids",
            "endpoint",
            "assay_family",
            "p_activity_median",
            "value_nm_median",
            "n_measurements",
            "sources",
            "activity_range_log10",
        ]
        top = top[[column for column in preferred if column in top.columns]]

    missingness_rows: list[dict[str, Any]] = []
    for dataset, frame, critical_columns in (
        (
            "menin",
            menin,
            (
                "smiles",
                "target_name",
                "target_id",
                "assay_description",
                "assay_type",
                "document_year",
                "standard_units",
                "p_value",
            ),
        ),
        (
            "herg",
            herg,
            (
                "smiles",
                "target_name",
                "target_id",
                "assay_description",
                "assay_type",
                "document_year",
                "standard_units",
                "p_value",
            ),
        ),
        (
            "pk_admet",
            pk,
            (
                "smiles",
                "admet_endpoint",
                "species",
                "matrix",
                "administration_route",
                "experimental_context",
                "standard_units",
            ),
        ),
    ):
        for column in critical_columns:
            if column not in frame.columns:
                continue
            missing = frame[column].isna() | frame[column].astype(str).str.strip().eq("")
            missingness_rows.append(
                {
                    "dataset": dataset,
                    "column": column,
                    "rows": int(len(frame)),
                    "missing_rows": int(missing.sum()),
                    "missing_fraction": float(missing.mean()) if len(frame) else np.nan,
                }
            )
    missingness = pd.DataFrame(missingness_rows)

    assay_completeness = pd.DataFrame()
    if not menin.empty:
        work = menin.copy()
        work["missing_assay_description"] = (
            work.get("assay_description", pd.Series("", index=work.index))
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
        )
        work["missing_assay_type"] = (
            work.get("assay_type", pd.Series("", index=work.index)).fillna("").astype(str).str.strip().eq("")
        )
        work["missing_date"] = pd.to_numeric(
            work.get("document_year", pd.Series(np.nan, index=work.index)),
            errors="coerce",
        ).isna()
        assay_completeness = (
            work.groupby("source", dropna=False)
            .agg(
                rows=("source", "size"),
                missing_assay_description=("missing_assay_description", "sum"),
                missing_assay_type=("missing_assay_type", "sum"),
                missing_date=("missing_date", "sum"),
            )
            .reset_index()
        )

    attrition_by_source = pd.DataFrame()
    if not menin.empty:
        eligible = _as_bool(menin.get("is_modeling_eligible", pd.Series(False, index=menin.index)))
        attrition_work = menin.assign(_eligible=eligible)
        attrition_by_source = (
            attrition_work.groupby("source", dropna=False)
            .agg(
                source_rows=("source", "size"),
                eligible_rows=("_eligible", "sum"),
                structures=(_identity_column(menin), "nunique"),
            )
            .reset_index()
        )
        attrition_by_source["excluded_rows"] = (
            attrition_by_source["source_rows"] - attrition_by_source["eligible_rows"]
        )

    quarantine_rows: list[pd.DataFrame] = []
    for dataset, filename in (
        ("menin", "menin_activity_quarantine.csv"),
        ("herg", "herg_activity_quarantine.csv"),
        ("pk_admet", "pk_admet_quarantine.csv"),
    ):
        quarantine = _read_csv(processed_dir / filename)
        if quarantine.empty:
            continue
        reason_column = (
            "exclusion_reason" if "exclusion_reason" in quarantine.columns else "admet_quality_flags"
        )
        source_column = "source" if "source" in quarantine.columns else None
        if reason_column not in quarantine.columns:
            continue
        exploded = quarantine.assign(
            _reason=quarantine[reason_column].fillna("").astype(str).str.split(";")
        ).explode("_reason")
        exploded["_reason"] = exploded["_reason"].replace("", "unspecified")
        grouped = (
            exploded.groupby(
                ([source_column] if source_column else []) + ["_reason"],
                dropna=False,
            )
            .size()
            .rename("rows")
            .reset_index()
            .rename(columns={"_reason": "reason", source_column or "": "source"})
        )
        if "source" not in grouped:
            grouped.insert(0, "source", "")
        grouped.insert(0, "dataset", dataset)
        quarantine_rows.append(grouped)
    quarantine_by_source = (
        pd.concat(quarantine_rows, ignore_index=True)
        if quarantine_rows
        else pd.DataFrame(columns=["dataset", "source", "reason", "rows"])
    )

    mirror_links = pd.DataFrame()
    if not menin.empty and "is_cross_source_mirror_candidate" in menin.columns:
        candidates = menin[_as_bool(menin["is_cross_source_mirror_candidate"])].copy()
        mirror_columns = [
            column
            for column in (
                "cross_source_mirror_group_id",
                "cross_source_mirror_preferred_source",
                "is_cross_source_mirror_redundant",
                "source",
                "source_record_id",
                "document_id",
                "assay_id",
                "structure_id",
                "endpoint",
                "assay_family",
                "relation",
                "value_nm",
                "p_value",
            )
            if column in candidates.columns
        ]
        mirror_links = candidates[mirror_columns].sort_values(
            ["cross_source_mirror_group_id", "source", "source_record_id"],
            kind="stable",
        )

    model_domain_performance = pd.DataFrame()
    model_source_performance = pd.DataFrame()
    model_temporal_performance = pd.DataFrame()
    model_scaffold_performance = pd.DataFrame()
    model_failure_cases = pd.DataFrame()
    herg_domain_performance = pd.DataFrame()
    split_coverage = pd.DataFrame()
    primary_prefix = _primary_menin_prefix(settings)
    primary_prediction = reports_dir / f"{primary_prefix}_model_test_predictions.csv"
    primary_split = reports_dir / f"{primary_prefix}_split_assignments.csv"
    prediction_candidates = [primary_prediction] if primary_prediction.exists() else []
    split_candidates = [primary_split] if primary_split.exists() else []
    predictions = _read_csv(prediction_candidates[0]) if prediction_candidates else pd.DataFrame()
    if not predictions.empty and "absolute_error" in predictions.columns:
        domain = _as_bool(
            predictions.get(
                "inside_applicability_domain",
                pd.Series(False, index=predictions.index),
            )
        )
        performance = predictions.assign(_inside_domain=domain)
        model_domain_performance = (
            performance.groupby("_inside_domain", dropna=False)
            .agg(
                holdout_rows=("absolute_error", "size"),
                mae=("absolute_error", "mean"),
                median_absolute_error=("absolute_error", "median"),
                mean_max_training_tanimoto=("max_training_tanimoto", "mean"),
            )
            .reset_index()
            .rename(columns={"_inside_domain": "inside_applicability_domain"})
        )
        if "sources" in performance.columns:
            source_work = performance.assign(
                source=performance["sources"].fillna("").astype(str).str.split(";")
            ).explode("source")
            model_source_performance = (
                source_work.groupby("source", dropna=False)
                .agg(
                    holdout_rows=("absolute_error", "size"),
                    mae=("absolute_error", "mean"),
                    median_absolute_error=("absolute_error", "median"),
                )
                .reset_index()
            )
        if "document_year" in performance.columns:
            years = pd.to_numeric(performance["document_year"], errors="coerce")
            temporal_work = performance.assign(
                year_band=np.where(
                    years.notna(),
                    (np.floor(years / 5) * 5).astype("Int64").astype(str)
                    + "-"
                    + (np.floor(years / 5) * 5 + 4).astype("Int64").astype(str),
                    "undated",
                )
            )
            model_temporal_performance = (
                temporal_work.groupby("year_band", dropna=False)
                .agg(
                    holdout_rows=("absolute_error", "size"),
                    mae=("absolute_error", "mean"),
                    median_absolute_error=("absolute_error", "median"),
                )
                .reset_index()
            )
        model_failure_cases = performance.sort_values(
            ["absolute_error", "max_training_tanimoto"],
            ascending=[False, True],
        ).head(100)
    if split_candidates:
        assignments = _read_csv(split_candidates[0])
        if not assignments.empty:
            group_columns = [
                column
                for column in (
                    "requested_split_strategy",
                    "actual_split_strategy",
                    "split",
                    "scaffold_grouping_method",
                )
                if column in assignments.columns
            ]
            row_column = "modeling_row" if "modeling_row" in assignments else assignments.columns[0]
            aggregation: dict[str, tuple[str, str]] = {"rows": (row_column, "size")}
            if "structure_group_key" in assignments:
                aggregation["structures"] = ("structure_group_key", "nunique")
            elif "structure_id" in assignments:
                aggregation["structures"] = ("structure_id", "nunique")
            if "bemis_murcko_group" in assignments:
                aggregation["scaffolds"] = ("bemis_murcko_group", "nunique")
            if group_columns and aggregation:
                split_coverage = (
                    assignments.groupby(group_columns, dropna=False).agg(**aggregation).reset_index()
                )
            join_key = (
                "structure_id"
                if "structure_id" in predictions.columns and "structure_id" in assignments.columns
                else "smiles"
            )
            if (
                not predictions.empty
                and join_key in predictions.columns
                and join_key in assignments.columns
                and "bemis_murcko_group" in assignments.columns
            ):
                scaffold_work = predictions.merge(
                    assignments[[join_key, "bemis_murcko_group"]].drop_duplicates(join_key),
                    on=join_key,
                    how="left",
                    validate="many_to_one",
                )
                model_scaffold_performance = (
                    scaffold_work.groupby("bemis_murcko_group", dropna=False)
                    .agg(
                        holdout_rows=("absolute_error", "size"),
                        mae=("absolute_error", "mean"),
                        median_absolute_error=("absolute_error", "median"),
                    )
                    .reset_index()
                    .sort_values(["mae", "holdout_rows"], ascending=[False, False])
                )
    herg_predictions = _read_csv(reports_dir / "herg_classifier_test_predictions.csv")
    if not herg_predictions.empty:
        herg_probability = pd.to_numeric(
            herg_predictions.get("predicted_herg_blocker_probability"), errors="coerce"
        )
        herg_observed = pd.to_numeric(herg_predictions.get("observed_herg_blocker_label"), errors="coerce")
        herg_inside = _as_bool(
            herg_predictions.get(
                "inside_applicability_domain",
                pd.Series(False, index=herg_predictions.index),
            )
        )
        herg_work = herg_predictions.assign(
            _probability=herg_probability,
            _observed=herg_observed,
            _inside=herg_inside,
            _brier=(herg_probability - herg_observed) ** 2,
            _correct=(herg_probability.ge(0.5).astype(float) == herg_observed).astype(float),
        ).dropna(subset=["_probability", "_observed"])
        herg_domain_performance = (
            herg_work.groupby("_inside", dropna=False)
            .agg(
                holdout_rows=("_observed", "size"),
                blocker_fraction=("_observed", "mean"),
                accuracy_0p5=("_correct", "mean"),
                brier_score=("_brier", "mean"),
                mean_probability=("_probability", "mean"),
            )
            .reset_index()
            .rename(columns={"_inside": "inside_applicability_domain"})
        )

    outputs = {
        "dataset_inventory": inventory,
        "menin_endpoint_assay_summary": endpoint_assay,
        "menin_source_endpoint_summary": source_endpoint,
        "menin_replicate_consistency": replicate,
        "herg_label_summary": herg_labels,
        "herg_primary_label_summary": primary_herg_labels,
        "herg_pooled_label_summary": pooled_herg_labels,
        "pk_admet_coverage": pk_coverage,
        "top_menin_tasks": top,
        "critical_field_missingness": missingness,
        "assay_context_completeness": assay_completeness,
        "curation_attrition_by_source": attrition_by_source,
        "quarantine_reasons_by_source": quarantine_by_source,
        "menin_cross_source_mirror_links": mirror_links,
        "menin_model_domain_performance": model_domain_performance,
        "menin_model_source_performance": model_source_performance,
        "menin_model_temporal_performance": model_temporal_performance,
        "menin_model_scaffold_performance": model_scaffold_performance,
        "menin_model_failure_cases": model_failure_cases,
        "menin_split_chemical_coverage": split_coverage,
        "herg_model_domain_performance": herg_domain_performance,
    }
    if analysis_dir is not None:
        chemical_tables = {
            "chemical_medicinal_profiles": "medicinal_chemistry_profiles.csv",
            "chemical_candidate_priorities": "candidate_priorities.csv",
            "chemical_priority_frontier": "priority_frontier.csv",
            "chemical_priority_data_gaps": "priority_data_gaps.csv",
            "chemical_priority_sensitivity": "priority_sensitivity.csv",
            "chemical_series_summary": "chemical_series_summary.csv",
            "chemical_activity_cliffs": "activity_cliffs.csv",
            "chemical_mmp_cliffs": "matched_molecular_pair_cliffs.csv",
            "chemical_connectivity_variant_summary": "connectivity_variant_summary.csv",
            "chemical_connectivity_variant_cliffs": "connectivity_variant_cliffs.csv",
            "chemical_similarity_cluster_summary": "similarity_cluster_summary.csv",
            "chemical_approved_reference_coverage": "approved_reference_coverage.csv",
            "chemical_prospective_selection_plan": "prospective_selection_plan.csv",
            "chemical_prospective_selection_summary": "prospective_selection_summary.csv",
        }
        for table_name, filename in chemical_tables.items():
            outputs[table_name] = _read_csv(analysis_dir / filename)
    empty_schemas = {
        "menin_endpoint_assay_summary": [
            "endpoint",
            "assay_family",
            "measurements",
            "unique_structures",
            "exact_measurements",
            "modeling_eligible",
            "median_p_activity",
        ],
        "menin_source_endpoint_summary": ["source", "endpoint", "measurements", "unique_structures"],
        "menin_replicate_consistency": [
            "structure_id",
            "endpoint",
            "assay_family",
            "measurements",
            "sources",
            "median_p_activity",
            "min_p_activity",
            "max_p_activity",
            "sd_p_activity",
            "range_log10",
        ],
        "herg_label_summary": ["scope", "endpoint", "assay_family", "label", "task_rows"],
        "herg_primary_label_summary": ["scope", "endpoint", "assay_family", "label", "task_rows"],
        "herg_pooled_label_summary": ["scope", "endpoint", "assay_family", "label", "task_rows"],
        "pk_admet_coverage": ["admet_endpoint", "observations", "compounds", "units"],
        "top_menin_tasks": ["structure_id", "smiles", "endpoint", "assay_family", "p_activity_median"],
        "critical_field_missingness": ["dataset", "column", "rows", "missing_rows", "missing_fraction"],
        "assay_context_completeness": [
            "source",
            "rows",
            "missing_assay_description",
            "missing_assay_type",
            "missing_date",
        ],
        "curation_attrition_by_source": [
            "source",
            "source_rows",
            "eligible_rows",
            "structures",
            "excluded_rows",
        ],
        "menin_cross_source_mirror_links": [
            "cross_source_mirror_group_id",
            "source",
            "source_record_id",
            "structure_id",
            "endpoint",
            "assay_family",
        ],
        "menin_model_domain_performance": [
            "inside_applicability_domain",
            "holdout_rows",
            "mae",
            "median_absolute_error",
            "mean_max_training_tanimoto",
        ],
        "menin_model_source_performance": ["source", "holdout_rows", "mae", "median_absolute_error"],
        "menin_model_temporal_performance": ["year_band", "holdout_rows", "mae", "median_absolute_error"],
        "menin_model_scaffold_performance": [
            "bemis_murcko_group",
            "holdout_rows",
            "mae",
            "median_absolute_error",
        ],
        "menin_model_failure_cases": [
            "structure_id",
            "smiles",
            "observed_p_activity_median",
            "predicted_p_activity_median",
            "absolute_error",
        ],
        "menin_split_chemical_coverage": [
            "requested_split_strategy",
            "actual_split_strategy",
            "split",
            "rows",
            "structures",
            "scaffolds",
        ],
        "herg_model_domain_performance": [
            "inside_applicability_domain",
            "holdout_rows",
            "blocker_fraction",
            "accuracy_0p5",
            "brier_score",
            "mean_probability",
        ],
    }
    for name, frame in outputs.items():
        if len(frame.columns) == 0:
            frame = pd.DataFrame(columns=empty_schemas.get(name, ["status"]))
            outputs[name] = frame
        frame.to_csv(tables_dir / f"{name}.csv", index=False)
    return outputs


def _plot_data_overview(
    processed_dir: Path,
    reports_dir: Path,
    settings: dict[str, Any] | None = None,
) -> None:
    figures = reports_dir / "figures"
    menin = _read_csv(processed_dir / "menin_activity_measurements.csv")
    compounds = _read_csv(processed_dir / "menin_compounds_curated.csv")
    quality = _read_csv(processed_dir / "data_quality_summary.csv")

    if not menin.empty and "endpoint" in menin:
        work = menin.copy()
        if "is_modeling_eligible" in work:
            work = work[_as_bool(work["is_modeling_eligible"])]
        counts = work["endpoint"].value_counts().sort_values(ascending=False)
        plt.figure(figsize=(7.4, 4.5))
        counts.plot.bar(color=COLORS["teal"])
        plt.ylabel("Eligible measurements")
        plt.xlabel("Endpoint")
        plt.title(f"Menin endpoint coverage after quality gates (n={int(counts.sum()):,})")
        plt.xticks(rotation=0)
        _save(figures / "menin_endpoint_counts.png")

        if {"source", "endpoint"}.issubset(work.columns):
            matrix = pd.crosstab(work["source"], work["endpoint"])
            plt.figure(figsize=(7.6, max(3.8, 0.6 * len(matrix))))
            image = plt.imshow(np.log10(matrix.to_numpy() + 1), cmap="YlGnBu", aspect="auto")
            plt.xticks(range(len(matrix.columns)), matrix.columns)
            plt.yticks(range(len(matrix.index)), matrix.index)
            plt.colorbar(image, label="log10(measurements + 1)")
            plt.title("Eligible Menin source × endpoint coverage")
            _save(figures / "menin_source_endpoint_heatmap.png")

        if "document_year" in work:
            years = pd.to_numeric(work["document_year"], errors="coerce").dropna().astype(int)
            years = years[(years >= 1900) & (years <= 2100)]
            if not years.empty:
                plt.figure(figsize=(8, 4.4))
                years.value_counts().sort_index().plot(color=COLORS["navy"], marker="o", ms=3)
                plt.ylabel("Measurements")
                plt.xlabel("Document year")
                plt.title("Temporal coverage of eligible Menin data")
                _save(figures / "menin_temporal_coverage.png")

    if not compounds.empty and "p_activity_median" in compounds:
        model_cfg = (settings or {}).get("modeling", {})
        endpoint = str(model_cfg.get("primary_menin_endpoint", "IC50"))
        assay_family = str(model_cfg.get("primary_menin_assay_family", "biochemical_binding"))
        primary = compounds[
            compounds.get("endpoint", pd.Series("", index=compounds.index))
            .fillna("")
            .astype(str)
            .str.casefold()
            .eq(endpoint.casefold())
            & compounds.get("assay_family", pd.Series("", index=compounds.index))
            .fillna("")
            .astype(str)
            .str.casefold()
            .eq(assay_family.casefold())
        ].copy()
        values = pd.to_numeric(primary["p_activity_median"], errors="coerce").dropna()
        plt.figure(figsize=(7.4, 4.5))
        plt.hist(values, bins=30, color=COLORS["navy"], edgecolor="white")
        plt.xlabel("Median pActivity")
        plt.ylabel("Unique primary-task structures")
        plt.title(f"Menin {endpoint} × {assay_family} potency distribution")
        _save(figures / "menin_potency_distribution.png")

        sampled = primary.dropna(subset=["smiles"]).copy()
        sampled = sampled[sampled["smiles"].astype(str).str.strip().ne("")]
        identity = _identity_column(sampled)
        if identity in sampled:
            sampled = sampled.sort_values(
                ["p_activity_median", identity], ascending=[False, True]
            ).drop_duplicates(identity, keep="first")
        if len(sampled) >= 10:
            if len(sampled) > 3000:
                sampled = sampled.sample(3000, random_state=13)
            matrix, backend = fingerprint_matrix(sampled["smiles"].astype(str), backend="auto", n_bits=512)
            coordinates = PCA(n_components=2, random_state=13).fit_transform(matrix)
            color = pd.to_numeric(sampled["p_activity_median"], errors="coerce")
            plt.figure(figsize=(7, 5.4))
            scatter = plt.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                c=color,
                cmap="viridis",
                s=12,
                alpha=0.72,
                linewidths=0,
            )
            plt.colorbar(scatter, label="Median pActivity")
            plt.xlabel("Fingerprint PC1")
            plt.ylabel("Fingerprint PC2")
            plt.title(f"Menin {endpoint} × {assay_family} chemical space ({backend}; n={len(sampled):,})")
            _save(figures / "menin_chemical_space.png")

    if not quality.empty:
        attrition = quality.groupby(["dataset", "reason"], as_index=False)["n_rows"].sum()
        attrition = attrition.sort_values("n_rows", ascending=False).head(12)
        if not attrition.empty:
            labels = attrition["dataset"].astype(str) + ": " + attrition["reason"].astype(str)
            plt.figure(figsize=(8, max(4.5, len(attrition) * 0.38)))
            plt.barh(labels[::-1], attrition["n_rows"][::-1], color=COLORS["orange"])
            plt.xlabel("Rows")
            plt.title("Largest eligibility and quarantine categories")
            _save(figures / "data_quality_attrition.png")


def _plot_model_diagnostics(
    reports_dir: Path,
    settings: dict[str, Any] | None = None,
    analysis_dir: Path | None = None,
) -> None:
    figures = reports_dir / "figures"
    primary = reports_dir / f"{_primary_menin_prefix(settings)}_model_test_predictions.csv"
    candidates = [primary] if primary.exists() else []
    if not candidates and (reports_dir / "menin_activity_model_test_predictions.csv").exists():
        candidates = [reports_dir / "menin_activity_model_test_predictions.csv"]
    regression = _read_csv(candidates[0]) if candidates and candidates[0].exists() else pd.DataFrame()
    regression_metrics = (
        _read_json(
            candidates[0].with_name(
                candidates[0].name.replace("_model_test_predictions.csv", "_model_metrics.json")
            )
        )
        if candidates and candidates[0].exists()
        else {}
    )
    if not regression.empty:
        observed = pd.to_numeric(regression.get("observed_p_activity_median"), errors="coerce")
        predicted = pd.to_numeric(regression.get("predicted_p_activity_median"), errors="coerce")
        valid = observed.notna() & predicted.notna()
        if valid.any():
            lower = float(min(observed[valid].min(), predicted[valid].min()))
            upper = float(max(observed[valid].max(), predicted[valid].max()))
            plt.figure(figsize=(5.7, 5.2))
            plt.scatter(observed[valid], predicted[valid], s=23, alpha=0.7, color=COLORS["teal"])
            plt.plot([lower, upper], [lower, upper], "--", color=COLORS["gray"])
            plt.xlabel("Observed pActivity")
            plt.ylabel("Predicted pActivity")
            task = regression_metrics.get("task", {})
            split = regression_metrics.get("split", {}).get("strategy", "unknown")
            plt.title(
                f"Menin {task.get('endpoint', '')} × {task.get('assay_family', '')} | "
                f"{split} holdout (n={int(valid.sum()):,})"
            )
            _save(figures / "menin_observed_vs_predicted.png")
        if "max_training_tanimoto" in regression and "absolute_error" in regression:
            plt.figure(figsize=(6.7, 4.7))
            plt.scatter(
                regression["max_training_tanimoto"],
                regression["absolute_error"],
                s=20,
                alpha=0.65,
                color=COLORS["orange"],
            )
            plt.xlabel("Maximum training-set Tanimoto")
            plt.ylabel("Absolute error (pActivity)")
            plt.title("Menin error versus chemical similarity")
            _save(figures / "menin_error_vs_similarity.png")

    herg = _read_csv(reports_dir / "herg_classifier_test_predictions.csv")
    herg_metrics = _read_json(reports_dir / "herg_classifier_metrics.json")
    if not herg.empty:
        y = pd.to_numeric(herg.get("observed_herg_blocker_label"), errors="coerce")
        probability = pd.to_numeric(herg.get("predicted_herg_blocker_probability"), errors="coerce")
        valid = y.notna() & probability.notna()
        if valid.sum() and y[valid].nunique() == 2:
            fpr, tpr, _ = roc_curve(y[valid], probability[valid])
            precision, recall, _ = precision_recall_curve(y[valid], probability[valid])
            plt.figure(figsize=(10.5, 4.8))
            plt.subplot(1, 2, 1)
            plt.plot(fpr, tpr, color=COLORS["teal"])
            plt.plot([0, 1], [0, 1], "--", color=COLORS["gray"])
            plt.xlabel("False-positive rate")
            plt.ylabel("True-positive rate")
            herg_task = herg_metrics.get("task", {})
            task_label = f"{herg_task.get('endpoint', '')} × {herg_task.get('assay_family', '')}"
            herg_split = herg_metrics.get("split", {}).get("strategy", "holdout")
            plt.title("ROC curve")
            plt.subplot(1, 2, 2)
            plt.plot(recall, precision, color=COLORS["red"])
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision–recall curve")
            plt.suptitle(f"hERG {task_label} | {herg_split} holdout (n={int(valid.sum()):,})")
            _save(figures / "herg_discrimination_curves.png")

    calibration = _read_csv(reports_dir / "herg_classifier_calibration_curve.csv")
    if not calibration.empty:
        plt.figure(figsize=(5.7, 5.2))
        plt.plot([0, 1], [0, 1], "--", color=COLORS["gray"])
        plt.plot(
            calibration["mean_predicted_probability"],
            calibration["observed_positive_fraction"],
            marker="o",
            color=COLORS["red"],
        )
        plt.xlabel("Predicted blocker probability")
        plt.ylabel("Observed blocker fraction")
        plt.title("hERG probability calibration")
        _save(figures / "herg_calibration.png")

    validation = _read_csv(reports_dir / "model_validation_summary.csv")
    if not validation.empty:
        regression_rows = validation[validation["task"].astype(str).str.startswith("menin_")]
        regression_rows = regression_rows.dropna(subset=["mae"])
        if not regression_rows.empty:
            labels = regression_rows["task"] + " / " + regression_rows["requested_split"]
            plt.figure(figsize=(8.5, max(4.2, 0.38 * len(regression_rows))))
            plt.barh(labels[::-1], regression_rows["mae"][::-1], color=COLORS["navy"])
            plt.xlabel("Holdout MAE (pActivity; lower is better)")
            plt.title("Menin validation sensitivity")
            _save(figures / "menin_validation_comparison.png")

    risk = _read_csv(reports_dir / "menin_with_predicted_herg_risk.csv")
    if not risk.empty and "predicted_herg_risk" in risk:
        evidence = (
            _read_csv(analysis_dir / "candidate_priorities.csv")
            if analysis_dir is not None
            else pd.DataFrame()
        )
        if not evidence.empty and "herg_evidence_status" in evidence:
            order = [
                "observed_non_blocker",
                "predicted_lower_concern",
                "predicted_indeterminate",
                "predicted_high_concern",
                "observed_blocker",
                "unknown_outside_applicability_domain",
                "unknown_missing_prediction",
            ]
            counts = evidence["herg_evidence_status"].value_counts().reindex(order).dropna()
            labels = [
                str(value).replace("predicted_", "").replace("unknown_", "unknown: ")
                for value in counts.index
            ]
            title = "hERG evidence status (out-of-domain is unknown)"
        else:
            order = ["low", "medium", "high", "unscored"]
            counts = risk["predicted_herg_risk"].value_counts().reindex(order).dropna()
            labels = list(counts.index)
            title = "Predicted hERG triage with applicability-domain flags"
        plt.figure(figsize=(6.8, 4.4))
        plt.bar(
            labels,
            counts.values,
            color=[
                COLORS["teal"],
                COLORS["teal"],
                COLORS["gold"],
                COLORS["red"],
                COLORS["red"],
                COLORS["gray"],
                COLORS["gray"],
            ][: len(counts)],
        )
        plt.ylabel("Menin task rows")
        plt.title(title)
        plt.xticks(rotation=25, ha="right")
        _save(figures / "predicted_herg_risk_counts.png")

        if {"p_activity_median", "predicted_herg_blocker_probability"}.issubset(risk.columns):
            plt.figure(figsize=(6.7, 5.0))
            inside = _as_bool(
                risk.get("herg_inside_applicability_domain", pd.Series(False, index=risk.index))
            )
            plt.scatter(
                risk.loc[~inside, "p_activity_median"],
                risk.loc[~inside, "predicted_herg_blocker_probability"],
                s=18,
                alpha=0.45,
                color=COLORS["gray"],
                label="outside domain",
            )
            plt.scatter(
                risk.loc[inside, "p_activity_median"],
                risk.loc[inside, "predicted_herg_blocker_probability"],
                s=18,
                alpha=0.65,
                color=COLORS["teal"],
                label="inside domain",
            )
            plt.axhline(0.7, ls="--", color=COLORS["red"], lw=1)
            plt.axhline(0.3, ls="--", color=COLORS["gold"], lw=1)
            plt.xlabel("Curated Menin pActivity")
            plt.ylabel("Predicted hERG blocker probability")
            plt.title("Potency–liability triage (hypothesis generation only)")
            plt.legend(frameon=False)
            _save(figures / "menin_potency_herg_triage.png")


def _plot_chemical_intelligence(analysis_dir: Path, reports_dir: Path) -> None:
    """Render primary-task medicinal-chemistry, SAR, and decision-layer diagnostics."""

    figures = reports_dir / "figures"
    profiles = _read_csv(analysis_dir / "medicinal_chemistry_profiles.csv")
    if not profiles.empty and {"mol_wt", "logp", "p_activity_median", "qed"}.issubset(profiles):
        valid = profiles.dropna(subset=["mol_wt", "logp", "p_activity_median", "qed"]).copy()
        if not valid.empty:
            plt.figure(figsize=(7.2, 5.4))
            scatter = plt.scatter(
                valid["logp"],
                valid["mol_wt"],
                c=valid["p_activity_median"],
                s=18 + 55 * pd.to_numeric(valid["qed"], errors="coerce").fillna(0),
                cmap="viridis",
                alpha=0.65,
                linewidths=0,
            )
            plt.colorbar(scatter, label="Observed primary-task pActivity")
            plt.xlabel("RDKit cLogP")
            plt.ylabel("Molecular weight (Da)")
            plt.title("Primary-task potency across medicinal-chemistry property space")
            _save(figures / "menin_medchem_property_landscape.png")

    series = _read_csv(analysis_dir / "chemical_series_summary.csv")
    required_series = {"series_id", "series_size", "best_p_activity", "minimum_p_activity"}
    if not series.empty and required_series.issubset(series):
        top = series.sort_values(["series_size", "best_p_activity"], ascending=[False, False]).head(20)
        labels = top["series_id"].astype(str).str.slice(0, 12)
        plt.figure(figsize=(8.0, 6.3))
        plt.barh(labels[::-1], top["series_size"][::-1], color=COLORS["teal"])
        plt.xlabel("Primary-task structures")
        plt.ylabel("Bemis–Murcko series")
        plt.title("Largest Menin chemical series")
        _save(figures / "menin_chemical_series_sizes.png")

    cliffs = _read_csv(analysis_dir / "activity_cliffs.csv")
    if not cliffs.empty and {
        "achiral_morgan_tanimoto",
        "absolute_delta_pactivity",
        "evidence_context_grade",
    }.issubset(cliffs):
        plt.figure(figsize=(7.0, 5.2))
        for grade, color in (
            ("cross_context", COLORS["gray"]),
            ("same_document", COLORS["gold"]),
            ("same_assay", COLORS["red"]),
        ):
            subset = cliffs[cliffs["evidence_context_grade"].astype(str).eq(grade)]
            if not subset.empty:
                plt.scatter(
                    subset["achiral_morgan_tanimoto"],
                    subset["absolute_delta_pactivity"],
                    s=18,
                    alpha=0.55,
                    color=color,
                    label=grade.replace("_", " "),
                )
        plt.xlabel("Achiral Morgan Tanimoto")
        plt.ylabel("Absolute ΔpActivity")
        plt.title("Primary-task activity cliffs requiring SAR review")
        plt.legend(frameon=False)
        _save(figures / "menin_activity_cliff_landscape.png")

    priorities = _read_csv(analysis_dir / "candidate_priorities.csv")
    if not priorities.empty and "experimental_followup_tier" in priorities:
        counts = priorities["experimental_followup_tier"].value_counts().sort_index()
        labels = [str(value).replace("priority_", "P").replace("_", " ") for value in counts.index]
        plt.figure(figsize=(8.0, 4.8))
        plt.bar(
            labels,
            counts.values,
            color=[COLORS["teal"], COLORS["gold"], COLORS["orange"], COLORS["red"], COLORS["gray"]][
                : len(counts)
            ],
        )
        plt.ylabel("Primary-task structures")
        plt.title("Evidence-first experimental follow-up tiers")
        plt.xticks(rotation=25, ha="right")
        _save(figures / "menin_followup_tier_counts.png")

        safety = priorities.dropna(subset=["p_activity_median", "predicted_herg_blocker_probability"]).copy()
        safety = safety[
            _as_bool(
                safety.get(
                    "herg_inside_applicability_domain",
                    pd.Series(False, index=safety.index),
                )
            )
        ]
        if not safety.empty:
            frontier = pd.to_numeric(safety.get("complete_evidence_pareto_rank"), errors="coerce").eq(1)
            plt.figure(figsize=(7.0, 5.2))
            plt.scatter(
                safety.loc[~frontier, "p_activity_median"],
                safety.loc[~frontier, "predicted_herg_blocker_probability"],
                s=18,
                alpha=0.45,
                color=COLORS["gray"],
                label="dominated trade-off",
            )
            plt.scatter(
                safety.loc[frontier, "p_activity_median"],
                safety.loc[frontier, "predicted_herg_blocker_probability"],
                s=35,
                alpha=0.8,
                color=COLORS["teal"],
                label="complete-evidence Pareto front",
            )
            plt.xlabel("Observed primary-task pActivity")
            plt.ylabel("Predicted hERG blocker probability")
            plt.title("Inside-domain potency–hERG trade-offs (not safety validation)")
            plt.legend(frameon=False)
            _save(figures / "menin_complete_evidence_frontier.png")


def _publication_readiness(
    processed_dir: Path,
    reports_dir: Path,
    models_dir: Path,
    settings: dict[str, Any] | None,
) -> pd.DataFrame:
    settings = settings or {}
    primary_endpoint = str(settings.get("modeling", {}).get("primary_menin_endpoint", "IC50"))
    primary_family = str(
        settings.get("modeling", {}).get("primary_menin_assay_family", "biochemical_binding")
    )
    metric_candidates = sorted(reports_dir.glob("menin_activity_*_model_metrics.json"))
    primary_metrics: dict[str, Any] = {}
    for path in metric_candidates:
        candidate = _read_json(path)
        task = candidate.get("task", {})
        if (
            str(task.get("endpoint", "")).casefold() == primary_endpoint.casefold()
            and str(task.get("assay_family", "")).casefold() == primary_family.casefold()
        ):
            primary_metrics = candidate
            break
    herg_metrics = _read_json(reports_dir / "herg_classifier_metrics.json")
    quality = _read_json(reports_dir / "quality" / "quality_gate.json")
    validation = _read_csv(reports_dir / "model_validation_summary.csv")
    processed_manifest = _read_json(reports_dir / "manifests" / "processed_manifest.json")
    software_manifest = _read_json(reports_dir / "manifests" / "software_manifest.json")
    models_manifest = _read_json(reports_dir / "manifests" / "models_manifest.json")
    analysis_manifest = _read_json(reports_dir / "manifests" / "analysis_manifest.json")
    reports_manifest = _read_json(reports_dir / "manifests" / "reports_manifest.json")
    analysis_enabled = settings.get("analysis", {}).get("enabled", False) is True
    verification_stages = (
        "raw",
        "processed",
        "software",
        "models",
        *(("analysis",) if analysis_enabled else ()),
        "reports",
    )
    verification = {
        stage: _read_json(reports_dir / "verification" / f"{stage}_verification.json")
        for stage in verification_stages
    }
    upstream_verified = all(
        verification[stage].get("valid") is True
        for stage in ("raw", "processed", "software", "models", *(("analysis",) if analysis_enabled else ()))
    )
    expected_splits = set(
        settings.get("modeling", {}).get("evaluation_splits", ["scaffold", "chemical", "temporal", "random"])
    )
    primary_task_name = f"menin_{primary_endpoint}_{primary_family}"
    primary_validation = (
        validation[validation.get("task", pd.Series(dtype=str)).astype(str).eq(primary_task_name)]
        if not validation.empty and "task" in validation.columns
        else pd.DataFrame()
    )
    completed_splits = (
        set(
            primary_validation.loc[
                primary_validation.get("status", pd.Series(index=primary_validation.index)).eq("trained"),
                "requested_split",
            ].astype(str)
        )
        if not primary_validation.empty and "requested_split" in primary_validation
        else set()
    )
    actual_splits_match = (
        bool(
            not primary_validation.empty
            and (
                primary_validation["requested_split"].astype(str)
                == primary_validation["actual_split"].astype(str)
            ).all()
        )
        if {"requested_split", "actual_split"}.issubset(primary_validation.columns)
        else False
    )
    provenance = primary_metrics.get("provenance", {})
    herg_provenance = herg_metrics.get("provenance", {})
    model_upstream = models_manifest.get("upstream", [])
    report_upstream = reports_manifest.get("upstream", [])

    def linked(upstream: list[Any], stage: str, digest: object) -> bool:
        return any(
            isinstance(item, dict) and item.get("stage") == stage and item.get("dataset_sha256") == digest
            for item in upstream
        )

    build_linked = bool(
        processed_manifest
        and software_manifest
        and models_manifest
        and reports_manifest
        and processed_manifest.get("build_id") == models_manifest.get("build_id")
        and processed_manifest.get("build_id") == software_manifest.get("build_id")
        and processed_manifest.get("build_id") == reports_manifest.get("build_id")
        and provenance.get("processed_dataset_sha256") == processed_manifest.get("dataset_sha256")
        and herg_provenance.get("processed_dataset_sha256") == processed_manifest.get("dataset_sha256")
        and provenance.get("software_dataset_sha256") == software_manifest.get("dataset_sha256")
        and herg_provenance.get("software_dataset_sha256") == software_manifest.get("dataset_sha256")
        and linked(model_upstream, "processed", processed_manifest.get("dataset_sha256"))
        and linked(model_upstream, "software", software_manifest.get("dataset_sha256"))
        and linked(report_upstream, "models", models_manifest.get("dataset_sha256"))
        and (
            not analysis_enabled
            or (
                analysis_manifest.get("build_id") == processed_manifest.get("build_id")
                and linked(
                    analysis_manifest.get("upstream", []),
                    "processed",
                    processed_manifest.get("dataset_sha256"),
                )
                and linked(
                    analysis_manifest.get("upstream", []),
                    "models",
                    models_manifest.get("dataset_sha256"),
                )
                and linked(
                    analysis_manifest.get("upstream", []),
                    "software",
                    software_manifest.get("dataset_sha256"),
                )
                and linked(
                    report_upstream,
                    "analysis",
                    analysis_manifest.get("dataset_sha256"),
                )
            )
        )
    )
    primary_trained = primary_metrics.get("status") == "trained"
    herg_trained = herg_metrics.get("status") == "trained"
    rdkit_features = (
        primary_metrics.get("features", {}).get("backend") == "rdkit_morgan"
        and herg_metrics.get("features", {}).get("backend") == "rdkit_morgan"
        and bool(primary_metrics.get("model"))
        and bool(herg_metrics.get("model"))
        and primary_metrics.get("artifact", {}).get("format") in {"skops", "joblib"}
        and herg_metrics.get("artifact", {}).get("format") in {"skops", "joblib"}
    )
    lock_manifested = (
        any(
            item.get("path") == "pipeline/environments/requirements.lock"
            for item in software_manifest.get("files", [])
            if isinstance(item, dict)
        )
        and verification["software"].get("valid") is True
    )
    checks = [
        (
            "Defined biological endpoint",
            primary_trained and herg_trained,
            f"Primary tasks are {primary_endpoint} × {primary_family} and scoped hERG IC50 × electrophysiology_functional.",
        ),
        (
            "Unambiguous RDKit algorithm",
            primary_trained and herg_trained and rdkit_features,
            "Both trained primary manifests must record RDKit Morgan features and selected algorithms.",
        ),
        (
            "Complete requested holdouts",
            expected_splits.issubset(completed_splits) and actual_splits_match,
            "Every configured split must train and must not silently fall back.",
        ),
        (
            "Applicability domain",
            bool(primary_metrics.get("applicability_domain"))
            and bool(herg_metrics.get("applicability_domain")),
            "Both primary tasks record nearest-neighbor domain policy and coverage.",
        ),
        (
            "Uncertainty",
            bool(primary_metrics.get("test_metric_bootstrap_95_ci"))
            and bool(primary_metrics.get("uncertainty")),
            "Primary Menin metrics include scaffold-group bootstrap intervals and conformal prediction intervals.",
        ),
        (
            "Build-linked provenance",
            build_linked,
            "Processed, model, and report manifests plus model provenance must share a content build.",
        ),
        (
            "Upstream manifest verification",
            upstream_verified,
            "Raw, processed, software, and model inputs must verify before report generation; the final report bundle is verified as the release's last step.",
        ),
        (
            "Data quality audit",
            quality.get("passed") is True,
            "The analysis-eligible quality gate must pass; source-inventory exclusions remain reported.",
        ),
        (
            "Environment lock",
            lock_manifested,
            "The frozen environment lock must be included in the verified software manifest.",
        ),
        (
            "Chemical-intelligence decision trace",
            bool(
                not analysis_enabled
                or (
                    analysis_manifest
                    and (reports_dir / "tables" / "chemical_candidate_priorities.csv").exists()
                    and (reports_dir / "tables" / "chemical_activity_cliffs.csv").exists()
                )
            ),
            "Primary-task properties, series, cliffs, evidence gaps, Pareto fronts, and applicability-aware tiers must be content-linked.",
        ),
        (
            "Clean source revision",
            provenance.get("git_dirty") is False and bool(provenance.get("git_revision")),
            "A publication release must be rebuilt from a clean committed revision.",
        ),
        (
            "Independent external validation",
            False,
            "No independent Menin or hERG external test set has been reserved.",
        ),
        ("Prospective experimental validation", False, "Requires new lab measurements after model lock."),
        (
            "Authorship and licensing approval",
            False,
            "Requires project-owner approval.",
        ),
    ]
    frame = pd.DataFrame(checks, columns=["criterion", "satisfied", "evidence_or_action"])
    frame.to_csv(reports_dir / "tables" / "publication_readiness_matrix.csv", index=False)
    return frame


def write_summary_report(
    processed_dir: Path,
    reports_dir: Path,
    models_dir: Path,
    *,
    settings: dict[str, Any] | None = None,
    analysis_dir: Path | None = None,
) -> Path:
    """Regenerate all descriptive outputs and a publication-readiness report."""

    settings = settings or {}
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables = _write_analysis_tables(processed_dir, reports_dir, settings, analysis_dir)
    _plot_data_overview(processed_dir, reports_dir, settings)
    _plot_model_diagnostics(reports_dir, settings, analysis_dir)
    if analysis_dir is not None:
        _plot_chemical_intelligence(analysis_dir, reports_dir)
    readiness = _publication_readiness(processed_dir, reports_dir, models_dir, settings)
    inventory = tables["dataset_inventory"]
    endpoint_assay = tables["menin_endpoint_assay_summary"]
    top = tables["top_menin_tasks"]
    domain_performance = tables["menin_model_domain_performance"]
    missingness = tables["critical_field_missingness"]
    mirror_links = tables["menin_cross_source_mirror_links"]
    herg_primary_labels = tables["herg_primary_label_summary"]
    herg_pooled_labels = tables["herg_pooled_label_summary"]
    model_summary = _read_csv(reports_dir / "model_validation_summary.csv")
    risk_predictions = _read_csv(reports_dir / "menin_with_predicted_herg_risk.csv")
    chemical_summary = _read_json(analysis_dir / "analysis_summary.json") if analysis_dir is not None else {}
    chemical_priorities = tables.get("chemical_candidate_priorities", pd.DataFrame())
    chemical_series = tables.get("chemical_series_summary", pd.DataFrame())
    chemical_cliffs = tables.get("chemical_activity_cliffs", pd.DataFrame())
    chemical_mmp_cliffs = tables.get("chemical_mmp_cliffs", pd.DataFrame())
    chemical_connectivity = tables.get("chemical_connectivity_variant_summary", pd.DataFrame())
    chemical_references = tables.get("chemical_approved_reference_coverage", pd.DataFrame())
    prospective_plan = tables.get("chemical_prospective_selection_plan", pd.DataFrame())
    prospective_summary = tables.get("chemical_prospective_selection_summary", pd.DataFrame())
    quality = _read_json(reports_dir / "quality" / "quality_gate.json")
    completed = int(readiness["satisfied"].sum())
    total = len(readiness)
    processed_manifest = _read_json(reports_dir / "manifests" / "processed_manifest.json")
    build_id = processed_manifest.get("build_id", "unmanifested")
    primary_ready = bool(
        quality.get("passed") is True
        and not model_summary.empty
        and readiness.loc[readiness["criterion"].eq("Defined biological endpoint"), "satisfied"]
        .astype(bool)
        .any()
    )
    bottom_line = (
        "The repository contains a traceable public-data evidence base and a chemically grouped, "
        "endpoint-scoped baseline QSAR evaluation. Menin labels are isolated by endpoint before "
        "aggregation; hERG labels are resolved at structure level; uncertainty, calibration, "
        "applicability domain, quarantine, and provenance outputs are explicit. These models remain "
        "hypothesis-generation tools until independent and prospective lab validation is complete."
        if primary_ready
        else "This build is incomplete: one or more primary models or the analysis-eligible quality "
        "gate is unavailable. Treat the repository as an auditable data pipeline, not as validated "
        "predictive evidence, until the readiness matrix is satisfied."
    )
    figure_labels = {
        "data_quality_attrition.png": "Data attrition",
        "menin_endpoint_counts.png": "Endpoint coverage",
        "menin_source_endpoint_heatmap.png": "Source × endpoint coverage",
        "menin_chemical_space.png": "Chemical space",
        "menin_observed_vs_predicted.png": "Observed vs predicted Menin activity",
        "menin_validation_comparison.png": "Menin validation sensitivity",
        "herg_discrimination_curves.png": "hERG discrimination",
        "herg_calibration.png": "hERG calibration",
        "menin_potency_herg_triage.png": "Potency–hERG triage",
        "menin_medchem_property_landscape.png": "Medicinal-chemistry property landscape",
        "menin_chemical_series_sizes.png": "Chemical-series coverage",
        "menin_activity_cliff_landscape.png": "Activity-cliff landscape",
        "menin_followup_tier_counts.png": "Experimental follow-up tiers",
        "menin_complete_evidence_frontier.png": "Complete-evidence Pareto frontier",
    }
    figure_lines = [
        f"- [{label}](figures/{filename})"
        for filename, label in figure_labels.items()
        if (reports_dir / "figures" / filename).exists()
    ] or ["_No figures were generated for this build._"]
    primary_task = f"menin_{settings.get('modeling', {}).get('primary_menin_endpoint', 'IC50')}_{settings.get('modeling', {}).get('primary_menin_assay_family', 'biochemical_binding')}"
    primary_herg_task = (
        f"herg_{settings.get('herg', {}).get('primary_endpoint', 'IC50')}_"
        f"{settings.get('herg', {}).get('primary_assay_family', 'electrophysiology_functional')}"
    )

    def validation_row(task: str, split: str) -> pd.Series | None:
        if model_summary.empty or not {"task", "requested_split"}.issubset(model_summary.columns):
            return None
        matches = model_summary[
            model_summary["task"].astype(str).eq(task)
            & model_summary["requested_split"].astype(str).eq(split)
            & model_summary.get("status", pd.Series("", index=model_summary.index)).astype(str).eq("trained")
        ]
        return matches.iloc[0] if not matches.empty else None

    temporal_menin = validation_row(primary_task, "temporal")
    temporal_herg = validation_row(primary_herg_task, "temporal")
    blocker_rows = (
        int(
            herg_primary_labels.loc[
                herg_primary_labels.get("label", pd.Series(dtype=str)).eq("blocker"),
                "task_rows",
            ].sum()
        )
        if not herg_primary_labels.empty
        else 0
    )
    nonblocker_rows = (
        int(
            herg_primary_labels.loc[
                herg_primary_labels.get("label", pd.Series(dtype=str)).eq("non_blocker"),
                "task_rows",
            ].sum()
        )
        if not herg_primary_labels.empty
        else 0
    )
    labeled_herg = blocker_rows + nonblocker_rows
    stress_lines: list[str] = []
    if temporal_menin is not None:
        stress_lines.append(
            "- Menin temporal holdout: "
            f"MAE {float(temporal_menin.get('mae')):.3f} pActivity, "
            f"R² {float(temporal_menin.get('r2')):.3f}, and Spearman "
            f"r {float(temporal_menin.get('spearman_r')):.3f}. This is the most "
            "prospective-like public stress test and should govern expectations under drift."
        )
    if temporal_herg is not None:
        stress_lines.append(
            "- hERG temporal holdout: "
            f"ROC AUC {float(temporal_herg.get('roc_auc')):.3f}, balanced accuracy "
            f"{float(temporal_herg.get('balanced_accuracy')):.3f}, and Brier score "
            f"{float(temporal_herg.get('brier_score')):.3f}."
        )
    if labeled_herg:
        stress_lines.append(
            f"- The primary hERG training population is {blocker_rows / labeled_herg:.1%} "
            "blockers. Precision–recall AUC must therefore be interpreted with the class "
            "prevalence; ROC AUC, balanced accuracy, specificity, and calibration are reported "
            "alongside it."
        )
    if not risk_predictions.empty:
        inside = _as_bool(
            risk_predictions.get(
                "herg_inside_applicability_domain",
                pd.Series(False, index=risk_predictions.index),
            )
        )
        observed = _as_bool(
            risk_predictions.get(
                "has_observed_primary_herg_record",
                pd.Series(False, index=risk_predictions.index),
            )
        )
        risk_counts = risk_predictions.get("predicted_herg_risk", pd.Series(dtype=str)).value_counts()
        observed_count = int(observed.sum())
        observed_noun = "structure has" if observed_count == 1 else "structures have"
        risk_count_text = ", ".join(f"{label}={int(count):,}" for label, count in risk_counts.items())
        stress_lines.append(
            f"- Of {len(risk_predictions):,} Menin structures scored for hERG, "
            f"{int(inside.sum()):,} ({inside.mean():.1%}) are inside the hERG applicability "
            f"domain and only {observed_count:,} {observed_noun} an observed primary hERG "
            f"record. Communication-band counts are {risk_count_text}; these are liability "
            "flags, not experimentally validated rankings."
        )
    if not stress_lines:
        stress_lines = ["_No trained temporal stress-test outputs are available._"]

    chemical_lines: list[str] = []
    if chemical_summary:
        tier_counts = chemical_summary.get("priority_tier_counts", {})
        tier_text = ", ".join(
            f"{str(label).replace('_', ' ')}={int(count):,}" for label, count in sorted(tier_counts.items())
        )
        chemical_lines = [
            (
                f"The primary chemical-intelligence population contains "
                f"{int(chemical_summary.get('candidate_count', 0)):,} unique structures across "
                f"{int(chemical_summary.get('unique_scaffold_series', 0)):,} Bemis–Murcko/exact-acyclic "
                f"series and {int(chemical_summary.get('similarity_clusters', 0)):,} Butina clusters."
            ),
            (
                f"Configured high-similarity analysis found "
                f"{int(chemical_summary.get('activity_cliffs', 0)):,} fingerprint activity cliffs; "
                f"single-cut MMP analysis found {int(chemical_summary.get('matched_molecular_pairs', 0)):,} "
                f"pairs, including {int(chemical_summary.get('matched_molecular_pair_cliffs', 0)):,} "
                "≥100-fold potency cliffs. These are SAR review targets, not causal transformations."
            ),
            (
                f"Evidence-first tier counts are {tier_text}. No out-of-domain hERG prediction receives "
                "safety credit, structural alerts are review flags rather than automatic exclusions, and "
                "the safety-free discovery score is labeled separately from the complete-evidence score."
            ),
        ]
    else:
        chemical_lines = ["_No content-linked chemical-intelligence stage is enabled for this build._"]

    lines = [
        "# Menin discovery data and modeling audit",
        "",
        f"Content build: `{build_id}`",
        "",
        "## Bottom line",
        "",
        bottom_line,
        "",
        "## Dataset inventory",
        "",
        _markdown_table(inventory),
        "",
        "## Menin endpoint and assay coverage",
        "",
        _markdown_table(endpoint_assay, limit=30),
        "",
        "## Critical-field completeness",
        "",
        _markdown_table(missingness, limit=40),
        "",
        "## Cross-source mirror linkage",
        "",
        f"The curation layer linked {mirror_links['cross_source_mirror_group_id'].nunique() if not mirror_links.empty else 0:,} potential cross-source mirror groups. Same-source replicates remain intact; a separate no-collapse model sensitivity analysis quantifies the heuristic's effect.",
        "",
        "## Applicability-domain performance",
        "",
        _markdown_table(domain_performance, limit=10),
        "",
        "## hERG task populations",
        "",
        "Primary functional-electrophysiology task:",
        "",
        _markdown_table(herg_primary_labels, limit=10),
        "",
        "Broader pooled sensitivity population (not the headline safety estimate):",
        "",
        _markdown_table(herg_pooled_labels, limit=10),
        "",
        "## Validation summary",
        "",
        _markdown_table(model_summary, limit=30),
        "",
        (
            "Random-split results are included as a sensitivity analysis, not the headline estimate. "
            "Scaffold, chemical-cluster, and temporal results better probe prospective chemical generalization; "
            "any requested strategy fallback is recorded in the `actual_split` column and split manifest."
        ),
        "",
        "## Generalization and safety stress test",
        "",
        *stress_lines,
        "",
        "## Chemical intelligence and experimental prioritization",
        "",
        *chemical_lines,
        "",
        "Approved Menin-inhibitor reference coverage:",
        "",
        _markdown_table(
            chemical_references[
                [
                    column
                    for column in (
                        "name",
                        "regulatory_status",
                        "has_exact_primary_task_structure",
                        "has_any_public_menin_measurement",
                        "public_menin_assay_families",
                        "maximum_primary_achiral_tanimoto",
                        "maximum_primary_chiral_tanimoto",
                    )
                    if column in chemical_references
                ]
            ]
            if not chemical_references.empty
            else chemical_references,
            limit=10,
        ),
        "",
        "The approved-reference panel is a dated coverage benchmark, not a comparator efficacy analysis. Regulatory status does not make public assay contexts interchangeable, and absent primary-task coverage is reported rather than imputed.",
        "",
        "Largest primary-task chemical series:",
        "",
        _markdown_table(
            chemical_series[
                [
                    column
                    for column in (
                        "series_id",
                        "series_size",
                        "median_p_activity",
                        "best_p_activity",
                        "activity_span_log10",
                    )
                    if column in chemical_series
                ]
            ]
            if not chemical_series.empty
            else chemical_series,
            limit=12,
        ),
        "",
        "Top same-context activity cliffs:",
        "",
        _markdown_table(
            chemical_cliffs[
                [
                    column
                    for column in (
                        "structure_id_a",
                        "structure_id_b",
                        "absolute_delta_pactivity",
                        "achiral_morgan_tanimoto",
                        "chiral_morgan_tanimoto",
                        "evidence_context_grade",
                    )
                    if column in chemical_cliffs
                ]
            ].sort_values(
                ["evidence_context_grade", "absolute_delta_pactivity"],
                ascending=[False, False],
            )
            if not chemical_cliffs.empty
            else chemical_cliffs,
            limit=12,
        ),
        "",
        (
            f"The analysis also exports {len(chemical_mmp_cliffs):,} MMP cliffs and "
            f"{len(chemical_connectivity):,} connectivity-equivalent variant groups for source-level review."
        ),
        "",
        "Experimental follow-up tiers:",
        "",
        _markdown_table(
            chemical_priorities[
                [
                    column
                    for column in (
                        "structure_id",
                        "p_activity_median",
                        "qed",
                        "property_window_violation_count",
                        "herg_evidence_status",
                        "experimental_followup_tier",
                        "discovery_rank_without_safety",
                        "complete_evidence_rank",
                    )
                    if column in chemical_priorities
                ]
            ]
            if not chemical_priorities.empty
            else chemical_priorities,
            limit=20,
        ),
        "",
        "These are transparent experimental follow-up categories, not validated drug candidates. Priority 2 explicitly means a potent public-data profile with a safety evidence gap; Priority 3 explicitly carries a modeled or observed liability flag.",
        "",
        "Prospective experimental design quotas:",
        "",
        _markdown_table(prospective_summary, limit=10),
        "",
        _markdown_table(
            prospective_plan[
                [
                    column
                    for column in (
                        "selection_order",
                        "selection_category",
                        "structure_id",
                        "p_activity_median",
                        "series_id",
                        "herg_evidence_status",
                        "selection_rationale",
                    )
                    if column in prospective_plan
                ]
            ]
            if not prospective_plan.empty
            else prospective_plan,
            limit=20,
        ),
        "",
        "This configurable selection is a pre-experimental design spanning exploitation, liability characterization, novelty, cliff confirmation, negative controls, and PK bridges. It must be frozen before testing and does not count as prospective validation until new blinded measurements are returned.",
        "",
        "## High-potency public tasks",
        "",
        _markdown_table(top, limit=20),
        "",
        "Rows above are endpoint–assay tasks, not interchangeable efficacy claims. Review the original records, replicate spread, assay context, censoring, and source rights before experimental prioritization.",
        "",
        "## Quality-gate status",
        "",
        "```json",
        json.dumps(quality, indent=2, sort_keys=True),
        "```",
        "",
        "Quality findings are not all fatal errors: the detailed CSVs distinguish missing public metadata, quarantined measurements, repeated-measurement conflicts, and schema violations. Modeling uses only rows that pass the explicit eligibility policy.",
        "",
        f"## Publication readiness ({completed}/{total} infrastructure criteria currently satisfied)",
        "",
        _markdown_table(readiness, limit=30),
        "",
        "## Figures",
        "",
        *figure_lines,
        "",
        "## Interpretation boundaries",
        "",
        "- Public assay values remain heterogeneous across biochemical, biophysical, and cellular contexts.",
        "- Censored values are preserved as bounds but excluded from point-regression labels; no censored-likelihood model is claimed.",
        "- hERG thresholds define a screening label, not clinical cardiotoxicity, and probabilities require external calibration checks.",
        "- Out-of-domain hERG probabilities are retained for audit but classified as unknown and receive no safety credit.",
        "- PAINS, Brenk, and NIH substructure matches are review alerts, not proof of interference, toxicity, or inactivity.",
        "- Apparent ligand-efficiency and lipophilic-efficiency values use IC50-derived pActivity and are not binding free energies.",
        "- PK/ADMET observations are coverage evidence only until endpoint, species, matrix, route, and units support separate validated tasks.",
        "- Model artifacts must be loaded only from this trusted build and matched to their manifests.",
        "",
        "## Target anchors",
        "",
        f"- Menin/MEN1: {MENIN_TARGET['chembl_id']}; UniProt {MENIN_TARGET['uniprot']}.",
        f"- hERG/KCNH2: {HERG_TARGET['chembl_id']}; UniProt {HERG_TARGET['uniprot']}.",
        "",
        "The complete methods, data dictionary, architecture, limitations, reproducibility instructions, and internal-data intake contract are maintained under `docs/`.",
        "",
    ]
    text = "\n".join(lines)
    publication_path = reports_dir / "publication_summary.md"
    publication_path.write_text(text, encoding="utf-8")
    (reports_dir / "summary.md").write_text(text, encoding="utf-8")
    return publication_path
