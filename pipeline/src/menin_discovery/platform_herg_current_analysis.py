"""Reproducible, non-causal analyses for the wild-type hERG paper data.

The builder consumes the immutable v1/v1.2 hERG artifacts and emits compact,
versioned statistical tables.  It performs no neural training and never turns
QT/QTc context into a molecular hERG label.  Associations and disagreement
statistics are descriptive; they are not causal effects or superiority claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from scipy.stats import rankdata, spearmanr

SCHEMA_VERSION = "platform-herg-current-analysis/1.1"
MANIFEST_NAME = "herg_current_analysis_manifest.json"
REPORT_NAME = "HERG_CURRENT_DATA_ANALYSIS.md"

OUTPUTS = (
    "dataset_inventory.parquet",
    "task_overlap.parquet",
    "measurement_distributions.parquet",
    "measurement_disagreements.parquet",
    "pic50_replicate_dispersion.parquet",
    "categorical_confounding.parquet",
    "q0_structure_descriptors.parquet",
    "descriptor_associations.parquet",
    "descriptor_bins.parquet",
    "descriptor_interactions.parquet",
    "scaffold_split_profile.parquet",
    "clinical_qt_coverage.parquet",
)

TASK_FILES = {
    "Q0": "q0_weak_fixed_dose_binary.parquet",
    "Q1": "q1_quantitative_pic50.parquet",
    "Q2": "q2_functional_assay_aware.parquet",
    "C0": "c0_clinical_development_context.parquet",
    "C1": "c1_qt_context_endpoints.parquet",
}

DESCRIPTORS = (
    "molecular_weight",
    "logp",
    "tpsa",
    "hbond_donors",
    "hbond_acceptors",
    "rotatable_bonds",
    "ring_count",
    "aromatic_ring_fraction",
    "formal_charge",
    "heavy_atom_count",
    "fraction_csp3",
)

INTERACTIONS = (
    ("logp", "formal_charge"),
    ("logp", "tpsa"),
    ("aromatic_ring_fraction", "formal_charge"),
    ("molecular_weight", "rotatable_bonds"),
    ("logp", "aromatic_ring_fraction"),
)


class HergCurrentAnalysisError(RuntimeError):
    """Raised when the statistical build or validation fails closed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_digest(body: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["manifest_sha256"] = hashlib.sha256(_canonical_json(result).encode()).hexdigest()
    return result


def _checked_parquet(path: Path, required: Iterable[str]) -> Path:
    if path.is_symlink() or not path.is_file() or path.suffix.casefold() != ".parquet":
        raise HergCurrentAnalysisError(f"missing, unsafe, or non-Parquet input: {path}")
    columns = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(set(required) - columns)
    if missing:
        raise HergCurrentAnalysisError(f"{path.name} missing required columns: {missing}")
    return path.resolve()


def _input_binding(role: str, path: Path) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "rows": pq.ParquetFile(path).metadata.num_rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _write_frame(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    clean = frame.reset_index(drop=True).replace({np.nan: None})
    table = pa.Table.from_pandas(clean, preserve_index=False)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        data_page_version="2.0",
    )
    return {
        "path": path.name,
        "rows": table.num_rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "arrow_schema_sha256": hashlib.sha256(table.schema.serialize().to_pybytes()).hexdigest(),
    }


def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _eligible_task(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    if task in {"Q0", "Q1", "Q2"} and "eligible" in frame:
        return frame[frame["eligible"].astype(bool)].copy()
    if task == "C1" and "context_eligible" in frame:
        return frame[frame["context_eligible"].astype(bool)].copy()
    return frame.copy()


def _inventory(tasks: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task, raw in sorted(tasks.items()):
        frame = _eligible_task(raw, task)
        structures = (
            frame["structure_id"].dropna().astype(str) if "structure_id" in frame else pd.Series(dtype=str)
        )
        labels = (
            frame["target_class"].dropna().astype(int) if "target_class" in frame else pd.Series(dtype=int)
        )
        positives = int((labels == 1).sum())
        rows.append(
            {
                "dataset": task,
                "all_rows": int(len(raw)),
                "eligible_rows": int(len(frame)),
                "unique_structures": int(structures.nunique()),
                "positive_labels": positives,
                "negative_labels": int((labels == 0).sum()),
                "gray_zone_labels": int((labels == 2).sum()),
                "positive_prevalence": _safe_rate(positives, int(labels.isin([0, 1]).sum())),
                "exact_pic50_rows": int(
                    (
                        (frame.get("target_relation", pd.Series(index=frame.index, dtype=object)) == "=")
                        & frame.get("target_pic50", pd.Series(index=frame.index, dtype=float)).notna()
                    ).sum()
                ),
                "interpretation": {
                    "Q0": "large weak fixed-dose binary screen",
                    "Q1": "quantitative pIC50 potency task",
                    "Q2": "functional assay-aware task",
                    "C0": "clinical-development context only",
                    "C1": "QT/QTc context only; never a hERG label",
                }[task],
            }
        )
    return pd.DataFrame(rows).sort_values("dataset")


def _task_overlap(tasks: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    sets = {
        task: set(_eligible_task(frame, task)["structure_id"].dropna().astype(str))
        for task, frame in tasks.items()
    }
    rows: list[dict[str, Any]] = []
    for left, right in combinations(sorted(sets), 2):
        shared = sets[left] & sets[right]
        union = sets[left] | sets[right]
        rows.append(
            {
                "left_task": left,
                "right_task": right,
                "left_structures": len(sets[left]),
                "right_structures": len(sets[right]),
                "shared_structures": len(shared),
                "union_structures": len(union),
                "jaccard": _safe_rate(len(shared), len(union)),
                "left_coverage": _safe_rate(len(shared), len(sets[left])),
                "right_coverage": _safe_rate(len(shared), len(sets[right])),
            }
        )
    return pd.DataFrame(rows)


def _measurement_distributions(modality: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    axes = (
        "source_family",
        "measurement_modality",
        "automation_class",
        "dose_design",
        "native_endpoint",
        "wild_type_evidence_scope",
        "modality_confidence",
        "automation_confidence",
        "dose_design_confidence",
    )
    total = len(modality)
    for axis in axes:
        for category, group in modality.groupby(axis, dropna=False, sort=True):
            rows.append(
                {
                    "axis": axis,
                    "category": "<missing>" if pd.isna(category) else str(category),
                    "source_family": None,
                    "observation_count": int(len(group)),
                    "unique_structures": int(group["structure_id"].dropna().nunique()),
                    "fraction_of_observations": _safe_rate(len(group), total),
                }
            )
    for (source, measurement), group in modality.groupby(
        ["source_family", "measurement_modality"], dropna=False, sort=True
    ):
        rows.append(
            {
                "axis": "source_by_modality",
                "category": str(measurement),
                "source_family": str(source),
                "observation_count": int(len(group)),
                "unique_structures": int(group["structure_id"].dropna().nunique()),
                "fraction_of_observations": _safe_rate(len(group), total),
            }
        )
    return pd.DataFrame(rows).sort_values(["axis", "source_family", "category"], na_position="first")


def _consensus(values: pd.Series) -> int | None:
    clean = values.dropna().astype(int)
    if clean.empty:
        return None
    counts = clean.value_counts()
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        return None
    return int(counts.index[0])


def _kappa(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) == 0:
        return None
    observed = float(np.mean(left == right))
    p_left = float(np.mean(left == 1))
    p_right = float(np.mean(right == 1))
    expected = p_left * p_right + (1 - p_left) * (1 - p_right)
    return (observed - expected) / (1 - expected) if expected < 1 else None


def _binary_pair_rows(frame: pd.DataFrame, axis: str) -> list[dict[str, Any]]:
    grouped = (
        frame.groupby(["structure_id", axis], sort=True)["derived_binary_label"]
        .agg(_consensus)
        .dropna()
        .reset_index()
    )
    groups = sorted(grouped[axis].astype(str).unique())
    rows: list[dict[str, Any]] = []
    for left, right in combinations(groups, 2):
        lframe = grouped[grouped[axis].astype(str) == left][["structure_id", "derived_binary_label"]]
        rframe = grouped[grouped[axis].astype(str) == right][["structure_id", "derived_binary_label"]]
        paired = lframe.merge(rframe, on="structure_id", suffixes=("_left", "_right"))
        if paired.empty:
            continue
        lv = paired["derived_binary_label_left"].to_numpy(dtype=int)
        rv = paired["derived_binary_label_right"].to_numpy(dtype=int)
        concordant = int(np.sum(lv == rv))
        rows.append(
            {
                "comparison_axis": axis,
                "left_group": left,
                "right_group": right,
                "outcome": "binary_consensus",
                "matched_structures": int(len(paired)),
                "concordant": concordant,
                "discordant": int(len(paired) - concordant),
                "agreement": _safe_rate(concordant, len(paired)),
                "cohen_kappa": _kappa(lv, rv),
                "mean_absolute_difference": None,
                "median_signed_difference": None,
                "spearman_rho": None,
                "caveat": "majority consensus; tied within-group labels excluded",
            }
        )
    return rows


def _continuous_pair_rows(frame: pd.DataFrame, axis: str) -> list[dict[str, Any]]:
    grouped = frame.groupby(["structure_id", axis], sort=True)["pic50_value"].mean().reset_index()
    groups = sorted(grouped[axis].astype(str).unique())
    rows: list[dict[str, Any]] = []
    for left, right in combinations(groups, 2):
        lframe = grouped[grouped[axis].astype(str) == left][["structure_id", "pic50_value"]]
        rframe = grouped[grouped[axis].astype(str) == right][["structure_id", "pic50_value"]]
        paired = lframe.merge(rframe, on="structure_id", suffixes=("_left", "_right"))
        if paired.empty:
            continue
        delta = paired["pic50_value_left"] - paired["pic50_value_right"]
        rho: float | None = None
        if (
            len(paired) >= 3
            and paired["pic50_value_left"].nunique() > 1
            and paired["pic50_value_right"].nunique() > 1
        ):
            value = spearmanr(paired["pic50_value_left"], paired["pic50_value_right"]).statistic
            rho = None if math.isnan(float(value)) else float(value)
        rows.append(
            {
                "comparison_axis": axis,
                "left_group": left,
                "right_group": right,
                "outcome": "exact_pic50_mean",
                "matched_structures": int(len(paired)),
                "concordant": None,
                "discordant": None,
                "agreement": None,
                "cohen_kappa": None,
                "mean_absolute_difference": float(delta.abs().mean()),
                "median_signed_difference": float(delta.median()),
                "spearman_rho": rho,
                "caveat": "descriptive matched means; protocol and source remain confounded",
            }
        )
    return rows


def _disagreements(ledger: pd.DataFrame, modality: pd.DataFrame) -> pd.DataFrame:
    keep = modality[["observation_id", "measurement_modality", "automation_class"]]
    merged = ledger.merge(keep, on="observation_id", how="inner", validate="one_to_one")
    labeled = merged[merged["derived_binary_label"].notna() & merged["structure_id"].notna()].copy()
    exact = merged[merged["pic50_value"].notna() & merged["structure_id"].notna()].copy()
    rows: list[dict[str, Any]] = []
    for axis in ("source_family", "measurement_modality", "automation_class"):
        rows.extend(_binary_pair_rows(labeled, axis))
        rows.extend(_continuous_pair_rows(exact, axis))
    if not rows:
        return pd.DataFrame(
            columns=[
                "comparison_axis",
                "left_group",
                "right_group",
                "outcome",
                "matched_structures",
                "concordant",
                "discordant",
                "agreement",
                "cohen_kappa",
                "mean_absolute_difference",
                "median_signed_difference",
                "spearman_rho",
                "caveat",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["outcome", "comparison_axis", "matched_structures", "left_group", "right_group"],
        ascending=[True, True, False, True, True],
    )


def _replicate_dispersion(q1: pd.DataFrame) -> pd.DataFrame:
    exact = q1[(q1["target_relation"] == "=") & q1["target_pic50"].notna()].copy()
    rows: list[dict[str, Any]] = []
    for structure_id, group in exact.groupby("structure_id", sort=True):
        if len(group) < 2:
            continue
        values = group["target_pic50"].to_numpy(dtype=float)
        q1v, q3v = np.quantile(values, [0.25, 0.75])
        rows.append(
            {
                "structure_id": str(structure_id),
                "observation_count": int(len(group)),
                "source_count": int(group["source_family"].nunique()),
                "assay_count": int(group["assay_id"].dropna().nunique()),
                "source_families_json": _canonical_json(sorted(group["source_family"].astype(str).unique())),
                "mean_pic50": float(np.mean(values)),
                "sample_sd_pic50": float(np.std(values, ddof=1)),
                "median_pic50": float(np.median(values)),
                "iqr_pic50": float(q3v - q1v),
                "range_pic50": float(np.max(values) - np.min(values)),
                "max_absolute_deviation": float(np.max(np.abs(values - np.median(values)))),
            }
        )
    return pd.DataFrame(rows).sort_values(["range_pic50", "structure_id"], ascending=[False, True])


def _cramers_v(left: pd.Series, right: pd.Series) -> tuple[int, int, int, float | None, float | None]:
    table = pd.crosstab(left.fillna("<missing>"), right.fillna("<missing>"))
    observed = table.to_numpy(dtype=float)
    n = int(observed.sum())
    r, k = observed.shape
    if n == 0 or r < 2 or k < 2:
        return n, r, k, None, None
    row = observed.sum(axis=1, keepdims=True)
    col = observed.sum(axis=0, keepdims=True)
    expected = row @ col / n
    chi2 = float(
        np.sum(
            np.divide((observed - expected) ** 2, expected, out=np.zeros_like(expected), where=expected > 0)
        )
    )
    phi2 = chi2 / n
    phi2_corrected = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    r_corrected = r - ((r - 1) ** 2) / max(n - 1, 1)
    k_corrected = k - ((k - 1) ** 2) / max(n - 1, 1)
    denominator = min(k_corrected - 1, r_corrected - 1)
    value = math.sqrt(phi2_corrected / denominator) if denominator > 0 else None
    return n, r, k, chi2, value


def _categorical_confounding(modality: pd.DataFrame) -> pd.DataFrame:
    working = modality.copy()
    counts = working["native_endpoint"].value_counts()
    working["endpoint_grouped"] = working["native_endpoint"].where(
        working["native_endpoint"].map(counts) >= 100, "other_rare_endpoint"
    )
    pairs = (
        ("source_family", "measurement_modality"),
        ("source_family", "automation_class"),
        ("source_family", "dose_design"),
        ("source_family", "endpoint_grouped"),
        ("measurement_modality", "automation_class"),
        ("measurement_modality", "dose_design"),
        ("measurement_modality", "endpoint_grouped"),
        ("automation_class", "dose_design"),
    )
    rows = []
    for left, right in pairs:
        n, lcats, rcats, chi2, effect = _cramers_v(working[left], working[right])
        rows.append(
            {
                "left_axis": left,
                "right_axis": right,
                "observations": n,
                "left_categories": lcats,
                "right_categories": rcats,
                "chi_square": chi2,
                "bias_corrected_cramers_v": effect,
                "interpretation": "association indicates confounding/entanglement, not causation",
            }
        )
    return pd.DataFrame(rows).sort_values("bias_corrected_cramers_v", ascending=False)


def _descriptor_record(row: Any) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(str(row.standardized_smiles))
    if mol is None:
        raise HergCurrentAnalysisError(f"RDKit could not parse standardized structure {row.structure_id}")
    rings = float(rdMolDescriptors.CalcNumRings(mol))
    aromatic_rings = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    return {
        "structure_id": str(row.structure_id),
        "standardized_smiles": str(row.standardized_smiles),
        "model_split": str(row.model_split),
        "scaffold_group_id": str(row.scaffold_group_id),
        "target_class": int(row.target_class),
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hbond_donors": float(Lipinski.NumHDonors(mol)),
        "hbond_acceptors": float(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
        "ring_count": rings,
        "aromatic_ring_fraction": aromatic_rings / rings if rings else 0.0,
        "formal_charge": float(Chem.GetFormalCharge(mol)),
        "heavy_atom_count": float(mol.GetNumHeavyAtoms()),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
    }


def _descriptors(q0: pd.DataFrame) -> pd.DataFrame:
    eligible = q0[q0["eligible"].astype(bool) & q0["target_class"].notna()].copy()
    if eligible["structure_id"].duplicated().any():
        raise HergCurrentAnalysisError("eligible Q0 structures are not unique")
    records = [_descriptor_record(row) for row in eligible.itertuples(index=False)]
    return pd.DataFrame(records).sort_values("structure_id")


def _descriptor_associations(frame: pd.DataFrame) -> pd.DataFrame:
    labels = frame["target_class"].to_numpy(dtype=int)
    n1 = int(np.sum(labels == 1))
    n0 = int(np.sum(labels == 0))
    p = n1 / len(labels)
    rows = []
    for descriptor in DESCRIPTORS:
        values = frame[descriptor].to_numpy(dtype=float)
        pos = values[labels == 1]
        neg = values[labels == 0]
        pooled_denominator = n1 + n0 - 2
        pooled_variance = (
            ((n1 - 1) * np.var(pos, ddof=1) + (n0 - 1) * np.var(neg, ddof=1)) / pooled_denominator
            if pooled_denominator > 0
            else 0.0
        )
        pooled_sd = math.sqrt(pooled_variance) if pooled_variance > 0 else 0.0
        total_sd = float(np.std(values, ddof=0))
        ranks = rankdata(values, method="average")
        auc = (float(ranks[labels == 1].sum()) - n1 * (n1 + 1) / 2) / (n1 * n0)
        rows.append(
            {
                "descriptor": descriptor,
                "observations": int(len(values)),
                "positive_count": n1,
                "negative_count": n0,
                "positive_mean": float(np.mean(pos)),
                "negative_mean": float(np.mean(neg)),
                "positive_median": float(np.median(pos)),
                "negative_median": float(np.median(neg)),
                "standardized_mean_difference": float((np.mean(pos) - np.mean(neg)) / pooled_sd)
                if pooled_sd
                else None,
                "point_biserial_r": float((np.mean(pos) - np.mean(neg)) * math.sqrt(p * (1 - p)) / total_sd)
                if total_sd
                else None,
                "univariate_auc_active_higher": auc,
                "exploratory_only": True,
            }
        )
    return pd.DataFrame(rows).sort_values("univariate_auc_active_higher", ascending=False)


def _quantile_codes(values: pd.Series, quantiles: int) -> pd.Series:
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=quantiles, labels=False, duplicates="drop").astype(int)


def _descriptor_bins(frame: pd.DataFrame) -> pd.DataFrame:
    overall = float(frame["target_class"].mean())
    rows = []
    for descriptor in DESCRIPTORS:
        working = frame[[descriptor, "target_class"]].copy()
        working["bin"] = _quantile_codes(working[descriptor], 10)
        for bin_id, group in working.groupby("bin", sort=True):
            positives = int(group["target_class"].sum())
            rate = positives / len(group)
            rows.append(
                {
                    "descriptor": descriptor,
                    "quantile_bin": int(bin_id) + 1,
                    "lower_observed": float(group[descriptor].min()),
                    "upper_observed": float(group[descriptor].max()),
                    "structures": int(len(group)),
                    "active_structures": positives,
                    "active_prevalence": rate,
                    "prevalence_lift": rate / overall if overall else None,
                }
            )
    return pd.DataFrame(rows).sort_values(["descriptor", "quantile_bin"])


def _smoothed_logit(positives: int, total: int) -> float:
    rate = (positives + 0.5) / (total + 1.0)
    return math.log(rate / (1.0 - rate))


def _descriptor_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    overall_positive = int(frame["target_class"].sum())
    overall_logit = _smoothed_logit(overall_positive, len(frame))
    overall_rate = overall_positive / len(frame)
    rows = []
    for left, right in INTERACTIONS:
        working = frame[[left, right, "target_class"]].copy()
        working["left_bin"] = _quantile_codes(working[left], 3)
        working["right_bin"] = _quantile_codes(working[right], 3)
        left_stats = working.groupby("left_bin")["target_class"].agg(["sum", "count"])
        right_stats = working.groupby("right_bin")["target_class"].agg(["sum", "count"])
        for (left_bin, right_bin), group in working.groupby(["left_bin", "right_bin"], sort=True):
            positives = int(group["target_class"].sum())
            total = len(group)
            cell_logit = _smoothed_logit(positives, total)
            left_logit = _smoothed_logit(
                int(left_stats.loc[left_bin, "sum"]), int(left_stats.loc[left_bin, "count"])
            )
            right_logit = _smoothed_logit(
                int(right_stats.loc[right_bin, "sum"]), int(right_stats.loc[right_bin, "count"])
            )
            rows.append(
                {
                    "left_descriptor": left,
                    "right_descriptor": right,
                    "left_tertile": int(left_bin) + 1,
                    "right_tertile": int(right_bin) + 1,
                    "structures": int(total),
                    "active_structures": positives,
                    "active_prevalence": positives / total,
                    "prevalence_lift": (positives / total) / overall_rate if overall_rate else None,
                    "additive_logit_residual": cell_logit - (left_logit + right_logit - overall_logit),
                    "exploratory_only": True,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["left_descriptor", "right_descriptor", "left_tertile", "right_tertile"]
    )


def _split_profile(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    train = frame[frame["model_split"] == "train"]
    for split, group in frame.groupby("model_split", sort=True):
        for metric in ("target_class", *DESCRIPTORS):
            values = group[metric].to_numpy(dtype=float)
            train_values = train[metric].to_numpy(dtype=float)
            train_sd = float(np.std(train_values, ddof=1))
            rows.append(
                {
                    "model_split": str(split),
                    "metric": metric,
                    "structures": int(len(group)),
                    "mean": float(np.mean(values)),
                    "sample_sd": float(np.std(values, ddof=1)),
                    "median": float(np.median(values)),
                    "q05": float(np.quantile(values, 0.05)),
                    "q95": float(np.quantile(values, 0.95)),
                    "standardized_mean_difference_vs_train": float(
                        (np.mean(values) - np.mean(train_values)) / train_sd
                    )
                    if train_sd
                    else None,
                }
            )
    return pd.DataFrame(rows).sort_values(["metric", "model_split"])


def _clinical_qt_coverage(tasks: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    structures = {
        task: set(_eligible_task(frame, task)["structure_id"].dropna().astype(str))
        for task, frame in tasks.items()
    }
    rows: list[dict[str, Any]] = []
    for context in ("C0", "C1"):
        for molecular in ("Q0", "Q1", "Q2"):
            overlap = structures[context] & structures[molecular]
            rows.append(
                {
                    "analysis_axis": "structure_overlap",
                    "category": f"{context}_with_{molecular}",
                    "record_count": len(overlap),
                    "denominator_count": len(structures[context]),
                    "coverage_fraction": _safe_rate(len(overlap), len(structures[context])),
                    "unique_structures": len(overlap),
                    "unique_trials": None,
                    "numeric_result_count": None,
                }
            )
    c1 = _eligible_task(tasks["C1"], "C1")
    for category, group in c1.groupby("candidate_classification", sort=True):
        rows.append(
            {
                "analysis_axis": "qt_endpoint_class",
                "category": str(category),
                "record_count": int(len(group)),
                "denominator_count": int(len(c1)),
                "coverage_fraction": _safe_rate(len(group), len(c1)),
                "unique_structures": int(group["structure_id"].nunique()),
                "unique_trials": int(group["nct_id"].nunique()),
                "numeric_result_count": int(group["reported_numeric_value_count"].sum()),
            }
        )
    for split, group in c1.groupby("model_split", sort=True):
        rows.append(
            {
                "analysis_axis": "qt_scaffold_split",
                "category": str(split),
                "record_count": int(len(group)),
                "denominator_count": int(len(c1)),
                "coverage_fraction": _safe_rate(len(group), len(c1)),
                "unique_structures": int(group["structure_id"].nunique()),
                "unique_trials": int(group["nct_id"].nunique()),
                "numeric_result_count": int(group["reported_numeric_value_count"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["analysis_axis", "category"])


def _fmt(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def _build_report(
    inventory: pd.DataFrame,
    overlaps: pd.DataFrame,
    distributions: pd.DataFrame,
    disagreements: pd.DataFrame,
    replicates: pd.DataFrame,
    confounding: pd.DataFrame,
    associations: pd.DataFrame,
    bins: pd.DataFrame,
    interactions: pd.DataFrame,
    split_profile: pd.DataFrame,
    clinical: pd.DataFrame,
) -> str:
    q0 = inventory[inventory["dataset"] == "Q0"].iloc[0]
    q1 = inventory[inventory["dataset"] == "Q1"].iloc[0]
    q2 = inventory[inventory["dataset"] == "Q2"].iloc[0]
    modality = distributions[distributions["axis"] == "measurement_modality"].sort_values(
        "observation_count", ascending=False
    )
    top_conf = confounding.iloc[0]
    top_assoc = associations.iloc[
        np.argmax(np.abs(associations["univariate_auc_active_higher"].to_numpy(dtype=float) - 0.5))
    ]
    largest_ranges = replicates.head(10)
    informative_disagreements = disagreements[disagreements["matched_structures"] >= 10].head(15)
    max_split = split_profile[split_profile["model_split"] != "train"].copy()
    max_shift = float(max_split["standardized_mean_difference_vs_train"].abs().max())
    strongest_cell = interactions.iloc[
        np.argmax(np.abs(interactions["additive_logit_residual"].to_numpy(dtype=float)))
    ]
    lines = [
        "# Current wild-type hERG data analysis",
        "",
        "## Executive result",
        "",
        f"The present hierarchy supports **{int(q0.unique_structures):,} Q0 structures**, "
        f"**{int(q1.eligible_rows):,} Q1 quantitative records**, and **{int(q2.eligible_rows):,} "
        "eligible Q2 functional records**. The core scientific advantage is not row count alone: "
        "target scope, evidence quality, assay technology, automation, dose design, scaffold split, "
        "clinical development, and QT/QTc context remain separately addressable.",
        "",
        "This analysis establishes data assets and testable hypotheses. It does **not** establish "
        "causality, clinical validity, or superiority over published models. Superiority must be shown "
        "on identical locked and prospective challenges.",
        "",
        "## Quality and prevalence",
        "",
        "| Layer | Eligible rows | Structures | Positive prevalence | Meaning |",
        "|---|---:|---:|---:|---|",
    ]
    for row in inventory.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {int(row.eligible_rows):,} | {int(row.unique_structures):,} | "
            f"{_fmt(row.positive_prevalence, 5)} | {row.interpretation} |"
        )
    lines.extend(
        [
            "",
            "Q0's low natural prevalence makes accuracy unsuitable as a headline metric. PR-AUC, "
            "calibration, false-negative rate, enrichment, and recall at a fixed testing budget are required. "
            "Q1 prevalence is conditional on the blocker/non-blocker zones; its 13,947 gray-zone records are "
            "excluded from that denominator and remain available for ordinal/regression analysis.",
            "",
            "## Measurement landscape",
            "",
            "| Modality | Observations | Structures |",
            "|---|---:|---:|",
        ]
    )
    for row in modality.itertuples(index=False):
        lines.append(f"| {row.category} | {int(row.observation_count):,} | {int(row.unique_structures):,} |")
    lines.extend(
        [
            "",
            f"The strongest categorical entanglement is **{top_conf.left_axis} × {top_conf.right_axis}** "
            f"(bias-corrected Cramér's V {_fmt(top_conf.bias_corrected_cramers_v)}; "
            f"n={int(top_conf.observations):,}). This is evidence that naive pooling can confound source, "
            "method, or protocol—not evidence that either variable causes potency.",
            "",
            "## Matched-structure disagreement",
            "",
            "Only exact structure matches are compared. Binary values are within-group majority consensuses; "
            "ties are excluded. Quantitative comparisons use exact pIC50 means and remain protocol-confounded.",
            "",
            "| Axis | Pair | Outcome | Matched | Agreement | Mean absolute ΔpIC50 | Spearman |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in informative_disagreements.itertuples(index=False):
        lines.append(
            f"| {row.comparison_axis} | {row.left_group} vs {row.right_group} | {row.outcome} | "
            f"{int(row.matched_structures):,} | {_fmt(row.agreement)} | "
            f"{_fmt(row.mean_absolute_difference)} | {_fmt(row.spearman_rho)} |"
        )
    lines.extend(
        [
            "",
            "## Quantitative replicate stability",
            "",
            f"There are **{len(replicates):,} structures with at least two exact pIC50 records**. Large "
            "ranges identify audit priorities, not automatic outliers: protocol, source, cell context, and "
            "temperature can create real measurement differences.",
            "",
            "| Structure | Records | Sources | pIC50 range | SD |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in largest_ranges.itertuples(index=False):
        lines.append(
            f"| {row.structure_id} | {int(row.observation_count)} | {int(row.source_count)} | "
            f"{row.range_pic50:.3f} | {row.sample_sd_pic50:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Fundamental molecular features",
            "",
            f"The strongest univariate separator in the weak Q0 screen is **{top_assoc.descriptor}** "
            f"(active-higher AUC {_fmt(top_assoc.univariate_auc_active_higher)}, standardized mean "
            f"difference {_fmt(top_assoc.standardized_mean_difference)}). These are exploratory structure–label "
            "associations and may partly reflect library composition or screen selection.",
            "",
            f"The largest prespecified two-feature additive-logit residual is "
            f"**{strongest_cell.left_descriptor} × {strongest_cell.right_descriptor}**, tertiles "
            f"{int(strongest_cell.left_tertile)}/{int(strongest_cell.right_tertile)} "
            f"(residual {_fmt(strongest_cell.additive_logit_residual)}, "
            f"prevalence lift {_fmt(strongest_cell.prevalence_lift)}; "
            f"n={int(strongest_cell.structures):,}). It is a hypothesis for matched-series and "
            "assay-aware confirmation, not mechanistic proof.",
            "",
            "Descriptor deciles, all prespecified interaction cells, and the full structure-level compact "
            "descriptor matrix are retained as machine-readable artifacts.",
            "",
            "## Scaffold-split shift",
            "",
            f"The largest absolute standardized train-versus-validation/test shift among class prevalence and "
            f"compact descriptors is **{max_shift:.3f} SD**. Small marginal shifts do not make the task easy: "
            "scaffold separation can preserve global property distributions while removing close analogs.",
            "",
            "## Clinical and QT/QTc coverage",
            "",
            "Clinical-development and QT/QTc records are context/evaluation layers only. They are never "
            "promoted into direct molecular hERG potency labels. Their overlap tables quantify how much of the "
            "molecular hierarchy can currently support downstream exposure/QT translation.",
            "",
            "## What this project can credibly emphasize",
            "",
            "1. Larger public coverage is combined with explicit WT scope and immutable provenance.",
            "2. Weak fixed-dose, quantitative potency, functional assays, clinical context, and QT/QTc are "
            "separate tasks instead of pooled labels.",
            "3. Measurement modality, automation, dose design, source, and scaffold are measurable evaluation axes.",
            "4. Natural-prevalence analyses expose the false-positive problem hidden by balanced benchmarks.",
            "5. Fundamental descriptors and prespecified interactions generate falsifiable hypotheses.",
            "6. These are design superiorities. Predictive superiority remains a future empirical result until "
            "competitors are reproduced on identical locked/prospective data.",
            "",
            "## Required next analyses",
            "",
            "- Reproduce published comparators on the frozen split and modality/source holdouts.",
            "- Add temperature, voltage protocol, cell line, incubation, and platform fields where source text supports them.",
            "- Validate feature interactions in matched molecular pairs and quantitative Q1/Q2 data.",
            "- Add unbound exposure, metabolites, and multi-ion-channel data before claiming QT-risk prediction.",
            "- Reserve a blinded multi-laboratory manual-patch panel for prospective validation.",
            "",
            "## Artifact contract",
            "",
            f"Schema `{SCHEMA_VERSION}`; RDKit `{rdBase.rdkitVersion}`. All Parquet outputs and their SHA-256 "
            "digests are recorded in the manifest. Inputs are content-bound; upstream artifacts are not modified.",
            "",
        ]
    )
    _ = overlaps, bins, clinical
    return "\n".join(lines)


def _paths(hierarchy_root: Path, quality_root: Path, modality_root: Path) -> dict[str, Path]:
    paths = {
        "observation_ledger": _checked_parquet(
            hierarchy_root / "observation_ledger.parquet",
            [
                "observation_id",
                "structure_id",
                "source_family",
                "derived_binary_label",
                "pic50_value",
                "native_endpoint",
            ],
        ),
        "modality_index": _checked_parquet(
            modality_root / "herg_measurement_modality_index.parquet",
            [
                "observation_id",
                "structure_id",
                "source_family",
                "native_endpoint",
                "measurement_modality",
                "automation_class",
                "dose_design",
                "wild_type_evidence_scope",
                "modality_confidence",
                "automation_confidence",
                "dose_design_confidence",
            ],
        ),
    }
    task_required = {
        "Q0": [
            "structure_id",
            "standardized_smiles",
            "target_class",
            "eligible",
            "model_split",
            "scaffold_group_id",
        ],
        "Q1": ["structure_id", "target_pic50", "target_relation", "source_family", "assay_id", "eligible"],
        "Q2": ["structure_id", "eligible"],
        "C0": ["structure_id"],
        "C1": [
            "structure_id",
            "context_eligible",
            "candidate_classification",
            "nct_id",
            "model_split",
            "reported_numeric_value_count",
        ],
    }
    for task, filename in TASK_FILES.items():
        paths[task] = _checked_parquet(quality_root / filename, task_required[task])
    return paths


def validate_herg_current_analysis(output_root: Path | str) -> dict[str, Any]:
    root = Path(output_root)
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise HergCurrentAnalysisError(f"missing analysis manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise HergCurrentAnalysisError("unexpected analysis schema version")
    supplied = manifest.get("manifest_sha256")
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    expected = hashlib.sha256(_canonical_json(body).encode()).hexdigest()
    if supplied != expected:
        raise HergCurrentAnalysisError("manifest digest mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or {item.get("path") for item in artifacts} != set(OUTPUTS):
        raise HergCurrentAnalysisError("artifact inventory mismatch")
    for artifact in artifacts:
        path = root / str(artifact["path"])
        if path.is_symlink() or not path.is_file():
            raise HergCurrentAnalysisError(f"missing or unsafe output: {path}")
        if _sha256_file(path) != artifact.get("sha256"):
            raise HergCurrentAnalysisError(f"artifact digest mismatch: {path.name}")
        if pq.ParquetFile(path).metadata.num_rows != artifact.get("rows"):
            raise HergCurrentAnalysisError(f"artifact row count mismatch: {path.name}")
    descriptor_rows = pq.ParquetFile(root / "q0_structure_descriptors.parquet").metadata.num_rows
    inventory = pq.read_table(root / "dataset_inventory.parquet").to_pandas()
    q0_structures = int(inventory.loc[inventory["dataset"] == "Q0", "unique_structures"].iloc[0])
    if descriptor_rows != q0_structures:
        raise HergCurrentAnalysisError("descriptor/Q0 structure count mismatch")
    if manifest.get("scientific_contract", {}).get("causal_claims_established") is not False:
        raise HergCurrentAnalysisError("scientific contract must deny causal claims")
    if manifest.get("scientific_contract", {}).get("qt_used_as_herg_label") is not False:
        raise HergCurrentAnalysisError("scientific contract must keep QT separate")
    return manifest


def build_herg_current_analysis(
    hierarchy_root: Path | str,
    quality_root: Path | str,
    modality_root: Path | str,
    output_root: Path | str,
    report_root: Path | str,
) -> dict[str, Any]:
    hierarchy = Path(hierarchy_root)
    quality = Path(quality_root)
    modality_path = Path(modality_root)
    output = Path(output_root)
    report = Path(report_root)
    paths = _paths(hierarchy, quality, modality_path)
    bindings = [_input_binding(role, path) for role, path in sorted(paths.items())]
    if output.exists():
        existing = validate_herg_current_analysis(output)
        if existing.get("inputs") == bindings:
            return existing
        raise HergCurrentAnalysisError("output exists but is bound to different inputs")

    tasks = {task: pq.read_table(paths[task]).to_pandas() for task in TASK_FILES}
    ledger = pq.read_table(
        paths["observation_ledger"],
        columns=[
            "observation_id",
            "structure_id",
            "source_family",
            "derived_binary_label",
            "pic50_value",
            "native_endpoint",
        ],
    ).to_pandas()
    modality = pq.read_table(paths["modality_index"]).to_pandas()

    inventory = _inventory(tasks)
    overlaps = _task_overlap(tasks)
    distributions = _measurement_distributions(modality)
    disagreements = _disagreements(ledger, modality)
    replicates = _replicate_dispersion(tasks["Q1"])
    confounding = _categorical_confounding(modality)
    descriptors = _descriptors(tasks["Q0"])
    associations = _descriptor_associations(descriptors)
    bins = _descriptor_bins(descriptors)
    interactions = _descriptor_interactions(descriptors)
    split_profile = _split_profile(descriptors)
    clinical = _clinical_qt_coverage(tasks)

    frames = {
        "dataset_inventory.parquet": inventory,
        "task_overlap.parquet": overlaps,
        "measurement_distributions.parquet": distributions,
        "measurement_disagreements.parquet": disagreements,
        "pic50_replicate_dispersion.parquet": replicates,
        "categorical_confounding.parquet": confounding,
        "q0_structure_descriptors.parquet": descriptors,
        "descriptor_associations.parquet": associations,
        "descriptor_bins.parquet": bins,
        "descriptor_interactions.parquet": interactions,
        "scaffold_split_profile.parquet": split_profile,
        "clinical_qt_coverage.parquet": clinical,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        artifacts = [_write_frame(temporary / name, frames[name]) for name in OUTPUTS]
        manifest = _manifest_digest(
            {
                "schema_version": SCHEMA_VERSION,
                "inputs": bindings,
                "artifacts": artifacts,
                "counts": {
                    "q0_eligible_structures": int(len(descriptors)),
                    "exact_pic50_replicate_structures": int(len(replicates)),
                    "matched_comparison_rows": int(len(disagreements)),
                    "descriptor_associations": int(len(associations)),
                    "prespecified_interaction_cells": int(len(interactions)),
                },
                "software": {
                    "rdkit_version": rdBase.rdkitVersion,
                    "numpy_version": np.__version__,
                    "pandas_version": pd.__version__,
                    "pyarrow_version": pa.__version__,
                },
                "scientific_contract": {
                    "wild_type_scope_inherited_from_upstream": True,
                    "qt_used_as_herg_label": False,
                    "causal_claims_established": False,
                    "predictive_superiority_established": False,
                    "associations_are_exploratory": True,
                    "comparisons_are_exact_structure_matched": True,
                },
            }
        )
        (temporary / MANIFEST_NAME).write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    report_text = _build_report(
        inventory,
        overlaps,
        distributions,
        disagreements,
        replicates,
        confounding,
        associations,
        bins,
        interactions,
        split_profile,
        clinical,
    )
    (report / REPORT_NAME).write_text(report_text, encoding="utf-8")
    return validate_herg_current_analysis(output)


def _default_roots() -> tuple[Path, Path, Path, Path, Path]:
    processed = Path("research/data/platform/processed/herg_hierarchy")
    return (
        processed / "v1",
        processed / "v1_2_quality_tasks",
        processed / "v1_2_modality_qt",
        processed / "v1_3_current_analysis",
        Path("research/reports/platform/herg_paper/current_analysis"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    defaults = _default_roots()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hierarchy-root", type=Path, default=defaults[0])
    parser.add_argument("--quality-root", type=Path, default=defaults[1])
    parser.add_argument("--modality-root", type=Path, default=defaults[2])
    parser.add_argument("--output-root", type=Path, default=defaults[3])
    parser.add_argument("--report-root", type=Path, default=defaults[4])
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        validate_herg_current_analysis(args.output_root)
    else:
        build_herg_current_analysis(
            args.hierarchy_root,
            args.quality_root,
            args.modality_root,
            args.output_root,
            args.report_root,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
