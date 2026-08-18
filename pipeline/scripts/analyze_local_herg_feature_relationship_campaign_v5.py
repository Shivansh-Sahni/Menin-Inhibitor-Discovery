#!/usr/bin/env python3
"""Train-only, scaffold-aware analysis of V5 hERG feature relationships.

The analyzer deliberately separates marginal association from incremental evidence.
Only paired, outer-held-out perturbation or ablation effects may support an
"incremental" relationship.  Fold summaries alone are retained as descriptive
evidence and are never assigned inferential confidence intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA = "platform-herg-feature-relationship-analysis-v5/1.0"
REQUIRED_OOF = {
    "structure_id",
    "scaffold_group_id",
    "outer_fold",
    "observed_pic50",
    "predicted_pic50",
}
REQUIRED_EFFECTS = {
    "hypothesis_id",
    "evidence_type",
    "outer_fold",
    "structure_id",
    "scaffold_group_id",
    "baseline_abs_error",
    "perturbed_abs_error",
}
CHEMISTRY_CANDIDATES = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "RingCount",
)
SUBGROUP_CANDIDATES = (
    "measurement_modality",
    "automation_class",
    "assay_family",
    "source_family",
    "protocol_completeness_tier",
)
ALLOWED_INCREMENTAL_EVIDENCE = {
    "block_ablation",
    "grouped_ablation",
    "conditional_permutation",
    "group_permutation",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, value: dict[str, Any], self_hash_key: str | None = None) -> None:
    document = dict(value)
    if self_hash_key:
        document.pop(self_hash_key, None)
        document[self_hash_key] = hashlib.sha256(_canonical_bytes(document)).hexdigest()
    _atomic_bytes(path, _canonical_bytes(document))


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def bh_qvalues(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg q-values, applied once at hypothesis grain."""
    values = np.asarray(pvalues, dtype=float)
    result = np.full(values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return result
    ordered = finite[np.argsort(values[finite], kind="stable")]
    adjusted = values[ordered] * len(ordered) / np.arange(1, len(ordered) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[ordered] = np.minimum(adjusted, 1.0)
    return result


def _scaffold_sufficient(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["delta_abs_error"] = work["perturbed_abs_error"] - work["baseline_abs_error"]
    return (
        work.groupby("scaffold_group_id", observed=True, sort=True)
        .agg(delta_sum=("delta_abs_error", "sum"), rows=("delta_abs_error", "size"))
        .reset_index()
    )


def paired_scaffold_bootstrap(frame: pd.DataFrame, *, replicates: int, seed: int) -> dict[str, float]:
    """Bootstrap paired error differences by resampling whole scaffolds."""
    if replicates < 10_000:
        raise ValueError("paired scaffold bootstrap requires at least 10,000 replicates")
    sufficient = _scaffold_sufficient(frame)
    if len(sufficient) < 2:
        raise ValueError("at least two scaffolds are required")
    sums = sufficient["delta_sum"].to_numpy(float)
    counts = sufficient["rows"].to_numpy(float)
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    batch = 512
    for start in range(0, replicates, batch):
        width = min(batch, replicates - start)
        indices = rng.integers(0, len(sums), size=(width, len(sums)))
        samples.append(sums[indices].sum(axis=1) / counts[indices].sum(axis=1))
    boot = np.concatenate(samples)
    point = float(sums.sum() / counts.sum())
    # A scaffold-level paired sign-flip test avoids a row-level independence claim.
    # Use the same row-weighted estimand as the point estimate and bootstrap.
    observed = abs(float(sums.sum() / counts.sum()))
    extreme = 0
    completed = 0
    for start in range(0, replicates, batch):
        width = min(batch, replicates - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(width, len(sums)))
        permuted = np.abs(np.sum(signs * sums, axis=1) / counts.sum())
        extreme += int(np.count_nonzero(permuted >= observed))
        completed += width
    return {
        "effect_mae_delta": point,
        "ci95_lower": float(np.quantile(boot, 0.025)),
        "ci95_upper": float(np.quantile(boot, 0.975)),
        "probability_effect_positive": float(np.mean(boot > 0)),
        "paired_sign_flip_p": float((extreme + 1) / (completed + 1)),
        "scaffolds": int(len(sufficient)),
        "rows": int(len(frame)),
    }


def _fold_stability(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold, part in frame.groupby("outer_fold", observed=True, sort=True):
        delta = part["perturbed_abs_error"] - part["baseline_abs_error"]
        rows.append(
            {
                "outer_fold": int(fold),
                "effect_mae_delta": float(delta.mean()),
                "rows": int(len(part)),
                "scaffolds": int(part["scaffold_group_id"].nunique()),
            }
        )
    folds = pd.DataFrame(rows)
    positive = int((folds["effect_mae_delta"] > 0).sum()) if len(folds) else 0
    negative = int((folds["effect_mae_delta"] < 0).sum()) if len(folds) else 0
    stable = len(folds) == 5 and max(positive, negative) >= 4
    direction = "beneficial" if positive >= 4 else "harmful" if negative >= 4 else "unstable"
    return folds, {
        "folds_observed": int(len(folds)),
        "positive_folds": positive,
        "negative_folds": negative,
        "direction_stable_4_of_5": bool(stable),
        "stable_direction": direction,
    }


def _coarsened_match(frame: pd.DataFrame, chemistry: Sequence[str]) -> pd.DataFrame:
    """Retain chemistry cells represented by at least two subgroup levels."""
    work = frame.copy()
    keys: list[str] = []
    for column in chemistry:
        values = pd.to_numeric(work[column], errors="coerce")
        if values.notna().sum() < 20 or values.nunique(dropna=True) < 4:
            continue
        try:
            bins = pd.qcut(values, q=min(5, values.nunique()), duplicates="drop")
        except ValueError:
            continue
        key = f"__match_{column}"
        work[key] = bins.astype("string").fillna("missing")
        keys.append(key)
    if not keys:
        return work.iloc[0:0].copy()
    work["__chemistry_cell"] = work[keys].astype(str).agg("|".join, axis=1)
    supported = (
        work.groupby("__chemistry_cell", observed=True)["__subgroup"].nunique().loc[lambda x: x >= 2].index
    )
    return work.loc[work["__chemistry_cell"].isin(supported)].copy()


def _subgroup_analysis(
    effects: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    chemistry = [column for column in CHEMISTRY_CANDIDATES if column in metadata]
    dimensions = [column for column in SUBGROUP_CANDIDATES if column in metadata]
    if not chemistry or not dimensions:
        return pd.DataFrame()
    joined = effects.merge(metadata, on="structure_id", how="left", validate="many_to_one")
    output: list[dict[str, Any]] = []
    for hypothesis, hpart in joined.groupby("hypothesis_id", observed=True, sort=True):
        for dimension in dimensions:
            work = hpart.loc[hpart[dimension].notna()].copy()
            work["__subgroup"] = work[dimension].astype(str)
            matched = _coarsened_match(work, chemistry)
            if matched.empty:
                continue
            for level, part in matched.groupby("__subgroup", observed=True, sort=True):
                if len(part) < 100 or part["scaffold_group_id"].nunique() < 20:
                    continue
                result = paired_scaffold_bootstrap(
                    part,
                    replicates=replicates,
                    seed=seed
                    + int(hashlib.sha256(f"{hypothesis}|{dimension}|{level}".encode()).hexdigest()[:8], 16),
                )
                output.append(
                    {
                        "hypothesis_id": hypothesis,
                        "dimension": dimension,
                        "level": level,
                        "chemistry_matching": "coarsened_exact_quantile_cells",
                        "matched_chemistry_columns": json.dumps(chemistry),
                        **result,
                    }
                )
    return pd.DataFrame(output)


def _validate_train_only(campaign_root: Path) -> dict[str, Any]:
    validation_path = campaign_root / "validation.json"
    if not validation_path.exists():
        raise FileNotFoundError(f"missing campaign validation: {validation_path}")
    document = json.loads(validation_path.read_text())
    text = json.dumps(document).lower()
    forbidden_true = (
        '"validation_labels_opened": true',
        '"test_labels_opened": true',
        '"validation_labels_used": true',
        '"test_labels_used": true',
    )
    if any(item in text for item in forbidden_true):
        raise ValueError("campaign validation indicates validation/test labels were opened")
    if '"source_partition": "train"' not in text and document.get("source_partition") != "train":
        raise ValueError("campaign does not establish train-only source_partition")
    return document


def _read_required(path: Path, columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    return frame


def _canonical_train(repo_root: Path) -> pd.DataFrame:
    prepared = repo_root / "research/local_runs/herg_discovery_campaign_v1/prepared"
    cache_path = prepared / "exact_train_cache.parquet"
    split_path = prepared / "nested_scaffold_splits.parquet"
    cache = pd.read_parquet(
        cache_path,
        columns=["structure_id", "scaffold_group_id", "target_pic50"],
    )
    splits = pd.read_parquet(split_path)
    heldout = splits.loc[
        (splits["source_partition"] == "train") & (splits["outer_role"] == "heldout"),
        ["structure_id", "scaffold_group_id", "outer_fold"],
    ]
    if cache["structure_id"].duplicated().any() or heldout["structure_id"].duplicated().any():
        raise ValueError("canonical train cache or held-out split has duplicate identities")
    canonical = cache.merge(
        heldout,
        on=["structure_id", "scaffold_group_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(canonical) != 18_801:
        raise ValueError(f"canonical exact train surface must contain 18,801 rows; found {len(canonical)}")
    return canonical


def _assert_oof_and_effect_alignment(
    oof: pd.DataFrame,
    canonical: pd.DataFrame,
    effects: pd.DataFrame | None = None,
) -> None:
    comparison = oof.merge(
        canonical,
        on="structure_id",
        how="outer",
        suffixes=("_oof", "_canonical"),
        indicator=True,
        validate="one_to_one",
    )
    if not (comparison["_merge"] == "both").all():
        raise ValueError("nested OOF identities differ from canonical exact train surface")
    if not np.allclose(
        comparison["observed_pic50"].to_numpy(float),
        comparison["target_pic50"].to_numpy(float),
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("nested OOF targets differ from canonical exact train targets")
    if not (
        comparison["scaffold_group_id_oof"].astype(str)
        == comparison["scaffold_group_id_canonical"].astype(str)
    ).all():
        raise ValueError("nested OOF scaffolds differ from canonical assignments")
    if not (comparison["outer_fold_oof"].astype(int) == comparison["outer_fold_canonical"].astype(int)).all():
        raise ValueError("nested OOF folds differ from canonical held-out assignments")
    if effects is None:
        return
    aligned = effects.merge(
        oof[["structure_id", "scaffold_group_id", "outer_fold"]],
        on="structure_id",
        how="left",
        suffixes=("_effect", "_oof"),
        validate="many_to_one",
    )
    if aligned["scaffold_group_id_oof"].isna().any():
        raise ValueError("paired effects contain identities outside nested train OOF")
    if not (
        aligned["scaffold_group_id_effect"].astype(str) == aligned["scaffold_group_id_oof"].astype(str)
    ).all():
        raise ValueError("paired-effect scaffold assignments differ from nested OOF")
    if not (aligned["outer_fold_effect"].astype(int) == aligned["outer_fold_oof"].astype(int)).all():
        raise ValueError("paired-effect fold assignments differ from nested OOF")


def analyze(
    *,
    repo_root: Path,
    campaign_root: Path,
    output_root: Path,
    metadata_path: Path | None,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    if bootstrap_replicates < 10_000:
        raise ValueError("--bootstrap-replicates must be >= 10000")
    campaign_validation = _validate_train_only(campaign_root)
    oof_path = campaign_root / "nested_oof_predictions.parquet"
    reference_path = campaign_root / "relationship_reference_oof.parquet"
    effects_path = campaign_root / "paired_effects.parquet"
    oof = _read_required(oof_path, REQUIRED_OOF)
    reference = _read_required(reference_path, REQUIRED_OOF)
    effects = _read_required(effects_path, REQUIRED_EFFECTS)
    canonical = _canonical_train(repo_root)
    if oof["structure_id"].duplicated().any():
        raise ValueError("nested OOF contains duplicate structure_id")
    if set(pd.to_numeric(oof["outer_fold"], errors="raise").astype(int).unique()) != set(range(5)):
        raise ValueError("nested OOF must contain outer folds 0..4")
    if len(oof) != 18_801:
        raise ValueError(f"nested OOF must contain exactly 18,801 train structures; found {len(oof)}")
    if reference["structure_id"].duplicated().any() or len(reference) != 18_801:
        raise ValueError("relationship reference OOF must contain 18,801 unique train structures")
    if (oof.groupby("scaffold_group_id", observed=True)["outer_fold"].nunique() != 1).any():
        raise ValueError("each scaffold must belong to exactly one outer fold")
    _assert_oof_and_effect_alignment(oof, canonical)
    _assert_oof_and_effect_alignment(reference, canonical, effects)
    unknown_evidence = set(effects["evidence_type"].astype(str)) - ALLOWED_INCREMENTAL_EVIDENCE
    if unknown_evidence:
        raise ValueError(f"unsupported incremental evidence_type: {sorted(unknown_evidence)}")
    evidence_counts = effects.groupby("hypothesis_id", observed=True)["evidence_type"].nunique()
    if (evidence_counts != 1).any():
        raise ValueError("each hypothesis_id must map to exactly one evidence_type")
    numeric = ["baseline_abs_error", "perturbed_abs_error"]
    if not np.isfinite(effects[numeric].to_numpy(float)).all():
        raise ValueError("paired effects contain nonfinite errors")
    if (effects[numeric] < 0).any().any():
        raise ValueError("paired effects contain negative absolute errors")
    expected_baseline = reference.assign(
        __baseline=(reference["observed_pic50"] - reference["predicted_pic50"]).abs()
    ).set_index("structure_id")["__baseline"]
    supplied_baseline = effects["structure_id"].map(expected_baseline)
    if not np.allclose(
        effects["baseline_abs_error"].to_numpy(float),
        supplied_baseline.to_numpy(float),
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("paired baseline errors do not reproduce nested OOF predictions")
    output_root.mkdir(parents=True, exist_ok=True)

    hypotheses: list[dict[str, Any]] = []
    fold_frames: list[pd.DataFrame] = []
    collapsed_effects: list[pd.DataFrame] = []
    for index, (hypothesis, part) in enumerate(effects.groupby("hypothesis_id", observed=True, sort=True)):
        evidence = str(part["evidence_type"].iloc[0])
        # Repeated conditional permutations are averaged within held-out identity first.
        identity = ["structure_id", "scaffold_group_id", "outer_fold"]
        collapsed = part.groupby(identity, observed=True, as_index=False).agg(
            baseline_abs_error=("baseline_abs_error", "mean"),
            perturbed_abs_error=("perturbed_abs_error", "mean"),
        )
        collapsed.insert(0, "evidence_type", evidence)
        collapsed.insert(0, "hypothesis_id", hypothesis)
        if len(collapsed) != 18_801 or set(collapsed["outer_fold"].astype(int)) != set(range(5)):
            raise ValueError(f"hypothesis {hypothesis!r} lacks complete five-fold 18,801-identity coverage")
        collapsed_effects.append(collapsed)
        inference = paired_scaffold_bootstrap(
            collapsed,
            replicates=bootstrap_replicates,
            seed=seed + index,
        )
        folds, stability = _fold_stability(collapsed)
        folds.insert(0, "evidence_type", evidence)
        folds.insert(0, "hypothesis_id", hypothesis)
        fold_frames.append(folds)
        hypotheses.append(
            {
                "hypothesis_id": hypothesis,
                "evidence_type": evidence,
                "evidence_class": "incremental_held_out",
                **inference,
                **stability,
            }
        )
    hypothesis_frame = pd.DataFrame(hypotheses)
    hypothesis_frame["bh_q_value"] = bh_qvalues(hypothesis_frame["paired_sign_flip_p"])
    hypothesis_frame["discovery_grade"] = (
        hypothesis_frame["direction_stable_4_of_5"]
        & (hypothesis_frame["bh_q_value"] <= 0.05)
        & ((hypothesis_frame["ci95_lower"] > 0) | (hypothesis_frame["ci95_upper"] < 0))
    )
    fold_frame = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    effects_for_subgroups = (
        pd.concat(collapsed_effects, ignore_index=True) if collapsed_effects else pd.DataFrame()
    )

    metadata = pd.DataFrame()
    if metadata_path is not None:
        metadata = pd.read_parquet(metadata_path)
        if "structure_id" not in metadata or metadata["structure_id"].duplicated().any():
            raise ValueError("metadata must have unique structure_id")
        if not set(metadata["structure_id"]).issubset(set(oof["structure_id"])):
            metadata = metadata.loc[metadata["structure_id"].isin(set(oof["structure_id"]))]
    subgroup = (
        _subgroup_analysis(
            effects_for_subgroups,
            metadata,
            replicates=bootstrap_replicates,
            seed=seed + 100_000,
        )
        if not metadata.empty
        else pd.DataFrame()
    )

    # Fold-level campaign summaries are included but explicitly non-inferential.
    ablation_path = campaign_root / "grouped_ablation.parquet"
    descriptive_ablation = pd.read_parquet(ablation_path) if ablation_path.exists() else pd.DataFrame()
    permutation_path = campaign_root / "conditional_permutation.parquet"
    descriptive_permutation = (
        pd.read_parquet(permutation_path) if permutation_path.exists() else pd.DataFrame()
    )

    paths = {
        "relationship_hypotheses": output_root / "relationship_hypotheses.parquet",
        "fold_stability": output_root / "fold_stability.parquet",
        "subgroup_replication": output_root / "subgroup_replication.parquet",
        "descriptive_block_ablation": output_root / "descriptive_block_ablation.parquet",
        "descriptive_conditional_permutation": output_root / "descriptive_conditional_permutation.parquet",
    }
    for name, frame in (
        ("relationship_hypotheses", hypothesis_frame),
        ("fold_stability", fold_frame),
        ("subgroup_replication", subgroup),
        ("descriptive_block_ablation", descriptive_ablation),
        ("descriptive_conditional_permutation", descriptive_permutation),
    ):
        _atomic_parquet(paths[name], frame)

    discovered = int(hypothesis_frame["discovery_grade"].sum())
    report = {
        "schema_version": SCHEMA,
        "status": "passed",
        "scientific_scope": {
            "source_partition": "train",
            "validation_labels_opened": False,
            "test_labels_opened": False,
            "performance_scope": "nested_outer_oof_internal_only",
            "causal_claims_supported": False,
        },
        "counts": {
            "nested_oof_rows": int(len(oof)),
            "paired_effect_rows": int(len(effects)),
            "hypotheses": int(len(hypothesis_frame)),
            "discovery_grade_hypotheses": discovered,
            "subgroup_replications": int(len(subgroup)),
        },
        "methods": {
            "paired_bootstrap_unit": "scaffold_group_id",
            "bootstrap_replicates": bootstrap_replicates,
            "null_test": "paired_scaffold_sign_flip",
            "multiple_testing": "Benjamini-Hochberg once at hypothesis grain",
            "direction_stability": "same signed held-out effect in at least 4 of 5 outer folds",
            "subgroup_matching": "coarsened exact matching on available chemistry quantile cells",
            "marginal_vs_incremental": "only paired held-out perturbation/ablation is incremental",
        },
        "limitations": [
            "Internal nested train OOF evidence is not external or prospective validation.",
            "Associations and perturbation effects do not establish biological causality.",
            "Fold-level aggregate campaign files are descriptive only.",
            "Subgroup results are omitted where chemistry overlap or scaffold support is inadequate.",
        ],
    }
    _atomic_json(output_root / "relationship_report.json", report, "relationship_report_sha256")
    markdown = [
        "# hERG V5 feature-relationship analysis",
        "",
        f"Status: passed. Analyzed {len(hypothesis_frame)} prespecified hypotheses using {bootstrap_replicates:,} paired scaffold bootstrap replicates.",
        "",
        f"Discovery-grade internal relationships: {discovered}. A relationship requires BH q <= 0.05, a bootstrap interval excluding zero, and the same direction in at least four of five held-out outer folds.",
        "",
        "Incremental findings come only from paired held-out block ablation or chemistry-conditioned/group permutation. Fold-level summaries and feature importance are descriptive, not independent evidence.",
        "",
        "All labels came from the train partition. Repository validation and test labels remained sealed. These results are internal and non-causal.",
    ]
    _atomic_bytes(output_root / "analysis.md", ("\n".join(markdown) + "\n").encode())

    artifacts = []
    for path in sorted(output_root.iterdir()):
        if path.is_file() and path.name not in {"manifest.json", "validation.json"}:
            artifacts.append({"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    validation = {
        "schema_version": f"{SCHEMA}-validation",
        "status": "passed",
        "source_partition": "train",
        "validation_labels_opened": False,
        "test_labels_opened": False,
        "nested_oof_exact_coverage": len(oof) == 18_801,
        "five_outer_folds": set(oof["outer_fold"].astype(int)) == set(range(5)),
        "bootstrap_replicates": bootstrap_replicates,
        "campaign_validation_bound": hashlib.sha256(_canonical_bytes(campaign_validation)).hexdigest(),
    }
    _atomic_json(output_root / "validation.json", validation, "validation_sha256")
    manifest: dict[str, Any] = {
        "schema_version": f"{SCHEMA}-manifest",
        "status": "passed",
        "inputs": [
            {"path": str(oof_path), "sha256": _sha256(oof_path)},
            {"path": str(reference_path), "sha256": _sha256(reference_path)},
            {"path": str(effects_path), "sha256": _sha256(effects_path)},
        ],
        "artifacts": artifacts,
    }
    if metadata_path is not None:
        manifest["inputs"].append({"path": str(metadata_path), "sha256": _sha256(metadata_path)})
    _atomic_json(output_root / "manifest.json", manifest, "manifest_sha256")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--workers", type=int, default=6, help="Reserved for compatible campaign launchers")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = analyze(
        repo_root=args.repo_root.resolve(),
        campaign_root=args.campaign_root.resolve(),
        output_root=args.output_root.resolve(),
        metadata_path=args.metadata_path.resolve() if args.metadata_path else None,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
