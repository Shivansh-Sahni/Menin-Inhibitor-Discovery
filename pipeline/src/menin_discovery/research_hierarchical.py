"""Compound-balanced hierarchical Gaussian models for repeated rat PK summaries.

This module implements the second PK model-ladder rung as a deliberately
modest variance-component model.  It does not claim that repeated summary PK
parameters identify absorption, metabolism, or distribution mechanisms.
Instead, it separates three statistical sources of variation that the current
data can at least partially inform:

* between-scaffold variation;
* between-compound variation within scaffold; and
* within-compound study variation when repeated studies exist.

Fixed effects are fitted to one equally weighted mean per compound.  This
prevents a compound with several reported studies from dominating structure
coefficients.  In leave-scaffold-out evaluation, both scaffold and compound
random effects are marginalized to their zero-mean population distributions;
no held-out group estimate is reused or inferred from its outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .research_modeling import regression_metrics


def _group_splits(groups: np.ndarray, requested: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(groups)
    n_splits = min(int(requested), len(unique))
    if n_splits < 2:
        raise ValueError("At least two distinct scaffolds are required for hierarchical evaluation")
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))


def _validate_compound_scaffolds(frame: pd.DataFrame, compound_column: str, scaffold_column: str) -> None:
    counts = frame.groupby(compound_column, dropna=False)[scaffold_column].nunique(dropna=False)
    inconsistent = counts[counts != 1]
    if not inconsistent.empty:
        raise ValueError(
            "Each compound must map to exactly one scaffold; inconsistent compounds: "
            f"{inconsistent.index.astype(str).tolist()[:5]}"
        )


def _compound_summary(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    compound_column: str,
    scaffold_column: str,
) -> pd.DataFrame:
    """Collapse study rows without allowing replicate count to weight compounds."""

    _validate_compound_scaffolds(frame, compound_column, scaffold_column)
    rows: list[dict[str, Any]] = []
    for compound_id, group in frame.groupby(compound_column, sort=True, dropna=False):
        values = pd.to_numeric(group[target_column], errors="coerce").to_numpy(dtype=float)
        values = np.log10(values[np.isfinite(values) & (values > 0)])
        if not len(values):
            continue
        row: dict[str, Any] = {
            "compound_id": str(compound_id),
            "scaffold": str(group[scaffold_column].iloc[0]),
            "target_log10": float(np.mean(values)),
            "n_evidence_rows": int(len(values)),
            # Population variance is unchanged if the complete set of study
            # rows is duplicated, unlike a degrees-of-freedom correction.
            "within_study_variance_log10_squared": float(np.mean((values - np.mean(values)) ** 2)),
        }
        for column in feature_columns:
            numeric = pd.to_numeric(group[column], errors="coerce")
            row[column] = float(numeric.mean()) if numeric.notna().any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class CompoundBalancedHierarchicalGaussian:
    """Ridge fixed effects plus nested scaffold/compound random intercepts.

    Variance components use transparent compound-balanced method-of-moments
    estimates after the fixed fit.  With the current sparse replication they
    should be interpreted as descriptive variance partitions, not precise
    biological constants.
    """

    alpha: float = 10.0
    feature_columns: list[str] = field(default_factory=list)
    compound_column: str = "compound_id"
    scaffold_column: str = "scaffold"
    target_column: str = "target_value"
    imputer_: SimpleImputer | None = None
    scaler_: StandardScaler | None = None
    fixed_model_: Ridge | None = None
    fixed_covariance_: np.ndarray | None = None
    scaffold_effects_: dict[str, float] = field(default_factory=dict)
    compound_effects_: dict[str, float] = field(default_factory=dict)
    scaffold_variance_: float = float("nan")
    compound_variance_: float = float("nan")
    study_variance_: float = float("nan")
    variance_floor_: float = 1e-8
    metadata_: dict[str, Any] = field(default_factory=dict)

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        feature_columns: list[str],
        target_column: str = "target_value",
        compound_column: str = "compound_id",
        scaffold_column: str = "scaffold",
    ) -> CompoundBalancedHierarchicalGaussian:
        required = {target_column, compound_column, scaffold_column, *feature_columns}
        if missing := sorted(required - set(frame.columns)):
            raise ValueError(f"Hierarchical PK model is missing columns: {missing}")
        eligible = frame.dropna(subset=[target_column, compound_column, scaffold_column]).copy()
        target = pd.to_numeric(eligible[target_column], errors="coerce")
        eligible = eligible[np.isfinite(target) & (target > 0)].reset_index(drop=True)
        if eligible[compound_column].nunique() < 3:
            raise ValueError("At least three compounds are required for hierarchical PK fitting")

        self.feature_columns = list(feature_columns)
        self.compound_column = compound_column
        self.scaffold_column = scaffold_column
        self.target_column = target_column
        summary = _compound_summary(
            eligible,
            feature_columns=self.feature_columns,
            target_column=target_column,
            compound_column=compound_column,
            scaffold_column=scaffold_column,
        )
        X_raw = summary[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        self.imputer_ = SimpleImputer()
        self.scaler_ = StandardScaler()
        X_imputed = self.imputer_.fit_transform(X_raw)
        X = self.scaler_.fit_transform(X_imputed)
        y = summary["target_log10"].to_numpy(dtype=float)
        self.fixed_model_ = Ridge(alpha=self.alpha)
        self.fixed_model_.fit(X, y)
        fixed_prediction = np.asarray(self.fixed_model_.predict(X), dtype=float)
        residual = y - fixed_prediction
        summary["fixed_residual"] = residual

        repeated = summary[summary["n_evidence_rows"] >= 2]
        repeated_compounds = int(len(repeated))
        if repeated_compounds >= 3:
            study_variance = float(repeated["within_study_variance_log10_squared"].mean())
            study_status = "weakly_identified_from_repeated_study_spread"
        else:
            # The current summaries cannot distinguish assay/study residual
            # variance from compound variance without repeated observations.
            study_variance = 0.0
            study_status = "not_separately_identified_absorbed_into_compound_component"

        scaffold_groups = [group for _, group in summary.groupby("scaffold", sort=True)]
        within_numerator = float(
            sum(
                np.sum((group["fixed_residual"] - group["fixed_residual"].mean()) ** 2)
                for group in scaffold_groups
            )
        )
        within_df = int(sum(max(len(group) - 1, 0) for group in scaffold_groups))
        within_scaffold_variance = within_numerator / within_df if within_df else float(np.var(residual))
        compound_variance = max(float(within_scaffold_variance - study_variance), 0.0)

        scaffold_means = summary.groupby("scaffold", sort=True)["fixed_residual"].mean()
        scaffold_sizes = summary.groupby("scaffold", sort=True).size().astype(float)
        between_scaffold_variance = (
            float(np.var(scaffold_means.to_numpy(dtype=float), ddof=1)) if len(scaffold_means) >= 2 else 0.0
        )
        mean_sampling_variance = float(
            np.mean((compound_variance + study_variance) / scaffold_sizes.to_numpy(dtype=float))
        )
        scaffold_variance = max(between_scaffold_variance - mean_sampling_variance, 0.0)

        self.study_variance_ = max(study_variance, 0.0)
        self.compound_variance_ = max(compound_variance, 0.0)
        self.scaffold_variance_ = max(scaffold_variance, 0.0)

        residual_variance = max(
            self.scaffold_variance_ + self.compound_variance_ + self.study_variance_,
            self.variance_floor_,
        )
        design = np.column_stack([np.ones(len(X)), X])
        penalty = np.diag([0.0, *([self.alpha] * X.shape[1])])
        information = design.T @ design + penalty
        self.fixed_covariance_ = residual_variance * np.linalg.pinv(information, hermitian=True)

        self.scaffold_effects_ = {}
        self.compound_effects_ = {}
        for scaffold, group in summary.groupby("scaffold", sort=True):
            residual_mean = float(group["fixed_residual"].mean())
            conditional_noise = (self.compound_variance_ + self.study_variance_) / max(len(group), 1)
            denominator = self.scaffold_variance_ + conditional_noise
            shrinkage = self.scaffold_variance_ / denominator if denominator > 0 else 0.0
            self.scaffold_effects_[str(scaffold)] = float(shrinkage * residual_mean)
        for row in summary.itertuples(index=False):
            scaffold_effect = self.scaffold_effects_.get(str(row.scaffold), 0.0)
            remainder = float(row.fixed_residual - scaffold_effect)
            denominator = self.compound_variance_ + self.study_variance_
            shrinkage = self.compound_variance_ / denominator if denominator > 0 else 0.0
            self.compound_effects_[str(row.compound_id)] = float(shrinkage * remainder)

        component_identifiability = "weak: variance partition is descriptive and repeated summaries do not identify PK process parameters"
        if within_df == 0:
            compound_status = "not_separable_from_scaffold_with_current_nesting"
        elif compound_variance == 0.0:
            compound_status = "boundary_estimate_zero"
        else:
            compound_status = "descriptive_method_of_moments"
        scaffold_status = (
            "boundary_estimate_zero" if scaffold_variance == 0.0 else "descriptive_method_of_moments"
        )
        self.metadata_ = {
            "model": "compound_balanced_hierarchical_gaussian",
            "track": "discovery-evaluation-only",
            "target_scale": "log10",
            "n_training_evidence_rows": int(len(eligible)),
            "n_training_compounds": int(len(summary)),
            "n_training_scaffolds": int(summary["scaffold"].nunique()),
            "n_compounds_with_repeated_studies": repeated_compounds,
            "compound_balance": "one equally weighted mean per compound for fixed effects",
            "random_effect_structure": "scaffold intercept + compound-within-scaffold intercept",
            "unseen_group_policy": "marginalize scaffold and compound effects to zero mean",
            "variance_estimator": "compound-balanced method of moments after ridge fixed effects",
            "overall_identifiability": component_identifiability,
            "study_variance_identifiability": study_status,
            "compound_variance_identifiability": compound_status,
            "scaffold_variance_identifiability": scaffold_status,
            "fixed_effect_alpha": float(self.alpha),
        }
        return self

    def _require_fitted(self) -> None:
        if self.imputer_ is None or self.scaler_ is None or self.fixed_model_ is None:
            raise RuntimeError("Hierarchical PK model is not fitted")

    def predict_distribution(
        self,
        frame: pd.DataFrame,
        *,
        marginalize_random_effects: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict log10 mean and standard deviation.

        Group-held-out evaluation must use ``marginalize_random_effects=True``.
        Conditional estimates are exposed only for diagnostic fitted-data
        summaries and must not be used for unseen-scaffold validation.
        """

        self._require_fitted()
        assert self.imputer_ is not None
        assert self.scaler_ is not None
        assert self.fixed_model_ is not None
        assert self.fixed_covariance_ is not None
        X_raw = frame[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        X = self.scaler_.transform(self.imputer_.transform(X_raw))
        mean = np.asarray(self.fixed_model_.predict(X), dtype=float)
        design = np.column_stack([np.ones(len(X)), X])
        fixed_variance = np.einsum("ij,jk,ik->i", design, self.fixed_covariance_, design)
        if marginalize_random_effects:
            random_variance = self.scaffold_variance_ + self.compound_variance_ + self.study_variance_
        else:
            scaffold = frame[self.scaffold_column].astype(str)
            compound = frame[self.compound_column].astype(str)
            mean = mean + scaffold.map(self.scaffold_effects_).fillna(0.0).to_numpy(dtype=float)
            mean = mean + compound.map(self.compound_effects_).fillna(0.0).to_numpy(dtype=float)
            random_variance = self.study_variance_
        variance = np.maximum(fixed_variance + random_variance, self.variance_floor_)
        return mean, np.sqrt(variance)

    def variance_components(self) -> pd.DataFrame:
        self._require_fitted()
        statuses = {
            "scaffold": self.metadata_["scaffold_variance_identifiability"],
            "compound_within_scaffold": self.metadata_["compound_variance_identifiability"],
            "within_compound_study": self.metadata_["study_variance_identifiability"],
        }
        values = {
            "scaffold": self.scaffold_variance_,
            "compound_within_scaffold": self.compound_variance_,
            "within_compound_study": self.study_variance_,
        }
        return pd.DataFrame(
            [
                {
                    "variance_component": component,
                    "variance_log10_squared": float(value),
                    "stddev_log10": float(np.sqrt(max(value, 0.0))),
                    "identifiability_status": statuses[component],
                    "estimate_method": self.metadata_["variance_estimator"],
                }
                for component, value in values.items()
            ]
        )


def _compound_balanced_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.groupby(["compound_id", "group", "fold"], as_index=False).agg(
        observed_log10=("observed_log10", "mean"),
        predicted_log10=("predicted_log10", "mean"),
        interval_lower_log10=("interval_lower_log10", "mean"),
        interval_upper_log10=("interval_upper_log10", "mean"),
        predictive_sigma_log10=("predictive_sigma_log10", "mean"),
    )


def _metric_row(
    frame: pd.DataFrame,
    *,
    model: str,
    evaluation_unit: str,
    primary: bool,
) -> dict[str, Any]:
    return {
        "model": model,
        "evaluation_unit": evaluation_unit,
        "primary_evaluation": primary,
        "n_unique_compounds": int(frame["compound_id"].nunique()),
        "n_unique_scaffolds": int(frame["group"].nunique()),
        **regression_metrics(
            frame["observed_log10"].to_numpy(dtype=float),
            frame["predicted_log10"].to_numpy(dtype=float),
            interval_lower=frame["interval_lower_log10"].to_numpy(dtype=float),
            interval_upper=frame["interval_upper_log10"].to_numpy(dtype=float),
            predictive_sigma=frame["predictive_sigma_log10"].to_numpy(dtype=float),
        ),
    }


def _cluster_bootstrap_metrics(
    compounds: pd.DataFrame,
    *,
    replicates: int,
    random_state: int,
) -> pd.DataFrame:
    groups = np.asarray(sorted(compounds["group"].astype(str).unique()))
    if len(groups) < 2 or replicates <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    for replicate in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        pieces = [compounds[compounds["group"].astype(str) == group] for group in sampled]
        sample = pd.concat(pieces, ignore_index=True)
        values = regression_metrics(
            sample["observed_log10"].to_numpy(dtype=float),
            sample["predicted_log10"].to_numpy(dtype=float),
            interval_lower=sample["interval_lower_log10"].to_numpy(dtype=float),
            interval_upper=sample["interval_upper_log10"].to_numpy(dtype=float),
            predictive_sigma=sample["predictive_sigma_log10"].to_numpy(dtype=float),
        )
        rows.append({"bootstrap_replicate": replicate, "bootstrap_unit": "heldout_scaffold", **values})
    return pd.DataFrame(rows)


def grouped_hierarchical_pk_benchmark(
    data: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str = "target_value",
    group_column: str = "scaffold",
    compound_column: str = "compound_id",
    folds: int = 5,
    interval_level: float = 0.90,
    alpha: float = 10.0,
    bootstrap_replicates: int = 250,
    random_state: int = 20260721,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leave-scaffold-out evaluation of compound-balanced partial pooling."""

    required = {target_column, group_column, compound_column, *feature_columns}
    if missing := sorted(required - set(data.columns)):
        raise ValueError(f"Hierarchical PK benchmark is missing columns: {missing}")
    frame = data.dropna(subset=[target_column, group_column, compound_column]).copy().reset_index(drop=True)
    target = pd.to_numeric(frame[target_column], errors="coerce")
    frame = frame[np.isfinite(target) & (target > 0)].reset_index(drop=True)
    _validate_compound_scaffolds(frame, compound_column, group_column)
    groups = frame[group_column].astype(str).to_numpy()
    splits = _group_splits(groups, folds)
    z_value = float(norm.ppf(0.5 + interval_level / 2.0))
    prediction_rows: list[dict[str, Any]] = []
    component_frames: list[pd.DataFrame] = []

    for fold_index, (train, test) in enumerate(splits):
        train_groups = sorted(frame.iloc[train][group_column].astype(str).unique())
        heldout_groups = sorted(frame.iloc[test][group_column].astype(str).unique())
        overlap = sorted(set(train_groups) & set(heldout_groups))
        if overlap:
            raise RuntimeError(f"Scaffold leakage detected before hierarchical fit: {overlap}")
        model = CompoundBalancedHierarchicalGaussian(alpha=alpha).fit(
            frame.iloc[train],
            feature_columns=feature_columns,
            target_column=target_column,
            compound_column=compound_column,
            scaffold_column=group_column,
        )
        predicted, sigma = model.predict_distribution(
            frame.iloc[test],
            marginalize_random_effects=True,
        )
        test_counts = (
            frame.iloc[test].groupby(compound_column)[compound_column].transform("size").to_numpy(dtype=float)
        )
        train_groups_text = ";".join(train_groups)
        heldout_groups_text = ";".join(heldout_groups)
        for position, index in enumerate(test):
            prediction_rows.append(
                {
                    "compound_id": str(frame.loc[index, compound_column]),
                    "sample_id": str(frame.loc[index, "sample_id"]) if "sample_id" in frame else str(index),
                    "model": "compound_balanced_hierarchical_gaussian",
                    "fold": fold_index,
                    "group": str(frame.loc[index, group_column]),
                    "observed_log10": float(np.log10(frame.loc[index, target_column])),
                    "predicted_log10": float(predicted[position]),
                    "interval_lower_log10": float(predicted[position] - z_value * sigma[position]),
                    "interval_upper_log10": float(predicted[position] + z_value * sigma[position]),
                    "predictive_sigma_log10": float(sigma[position]),
                    "compound_weight": float(1.0 / test_counts[position]),
                    "train_groups": train_groups_text,
                    "heldout_groups": heldout_groups_text,
                    "heldout_group_seen_in_training": False,
                    "random_effect_prediction": "marginalized_zero_unseen_scaffold_and_compound",
                    "track": "discovery-evaluation-only",
                }
            )
        components = model.variance_components()
        for key, value in model.metadata_.items():
            components[key] = value
        components["fold"] = fold_index
        components["train_groups"] = train_groups_text
        components["heldout_groups"] = heldout_groups_text
        component_frames.append(components)

    predictions = pd.DataFrame(prediction_rows)
    components = pd.concat(component_frames, ignore_index=True)
    compounds = _compound_balanced_predictions(predictions)
    evidence_metrics = _metric_row(
        predictions,
        model="compound_balanced_hierarchical_gaussian",
        evaluation_unit="evidence_row_secondary",
        primary=False,
    )
    compound_metrics = _metric_row(
        compounds,
        model="compound_balanced_hierarchical_gaussian",
        evaluation_unit="compound",
        primary=True,
    )
    bootstrap = _cluster_bootstrap_metrics(
        compounds,
        replicates=bootstrap_replicates,
        random_state=random_state,
    )
    if not bootstrap.empty:
        for metric in ("log_mae", "log_rmse", "median_fold_error", "prediction_interval_coverage"):
            compound_metrics[f"bootstrap_{metric}_lower_95"] = float(bootstrap[metric].quantile(0.025))
            compound_metrics[f"bootstrap_{metric}_upper_95"] = float(bootstrap[metric].quantile(0.975))
    common = {
        "track": "discovery-evaluation-only",
        "compound_balance": "one equally weighted outcome mean per compound",
        "random_effect_prediction": "marginalized for unseen heldout scaffold and compound",
        "uncertainty_method": "fixed-effect covariance plus marginal scaffold/compound/study variance",
        "bootstrap_unit": "heldout_scaffold",
        "interval_level": interval_level,
    }
    metrics = pd.DataFrame([{**evidence_metrics, **common}, {**compound_metrics, **common}])
    return metrics, predictions, components, bootstrap


def compound_balanced_conventional_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Re-score conventional OOF predictions at the same compound-level unit."""

    required = {
        "compound_id",
        "group",
        "fold",
        "model",
        "observed_log10",
        "predicted_log10",
        "interval_lower_log10",
        "interval_upper_log10",
        "predictive_sigma_log10",
    }
    if missing := sorted(required - set(predictions.columns)):
        raise ValueError(f"Conventional predictions are missing comparison columns: {missing}")
    rows: list[dict[str, Any]] = []
    for model_name, group in predictions.groupby("model", sort=True):
        compounds = _compound_balanced_predictions(group)
        rows.append(
            {
                **_metric_row(
                    compounds,
                    model=str(model_name),
                    evaluation_unit="compound",
                    primary=True,
                ),
                "track": "conventional-baseline-evaluation",
                "compound_balance": "one equally weighted outcome mean per compound",
            }
        )
    return pd.DataFrame(rows).sort_values(["log_mae", "model"]).reset_index(drop=True)
