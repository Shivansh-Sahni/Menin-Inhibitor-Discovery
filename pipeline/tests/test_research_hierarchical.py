from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from menin_discovery.research_hierarchical import (
    CompoundBalancedHierarchicalGaussian,
    grouped_hierarchical_pk_benchmark,
)


def _synthetic_repeated_pk() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    compound_index = 0
    for scaffold_index in range(6):
        scaffold_effect = (scaffold_index - 2.5) * 0.035
        for member in range(3):
            feature = -1.2 + 0.14 * compound_index
            compound_effect = (member - 1) * 0.025
            for study_index, study_effect in enumerate((-0.03, 0.03)):
                log_target = 2.0 + 0.24 * feature + scaffold_effect + compound_effect + study_effect
                rows.append(
                    {
                        "compound_id": f"C{compound_index}",
                        "sample_id": f"C{compound_index}-R{study_index}",
                        "scaffold": f"S{scaffold_index}",
                        "feature": feature,
                        "target_value": 10.0**log_target,
                    }
                )
            compound_index += 1
    return pd.DataFrame(rows)


def test_grouped_hierarchy_marginalizes_heldout_random_effects_without_leakage() -> None:
    frame = _synthetic_repeated_pk()
    metrics, predictions, components, bootstrap = grouped_hierarchical_pk_benchmark(
        frame,
        feature_columns=["feature"],
        folds=3,
        bootstrap_replicates=20,
        random_state=17,
    )

    assert not predictions["heldout_group_seen_in_training"].any()
    for row in predictions.itertuples(index=False):
        assert row.group not in set(row.train_groups.split(";"))
    assert set(predictions["random_effect_prediction"]) == {"marginalized_zero_unseen_scaffold_and_compound"}
    assert predictions.groupby("compound_id")["compound_weight"].sum().eq(1.0).all()
    assert (predictions["interval_lower_log10"] < predictions["predicted_log10"]).all()
    assert (predictions["predicted_log10"] < predictions["interval_upper_log10"]).all()
    assert np.isfinite(predictions["predictive_sigma_log10"]).all()

    primary = metrics[metrics["primary_evaluation"]].iloc[0]
    assert primary["evaluation_unit"] == "compound"
    assert np.isfinite(primary["log_mae"])
    assert np.isfinite(primary["bootstrap_log_mae_lower_95"])
    assert len(bootstrap) == 20
    assert set(components["variance_component"]) == {
        "scaffold",
        "compound_within_scaffold",
        "within_compound_study",
    }
    assert components["variance_log10_squared"].ge(0.0).all()
    assert components["overall_identifiability"].str.startswith("weak:").all()


def test_fixed_fit_is_invariant_to_complete_replication_of_one_compound() -> None:
    frame = _synthetic_repeated_pk()
    duplicated = pd.concat(
        [frame, frame[frame["compound_id"] == "C0"]],
        ignore_index=True,
    )
    first = CompoundBalancedHierarchicalGaussian(alpha=2.0).fit(
        frame,
        feature_columns=["feature"],
    )
    second = CompoundBalancedHierarchicalGaussian(alpha=2.0).fit(
        duplicated,
        feature_columns=["feature"],
    )
    unseen = pd.DataFrame({"compound_id": ["NEW"], "scaffold": ["UNSEEN"], "feature": [0.2]})
    first_mean, first_sigma = first.predict_distribution(unseen, marginalize_random_effects=True)
    second_mean, second_sigma = second.predict_distribution(unseen, marginalize_random_effects=True)

    assert first_mean == pytest.approx(second_mean, abs=1e-12)
    assert first_sigma == pytest.approx(second_sigma, abs=1e-12)
    assert first.metadata_["compound_balance"] == "one equally weighted mean per compound for fixed effects"
    assert first.metadata_["unseen_group_policy"].startswith("marginalize")


def test_hierarchy_rejects_compound_assigned_to_multiple_scaffolds() -> None:
    frame = _synthetic_repeated_pk()
    frame.loc[frame.index[0], "scaffold"] = "CONFLICT"
    with pytest.raises(ValueError, match="exactly one scaffold"):
        grouped_hierarchical_pk_benchmark(
            frame,
            feature_columns=["feature"],
            folds=3,
            bootstrap_replicates=0,
        )
