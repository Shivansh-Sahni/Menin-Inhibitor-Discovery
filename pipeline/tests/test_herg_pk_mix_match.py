from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_herg_binary_mix_match import (  # noqa: E402
    _class_balanced_weights,
    _decisive_interval_frame,
)
from run_herg_pk_mix_match import (  # noqa: E402
    _feature_contract,
    _regression_metrics,
    _source_balanced_weights,
)


def test_source_balancing_gives_internal_and_public_equal_total_mass() -> None:
    frame = pd.DataFrame(
        {
            "source_group": ["internal"] * 3 + ["public_regression_train"] * 12,
        }
    )
    weights = _source_balanced_weights(frame, internal_sources={"internal"})

    assert weights.iloc[:3].sum() == pytest.approx(3.0)
    assert weights.iloc[3:].sum() == pytest.approx(3.0)


def test_liability_metrics_exclude_intermediate_ic50_class() -> None:
    frame = pd.DataFrame(
        {
            "observed_pic50": [5.5, 5.2, 4.8, 4.4],
            "predicted_pic50": [5.4, 5.1, 5.8, 4.3],
            "scaffold": ["A", "B", "C", "D"],
            "max_internal_like_train_tanimoto": [0.9, 0.8, 0.7, 0.6],
        }
    )

    metrics = _regression_metrics(frame)

    # pIC50 4.8 lies between the 10 and 30 uM decision bounds and is not
    # invented as either a blocker or a nonblocker.
    assert metrics["decisive_n"] == 3
    assert metrics["n_blockers"] == 2
    assert metrics["n_nonblockers"] == 1


def test_feature_contract_does_not_relabel_proxies_as_fundamental() -> None:
    contract = _feature_contract().set_index("feature_layer")

    assert "not fundamental" in contract.loc["compact_proxies", "prohibited_claim"]
    assert "must not be imputed" in contract.loc["state_path_flux", "prohibited_claim"]
    assert np.isfinite(len(contract))


def test_binary_source_and_class_cells_receive_equal_mass() -> None:
    frame = pd.DataFrame(
        {
            "source_group": ["internal"] * 5
            + ["angelo_same_series_extension"] * 3
            + ["public_classification_train"] * 12,
            "target_class": [0, 1, 1, 1, 1, 0, 1, 1] + [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    )
    weights = _class_balanced_weights(frame, source_balance=True)
    buckets = np.where(
        frame["source_group"].isin({"internal", "angelo_same_series_extension"}),
        "internal_like",
        "public",
    )
    cell_mass = (
        pd.DataFrame(
            {
                "cell": [
                    f"{bucket}:{label}"
                    for bucket, label in zip(
                        buckets,
                        frame["target_class"],
                        strict=True,
                    )
                ],
                "weight": weights,
            }
        )
        .groupby("cell")["weight"]
        .sum()
    )

    assert cell_mass.nunique() == 1
    assert weights.mean() == pytest.approx(1.0)


def test_interval_decision_does_not_force_intermediate_values() -> None:
    frame = pd.DataFrame(
        {
            "record_id": ["a", "b", "c"],
            "compound_id": ["a", "b", "c"],
            "standardized_smiles": ["CC", "CCC", "CCCC"],
            "scaffold": ["", "", ""],
            "source_group": ["internal"] * 3,
            "pic50_lower": [5.1, -np.inf, 4.7],
            "pic50_upper": [5.1, 4.5, 4.7],
            "source_conflict": [False, False, False],
        }
    )

    decisive, audit = _decisive_interval_frame(
        frame,
        dataset_role="test",
    )

    assert decisive["target_class"].tolist() == [1, 0]
    assert (
        audit.loc[audit["record_id"].eq("c"), "binary_label_basis"].item()
        == "excluded_intermediate_or_ambiguous_interval"
    )
