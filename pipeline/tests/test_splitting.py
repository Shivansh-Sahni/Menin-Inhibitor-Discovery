import numpy as np
import pandas as pd
from menin_discovery.features import scaffold_key
from menin_discovery.splitting import make_cv_folds, make_split


def _classification_frame(n_rows=60):
    return pd.DataFrame(
        {
            "smiles": [f"{'C' * (index + 1)}N" for index in range(n_rows)],
            "label": [index % 2 for index in range(n_rows)],
        }
    )


def test_random_split_is_deterministic_and_stratified():
    data = _classification_frame()
    first = make_split(
        data,
        strategy="random",
        target_column="label",
        task_type="classification",
        random_state=7,
    )
    second = make_split(
        data,
        strategy="random",
        target_column="label",
        task_type="classification",
        random_state=7,
    )
    assert np.array_equal(first.train_indices, second.train_indices)
    assert np.array_equal(first.test_indices, second.test_indices)
    assert first.metadata["split_sha256"] == second.metadata["split_sha256"]
    assert not set(first.train_indices) & set(first.test_indices)
    assert set(data.iloc[first.test_indices]["label"]) == {0, 1}


def test_random_and_cv_splits_keep_repeated_structure_ids_together():
    structures = [f"STR-{index:03d}" for index in range(20)]
    data = pd.DataFrame(
        {
            "structure_id": np.repeat(structures, 2),
            "smiles": np.repeat([f"C{'N' * (index + 1)}" for index in range(20)], 2),
            "label": np.tile([0, 1], 20),
        }
    )
    split = make_split(
        data,
        strategy="random",
        target_column="label",
        task_type="classification",
        random_state=5,
    )
    train_structures = set(data.iloc[split.train_indices]["structure_id"])
    test_structures = set(data.iloc[split.test_indices]["structure_id"])
    assert not train_structures & test_structures
    assert split.metadata["structure_identity_overlap"] == 0

    folds, metadata = make_cv_folds(
        data,
        strategy="random",
        n_splits=3,
        random_state=5,
        target_column="label",
        task_type="classification",
    )
    assert metadata["maximum_structure_overlap"] == 0
    for train, test in folds:
        assert not set(data.iloc[train]["structure_id"]) & set(data.iloc[test]["structure_id"])


def test_scaffold_split_has_no_scaffold_overlap():
    data = pd.DataFrame(
        {
            "smiles": [
                "c1ccccc1O",
                "c1ccccc1N",
                "c1ccccc1Cl",
                "c1ccncc1O",
                "c1ccncc1N",
                "C1CCCCC1O",
                "C1CCCCC1N",
                "C1CCNCC1",
                "C1CCOCC1",
                "CCO",
                "CCN",
                "CCCl",
            ],
            "target": np.linspace(5, 8, 12),
        }
    )
    split = make_split(data, strategy="scaffold", target_column="target", task_type="regression")
    train_keys = {scaffold_key(value)[0] for value in data.iloc[split.train_indices]["smiles"]}
    test_keys = {scaffold_key(value)[0] for value in data.iloc[split.test_indices]["smiles"]}
    assert not train_keys & test_keys
    assert split.metadata["structure_group_overlap"] == 0


def test_temporal_split_holds_out_the_latest_years():
    data = pd.DataFrame(
        {
            "smiles": [f"C{'N' * (index + 1)}" for index in range(20)],
            "target": np.linspace(4, 9, 20),
            "document_year": np.repeat(np.arange(2015, 2020), 4),
        }
    )
    split = make_split(data, strategy="temporal", target_column="target", task_type="regression")
    train_years = data.iloc[split.train_indices]["document_year"]
    test_years = data.iloc[split.test_indices]["document_year"]
    assert test_years.min() > train_years.max()
    assert split.metadata["cutoff_year"] == test_years.min()


def test_temporal_split_uses_first_structure_year_and_prevents_reentry():
    data = pd.DataFrame(
        {
            "structure_id": ["old", "old", "middle", "middle", "new", "new"],
            "smiles": ["CCO", "CCO", "CCN", "CCN", "CCC", "CCC"],
            "target": [5.0, 5.1, 6.0, 6.1, 7.0, 7.1],
            "document_year": [2015, 2022, 2019, 2020, 2023, 2024],
        }
    )
    split = make_split(
        data,
        strategy="temporal",
        test_size=0.33,
        target_column="target",
        task_type="regression",
    )
    train_structures = set(data.iloc[split.train_indices]["structure_id"])
    test_structures = set(data.iloc[split.test_indices]["structure_id"])
    assert not train_structures & test_structures
    assert "old" in train_structures
    assert split.metadata["time_basis"] == "earliest known year per structure"


def test_temporal_split_falls_back_when_fewer_than_half_the_rows_are_dated():
    data = pd.DataFrame(
        {
            "smiles": [f"C{'N' * (index + 1)}" for index in range(10)],
            "target": np.linspace(4, 9, 10),
            "document_year": [2019] * 4 + [np.nan] * 6,
        }
    )
    split = make_split(
        data,
        strategy="temporal",
        test_size=0.2,
        target_column="target",
        task_type="regression",
    )
    assert split.metadata["requested_strategy"] == "temporal"
    assert split.metadata["strategy"] == "random"
    assert split.metadata["fallback_reason"].startswith(
        "Insufficient dated structures for temporal holdout: 4/10"
    )


def test_temporal_split_falls_back_when_dated_holdout_is_below_minimum_size():
    data = pd.DataFrame(
        {
            "smiles": [f"C{'N' * (index + 1)}" for index in range(10)],
            "target": np.linspace(4, 9, 10),
            # Exactly half are dated, but the latest period contains only one
            # row, below half of the requested two-row holdout.
            "document_year": [2019] * 4 + [2020] + [np.nan] * 5,
        }
    )
    split = make_split(
        data,
        strategy="temporal",
        test_size=0.2,
        target_column="target",
        task_type="regression",
    )
    assert split.metadata["strategy"] == "random"
    assert split.metadata["fallback_reason"] == "No viable temporal cutoff was found"


def test_chemical_cv_folds_are_deterministic():
    data = _classification_frame(48)
    first, first_meta = make_cv_folds(
        data,
        strategy="chemical",
        n_splits=3,
        random_state=11,
        target_column="label",
        task_type="classification",
    )
    second, second_meta = make_cv_folds(
        data,
        strategy="chemical",
        n_splits=3,
        random_state=11,
        target_column="label",
        task_type="classification",
    )
    assert first_meta["cv_sha256"] == second_meta["cv_sha256"]
    assert len(first) == len(second) >= 2
