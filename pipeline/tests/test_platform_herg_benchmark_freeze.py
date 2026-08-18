from __future__ import annotations

import pandas as pd
from menin_discovery.platform_herg_benchmark_freeze import (
    INPUT_COLUMNS,
    MEMBERSHIP_COLUMNS,
    _materialized_challenges,
    _read_memberships,
    _registry_rows,
)


def _memberships() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "membership_id": ["a", "b", "c"],
            "task_id": [
                "Q0_WEAK_FIXED_DOSE_BINARY",
                "Q2_FUNCTIONAL_ASSAY_AWARE",
                "C1_QT_CONTEXT_EVALUATION",
            ],
            "quality_level": ["Q0", "Q2", "C1"],
            "source_artifact": ["q0", "q2", "c1"],
            "record_id": ["r1", "r2", "r3"],
            "observation_id": ["o1", "o2", None],
            "structure_id": ["s1", "s2", "s3"],
            "target_scope": ["wild_type", "wild_type_or_unspecified", "clinical_context_not_target_variant"],
            "source_family": [
                "pubchem_aid720551",
                "chembl_herg_specialized_view",
                "ClinicalTrials.gov_posted_results",
            ],
            "measurement_technology": [
                "automated_fluxor_qhts",
                "manual_patch_clamp",
                "clinical_ECG_QT_QTc_context",
            ],
            "model_split": ["test", "test", "test"],
            "scaffold_group_id": ["g1", "g2", "g3"],
            "eligible": [True, True, True],
            "direct_herg_label": [True, True, False],
            "use_as_training_label": [True, True, False],
            "clinical_context_only": [False, False, True],
        }
    )


def test_challenge_masks_separate_clinical_context() -> None:
    masks = _materialized_challenges(_memberships())
    assert masks["Q0_OFFICIAL_SCAFFOLD"].tolist() == [True, False, False]
    assert masks["Q2_MANUAL_PATCH_STRESS"].tolist() == [False, True, False]
    assert masks["QT_TRANSLATION_CONTEXT"].tolist() == [False, False, True]


def test_registry_is_explicitly_not_gold_or_superiority_ready() -> None:
    source = _memberships()
    parts = []
    for challenge_id, mask in _materialized_challenges(source).items():
        part = source.loc[mask, [c for c in MEMBERSHIP_COLUMNS if c != "challenge_id"]].copy()
        part.insert(0, "challenge_id", challenge_id)
        parts.append(part)
    registry = _registry_rows(pd.concat(parts, ignore_index=True))
    assert registry
    assert all(not row["labels_embedded"] for row in registry)
    assert all(not row["adjudicated_gold_standard"] for row in registry)
    assert all(not row["ready_for_superiority_claim"] for row in registry)


def test_reader_projects_only_label_blind_routing_columns(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_read_parquet(path, *, columns):
        captured["path"] = path
        captured["columns"] = columns
        return _memberships()

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    result = _read_memberships(tmp_path / "membership.parquet")
    assert len(result) == 3
    assert captured["columns"] == INPUT_COLUMNS
    assert not {
        "target_relation_pic50",
        "target_pic50_point",
        "target_pic50_lower_bound",
        "target_pic50_upper_bound",
        "target_class",
    }.intersection(captured["columns"])
