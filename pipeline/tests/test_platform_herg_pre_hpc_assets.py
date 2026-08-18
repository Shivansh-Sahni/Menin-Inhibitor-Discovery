from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from menin_discovery.platform_herg_pre_hpc_assets import (
    _conflict_queue,
    _gold_candidates,
    _protocol_queue,
)


def _observation(observation_id: str, value: float, **updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "observation_id": observation_id,
        "structure_id": "S1",
        "standardized_smiles": "CCN",
        "standard_inchi_key": "KEY",
        "wild_type_evidence_scope": "wild_type_or_unspecified",
        "source_family": "source",
        "source_record_id": f"R-{observation_id}",
        "assay_id": "A1",
        "assay_family": "functional",
        "measurement_modality": "patch_clamp_electrophysiology",
        "automation_class": "manual",
        "native_endpoint": "IC50",
        "native_relation": "=",
        "native_value": 10.0,
        "native_unit": "nM",
        "endpoint_class": "potency_ic50",
        "endpoint_standardization_status": "exact_standardized",
        "potency_relation_pic50": "=",
        "potency_pic50_point": value,
        "potency_pic50_lower_bound": value,
        "potency_pic50_upper_bound": value,
        "potency_censoring": "exact",
        "model_split": "test",
        "scaffold_group_id": "G1",
        "native_aux_json": "{}",
    }
    row.update(updates)
    return row


def test_gold_candidates_are_explicitly_nonadjudicated_and_preserve_censoring() -> None:
    exact = _observation("O1", 7.0)
    censored = _observation(
        "O2",
        6.0,
        endpoint_standardization_status="censored_standardized",
        potency_relation_pic50="<",
        potency_pic50_point=None,
        potency_pic50_lower_bound=None,
        potency_pic50_upper_bound=6.0,
        potency_censoring="pic50_upper_bounded",
    )
    protocol = {
        ("source", "A1", "functional"): {
            "protocol_completeness_score": 5,
            "unresolved_fields_json": '["temperature"]',
        }
    }
    rows = _gold_candidates([exact, censored], {"O1", "O2"}, protocol)
    assert len(rows) == 2
    assert all(
        row["candidate_status"] == "evaluation_candidate_not_adjudicated_gold_standard" for row in rows
    )
    censored_out = next(row for row in rows if row["observation_id"] == "O2")
    assert censored_out["potency_pic50_point"] is None
    assert censored_out["potency_pic50_upper_bound"] == 6.0


def test_conflict_queue_prioritizes_large_exact_ranges() -> None:
    protocols = {("source", "A1", "functional"): {"protocol_completeness_score": 4}}
    queue = _conflict_queue(
        [_observation("O1", 5.0), _observation("O2", 7.2), _observation("O3", 7.0)],
        protocols,
    )
    assert len(queue) == 1
    assert queue[0]["review_priority"] == "critical"
    assert queue[0]["pic50_range"] == 2.2
    assert queue[0]["exact_replicate_count"] == 3


def test_conflict_queue_excludes_conversion_noise() -> None:
    protocols = {("source", "A1", "functional"): {"protocol_completeness_score": 4}}
    queue = _conflict_queue(
        [_observation("O1", 7.0), _observation("O2", 7.0 + 5e-10)],
        protocols,
    )
    assert queue == []


def test_protocol_queue_only_contains_unresolved_assays_and_production_validates(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.parquet"
    pq.write_table(
        pa.table(
            {
                "assay_catalog_id": ["A", "B"],
                "source_family": ["s", "s"],
                "assay_id": ["1", "2"],
                "assay_family": ["functional", "functional"],
                "observation_count": [1000, 2],
                "unresolved_fields_json": ['["voltage"]', "[]"],
                "protocol_completeness_score": pa.array([5, 6], type=pa.int8()),
                "host_systems_json": ["[]", "[]"],
                "named_platforms_json": ["[]", "[]"],
                "recording_configurations_json": ["[]", "[]"],
                "source_automation_classes_json": ["{}", "{}"],
                "raw_protocol_text_json": ["[]", "[]"],
                "source_contract_evidence_json": ["[]", "[]"],
            }
        ),
        protocol,
    )
    queue = _protocol_queue(protocol)
    assert [row["assay_catalog_id"] for row in queue] == ["A"]
