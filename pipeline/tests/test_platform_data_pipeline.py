from __future__ import annotations

import math

import pandas as pd
from menin_discovery.platform_data_pipeline import (
    GAS_CONSTANT_KCAL_MOL_K,
    PUBLIC_ACCESS_CLASS,
    REFERENCE_TEMPERATURE_K,
    _task_registry,
    _task_view,
    binding_free_energy_view,
)


def _observations() -> pd.DataFrame:
    rows = []
    for index, (domain, endpoint, assay, unit, kind) in enumerate(
        [
            ("binding", "Kd", "A1", "nM", "experimental_summary"),
            ("binding", "Kd", "A1", "nM", "experimental_summary"),
            ("herg", "IC50", "A2", "nM", "experimental_summary"),
            ("herg", "Kd", "A2", "nM", "experimental_summary"),
            ("qt", "QTcF", "A3", "ms", "experimental_summary"),
            ("pk_adme", "Cmax", "A4", "ng/mL", "experimental_summary"),
            ("binding", "standard_binding_free_energy", "A1", "kcal/mol", "derived"),
        ]
    ):
        value = -7.0 if kind == "derived" else float(index + 1)
        rows.append(
            {
                "observation_id": f"OBS-{index}",
                "source_id": "SRC-X",
                "snapshot_id": "SNP-X",
                "source_record_id": f"R-{index}",
                "molecule_id": f"M-{index}",
                "protein_id": "P-1",
                "assay_id": assay,
                "evidence_domain": domain,
                "endpoint": endpoint,
                "endpoint_family": "x",
                "relation": "=",
                "canonical_value": value,
                "canonical_unit": unit,
                "lower_bound": value,
                "upper_bound": value,
                "observation_kind": kind,
                "document_year": 2020,
                "quality_grade": "protocol_sufficient",
                "access_class": PUBLIC_ACCESS_CLASS,
                "inclusion_status": "included",
                "potential_leakage": kind == "derived",
                "value_provenance": "lineage",
            }
        )
    return pd.DataFrame(rows)


def _assays() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "assay_id": ["A1", "A2", "A3", "A4"],
            "assay_family": ["binding", "herg_functional", "cardiac_electrophysiology_qt_apd", "pk_adme"],
        }
    )


def test_default_tasks_share_contract_ids_and_fail_closed_on_domains() -> None:
    tasks = _task_view(_observations(), _assays())
    assert set(tasks["endpoint"]) == {"Kd", "IC50"}
    assert set(tasks.loc[tasks["endpoint"] == "Kd", "task_id"]).__len__() == 1
    assert len(tasks.loc[tasks["endpoint"] == "Kd"]) == 2
    assert tasks["default_task_eligible"].map(type).eq(bool).all()
    assert tasks["default_task_eligible"].all()
    assert tasks["required_modalities"].eq("small_molecule_structure;protein_sequence").all()
    assert not tasks["sensitivity_task_eligible"].any()
    registry = _task_registry(tasks)
    assert int(registry["row_count"].sum()) == len(tasks)
    assert registry["task_id"].is_unique


def test_derived_task_is_separate_sensitivity_policy() -> None:
    tasks = _task_view(_observations(), _assays(), derived_sensitivity=True)
    assert len(tasks) == 1
    assert tasks.iloc[0]["observation_kind"] == "derived"
    assert bool(tasks.iloc[0]["default_task_eligible"]) is False
    assert bool(tasks.iloc[0]["sensitivity_task_eligible"]) is True


def test_herg_continuous_and_exact_binary_threshold_contracts_are_distinct() -> None:
    template = _observations().loc[_observations()["evidence_domain"].eq("herg")].iloc[0].to_dict()
    rows = []
    for observation_id, value, relation in (
        ("HERG-LOW", 5_000.0, "="),
        ("HERG-MID", 20_000.0, "="),
        ("HERG-HIGH", 40_000.0, "="),
        ("HERG-CENSORED", 5_000.0, "<"),
    ):
        row = dict(template)
        row.update(
            {
                "observation_id": observation_id,
                "source_record_id": observation_id,
                "canonical_value": value,
                "lower_bound": value if relation in {"=", ">", ">="} else math.nan,
                "upper_bound": value if relation in {"=", "<", "<="} else math.nan,
                "relation": relation,
            }
        )
        rows.append(row)
    tasks = _task_view(pd.DataFrame(rows), _assays())
    binary = tasks[tasks["label_kind"].eq("categorical")]
    continuous = tasks[tasks["label_kind"].str.startswith("continuous_")]
    assert set(binary["observation_id"]) == {"HERG-LOW", "HERG-HIGH"}
    assert dict(zip(binary["observation_id"], binary["label_text"], strict=True)) == {
        "HERG-HIGH": "nonblocker",
        "HERG-LOW": "blocker",
    }
    assert binary["task_id"].nunique() == 1
    assert set(continuous["observation_id"]) == {
        "HERG-LOW",
        "HERG-MID",
        "HERG-HIGH",
        "HERG-CENSORED",
    }
    assert continuous["task_id"].nunique() == 2
    assert not tasks[["task_id", "observation_id"]].duplicated().any()


def test_binding_free_energy_is_exact_kd_only_and_roundtrips() -> None:
    observations = _observations().iloc[:1].copy()
    observations["inclusion_status"] = "included"
    observations["canonical_value"] = 10.0
    observations["canonical_unit"] = "nM"
    observations["relation"] = "="
    assays = pd.DataFrame({"assay_id": ["A1"], "temperature_c": [math.nan]})
    proteins = pd.DataFrame({"protein_id": ["P-1"], "entity_type": ["single_protein"]})
    derivations, derived = binding_free_energy_view(observations, assays, proteins)
    assert len(derivations) == len(derived) == 1
    expected = GAS_CONSTANT_KCAL_MOL_K * REFERENCE_TEMPERATURE_K * math.log(10e-9)
    assert math.isclose(float(derivations.iloc[0]["delta_g_kcal_mol"]), expected, rel_tol=1e-12)
    assert float(derivations.iloc[0]["roundtrip_relative_error"]) < 1e-12
    assert derivations.iloc[0]["temperature_source"] == "reference_temperature_approximation"
