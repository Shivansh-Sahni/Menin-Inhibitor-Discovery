from __future__ import annotations

from menin_discovery.platform_herg_hpc_preflight import _matches_constraint, _nr_paths


def test_nr_paths_find_only_explicitly_unresolved_contract_fields() -> None:
    payload = {
        "ready": "2026.03.3",
        "missing": "NR",
        "nested": [{"shape": "model_dimension_NR"}, {"shape": [2, 3]}],
    }
    assert _nr_paths(payload) == ["missing", "nested[0].shape"]


def test_nr_paths_ignore_narrative_text_without_nr_token() -> None:
    assert _nr_paths({"status": "not reported", "gate": "all_versions_non_NR", "value": None}) == []


def test_version_constraints_normalize_zero_padding_and_wildcards() -> None:
    assert _matches_constraint("2026.3.3", "2026.03.3")
    assert _matches_constraint("2.7.4", "2.7.*")
    assert not _matches_constraint("2.13.0", "2.7.*")
