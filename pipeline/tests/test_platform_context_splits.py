from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest
from menin_discovery import platform_context_splits as context
from menin_discovery.platform_features import stable_json_digest


def _row(index: int, *, year: int | None = 2000) -> dict[str, object]:
    return {
        "record_id": f"record-{index:03d}",
        "molecule_id": f"molecule-{index:03d}",
        "protein_id": "protein-1",
        "target_id": "target-1",
        "source_id": "source-1",
        "assay_id": f"assay-{index:03d}",
        "document_id": f"document-{index:03d}",
        "document_year": year,
    }


def _write_empty_acceptance(root: Path, config: context.ContextSplitConfig) -> None:
    root.mkdir()
    payload = asdict(config)
    context._atomic_json(
        root / "acceptance.json",
        {
            "schema_version": context.SCHEMA_VERSION,
            "configuration": payload,
            "configuration_sha256": stable_json_digest(payload),
            "component_inventory": {},
            "component_inventory_sha256": stable_json_digest({}),
            "source_binding": {
                "implementation_sha256": context.sha256_file(Path(context.__file__).resolve())
            },
            "tasks": [],
            "candidate_only": True,
            "official_split_replaced": False,
            "substantive_training_ready": False,
            "substantive_training_authorized": False,
            "substantive_training_started": False,
        },
    )


@pytest.mark.parametrize("column", ["label_value", "LABEL_TEXT", "label_upper_bound"])
def test_label_columns_are_rejected_before_parquet_access(column: str) -> None:
    with pytest.raises(ValueError, match="label columns forbidden"):
        context._guard_columns(("observation_id", column), "poison.parquet")
    assert context._guard_columns(context.CONTEXT_COLUMNS, "safe.parquet") == context.CONTEXT_COLUMNS


def test_temporal_rule_is_strict_deterministic_and_excludes_missing_years() -> None:
    rows = [_row(index, year=year) for index, year in enumerate([1998, 1998, 1999, 2000, 2001, 2002, None])]
    first, metadata = context._route_rows(rows, "strict_temporal", context.ContextSplitConfig())
    second, _ = context._route_rows(list(reversed(rows)), "strict_temporal", context.ContextSplitConfig())
    assert sorted(first, key=lambda row: row["record_id"]) == sorted(second, key=lambda row: row["record_id"])
    unknown = [row for row in first if row["document_year"] is None]
    assert [row["split"] for row in unknown] == ["excluded_unknown"]
    audit = context._audit_rows(first, metadata["grouping_field"])
    assert audit["strict_chronology_passed"] is True
    assert audit["group_exclusion_passed"] is True
    assert all(audit["row_counts"][partition] > 0 for partition in context.PARTITIONS)


def test_temporal_boundaries_fail_closed_with_insufficient_years() -> None:
    with pytest.raises(ValueError, match="at least three distinct years"):
        context._temporal_boundaries({2000: 4, 2001: 3})


def test_hash_group_candidates_are_deterministic_and_group_exclusive() -> None:
    rows = [_row(index, year=2000 + index % 5) for index in range(100)]
    config = context.ContextSplitConfig()
    first, metadata = context._route_rows(rows, "assay_group_holdout", config)
    second, _ = context._route_rows(rows, "assay_group_holdout", config)
    assert first == second
    audit = context._audit_rows(first, metadata["grouping_field"])
    assert audit["group_exclusion_passed"] is True
    assert sum(audit["row_counts"].values()) == 100


def test_false_chronology_is_not_claimed() -> None:
    rows = []
    for index, (partition, year) in enumerate([("train", 2000), ("validation", 1999), ("test", 2001)]):
        row = _row(index, year=year)
        row.update(
            {
                "split": partition,
                "group_id": f"year:{year}",
                "strategy": "strict_temporal",
            }
        )
        rows.append(row)
    audit = context._audit_rows(rows, "document_year")
    assert audit["strict_chronology_passed"] is False


def test_configuration_drift_and_unbound_files_are_rejected(tmp_path: Path) -> None:
    frozen = context.ContextSplitConfig()
    output = tmp_path / "context"
    _write_empty_acceptance(output, frozen)
    assert context.verify_context_split_candidates(output, expected_config=frozen)["status"] == "verified"
    with pytest.raises(ValueError, match="configuration drift"):
        context.verify_context_split_candidates(
            output,
            expected_config=replace(frozen, seed=frozen.seed + 1),
        )
    (output / "unbound.txt").write_text("unbound\n", encoding="utf-8")
    with pytest.raises(ValueError, match="closed inventory membership mismatch"):
        context.verify_context_split_candidates(output)


def test_verifier_rejects_symlinks_even_when_they_resolve_to_regular_files(tmp_path: Path) -> None:
    output = tmp_path / "context"
    _write_empty_acceptance(output, context.ContextSplitConfig())
    target = tmp_path / "outside.txt"
    target.write_text("outside\n", encoding="utf-8")
    (output / "alias.txt").symlink_to(target)
    with pytest.raises(ValueError, match="symlink forbidden"):
        context.verify_context_split_candidates(output)


def test_unsafe_paths_and_output_schema_are_label_free() -> None:
    for value in ("../split.parquet", "/tmp/split.parquet", ""):
        with pytest.raises(ValueError, match="unsafe"):
            context._safe_relative(value, "component")
    assert not any(name.casefold().startswith("label") for name in context.OUTPUT_SCHEMA.names)


def test_fixed_seed_assignment_changes_only_when_seed_changes() -> None:
    value = "assay-stable"
    first = context._hash_partition(value, context.ContextSplitConfig(seed=1))
    assert first == context._hash_partition(value, context.ContextSplitConfig(seed=1))
    observed = {
        context._hash_partition(value, context.ContextSplitConfig(seed=seed)) for seed in range(1, 100)
    }
    assert observed == set(context.PARTITIONS)
