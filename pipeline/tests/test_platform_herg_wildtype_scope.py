from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from menin_discovery.platform_herg_wildtype_scope import (
    HergWildtypeScopeError,
    build_herg_wildtype_scope,
    validate_herg_wildtype_scope,
)


def _ledger(path: Path, variants: list[str]) -> Path:
    count = len(variants)
    table = pa.table(
        {
            "observation_id": [f"OBS-{index}" for index in range(count)],
            "source_family": ["fixture"] * count,
            "structure_id": [f"STR-{index}" for index in range(count)],
            "target_variant": variants,
            "assay_id": [f"ASSAY-{index}" for index in range(count)],
            "native_aux_json": [json.dumps({"description": variant}) for variant in variants],
            "pic50_value": [5.0] * count,
            "derived_binary_label": pa.array([1] * count, type=pa.int8()),
        }
    )
    pq.write_table(table, path)
    return path


def test_admits_confirmed_and_unspecified_but_excludes_mutants(tmp_path: Path) -> None:
    ledger = _ledger(
        tmp_path / "ledger.parquet", ["wild_type", "wild_type_or_unspecified", "mutant_or_variant"]
    )
    output = tmp_path / "out"
    manifest = build_herg_wildtype_scope(observation_ledger_path=ledger, output_root=output)
    admitted = pq.read_table(output / "wildtype_observation_index.parquet").to_pylist()
    excluded = pq.read_table(output / "explicit_mutant_exclusions.parquet").to_pylist()

    assert {row["wildtype_scope"] for row in admitted} == {"confirmed_wild_type", "wild_type_or_unspecified"}
    assert [row["target_variant_original"] for row in excluded] == ["mutant_or_variant"]
    assert manifest["counts"]["admitted_observations"] == 2
    assert manifest["counts"]["excluded_explicit_mutant_observations"] == 1
    assert validate_herg_wildtype_scope(output) == manifest


def test_unknown_variant_fails_closed_and_does_not_replace_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    ledger = _ledger(tmp_path / "ledger.parquet", ["unknown"])
    with pytest.raises(HergWildtypeScopeError, match="Unrecognized"):
        build_herg_wildtype_scope(observation_ledger_path=ledger, output_root=output)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_validation_detects_artifact_tampering(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path / "ledger.parquet", ["wild_type", "mutant_or_variant"])
    output = tmp_path / "out"
    build_herg_wildtype_scope(observation_ledger_path=ledger, output_root=output)
    with (output / "explicit_mutant_exclusions.parquet").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(HergWildtypeScopeError, match="digest mismatch"):
        validate_herg_wildtype_scope(output)
