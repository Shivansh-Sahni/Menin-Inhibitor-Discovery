from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from menin_discovery.platform_herg_split import (
    ARTIFACT_NAME,
    HergSplitError,
    build_herg_scaffold_split,
    main,
    validate_herg_scaffold_split,
)


def _write_consensus(path: Path, *, duplicate: bool = False) -> Path:
    # One hundred acyclic structures exercise the exact-structure proxy without
    # collapsing all empty Murcko scaffolds. The aromatic rows share one true
    # Murcko scaffold and therefore must remain in one partition.
    smiles = ["C" * length for length in range(1, 101)]
    smiles.extend(["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "Oc1ccccc1"])
    structure_ids = [f"HSTR-{index:04d}" for index in range(len(smiles))]
    inchi_keys = [f"FIXTURE-INCHI-{index:04d}" for index in range(len(smiles))]
    if duplicate:
        structure_ids[-1] = structure_ids[0]
    table = pa.table(
        {
            "structure_id": pa.array(structure_ids, type=pa.large_string()),
            "standardized_smiles": pa.array(smiles, type=pa.large_string()),
            "standard_inchi_key": pa.array(inchi_keys, type=pa.large_string()),
            "herg_blocker_label": pa.array(
                [1 if index % 26 == 0 else 0 for index in range(len(smiles))], type=pa.int8()
            ),
        }
    )
    pq.write_table(table, path)
    return path


def test_split_is_deterministic_group_isolated_and_imbalance_preserving(tmp_path: Path) -> None:
    source = _write_consensus(tmp_path / "consensus.parquet")
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = build_herg_scaffold_split(consensus_path=source, output_root=first)
    second_manifest = build_herg_scaffold_split(consensus_path=source, output_root=second)

    assert first_manifest == second_manifest
    assert (first / ARTIFACT_NAME).read_bytes() == (second / ARTIFACT_NAME).read_bytes()
    output = pq.read_table(first / ARTIFACT_NAME).to_pandas()
    assert list(output.columns) == [
        "structure_id",
        "standardized_smiles",
        "standard_inchi_key",
        "herg_blocker_label",
        "split",
        "scaffold_group_id",
    ]
    assert set(output["split"]) == {"train", "validation", "test"}
    assert output.groupby("scaffold_group_id")["split"].nunique().max() == 1
    assert output.groupby("standard_inchi_key")["split"].nunique().max() == 1

    phenyl = output[output["standardized_smiles"].str.contains("c1ccccc1", regex=False)]
    assert phenyl["scaffold_group_id"].nunique() == 1
    assert phenyl["split"].nunique() == 1
    acyclic = output[output["standardized_smiles"].isin(["C", "CC", "CCC"])]
    assert acyclic["scaffold_group_id"].nunique() == 3

    qc = first_manifest["qc"]
    assert qc["class_counts"] == {"0": 100, "1": 4}
    assert qc["scaffold_group_overlap_count"] == 0
    assert qc["exact_structure_overlap_count"] == 0
    assert qc["acyclic_exact_proxy_rows"] == 100
    assert first_manifest["split_policy"]["label_stratification"] is False
    assert first_manifest["split_policy"]["seed_search"] is False
    assert main(["--output-root", str(first), "--validate-only"]) == 0


def test_duplicate_consensus_structure_fails_closed_without_output(tmp_path: Path) -> None:
    source = _write_consensus(tmp_path / "duplicate.parquet", duplicate=True)
    output = tmp_path / "failed"
    with pytest.raises(HergSplitError, match="Duplicate structure_id"):
        build_herg_scaffold_split(consensus_path=source, output_root=output)
    assert not output.exists()


def test_manifest_and_artifact_tampering_are_detected(tmp_path: Path) -> None:
    source = _write_consensus(tmp_path / "consensus.parquet")
    output = tmp_path / "output"
    build_herg_scaffold_split(consensus_path=source, output_root=output)

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["qc"]["rows"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(HergSplitError, match="manifest digest mismatch"):
        validate_herg_scaffold_split(output)

    # Restore the manifest, then mutate the Parquet bytes independently.
    output_two = tmp_path / "output_two"
    build_herg_scaffold_split(consensus_path=source, output_root=output_two)
    artifact_path = output_two / ARTIFACT_NAME
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tamper")
    with pytest.raises(HergSplitError, match="artifact hash mismatch"):
        validate_herg_scaffold_split(output_two)
