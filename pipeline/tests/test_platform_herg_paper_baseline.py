from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from menin_discovery.platform_herg_paper_baseline import (
    HergPaperBaselineError,
    run_herg_paper_baseline,
    validate_herg_paper_baseline,
)


def _split(path: Path) -> Path:
    smiles = ["CC", "CCC", "CCCC", "CCO", "CCN", "CCCl", "CCBr", "CCF", "CO", "CN", "C=O", "C#N"]
    partitions = ["train"] * 6 + ["validation"] * 3 + ["test"] * 3
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    pq.write_table(
        pa.table(
            {
                "structure_id": [f"STR-{i}" for i in range(12)],
                "standardized_smiles": smiles,
                "herg_blocker_label": pa.array(labels, type=pa.int8()),
                "split": partitions,
                "scaffold_group_id": [f"GROUP-{i}" for i in range(12)],
            }
        ),
        path,
    )
    return path


def test_runs_locked_baseline_and_validates(tmp_path: Path) -> None:
    split = _split(tmp_path / "split.parquet")
    output = tmp_path / "out"
    manifest = run_herg_paper_baseline(split_path=split, output_root=output)
    assert manifest["scientific_contract"]["superiority_established"] is False
    assert manifest["counts"]["test"]["rows"] == 3
    assert pq.read_table(output / "baseline_metrics.parquet").num_rows == 4
    assert validate_herg_paper_baseline(output) == manifest


def test_scaffold_leakage_fails_closed(tmp_path: Path) -> None:
    split = _split(tmp_path / "split.parquet")
    table = pq.read_table(split).to_pydict()
    table["scaffold_group_id"][6] = table["scaffold_group_id"][0]
    pq.write_table(pa.table(table), split)
    with pytest.raises(HergPaperBaselineError, match="leaked"):
        run_herg_paper_baseline(split_path=split, output_root=tmp_path / "out")


def test_artifact_tampering_is_detected(tmp_path: Path) -> None:
    output = tmp_path / "out"
    run_herg_paper_baseline(split_path=_split(tmp_path / "split.parquet"), output_root=output)
    with (output / "morgan_sgd_logistic.joblib").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(HergPaperBaselineError, match="digest mismatch"):
        validate_herg_paper_baseline(output)
