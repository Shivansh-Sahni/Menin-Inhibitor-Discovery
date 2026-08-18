from __future__ import annotations

import argparse
import json
import stat
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_local_herg_chemprop_unit import (  # noqa: E402
    ChempropUnitError,
    _bounded_spec,
    _prepare_inputs,
)


def _prepared(root: Path, model_split: str = "train") -> Path:
    prepared = root / "prepared"
    prepared.mkdir(parents=True)
    rows = []
    smiles = ["CC", "CCC", "CCCC", "CCO", "CCN", "c1ccccc1", "CCF", "CCCl", "CCBr"]
    for index, smiles_value in enumerate(smiles):
        rows.append(
            {
                "structure_id": f"S{index}",
                "standardized_smiles": smiles_value,
                "target_pic50": 4.5 + index / 10,
                "scaffold_group_id": f"G{index}",
                "model_split": model_split,
                "inner_fold": index % 3,
            }
        )
    pd.DataFrame(rows).to_parquet(prepared / "prepared_exact_train.parquet", index=False)
    return prepared


def _fake_chemprop(repo: Path) -> None:
    executable = repo / ".venv" / "bin" / "chemprop"
    executable.parent.mkdir(parents=True)
    executable.write_text(
        f"""#!{sys.executable}
import pathlib, sys
import pandas as pd
args = sys.argv[1:]
data = pathlib.Path(args[args.index('--data-path') + 1])
out = pathlib.Path(args[args.index('--output-dir') + 1]) / 'model_0'
out.mkdir(parents=True, exist_ok=True)
frame = pd.read_csv(data)
held = frame.loc[frame['__chemprop_split'].eq('test')]
pd.DataFrame({{'smiles': held['smiles'], 'target_pic50': held['target_pic50'] + 0.1}}).to_csv(out / 'test_predictions.csv', index=False)
(out / 'best.pt').write_bytes(b'model')
""",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)


def test_input_contract_rejects_repository_test_rows(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path, model_split="test")
    with pytest.raises(ChempropUnitError, match="nontraining"):
        _prepare_inputs(prepared, {"outer_fold": 0})


def test_json_assignment_mapping_is_accepted(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    data = pd.read_parquet(prepared / "prepared_exact_train.parquet").drop(columns="inner_fold")
    data.to_parquet(prepared / "prepared_exact_train.parquet", index=False)
    mapping = {f"S{index}": index % 3 for index in range(9)}
    (prepared / "fold_assignments.json").write_text(json.dumps(mapping), encoding="utf-8")
    frame, _, assignment, metadata = _prepare_inputs(prepared, {"outer_fold": 0})
    assert assignment == prepared / "fold_assignments.json"
    assert set(frame["inner_role"]) == {"train", "validation", "holdout"}
    assert metadata["assignment"]["method"] == "explicit_scaffold_folds"


def test_bounded_configuration_rejects_runaway_epochs() -> None:
    with pytest.raises(ChempropUnitError, match="epochs"):
        _bounded_spec({"epochs": 1000}, "unit")


def test_fake_chemprop_passes_writes_oof_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    prepared = _prepared(repo)
    _fake_chemprop(repo)
    monkeypatch.setattr(
        "run_local_herg_chemprop_unit.importlib.metadata.version",
        lambda name: "2.3.0" if name == "chemprop" else "0",
    )
    from run_local_herg_chemprop_unit import _run  # noqa: PLC0415

    args = argparse.Namespace(
        repo_root=str(repo),
        prepared_root=str(prepared),
        output_root=str(repo / "output"),
        unit_id="chemprop-fold-0",
        unit_spec=json.dumps(
            {
                "outer_fold": 0,
                "validation_fold": 1,
                "epochs": 2,
                "patience": 1,
                "maximum_minutes": 1,
            }
        ),
        workers=2,
    )
    result, code = _run(args)
    assert code == 0
    assert result["status"] == "passed"
    unit_root = repo / "output" / "chemprop_units" / "chemprop-fold-0"
    predictions = pd.read_parquet(unit_root / "oof_predictions.parquet")
    assert set(predictions["source_partition"]) == {"train"}
    assert set(predictions["outer_fold"]) == {0}
    assert predictions["absolute_error_pic50"].round(8).eq(0.1).all()
    again, again_code = _run(args)
    assert again_code == 0
    assert again["status"] == "skipped_validated_complete"


def test_capability_cli_is_machine_readable(tmp_path: Path) -> None:
    script = SCRIPTS / "run_local_herg_chemprop_unit.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--repo-root", str(tmp_path), "--capabilities-json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["cpu_only"] is True
    assert payload["macos_num_workers"] == 0
