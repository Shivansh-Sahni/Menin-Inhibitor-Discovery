from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_local_herg_discovery_worker.py"
SPEC = importlib.util.spec_from_file_location("run_local_herg_discovery_worker", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def test_fingerprint_unpack_is_little_endian_and_exact_length() -> None:
    bits = worker._unpack_binary_fingerprint(bytes([0b10000001, 0b00000010]), 10)
    assert bits.tolist() == [1, 0, 0, 0, 0, 0, 0, 1, 0, 1]
    with pytest.raises(worker.DiscoveryWorkerError, match="byte length"):
        worker._unpack_binary_fingerprint(b"\x01", 10)


def test_nested_splits_are_scaffold_exclusive_and_complete() -> None:
    frame = pd.DataFrame(
        {
            "structure_id": [f"S{index:02d}" for index in range(20)],
            "scaffold_group_id": [f"G{index // 2:02d}" for index in range(20)],
        }
    )
    splits = worker._make_nested_splits(frame, outer_folds=5, inner_folds=3)
    assert len(splits) == 100
    assert set(splits["source_partition"]) == {"train"}
    for _, fold in splits.groupby("outer_fold"):
        fit = set(fold.loc[fold["outer_role"].eq("fit"), "scaffold_group_id"])
        heldout = set(fold.loc[fold["outer_role"].eq("heldout"), "scaffold_group_id"])
        assert fit.isdisjoint(heldout)
        assert set(fold.loc[fold["outer_role"].eq("fit"), "inner_fold"]) == {0, 1, 2}


def _synthetic_inputs(root: Path) -> tuple[Path, Path]:
    matrix_root = root / "matrix"
    matrix_root.mkdir()
    rows = 20
    matrix = pd.DataFrame(
        {
            "structure_id": [f"S{index:02d}" for index in range(rows)] + ["VALIDATION-POISON"],
            "standard_inchi_key": [f"KEY{index:02d}" for index in range(rows)] + ["POISON"],
            "model_split": ["train"] * rows + ["validation"],
            "scaffold_group_id": [f"G{index // 2:02d}" for index in range(rows)] + ["GV"],
            "f2d_feature_id": [f"F{index:02d}" for index in range(rows)] + ["FV"],
            "morgan_r2_2048": [bytes([index % 256]) + bytes(255) for index in range(rows)] + [bytes(256)],
            "maccs_167": [bytes([index % 256]) + bytes(20) for index in range(rows)] + [bytes(21)],
            "rdkit2d__MolLogP": np.linspace(1, 4, rows + 1),
            "rdkit2d__TPSA": np.linspace(20, 100, rows + 1),
            "rdkit2d__NumAromaticRings": np.tile([1, 2, 3], 7)[: rows + 1],
            "rdkit2d__fr_benzene": np.tile([0, 1], 11)[: rows + 1],
            "rdkit2d__Chi0": np.linspace(2, 10, rows + 1),
            "f3d__formal_charge": np.tile([-1, 0, 1], 7)[: rows + 1],
            "f3d__basic_site_proxy_count": np.tile([0, 1], 11)[: rows + 1],
            "f3d__acidic_site_proxy_count": np.tile([1, 0], 11)[: rows + 1],
            "f3d__tautomer_count_capped16": np.tile([1, 2], 11)[: rows + 1],
            "f3d__rotatable_bond_count": np.tile([1, 2, 3], 7)[: rows + 1],
            "f3d__energy_range_kcal_mol": np.linspace(1, 6, rows + 1),
            "f3d__effective_conformer_count": np.linspace(1, 3, rows + 1),
            "f3d__dominant_conformer_weight": np.linspace(0.3, 1, rows + 1),
            "f3d__embedded_conformer_count": np.full(rows + 1, 6),
            "f3d__retained_conformer_count": np.full(rows + 1, 3),
            "f3d__unconverged_retained_count": np.tile([0, 1], 11)[: rows + 1],
            "f3d__energy_polar_exposure_correlation": np.linspace(-0.5, 0.5, rows + 1),
            "f3d__ensemble_pmi1__mean": np.linspace(1, 5, rows + 1),
            "f3d__ensemble_gasteiger_dipole_proxy_eA__mean": np.linspace(1, 4, rows + 1),
            "f3d__ensemble_absolute_charge_radius_A__mean": np.linspace(1, 3, rows + 1),
            "f3d__ensemble_polar_radial_exposure__mean": np.linspace(0, 1, rows + 1),
            "f3d__ensemble_internal_polar_contact_count__mean": np.tile([0, 1], 11)[: rows + 1],
            "f3d__ensemble_sasa__mean": np.linspace(10, 40, rows + 1),
            "f3d__dominant_autocorr3d__000": np.linspace(-1, 1, rows + 1),
            "f3d__dominant_whim__000": np.linspace(0, 2, rows + 1),
        }
    )
    matrix.to_parquet(matrix_root / "combined_feature_matrix.parquet", index=False)
    (matrix_root / "validation.json").write_text(json.dumps({"status": "passed"}))

    observation_rows: list[dict[str, object]] = []
    for index in range(rows):
        observation_rows.append(
            {
                "structure_id": f"S{index:02d}",
                "standardized_smiles": "C" * (index % 4 + 1),
                "standard_inchi_key": f"KEY{index:02d}",
                "model_split": "train",
                "scaffold_group_id": f"G{index // 2:02d}",
                "target_variant": "wild_type_or_unspecified",
                "wild_type_evidence_scope": "wild_type_or_unspecified",
                "master_confirmed_wild_type_scope": False,
                "measurement_modality": "unresolved",
                "automation_class": "unresolved",
                "assay_family": "mixed_unresolved_compilation",
                "source_family": "synthetic",
                "protocol_completeness_score": 0,
                "potency_relation_pic50": "=",
                "potency_pic50_point": 4.0 + index / 10,
                "standardized_pic50_primary": True,
            }
        )
    for split in ("validation", "test"):
        poison = dict(observation_rows[0])
        poison.update(
            structure_id=f"{split}-poison",
            model_split=split,
            scaffold_group_id=f"{split}-group",
            potency_pic50_point=999.0,
        )
        observation_rows.append(poison)
    observations_path = root / "observations.parquet"
    table = pa.Table.from_pandas(pd.DataFrame(observation_rows), preserve_index=False)
    pq.write_table(table, observations_path)
    return matrix_root, observations_path


def test_prepare_excludes_repository_holdouts_and_unpacks_fingerprints(tmp_path: Path) -> None:
    matrix_root, observations = _synthetic_inputs(tmp_path)
    output = tmp_path / "prepared"
    result = worker._prepare(
        argparse.Namespace(
            repo_root=str(tmp_path),
            matrix_root=str(matrix_root),
            observations=str(observations),
            output_root=str(output),
            outer_folds=5,
            inner_folds=3,
        )
    )
    assert result["status"] == "passed"
    cache = pd.read_parquet(output / "exact_train_cache.parquet")
    assert len(cache) == 20
    assert not cache["structure_id"].str.contains("poison", case=False).any()
    assert len([column for column in cache if column.startswith("morgan__")]) == 2048
    assert len([column for column in cache if column.startswith("maccs__")]) == 167
    assert cache.loc[1, "morgan__0000"] == 1
    assert cache.loc[1, "morgan__0001"] == 0
    summary = json.loads((output / "source_summary.json").read_text())
    assert "must not be numerically pooled" in " ".join(summary["limitations"])


def test_classical_unit_writes_train_only_atomic_artifacts(tmp_path: Path) -> None:
    matrix_root, observations = _synthetic_inputs(tmp_path)
    prepared = tmp_path / "prepared"
    worker._prepare(
        argparse.Namespace(
            repo_root=str(tmp_path),
            matrix_root=str(matrix_root),
            observations=str(observations),
            output_root=str(prepared),
            outer_folds=5,
            inner_folds=3,
        )
    )
    result = worker._classical_unit(
        argparse.Namespace(
            prepared_root=str(prepared),
            output_root=str(tmp_path / "results"),
            unit_id="ridge_outer_0",
            model_id="ridge_safe",
            stage="outer",
            outer_fold=0,
            inner_fold=None,
            seed=4,
            model="ridge",
            groups="safe_classical",
            params_json='{"alpha": 1.0}',
            workers=1,
            maximum_features=768,
        )
    )
    assert result["status"] == "passed"
    assert result["scientific_scope"]["repository_test_outcomes_loaded"] is False
    predictions = pd.read_parquet(result["prediction_artifact"]["path"])
    assert set(predictions["source_partition"]) == {"train"}
    assert len(predictions) == result["evaluation_structures"]


def _chemprop_analysis_fixture(root: Path, folds: int = 5) -> tuple[Path, Path]:
    prepared = root / "prepared"
    results = root / "results"
    prepared.mkdir()
    rows = pd.DataFrame(
        {
            "structure_id": [f"S{index:02d}" for index in range(10)],
            "scaffold_group_id": [f"G{index:02d}" for index in range(10)],
            "target_pic50": np.linspace(4.0, 5.8, 10),
        }
    )
    rows.to_parquet(prepared / "exact_train_cache.parquet", index=False)
    split_frames = []
    for fold in range(5):
        heldout = {f"S{2 * fold:02d}", f"S{2 * fold + 1:02d}"}
        split = rows[["structure_id", "scaffold_group_id"]].copy()
        split["source_partition"] = "train"
        split["outer_fold"] = fold
        split["outer_role"] = np.where(split["structure_id"].isin(heldout), "heldout", "fit")
        split["inner_fold"] = np.where(split["outer_role"].eq("heldout"), -1, 0)
        split_frames.append(split)
    pd.concat(split_frames, ignore_index=True).to_parquet(
        prepared / "nested_scaffold_splits.parquet", index=False
    )

    for fold in range(folds):
        selected = rows.iloc[2 * fold : 2 * fold + 2].copy()
        prediction = pd.DataFrame(
            {
                "schema_version": "platform-local-herg-chemprop-oof/1.0",
                "unit_id": f"chemprop_outer_{fold}",
                "structure_id": selected["structure_id"],
                "scaffold_group_id": selected["scaffold_group_id"],
                "source_partition": "train",
                "inner_role": "holdout",
                "outer_fold": fold,
                "observed_pic50": selected["target_pic50"],
                "predicted_pic50": selected["target_pic50"] + 0.1,
            }
        )
        unit_root = results / "chemprop_units" / f"chemprop_outer_{fold}"
        unit_root.mkdir(parents=True)
        prediction_path = unit_root / "oof_predictions.parquet"
        prediction.to_parquet(prediction_path, index=False)
        spec = {"outer_fold": fold, "seed": fold}
        canonical_spec = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
        document = {
            "status": "passed",
            "unit_id": f"chemprop_outer_{fold}",
            "resolved_spec": spec,
            "resolved_spec_sha256": hashlib.sha256(canonical_spec).hexdigest(),
            "inputs": [],
            "artifacts": [worker._binding(prediction_path)],
            "oof_predictions_path": str(prediction_path.resolve()),
            "preparation": {"role_counts": {"train": 6, "validation": 2, "holdout": 2}},
            "runtime": {"elapsed_seconds": 1.0},
            "scientific_contract": {
                "data_scope": "exact_pic50_repository_train_partition_only",
                "repository_validation_labels_opened": False,
                "repository_test_labels_opened": False,
            },
        }
        canonical = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        document["unit_json_sha256"] = hashlib.sha256(canonical).hexdigest()
        (unit_root / "unit.json").write_text(json.dumps(document), encoding="utf-8")
    return prepared, results


def test_complete_chemprop_outer_oof_is_integrated_only_with_all_five_folds(
    tmp_path: Path,
) -> None:
    prepared, results = _chemprop_analysis_fixture(tmp_path)
    metrics, predictions, integration = worker._complete_chemprop_outer_oof(prepared, results)
    assert integration["status"] == "integrated"
    assert integration["valid_outer_folds"] == [0, 1, 2, 3, 4]
    assert len(metrics) == 5
    combined = pd.concat(predictions, ignore_index=True)
    assert set(combined["model_id"]) == {worker.CHEMPROP_MODEL_ID}
    assert combined["structure_id"].nunique() == 10


def test_incomplete_chemprop_outer_oof_is_skipped_without_partial_ranking(tmp_path: Path) -> None:
    prepared, results = _chemprop_analysis_fixture(tmp_path, folds=4)
    metrics, predictions, integration = worker._complete_chemprop_outer_oof(prepared, results)
    assert integration["status"] == "incomplete"
    assert integration["valid_outer_folds"] == [0, 1, 2, 3]
    assert metrics == []
    assert predictions == []
