from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_local_herg_feature_relationship_campaign_v5.py"
SPEC = importlib.util.spec_from_file_location("relationship_v5", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_bh_qvalues_at_hypothesis_grain() -> None:
    result = MODULE.bh_qvalues([0.01, 0.04, 0.03, np.nan])
    assert result[:3] == pytest.approx([0.03, 0.04, 0.04])
    assert np.isnan(result[3])


def test_paired_scaffold_bootstrap_is_deterministic_and_detects_effect() -> None:
    frame = pd.DataFrame(
        {
            "scaffold_group_id": np.repeat([f"s{i}" for i in range(30)], 3),
            "baseline_abs_error": 0.3,
            "perturbed_abs_error": 0.5,
        }
    )
    first = MODULE.paired_scaffold_bootstrap(frame, replicates=10_000, seed=7)
    second = MODULE.paired_scaffold_bootstrap(frame, replicates=10_000, seed=7)
    assert first == second
    assert first["effect_mae_delta"] == pytest.approx(0.2)
    assert first["ci95_lower"] > 0
    assert first["paired_sign_flip_p"] < 0.01


def test_bootstrap_rejects_less_than_ten_thousand() -> None:
    frame = pd.DataFrame(
        {
            "scaffold_group_id": ["a", "b"],
            "baseline_abs_error": [0.1, 0.2],
            "perturbed_abs_error": [0.2, 0.3],
        }
    )
    with pytest.raises(ValueError, match="10,000"):
        MODULE.paired_scaffold_bootstrap(frame, replicates=9_999, seed=1)


def test_fold_direction_requires_four_of_five() -> None:
    frame = pd.DataFrame(
        {
            "outer_fold": np.repeat(range(5), 2),
            "scaffold_group_id": [f"s{i}" for i in range(10)],
            "baseline_abs_error": 0.2,
            "perturbed_abs_error": [0.4] * 8 + [0.1] * 2,
        }
    )
    _, summary = MODULE._fold_stability(frame)
    assert summary["direction_stable_4_of_5"] is True
    assert summary["stable_direction"] == "beneficial"


def test_coarsened_matching_retains_only_cells_with_overlap() -> None:
    frame = pd.DataFrame(
        {
            "MolWt": list(range(100)),
            "__subgroup": ["A"] * 50 + ["B"] * 50,
        }
    )
    matched = MODULE._coarsened_match(frame, ["MolWt"])
    assert 0 < len(matched) < len(frame)
    assert set(matched["__subgroup"]) == {"A", "B"}
    assert matched.groupby("__chemistry_cell")["__subgroup"].nunique().min() == 2


def _campaign(tmp_path: Path, *, opened: bool = False) -> tuple[Path, Path]:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "validation.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "source_partition": "train",
                "validation_labels_opened": opened,
                "test_labels_opened": False,
            }
        )
    )
    n = 18_801
    identities = np.array([f"m{i}" for i in range(n)])
    scaffolds = np.array([f"s{i // 10}" for i in range(n)])
    folds = np.array([(i // 10) % 5 for i in range(n)])
    oof = pd.DataFrame(
        {
            "structure_id": identities,
            "scaffold_group_id": scaffolds,
            "outer_fold": folds,
            "observed_pic50": 5.0,
            "predicted_pic50": 4.8,
        }
    )
    oof.to_parquet(campaign / "nested_oof_predictions.parquet", index=False)
    oof.to_parquet(campaign / "relationship_reference_oof.parquet", index=False)
    pd.DataFrame(
        {
            "hypothesis_id": "polarity_block",
            "evidence_type": "grouped_ablation",
            "outer_fold": folds,
            "structure_id": identities,
            "scaffold_group_id": scaffolds,
            "baseline_abs_error": 0.2,
            "perturbed_abs_error": 0.35,
        }
    ).to_parquet(campaign / "paired_effects.parquet", index=False)
    prepared = tmp_path / "research/local_runs/herg_discovery_campaign_v1/prepared"
    prepared.mkdir(parents=True)
    pd.DataFrame(
        {
            "structure_id": identities,
            "scaffold_group_id": scaffolds,
            "target_pic50": 5.0,
        }
    ).to_parquet(prepared / "exact_train_cache.parquet", index=False)
    pd.DataFrame(
        {
            "structure_id": identities,
            "scaffold_group_id": scaffolds,
            "source_partition": "train",
            "outer_fold": folds,
            "outer_role": "heldout",
            "inner_fold": -1,
        }
    ).to_parquet(prepared / "nested_scaffold_splits.parquet", index=False)
    metadata = tmp_path / "metadata.parquet"
    pd.DataFrame(
        {
            "structure_id": identities,
            "MolWt": np.resize(np.arange(100, 200), n),
            "MolLogP": np.resize(np.linspace(-1, 4, 100), n),
            "assay_family": np.where(np.arange(n) % 2, "patch", "flux"),
        }
    ).to_parquet(metadata, index=False)
    return campaign, metadata


def test_end_to_end_train_only_analysis(tmp_path: Path) -> None:
    campaign, metadata = _campaign(tmp_path)
    output = tmp_path / "analysis"
    report = MODULE.analyze(
        repo_root=tmp_path,
        campaign_root=campaign,
        output_root=output,
        metadata_path=metadata,
        bootstrap_replicates=10_000,
        seed=13,
    )
    assert report["status"] == "passed"
    assert report["counts"]["nested_oof_rows"] == 18_801
    hypotheses = pd.read_parquet(output / "relationship_hypotheses.parquet")
    assert hypotheses.loc[0, "direction_stable_4_of_5"]
    assert hypotheses.loc[0, "bh_q_value"] <= 0.05
    validation = json.loads((output / "validation.json").read_text())
    assert validation["validation_labels_opened"] is False
    assert (output / "manifest.json").exists()
    assert (output / "analysis.md").exists()


def test_rejects_opened_validation_labels(tmp_path: Path) -> None:
    campaign, metadata = _campaign(tmp_path, opened=True)
    with pytest.raises(ValueError, match="validation/test labels"):
        MODULE.analyze(
            repo_root=tmp_path,
            campaign_root=campaign,
            output_root=tmp_path / "analysis",
            metadata_path=metadata,
            bootstrap_replicates=10_000,
            seed=13,
        )


def test_rejects_effect_fold_mismatch(tmp_path: Path) -> None:
    campaign, metadata = _campaign(tmp_path)
    effects = pd.read_parquet(campaign / "paired_effects.parquet")
    effects.loc[0, "outer_fold"] = (int(effects.loc[0, "outer_fold"]) + 1) % 5
    effects.to_parquet(campaign / "paired_effects.parquet", index=False)
    with pytest.raises(ValueError, match="fold assignments"):
        MODULE.analyze(
            repo_root=tmp_path,
            campaign_root=campaign,
            output_root=tmp_path / "analysis",
            metadata_path=metadata,
            bootstrap_replicates=10_000,
            seed=13,
        )


def test_rejects_unverified_incremental_evidence_type(tmp_path: Path) -> None:
    campaign, metadata = _campaign(tmp_path)
    effects = pd.read_parquet(campaign / "paired_effects.parquet")
    effects["evidence_type"] = "feature_importance"
    effects.to_parquet(campaign / "paired_effects.parquet", index=False)
    with pytest.raises(ValueError, match="unsupported incremental"):
        MODULE.analyze(
            repo_root=tmp_path,
            campaign_root=campaign,
            output_root=tmp_path / "analysis",
            metadata_path=metadata,
            bootstrap_replicates=10_000,
            seed=13,
        )


def test_rejects_canonical_target_mismatch(tmp_path: Path) -> None:
    campaign, metadata = _campaign(tmp_path)
    oof = pd.read_parquet(campaign / "nested_oof_predictions.parquet")
    oof.loc[0, "observed_pic50"] = 6.0
    oof.to_parquet(campaign / "nested_oof_predictions.parquet", index=False)
    with pytest.raises(ValueError, match="canonical exact train targets"):
        MODULE.analyze(
            repo_root=tmp_path,
            campaign_root=campaign,
            output_root=tmp_path / "analysis",
            metadata_path=metadata,
            bootstrap_replicates=10_000,
            seed=13,
        )


def test_relationship_reference_is_distinct_from_performance_oof(tmp_path: Path) -> None:
    campaign, metadata = _campaign(tmp_path)
    performance = pd.read_parquet(campaign / "nested_oof_predictions.parquet")
    performance["predicted_pic50"] = 4.1
    performance.to_parquet(campaign / "nested_oof_predictions.parquet", index=False)
    report = MODULE.analyze(
        repo_root=tmp_path,
        campaign_root=campaign,
        output_root=tmp_path / "analysis",
        metadata_path=metadata,
        bootstrap_replicates=10_000,
        seed=13,
    )
    assert report["status"] == "passed"
    manifest = json.loads((tmp_path / "analysis/manifest.json").read_text())
    names = {Path(item["path"]).name for item in manifest["inputs"]}
    assert "nested_oof_predictions.parquet" in names
    assert "relationship_reference_oof.parquet" in names
