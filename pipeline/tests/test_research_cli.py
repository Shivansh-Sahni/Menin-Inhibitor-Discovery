from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from menin_discovery.research_cli import (
    _conservative_herg_pilot_classes,
    _heavy_physics_status,
    _hpc_stage,
    _model_reporting_summary,
    _physics_stage,
    _promote_directory,
)


def test_heavy_physics_status_requires_manifest(tmp_path: Path) -> None:
    simulations = tmp_path / "simulations"
    config = {"paths": {"simulations": simulations}}

    pending = _heavy_physics_status(config)
    assert pending["manifest_present"] is False
    assert "pending" in pending["status"]

    simulations.mkdir()
    assert _heavy_physics_status(config)["manifest_present"] is False

    (simulations / "manifest.json").write_text("{}\n", encoding="utf-8")
    generated = _heavy_physics_status(config)
    assert generated["manifest_present"] is True
    assert "generated" in generated["status"]


def test_promote_directory_rebases_returned_staging_paths(tmp_path: Path) -> None:
    target = tmp_path / "stable" / "pk"

    def build(staging: Path) -> dict[str, object]:
        artifact = staging / "nested" / "metrics.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("{}\n", encoding="utf-8")
        return {
            "output_dir": str(staging),
            "artifact": artifact,
            "nested": [str(artifact), {"path": str(staging / "table.parquet")}],
            "ordinary_text": f"diagnostic mentions {staging} but is not a path",
            "external_path": str(tmp_path / "external"),
        }

    result = _promote_directory(target, build, required=("nested/metrics.json",))

    assert result["output_dir"] == str(target)
    assert result["artifact"] == target / "nested" / "metrics.json"
    assert result["nested"] == [
        str(target / "nested" / "metrics.json"),
        {"path": str(target / "table.parquet")},
    ]
    assert result["ordinary_text"].startswith("diagnostic mentions ")
    assert ".pk-stage-" in result["ordinary_text"]
    assert result["external_path"] == str(tmp_path / "external")
    assert (target / "nested" / "metrics.json").is_file()


def test_herg_pilot_classes_use_decisive_censored_limits_and_quarantine_conflicts() -> None:
    measurements = pd.DataFrame(
        {
            "compound_id": [
                "exact_blocker",
                "upper_blocker",
                "lower_nonblocker",
                "middle",
                "conflict",
                "conflict",
                "weak_limit",
                "excluded",
            ],
            "endpoint": ["herg_ic50"] * 8,
            "value": [5.0, 0.4, 30.0, 20.0, 2.0, 30.0, 20.0, 30.0],
            "unit": ["uM", "µM", "μM", "uM", "uM", "uM", "uM", "uM"],
            "relation": ["=", "<", ">", "~", "=", ">", ">", ">"],
            "model_eligible": [True, True, True, True, True, True, True, False],
        }
    )

    result = _conservative_herg_pilot_classes(measurements).set_index("compound_id")["herg_class"]

    assert result.to_dict() == {
        "conflict": "conflicting",
        "exact_blocker": "blocker",
        "lower_nonblocker": "nonblocker",
        "middle": "intermediate",
        "upper_blocker": "blocker",
        "weak_limit": "indeterminate",
    }


def test_physics_stage_can_defer_without_enumeration_or_model_facing_table(
    tmp_path: Path, monkeypatch
) -> None:
    physics = tmp_path / "physics"
    config = {
        "run": {"execute_local_fast_physics": False},
        "paths": {"physics": physics},
    }

    def forbidden_enumeration(*args, **kwargs):
        raise AssertionError("molecular enumeration must not run in deferred mode")

    monkeypatch.setattr("menin_discovery.research_cli.run_fast_physics", forbidden_enumeration)
    result = _physics_stage(config, smoke=False)

    assert result["status"] == "deferred_to_hpc"
    assert result["model_facing_physics_features_available"] is False
    assert sorted(path.name for path in physics.iterdir()) == [
        "fast_physics_feature_ontology.csv",
        "fast_physics_run_summary.json",
        "physics_deferred.json",
    ]
    assert not (physics / "fast_physics_summary.parquet").exists()
    assert not list(physics.glob("*.parquet"))


def test_model_reporting_tolerates_headerless_empty_physics_promotion_csv(
    tmp_path: Path,
) -> None:
    pk_root = tmp_path / "models" / "pk"
    pk_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "domain": "pk",
                "endpoint": "iv_auc_dose_normalized",
                "feature_layer": "state_conformer_physics",
                "status": "deferred_to_hpc",
                "best_model": None,
                "primary_metric": None,
                "primary_value": None,
            }
        ]
    ).to_csv(pk_root / "pk_model_ladder_summary.csv", index=False)
    (pk_root / "pk_physics_promotion_gates.csv").write_text("", encoding="utf-8")

    result = _model_reporting_summary({"paths": {"models": tmp_path / "models"}})

    assert len(result) == 1
    assert result.loc[0, "feature_layer"] == "state_conformer_physics"
    assert result.loc[0, "status"] == "deferred_to_hpc"


def test_hpc_fallback_uses_six_2d_axes_and_writes_uncapped_sampling_protocol(
    tmp_path: Path, monkeypatch
) -> None:
    simulations = tmp_path / "simulations"
    captured: dict[str, pd.DataFrame] = {}
    compounds = pd.DataFrame(
        {
            "compound_id": ["C1", "C2"],
            "standardized_smiles": ["CCN", "CCO"],
            "mw": [700.0, 760.0],
            "series_id": ["S1", "S2"],
        }
    )
    measurements = pd.DataFrame(columns=["compound_id", "species", "endpoint", "value", "unit", "relation"])

    monkeypatch.setattr(
        "menin_discovery.research_cli.load_canonical_tables",
        lambda _root: {"compounds": compounds, "measurements": measurements},
    )
    monkeypatch.setattr(
        "menin_discovery.research_cli.compound_model_frame",
        lambda frame: frame.copy(),
    )
    monkeypatch.setattr(
        "menin_discovery.research_cli._conservative_herg_pilot_classes",
        lambda *args, **kwargs: pd.DataFrame(columns=["compound_id", "herg_class"]),
    )

    def fake_generate(_compounds, physics, staging, *, config):
        captured["physics"] = physics.copy()
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "manifest.json").write_text('{"required_bundle_files": []}\n', encoding="utf-8")
        pd.DataFrame([{"compound_id": "C1"}]).to_parquet(staging / "pilot_selection.parquet", index=False)
        for workflow in (
            "environment_md",
            "membrane_pmf",
            "herg_ensemble",
            "relative_free_energy",
        ):
            directory = staging / workflow
            directory.mkdir()
            (directory / "protocol.yaml").write_text("launch: false\n", encoding="utf-8")
        return {"pilot_count": 1}

    monkeypatch.setattr("menin_discovery.research_cli.generate_hpc_bundles", fake_generate)
    config = {
        "paths": {
            "canonical": tmp_path / "canonical",
            "reports": tmp_path / "reports",
            "physics": tmp_path / "physics",
            "simulations": simulations,
        },
        "validation": {"blocker_threshold_um": 10.0, "nonblocker_threshold_um": 30.0},
        "hpc": {
            "state_conformer_sampling": {
                "fixed_state_count_cap": None,
                "nominal_population_threshold": 0.001,
                "threshold_sensitivity": [0.01, 0.001, 0.0001],
                "conformer_depth_ladder": [25, 50, 100, 250, 500],
            }
        },
    }

    result = _hpc_stage(config)

    assert result["pilot_selection_basis"] == "prephysics_2d_mechanistic_coverage_only"
    assert captured["physics"].columns.tolist() == [
        "compound_id",
        "selection_size_mw",
        "selection_formal_charge",
        "selection_flexibility_rotatable_bonds",
        "selection_polarity_tpsa",
        "selection_partition_logp",
        "selection_shape_fraction_csp3",
    ]
    protocol = yaml.safe_load(
        (simulations / "state_conformer_sampling" / "protocol.yaml").read_text(encoding="utf-8")
    )
    assert protocol["fixed_state_count_cap"] is None
    assert "max_states" not in protocol
    assert protocol["conformer_depth_ladder"] == [25, 50, 100, 250, 500]
    assert protocol["execution_status"] == "deferred_not_executed"
