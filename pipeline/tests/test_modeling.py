import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
from menin_discovery.curation import annotate_cross_source_mirrors
from menin_discovery.modeling import (
    prepare_herg_task,
    prepare_menin_task,
    train_herg_classifier_and_predict,
    train_menin_activity_model,
)


def test_prepare_herg_task_deduplicates_and_excludes_conflicting_structures():
    frame = pd.DataFrame(
        {
            "structure_id": ["a", "a", "b", "b", "c"],
            "smiles": ["CC", "CC", "CCC", "CCC", "CCCC"],
            "endpoint": ["IC50", "Ki", "IC50", "Ki", "IC50"],
            "herg_blocker_label": [1, 1, 0, 1, np.nan],
        }
    )
    task = prepare_herg_task(frame)
    assert task["structure_id"].tolist() == ["a"]
    assert task.iloc[0]["n_task_rows"] == 2
    assert task.attrs["task_metadata"]["n_conflicting_structures_excluded"] == 1


def test_prepare_herg_task_filters_endpoint_and_assay_before_conflict_resolution():
    frame = pd.DataFrame(
        {
            "structure_id": ["a", "a", "b"],
            "smiles": ["CC", "CC", "CCC"],
            "endpoint": ["IC50", "Ki", "IC50"],
            "assay_family": [
                "electrophysiology_functional",
                "biochemical_binding",
                "biochemical_binding",
            ],
            "herg_blocker_label": [1, 0, 0],
        }
    )
    task = prepare_herg_task(
        frame,
        endpoint="IC50",
        assay_family="electrophysiology_functional",
    )
    assert task["structure_id"].tolist() == ["a"]
    assert task.attrs["task_metadata"]["n_conflicting_structures_excluded"] == 0


def test_prepare_menin_task_filters_endpoint_before_aggregation():
    measurements = pd.DataFrame(
        {
            "smiles": ["CCO", "CCO", "CCO", "CCN"],
            "endpoint": ["IC50", "IC50", "Ki", "IC50"],
            "p_value": [7.0, 8.0, 5.0, 6.0],
            "is_exact": [True, True, True, True],
            "is_core_endpoint": [True, True, True, True],
            "document_year": [2020, 2021, 2022, 2020],
        }
    )
    task = prepare_menin_task(measurements, endpoint="IC50")
    ethanol = task.loc[task["smiles"].eq("CCO")].iloc[0]
    assert ethanol["p_activity_median"] == 7.5
    assert ethanol["n_measurements"] == 2
    assert task.attrs["task_metadata"]["input_level"] == "measurement"


def test_prepare_menin_task_excludes_quarantine_and_selects_assay_family():
    measurements = pd.DataFrame(
        {
            "structure_id": ["a", "a", "a", "b"],
            "smiles": ["CCO", "CCO", "CCO", "CCN"],
            "endpoint": ["IC50"] * 4,
            "assay_family": [
                "biochemical_inhibition",
                "biochemical_inhibition",
                "cellular_functional",
                "biochemical_inhibition",
            ],
            "p_value": [7.0, 3.0, 5.0, 6.0],
            "is_exact": [True] * 4,
            "is_core_endpoint": [True] * 4,
            "is_modeling_eligible": [True, False, True, True],
        }
    )
    task = prepare_menin_task(
        measurements,
        endpoint="IC50",
        assay_family="biochemical_inhibition",
    )
    assert set(task["structure_id"]) == {"a", "b"}
    assert task.loc[task["structure_id"] == "a", "p_activity_median"].item() == 7.0
    assert task.attrs["task_metadata"]["ineligible_rows_excluded"] == 1


def test_prepare_menin_task_mirror_collapse_is_adjustable_and_keeps_provenance():
    measurements = pd.DataFrame(
        {
            "structure_id": ["STR-1"] * 4,
            "smiles": ["CCO"] * 4,
            "source": ["ChEMBL", "ChEMBL", "BindingDB", "PubChem"],
            "source_record_id": ["c1", "c2", "b1", "p1"],
            "endpoint": ["IC50"] * 4,
            "assay_family": ["biochemical_binding"] * 4,
            "relation": ["="] * 4,
            "p_value": [7.0] * 4,
            "is_exact": [True] * 4,
            "is_core_endpoint": [True] * 4,
            "is_modeling_eligible": [True] * 4,
            "document_year": [2020, 2020, 2020, 2020],
        }
    )
    measurements = annotate_cross_source_mirrors(measurements)

    collapsed = prepare_menin_task(
        measurements,
        endpoint="IC50",
        assay_family="biochemical_binding",
    )
    retained = prepare_menin_task(
        measurements,
        endpoint="IC50",
        assay_family="biochemical_binding",
        collapse_cross_source_mirrors=False,
    )

    # Both same-source ChEMBL replicates remain in the central label. The two
    # lower-priority cross-source mirrors are excluded only in the default run.
    assert collapsed.loc[0, "n_measurements"] == 2
    assert retained.loc[0, "n_measurements"] == 4
    # Measurement provenance still describes every linked source row.
    assert collapsed.loc[0, "n_source_rows"] == 4
    assert collapsed.loc[0, "n_sources"] == 3
    assert collapsed.attrs["task_metadata"]["cross_source_mirror_rows_collapsed"] == 2
    assert retained.attrs["task_metadata"]["cross_source_mirror_rows_collapsed"] == 0
    assert retained.attrs["task_metadata"]["collapse_cross_source_mirrors"] is False


def test_model_json_serializes_yaml_dates_in_provenance_as_iso_strings(tmp_path):
    provenance = {
        "resolved_settings": {
            "analysis": {
                "reference_compounds": [
                    {
                        # PyYAML resolves an unquoted YYYY-MM-DD scalar to
                        # datetime.date, matching the canonical-build failure.
                        "source_checked_at": date(2026, 7, 14),
                    }
                ]
            }
        },
        "generated_at": datetime(2026, 7, 14, 12, 34, 56, tzinfo=timezone.utc),
    }
    metrics = train_menin_activity_model(
        pd.DataFrame({"smiles": ["CCO"], "p_activity_median": [7.0]}),
        tmp_path / "models",
        tmp_path / "reports",
        min_samples=2,
        provenance_context=provenance,
    )
    assert metrics["status"] == "insufficient_data"

    payload = json.loads(
        (tmp_path / "reports" / "menin_activity_model_metrics.json").read_text(encoding="utf-8")
    )
    serialized = payload["provenance"]
    assert (
        serialized["resolved_settings"]["analysis"]["reference_compounds"][0]["source_checked_at"]
        == "2026-07-14"
    )
    assert serialized["generated_at"] == "2026-07-14T12:34:56+00:00"


def test_regression_model_emits_comparison_uncertainty_and_manifest(tmp_path):
    n_rows = 48
    compounds = pd.DataFrame(
        {
            "smiles": ["C" * (index + 1) for index in range(n_rows)],
            "p_activity_median": 4.5 + np.arange(n_rows) * 0.08,
            "endpoints": ["IC50"] * n_rows,
            "document_years": [str(2010 + index % 10) for index in range(n_rows)],
        }
    )
    models = tmp_path / "models"
    reports = tmp_path / "reports"
    metrics = train_menin_activity_model(
        compounds,
        models,
        reports,
        split_strategy="random",
        feature_backend="hashed",
        feature_n_bits=128,
        cv_folds=2,
        bootstrap_iterations=20,
        tree_estimators=16,
        min_samples=20,
    )
    assert metrics["status"] == "trained"
    assert metrics["features"]["backend"] == "hashed_smiles"
    assert "test_metric_bootstrap_95_ci" in metrics
    assert "uncertainty" in metrics
    assert (reports / "menin_activity_model_comparison.csv").exists()
    assert (reports / "menin_activity_split_assignments.csv").exists()
    manifest = json.loads((models / "menin_activity_manifest.json").read_text())
    assert len(manifest["dataset_sha256"]) == 64
    assert len(manifest["artifact"]["sha256"]) == 64


def test_classifier_emits_calibration_and_application_domain(tmp_path):
    n_rows = 60
    herg = pd.DataFrame(
        {
            "smiles": [f"N{'C' * (index + 1)}" for index in range(n_rows)],
            "herg_blocker_label": [index % 2 for index in range(n_rows)],
            "document_years": [str(2010 + index % 10) for index in range(n_rows)],
        }
    )
    menin = pd.DataFrame({"smiles": herg["smiles"].iloc[:10].tolist() + [""]})
    models = tmp_path / "models"
    reports = tmp_path / "reports"
    metrics = train_herg_classifier_and_predict(
        herg,
        menin,
        models,
        reports,
        split_strategy="random",
        feature_backend="hashed",
        feature_n_bits=128,
        cv_folds=2,
        bootstrap_iterations=20,
        tree_estimators=16,
        min_samples=20,
    )
    assert metrics["status"] == "trained"
    assert "calibration" in metrics
    assert "pr_auc" in metrics["test_metrics"]
    assert "applicability_domain" in metrics
    predictions = pd.read_csv(reports / "menin_with_predicted_herg_risk.csv")
    assert "herg_inside_applicability_domain" in predictions.columns
    assert predictions.iloc[-1]["predicted_herg_risk"] == "unscored"
    assert (reports / "herg_classifier_calibration_curve.csv").exists()
