from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest
from menin_discovery.platform_clinical_results import (
    DEFAULT_SNAPSHOT_KEY,
    MANIFEST_NAME,
    ClinicalResultsError,
    _write_verification_report,
    build_candidates,
    canonical_json_bytes,
    classify_endpoint_text,
    document_with_sha256,
    sha256_file,
    verify_candidates,
    verify_document_sha256,
)


def _protocol(nct_id: str, title: str) -> dict[str, object]:
    return {
        "identificationModule": {"nctId": nct_id, "briefTitle": title},
        "statusModule": {"overallStatus": "COMPLETED"},
        "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE1"]},
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Fixture", "class": "OTHER"}},
        "conditionsModule": {"conditions": ["Fixture condition"]},
        "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": "Reported drug text"}]},
    }


def _outcome(
    title: str,
    unit: str,
    value: str,
    *,
    description: str = "",
) -> dict[str, object]:
    return {
        "type": "SECONDARY",
        "title": title,
        "description": description,
        "reportingStatus": "POSTED",
        "paramType": "MEAN",
        "unitOfMeasure": unit,
        "timeFrame": "Day 1",
        "groups": [{"id": "OG000", "title": "Reported group"}],
        "denoms": [{"units": "Participants", "counts": [{"groupId": "OG000", "value": "10"}]}],
        "classes": [
            {
                "title": "Day 1",
                "categories": [{"measurements": [{"groupId": "OG000", "value": value}]}],
            }
        ],
    }


def _fixture_source(tmp_path: Path) -> Path:
    source = tmp_path / "clinicaltrials_gov_v2"
    pages = source / "cardiac_safety_heuristic_cohorts" / DEFAULT_SNAPSHOT_KEY / "pages"
    pages.mkdir(parents=True)
    studies = [
        {
            "protocolSection": _protocol("NCT00000001", "QT Questionnaire feasibility study"),
            "hasResults": False,
        },
        {
            "protocolSection": _protocol("NCT00000002", "Posted QT study"),
            "resultsSection": {
                "outcomeMeasuresModule": {
                    "outcomeMeasures": [
                        _outcome("Change in QTcF", "ms", "4.2", description="Corrected QT interval")
                    ]
                },
                "adverseEventsModule": {"eventGroups": []},
            },
            "hasResults": True,
        },
        {
            "protocolSection": _protocol("NCT00000003", "Posted PK study"),
            "resultsSection": {
                "outcomeMeasuresModule": {
                    "outcomeMeasures": [
                        _outcome("Cmax", "ng/mL", "125", description="Maximum plasma concentration"),
                        _outcome("Participants With PK Sampling", "Participants", "10"),
                        _outcome("Creatinine clearance", "mL/min", "90"),
                    ]
                },
                "adverseEventsModule": {"eventGroups": []},
            },
            "hasResults": True,
        },
        {
            "protocolSection": _protocol("NCT00000004", "Posted adverse event study"),
            "resultsSection": {
                "outcomeMeasuresModule": {"outcomeMeasures": []},
                "adverseEventsModule": {
                    "timeFrame": "Day 1 through Day 7",
                    "frequencyThreshold": "5",
                    "eventGroups": [
                        {
                            "id": "EG000",
                            "title": "Reported group",
                            "seriousNumAffected": 0,
                            "seriousNumAtRisk": 10,
                            "otherNumAffected": 1,
                            "otherNumAtRisk": 10,
                        }
                    ],
                    "seriousEvents": [
                        {
                            "term": "Torsades de pointes",
                            "organSystem": "Cardiac disorders",
                            "stats": [{"groupId": "EG000", "numAffected": 0, "numAtRisk": 10}],
                        }
                    ],
                },
            },
            "hasResults": True,
        },
    ]
    entries: list[dict[str, object]] = []
    concatenated = hashlib.sha256()
    for index, study in enumerate(studies):
        page = pages / f"page_{index:06d}.json"
        page.write_bytes(canonical_json_bytes({"totalCount": 4, "studies": [study]}))
        page_bytes = page.read_bytes()
        concatenated.update(page_bytes)
        relative = page.relative_to(source).as_posix()
        entries.append(
            {
                "artifact_role": "source_HTTP_artifact",
                "bytes": len(page_bytes),
                "path": relative,
                "sha256": hashlib.sha256(page_bytes).hexdigest(),
            }
        )
    bundle = {
        "entries": entries,
        "entries_sha256": hashlib.sha256(canonical_json_bytes(entries)).hexdigest(),
        "entry_count": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "excluded_paths": [MANIFEST_NAME],
    }
    manifest = document_with_sha256(
        {
            "schema_version": "platform-external-acquisition/1.0",
            "source_id": "clinicaltrials_gov_v2",
            "release_id": "api-2.0.5",
            "api_version": "2.0.5",
            "dataTimestamp": "2026-08-04T09:00:05",
            "cardiac_safety_heuristic_cohort": {
                "cohort_snapshot_key": DEFAULT_SNAPSHOT_KEY,
                "page_count": 4,
                "reported_total_count": 4,
                "concatenated_raw_page_bytes_sha256": concatenated.hexdigest(),
            },
            "bundle_inventory": bundle,
        }
    )
    (source / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return source


def _read_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, mode="rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_conservative_classifier_rejects_ambiguous_qt_and_non_drug_clearance() -> None:
    assert (
        classify_endpoint_text(
            title="QT Questionnaire", description="Quality tool", unit="score", record_kind="outcome_measure"
        )
        == []
    )
    assert (
        classify_endpoint_text(
            title="Creatinine clearance",
            description="Renal function",
            unit="mL/min",
            record_kind="outcome_measure",
        )
        == []
    )
    qtc = classify_endpoint_text(
        title="Change in QTcF", description="Corrected QT interval", unit="ms", record_kind="outcome_measure"
    )
    assert qtc[0]["candidate_classification"] == "qt_qtc_interval_measure_candidate"
    assert qtc[0]["genuine_endpoint_candidate"] is True


def test_build_preserves_absence_ambiguity_denominators_and_zero_semantics(tmp_path: Path) -> None:
    source = _fixture_source(tmp_path)
    output = tmp_path / "candidates"
    manifest = build_candidates(source, output)
    assert manifest["row_counts"] == {
        "adverse_event_groups.csv.gz": 1,
        "arms_groups.csv.gz": 4,
        "endpoint_candidates.csv.gz": 4,
        "interventions.csv.gz": 4,
        "outcome_measures.csv.gz": 4,
        "studies.csv.gz": 4,
    }
    studies = _read_csv(output / "studies.csv.gz")
    absent = next(row for row in studies if row["nct_id"] == "NCT00000001")
    assert absent["has_results_reported"] == "false"
    assert "unknown_not_negative" in absent["absence_semantics"]
    assert absent["target_endpoint_candidate_count"] == "0"

    outcomes = _read_csv(output / "outcome_measures.csv.gz")
    cmax = next(row for row in outcomes if row["title"] == "Cmax")
    assert json.loads(cmax["denominator_records_json"])[0]["value"] == "10"
    assert json.loads(cmax["candidate_classifications_json"]) == ["pk_genuine_metric_candidate"]
    creatinine = next(row for row in outcomes if row["title"] == "Creatinine clearance")
    assert json.loads(creatinine["candidate_domains_json"]) == []

    candidates = _read_csv(output / "endpoint_candidates.csv.gz")
    classes = {row["candidate_classification"] for row in candidates}
    assert classes == {
        "pk_context_or_safety_count_not_genuine_metric",
        "pk_genuine_metric_candidate",
        "qt_qtc_event_or_threshold_candidate",
        "qt_qtc_interval_measure_candidate",
    }
    torsade = next(row for row in candidates if row["title_or_term"] == "Torsades de pointes")
    assert json.loads(torsade["value_records_json"])[0] == {
        "group_id": "EG000",
        "num_affected": "0",
        "num_at_risk": "10",
    }
    assert torsade["zero_counts_are_reported_values_not_study_level_negatives"] == "true"
    assert verify_candidates(output, source_root=source)["verification_status"] == "pass"


def test_source_and_output_tamper_fail_closed(tmp_path: Path) -> None:
    source = _fixture_source(tmp_path)
    page = source / "cardiac_safety_heuristic_cohorts" / DEFAULT_SNAPSHOT_KEY / "pages/page_000003.json"
    page.write_bytes(page.read_bytes() + b" ")
    with pytest.raises(ClinicalResultsError, match="byte count changed|SHA-256 changed"):
        build_candidates(source, tmp_path / "should_not_exist")

    source = _fixture_source(tmp_path / "second")
    output = tmp_path / "candidates"
    build_candidates(source, output)
    with (output / "studies.csv.gz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ClinicalResultsError, match="byte count changed|SHA-256 changed"):
        verify_candidates(output, source_root=source)


def test_output_symlink_fails_closed(tmp_path: Path) -> None:
    source = _fixture_source(tmp_path)
    output = tmp_path / "candidates"
    build_candidates(source, output)
    target = tmp_path / "outside.txt"
    target.write_text("outside\n", encoding="utf-8")
    (output / "alias.txt").symlink_to(target)
    with pytest.raises(ClinicalResultsError, match="contains a symlink"):
        verify_candidates(output, source_root=source)


def test_independent_builds_are_byte_identical(tmp_path: Path) -> None:
    source = _fixture_source(tmp_path)
    output_a = tmp_path / "a"
    output_b = tmp_path / "b"
    build_candidates(source, output_a)
    build_candidates(source, output_b)
    names_a = sorted(path.name for path in output_a.iterdir())
    names_b = sorted(path.name for path in output_b.iterdir())
    assert names_a == names_b
    assert {name: sha256_file(output_a / name) for name in names_a} == {
        name: sha256_file(output_b / name) for name in names_b
    }


def test_verification_report_keeps_candidate_and_report_digests_distinct(tmp_path: Path) -> None:
    report_path = tmp_path / "verification.json"
    _write_verification_report(report_path, {"manifest_sha256": "a" * 64, "verification_status": "pass"})
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["candidate_manifest_internal_sha256"] == "a" * 64
    assert report["manifest_sha256"] != "a" * 64
    assert verify_document_sha256(report)
