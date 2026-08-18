import json

import pandas as pd
from menin_discovery.quality import (
    QualityConfig,
    audit_menin_table,
    audit_pk_table,
    write_quality_outputs,
)


def _menin_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["source-a"] * 5,
            "source_record_id": ["r1", "r1", "r3", "r4", "r5"],
            "compound_id": ["cmp-1", "cmp-1", "cmp-2", "cmp-3", "cmp-4"],
            "smiles": ["CC", "CC", "CCC", "CCCC", "CCO"],
            "target_name": ["Menin", "Menin", "Other protein", "Menin", ""],
            "target_id": ["CHEMBL1615381", "CHEMBL1615381", "OTHER", "O00255,Q03164", ""],
            "endpoint": ["IC50"] * 5,
            "relation": ["=", "=", "=", "=", "="],
            "value_raw": [10, 2_000, 100, -1, "not measured"],
            "standard_units": ["nM", "nM", "furlong", "", "nM"],
            "assay_description": ["binding", "binding", "binding", "unknown origin", ""],
            "assay_type": ["B", "B", "B", "B", ""],
            "document_year": [2020, 2020, 1492, 2022, 2022],
        }
    )


def test_activity_audit_finds_schema_values_identity_and_conflicts(tmp_path):
    report = audit_menin_table(_menin_frame(), generated_at="2026-01-02T03:04:05Z")
    codes = {finding.code for finding in report.findings}

    assert {
        "ambiguous_target",
        "conflicting_identifier",
        "conflicting_repeated_measurement",
        "duplicate_identifier",
        "incompatible_unit",
        "invalid_numeric_value",
        "missing_assay_description",
        "missing_assay_type",
        "missing_target",
        "missing_unit",
        "nonpositive_value",
        "unexpected_target",
        "unknown_unit",
        "value_out_of_range",
    }.issubset(codes)
    assert not report.passed
    assert report.row_count == 5
    repeated = report.summary_frame().query("code == 'conflicting_repeated_measurement'").iloc[0]
    assert repeated["affected_row_count"] == 2

    paths = write_quality_outputs(report, tmp_path)
    assert set(paths) == {"json", "findings_csv", "summary_csv"}
    payload = json.loads(paths["json"].read_text())
    findings = pd.read_csv(paths["findings_csv"])
    summary = pd.read_csv(paths["summary_csv"])
    assert payload["generated_at"] == "2026-01-02T03:04:05Z"
    assert len(findings) == report.finding_count
    assert "unknown_unit" in set(summary["code"])


def test_missing_required_column_is_table_level_finding():
    report = audit_menin_table(_menin_frame().drop(columns=["assay_type"]))
    finding = next(
        finding
        for finding in report.findings
        if finding.code == "missing_required_column" and finding.column == "assay_type"
    )
    assert finding.scope == "table"
    assert finding.row_number is None


def test_conflict_threshold_is_adjustable():
    default = audit_menin_table(_menin_frame())
    strict = audit_menin_table(_menin_frame(), config=QualityConfig(conflict_log10_threshold=3.0))
    assert any(finding.code == "conflicting_repeated_measurement" for finding in default.findings)
    assert not any(finding.code == "conflicting_repeated_measurement" for finding in strict.findings)


def test_pk_checks_are_endpoint_aware():
    data = pd.DataFrame(
        {
            "source": ["ChEMBL"] * 3,
            "activity_id": [1, 2, 3],
            "molecule_chembl_id": ["A", "B", "C"],
            "smiles": ["CC", "CCC", "CCCC"],
            "standard_type": ["LogD7.4", "CL", "IC50"],
            "standard_relation": ["=", "=", "="],
            "standard_value": [-1.2, -5.0, 10.0],
            "standard_units": ["", "mL.min-1.kg-1", "furlong"],
            "assay_description": ["log distribution", "clearance", "binding"],
            "assay_type": ["P", "A", "B"],
            "target_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
            "target_pref_name": ["system 1", "system 2", "target 3"],
        }
    )
    report = audit_pk_table(data)
    row_codes = {(finding.row_number, finding.code) for finding in report.findings}

    # Signed LogD is valid and unitless; clearance must be positive; IC50 must
    # have a recognized concentration unit.
    assert (0, "nonpositive_value") not in row_codes
    assert (0, "missing_unit") not in row_codes
    assert (1, "nonpositive_value") in row_codes
    assert (2, "unknown_unit") in row_codes
    assert (2, "incompatible_unit") in row_codes


def test_audit_does_not_mutate_input():
    data = _menin_frame()
    before = data.copy(deep=True)
    audit_menin_table(data)
    pd.testing.assert_frame_equal(data, before)


def test_recorded_target_relevance_supports_rows_without_direct_target_fields():
    data = _menin_frame().iloc[[4]].copy()
    data["is_target_relevant"] = True
    data["target_relevance"] = "confirmed_target_gene_lookup"
    data["target_relevance_reason"] = "PubChem target-gene lookup matched MEN1"
    report = audit_menin_table(data)
    codes = {finding.code for finding in report.findings}
    assert "missing_target" not in codes
    assert "unexpected_target" not in codes
    assert "target_identity_derived" in codes
