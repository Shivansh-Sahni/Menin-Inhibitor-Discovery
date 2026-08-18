from __future__ import annotations

import math

import pytest
from menin_discovery.platform_qt_exposure_prep import (
    NO_LABEL_SEMANTICS,
    PositiveInterval,
    _correction_status,
    _margin_contract,
    _text_evidence,
    concentration_to_micromolar,
    fraction_unbound_interval,
    multiply_positive_intervals,
    pic50_interval_to_ic50_micromolar,
    quotient_positive_intervals,
    relation_interval,
)


def test_clinical_text_is_preserved_as_candidate_not_adjudicated_dose() -> None:
    evidence = _text_evidence(
        {
            "intervention_candidate_id": "CTI-1",
            "intervention_name": "Example drug",
            "intervention_description": "Participants received 25 mg orally twice daily.",
            "raw_json_pointer": "/studies/0/interventions/0",
            "source_page_path": "pages/page_000001.json",
            "source_page_sha256": "a" * 64,
        }
    )
    assert evidence["dose"][0]["matched_text"] == "25 mg"
    assert evidence["route"][0]["normalized_route_candidate"] == "oral"
    assert evidence["regimen"][0]["matched_text"].casefold() == "twice daily"
    assert "adjudicated" not in evidence["dose"][0]


def test_concentration_conversion_requires_analyte_molecular_weight_for_mass_units() -> None:
    assert concentration_to_micromolar(1_000.0, "nM") == pytest.approx(1.0)
    assert concentration_to_micromolar(100.0, "ng/mL", analyte_molecular_weight_g_mol=500.0) == pytest.approx(
        0.2
    )
    with pytest.raises(ValueError, match="molecular weight"):
        concentration_to_micromolar(100.0, "ng/mL")


def test_censoring_propagates_through_unbound_cmax_and_margin() -> None:
    ic50 = relation_interval(2.0, ">=")
    total_cmax = relation_interval(0.5, "=")
    fu = fraction_unbound_interval(10.0, "%")
    unbound_cmax = multiply_positive_intervals(total_cmax, fu)
    margin = quotient_positive_intervals(ic50, unbound_cmax)
    assert unbound_cmax == PositiveInterval(0.05, 0.05)
    assert margin.lower == pytest.approx(40.0)
    assert margin.upper is None


def test_censored_fraction_unbound_uses_natural_unit_upper_bound() -> None:
    assert fraction_unbound_interval(10.0, "%", ">") == PositiveInterval(
        0.1, 1.0, lower_inclusive=False, upper_inclusive=True
    )
    with pytest.raises(ValueError, match="empty"):
        fraction_unbound_interval(100.0, "%", ">")


def test_pic50_bound_conversion_reverses_bounds() -> None:
    interval = pic50_interval_to_ic50_micromolar(6.0, 7.0)
    assert interval.lower == pytest.approx(0.1)
    assert interval.upper == pytest.approx(1.0)
    assert math.isclose(interval.lower or 0.0, 10**-1)


def test_margin_contract_blocks_candidate_only_inputs_and_imputation() -> None:
    contract = _margin_contract()
    assert contract["label_policy"] == NO_LABEL_SEMANTICS
    assert "No PK" in contract["imputation_policy"]
    assert any("candidate-only" in value for value in contract["hard_blocks"])
    assert contract["missing_input_output"]["margin_point"] is None


def test_qt_correction_status_preserves_unresolved_context() -> None:
    assert _correction_status(["unresolved"]) == "all_endpoints_unresolved"
    assert _correction_status(["QTcF", "unresolved"]) == "mixed_resolved_and_unresolved"
    assert _correction_status(["QTcF", "QTcB"]) == "all_endpoints_have_explicit_correction_method"
