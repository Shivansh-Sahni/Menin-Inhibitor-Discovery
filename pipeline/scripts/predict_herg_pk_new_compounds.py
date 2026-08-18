#!/usr/bin/env python3
"""Run the guarded hERG and rat-PK interfaces for a blinded Menin batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline/src"))

from menin_discovery.research_common import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
)
from predict_herg_new_compounds import predict as predict_herg  # noqa: E402
from predict_pk_new_compounds import predict as predict_pk  # noqa: E402

DEFAULT_OUTPUT = ROOT / "research/reports/pk_herg/new_compound_predictions/integrated"


def _ic50_um(pic50: float) -> float:
    return float(np.power(10.0, 6.0 - pic50))


def _integrated_status(row: pd.Series, pk: pd.DataFrame) -> str:
    if row["prediction_eligibility"] != "eligible_same_series_discovery_hypothesis":
        return "withheld_hERG_scope_or_domain_gate"
    pk_core = pk[
        pk["endpoint"].isin(
            [
                "iv_auc_dose_normalized",
                "po_auc_dose_normalized",
                "po_cmax_dose_normalized",
                "vdss",
            ]
        )
    ]
    if not pk_core["prediction_status"].eq("eligible_same_series_discovery_hypothesis").all():
        return "hERG_available_PK_partly_withheld"
    if row["threshold_interval_status"] == "envelope_crosses_threshold":
        return "prioritize_hERG_measurement; PK_hypotheses_available"
    return "same_series_discovery_hypotheses_available; experimental_confirmation_required"


def _report(
    herg: pd.DataFrame,
    pk: pd.DataFrame,
    derived: pd.DataFrame,
) -> str:
    lines = [
        "# Integrated Menin-inhibitor hERG + rat-PK prediction",
        "",
        "This is the readout to discuss with Angelo for a blinded, similar-series "
        "structure. It does not predict Menin potency and is not an optimization score.",
        "",
    ]
    for row in herg.itertuples(index=False):
        compound_pk = pk[pk["compound_id"].eq(row.compound_id)]
        compound_derived = derived[derived["compound_id"].eq(row.compound_id)]
        point_min = float(row.retained_model_pic50_min)
        point_max = float(row.retained_model_pic50_max)
        envelope_low = float(row.conservative_model_envelope_pic50_lower)
        envelope_high = float(row.conservative_model_envelope_pic50_upper)
        status = _integrated_status(
            pd.Series(row._asdict()),
            compound_pk,
        )
        lines.extend(
            [
                f"## {row.compound_id}",
                "",
                "### hERG",
                "",
                f"- Retained model range: **pIC50 {point_min:.2f}–{point_max:.2f}** "
                f"(approximately IC50 {_ic50_um(point_max):.2g}–{_ic50_um(point_min):.2g} µM).",
                f"- Conservative residual envelope: **pIC50 {envelope_low:.2f}–{envelope_high:.2f}** "
                f"(approximately IC50 {_ic50_um(envelope_high):.2g}–{_ic50_um(envelope_low):.2g} µM).",
                f"- Nearest measured similarity: **{row.nearest_training_tanimoto:.3f}**; "
                f"{int(row.neighbor_structures_ge_0p80)} measured neighbors at Tanimoto ≥0.80.",
                f"- Decision: `{row.decision_status}`.",
                "",
                "### Rat PK",
                "",
                "| Endpoint | Primary estimate | Model range | Empirical 90% envelope | Unit | Status |",
                "|---|---:|---:|---:|---|---|",
            ]
        )
        for record in compound_pk.itertuples(index=False):
            lines.append(
                f"| {record.endpoint} | {record.primary_predicted_value:.3g} | "
                f"{record.model_point_min:.3g}–{record.model_point_max:.3g} | "
                f"{record.conservative_90pct_lower:.3g}–{record.conservative_90pct_upper:.3g} | "
                f"{record.unit} | {record.prediction_status} |"
            )
        lines.extend(["", "Derived, non-independent closure checks:", ""])
        for record in compound_derived.itertuples(index=False):
            lines.append(
                f"- {record.derived_endpoint}: {record.predicted_value:.3g} "
                f"{record.unit} (envelope {record.conservative_90pct_lower:.3g}–"
                f"{record.conservative_90pct_upper:.3g}); `{record.status}`."
            )
        lines.extend(
            [
                "",
                f"**Integrated status:** `{status}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "The hERG result is a same-series assay-prioritization hypothesis. PK is "
            "weaker: it is trained on 46 internal structures and summary parameters, "
            "has no raw concentration-time profiles, and cannot identify Fa, Fg, Fh, "
            "distribution, or clearance mechanisms separately. Tmax is reported but "
            "withheld from ranking because its held-scaffold rank correlation is near zero.",
            "",
        ]
    )
    return "\n".join(lines)


def predict(input_path: Path, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    herg_output = output / "herg"
    pk_output = output / "pk"
    herg = predict_herg(input_path, herg_output)
    pk, derived = predict_pk(input_path, pk_output)
    summaries: list[dict[str, object]] = []
    for row in herg.itertuples(index=False):
        compound_pk = pk[pk["compound_id"].eq(row.compound_id)]
        summaries.append(
            {
                "compound_id": row.compound_id,
                "hERG_pic50_model_min": row.retained_model_pic50_min,
                "hERG_pic50_model_max": row.retained_model_pic50_max,
                "hERG_conservative_pic50_lower": row.conservative_model_envelope_pic50_lower,
                "hERG_conservative_pic50_upper": row.conservative_model_envelope_pic50_upper,
                "hERG_blocker_probability_min": row.retained_blocker_probability_min,
                "hERG_blocker_probability_max": row.retained_blocker_probability_max,
                "hERG_nearest_tanimoto": row.nearest_training_tanimoto,
                "hERG_decision_status": row.decision_status,
                "pk_eligible_core_endpoints": int(
                    compound_pk[~compound_pk["endpoint"].eq("po_tmax")]["prediction_status"]
                    .eq("eligible_same_series_discovery_hypothesis")
                    .sum()
                ),
                "integrated_status": _integrated_status(pd.Series(row._asdict()), compound_pk),
            }
        )
    summary = pd.DataFrame(summaries)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output / "integrated_summary.csv", summary)
    atomic_write_text(
        output / "integrated_prediction_report.md",
        _report(herg, pk, derived),
    )
    atomic_write_json(
        output / "integrated_prediction_summary.json",
        {
            "compounds": len(summary),
            "hERG_target": "liability_for_Menin_inhibitors_not_Menin_potency",
            "rat_PK_core_endpoints": 4,
            "prospective_status": "discovery_only_until_protocol_matched_outcomes",
            "results": summary.to_dict(orient="records"),
        },
    )
    return summary, pk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, pk = predict(args.input, args.output)
    print(
        json.dumps(
            {
                "compounds": len(summary),
                "pk_endpoint_rows": len(pk),
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
