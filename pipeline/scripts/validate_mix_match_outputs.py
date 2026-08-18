"""Validate the final Menin hERG/PK mix-and-match analysis artifacts.

The checks here are deliberately independent of model fitting. They verify that
the persisted prediction matrices are complete, keyed uniquely, numerically
valid, internally anchored, and consistent with their published summaries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "research" / "reports" / "pk_herg" / "mix_match"
WORKBOOK = (
    ROOT / "research" / "outputs" / "pk_herg_professor_briefing" / "pk_herg_mix_match_professor_briefing.xlsx"
)


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORT_DIR / name)


def _read_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(REPORT_DIR / name)


def _empty_data_rows(name: str) -> int:
    try:
        frame = _read_csv(name)
    except pd.errors.EmptyDataError:
        return 0
    return len(frame)


def _record(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    observed: Any,
    requirement: str,
) -> None:
    checks.append(
        {
            "check": name,
            "passed": bool(passed),
            "observed": observed,
            "requirement": requirement,
        }
    )


def validate() -> dict[str, Any]:
    continuous = _read_parquet("herg_mix_match_predictions.parquet")
    binary = _read_parquet("herg_binary_predictions.parquet")
    pk = _read_parquet("pk_feature_model_predictions.parquet")
    continuous_summary = _read_csv("herg_mix_match_summary.csv")
    binary_summary = _read_csv("herg_binary_summary.csv")
    pk_summary = _read_csv("pk_feature_model_summary.csv")
    inventory = _read_csv("experiment_inventory.csv")
    model_registry = _read_csv("model_registry.csv")

    checks: list[dict[str, Any]] = []

    continuous_key = [
        "evaluation",
        "fold",
        "record_id",
        "data_regime",
        "feature_layer",
        "model",
    ]
    binary_key = [
        "evaluation",
        "fold",
        "record_id",
        "data_regime",
        "feature_layer",
        "model",
    ]
    pk_key = ["endpoint", "fold", "record_id", "feature_layer", "model"]

    for name, frame, key in (
        ("continuous prediction key", continuous, continuous_key),
        ("binary prediction key", binary, binary_key),
        ("PK prediction key", pk, pk_key),
    ):
        duplicate_count = int(frame.duplicated(key, keep=False).sum())
        _record(
            checks,
            name,
            duplicate_count == 0,
            duplicate_count,
            "zero duplicate prediction keys",
        )

    continuous_finite = bool(np.isfinite(continuous[["observed_pic50", "predicted_pic50"]]).all().all())
    binary_finite = bool(np.isfinite(binary[["observed_class", "predicted_blocker_probability"]]).all().all())
    pk_finite = bool(np.isfinite(pk[["observed_log10", "predicted_log10"]]).all().all())
    _record(
        checks,
        "continuous values finite",
        continuous_finite,
        continuous_finite,
        "all observed and predicted pIC50 values finite",
    )
    _record(
        checks,
        "binary values finite",
        binary_finite,
        binary_finite,
        "all labels and probabilities finite",
    )
    _record(
        checks,
        "PK values finite",
        pk_finite,
        pk_finite,
        "all observed and predicted log endpoints finite",
    )

    probability_in_range = bool(binary["predicted_blocker_probability"].between(0.0, 1.0).all())
    _record(
        checks,
        "binary probabilities bounded",
        probability_in_range,
        [
            float(binary["predicted_blocker_probability"].min()),
            float(binary["predicted_blocker_probability"].max()),
        ],
        "every blocker probability lies in [0, 1]",
    )

    internal_anchor_present = bool(
        (continuous["training_internal_like_structures"] > 0).all()
        and (binary["training_internal_like_structures"] > 0).all()
    )
    _record(
        checks,
        "internal evidence in every hERG fit",
        internal_anchor_present,
        internal_anchor_present,
        "all candidate hERG training regimes include internal-like evidence",
    )

    internal_continuous = continuous[continuous["evaluation"].eq("internal_scaffold_cv")]
    internal_binary = binary[binary["evaluation"].eq("internal_scaffold_cv")]
    exact_train_match_count = int(
        internal_continuous["max_internal_like_train_tanimoto"].ge(1.0).sum()
        + internal_binary["max_internal_like_train_tanimoto"].ge(1.0).sum()
    )
    _record(
        checks,
        "no exact internal train/test structure matches in scaffold CV",
        exact_train_match_count == 0,
        exact_train_match_count,
        "zero held-out internal predictions have Tanimoto 1.0 to internal training",
    )

    expected_inventory = len(continuous_summary) + len(binary_summary) + len(pk_summary)
    _record(
        checks,
        "experiment inventory complete",
        len(inventory) == expected_inventory,
        {
            "inventory_rows": len(inventory),
            "summary_rows": expected_inventory,
        },
        "inventory row count equals all persisted model-summary rows",
    )

    continuous_groups = continuous.groupby(
        ["evaluation", "data_regime", "feature_layer", "model"], dropna=False
    ).ngroups
    binary_groups = binary.groupby(
        ["evaluation", "data_regime", "feature_layer", "model"], dropna=False
    ).ngroups
    pk_groups = pk.groupby(["endpoint", "feature_layer", "model"], dropna=False).ngroups
    _record(
        checks,
        "continuous summary/prediction groups agree",
        continuous_groups == len(continuous_summary),
        [continuous_groups, len(continuous_summary)],
        "one summary row per continuous prediction group",
    )
    _record(
        checks,
        "binary summary/prediction groups agree",
        binary_groups == len(binary_summary),
        [binary_groups, len(binary_summary)],
        "one summary row per binary prediction group",
    )
    _record(
        checks,
        "PK summary/prediction groups agree",
        pk_groups == len(pk_summary),
        [pk_groups, len(pk_summary)],
        "one summary row per PK prediction group",
    )

    _record(
        checks,
        "continuous fit-failure ledger empty",
        _empty_data_rows("herg_fit_failures.csv") == 0,
        _empty_data_rows("herg_fit_failures.csv"),
        "zero recorded continuous fit failures",
    )
    _record(
        checks,
        "binary fit-failure ledger empty",
        _empty_data_rows("herg_binary_failure_ledger.csv") == 0,
        _empty_data_rows("herg_binary_failure_ledger.csv"),
        "zero recorded binary fit failures",
    )

    _record(
        checks,
        "bounded canonical model registry",
        len(model_registry) <= 10,
        len(model_registry),
        "no more than 10 retained canonical model roles",
    )
    _record(
        checks,
        "professor workbook present",
        WORKBOOK.exists() and WORKBOOK.stat().st_size > 0,
        {
            "exists": WORKBOOK.exists(),
            "bytes": WORKBOOK.stat().st_size if WORKBOOK.exists() else 0,
        },
        "nonempty final workbook exists",
    )

    passed = all(check["passed"] for check in checks)
    return {
        "status": "passed" if passed else "failed",
        "checks_passed": sum(check["passed"] for check in checks),
        "checks_total": len(checks),
        "prediction_rows": {
            "continuous_hERG": len(continuous),
            "binary_hERG": len(binary),
            "rat_PK": len(pk),
        },
        "summary_rows": {
            "continuous_hERG": len(continuous_summary),
            "binary_hERG": len(binary_summary),
            "rat_PK": len(pk_summary),
            "all_experiments": len(inventory),
        },
        "checks": checks,
    }


def write_report(result: dict[str, Any]) -> None:
    json_path = REPORT_DIR / "final_validation_summary.json"
    markdown_path = REPORT_DIR / "final_validation_report.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# Final mix-and-match validation",
        "",
        f"**Status:** {result['status'].upper()} ({result['checks_passed']}/{result['checks_total']} checks)",
        "",
        "## Persisted scale",
        "",
        "| Artifact | Rows |",
        "|---|---:|",
        *[
            f"| {name.replace('_', ' ')} predictions | {count:,} |"
            for name, count in result["prediction_rows"].items()
        ],
        *[
            f"| {name.replace('_', ' ')} summaries | {count:,} |"
            for name, count in result["summary_rows"].items()
        ],
        "",
        "## Checks",
        "",
        "| Check | Result | Observed | Requirement |",
        "|---|---|---|---|",
    ]
    for check in result["checks"]:
        observed = json.dumps(check["observed"], sort_keys=True)
        lines.append(
            f"| {check['check']} | "
            f"{'PASS' if check['passed'] else 'FAIL'} | "
            f"`{observed}` | {check['requirement']} |"
        )
    lines.extend(
        [
            "",
            "This validates persisted artifact integrity and split-level structure "
            "isolation. It does not convert retrospective evidence into prospective "
            "or unrelated-scaffold validation.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines))


def main() -> None:
    result = validate()
    write_report(result)
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
