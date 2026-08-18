"""Data-quality auditing for Menin, hERG, and PK/ADMET tables.

The functions in this module do not mutate their inputs.  They produce both
row-level findings (suitable for review or quarantine) and deterministic
aggregate summaries (suitable for build gates and publication supplements).
The defaults reflect the project's current long-form activity and PK schemas,
but :class:`QualityConfig` makes thresholds and unit vocabularies adjustable.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

TableKind = Literal["menin", "herg", "pk"]
Severity = Literal["info", "warning", "error"]

ACTIVITY_REQUIRED_COLUMNS = (
    "source",
    "source_record_id",
    "compound_id",
    "smiles",
    "target_name",
    "target_id",
    "endpoint",
    "relation",
    "value_raw",
    "standard_units",
    "assay_description",
    "assay_type",
)

PK_REQUIRED_COLUMNS = (
    "source",
    "activity_id",
    "molecule_chembl_id",
    "smiles",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "assay_description",
    "assay_type",
    "target_chembl_id",
    "target_pref_name",
)

CONCENTRATION_ENDPOINTS = frozenset(
    {
        "ac50",
        "ec50",
        "gi50",
        "ic20",
        "ic25",
        "ic50",
        "ic90",
        "kd",
        "ki",
        "km",
        "mic",
        "potency",
        "solubility",
        "xc50",
    }
)

# Values for these endpoints may legitimately be zero or negative.
SIGNED_OR_ZERO_ALLOWED_ENDPOINTS = frozenset(
    {
        "% ctrl",
        "% maximum response",
        "activity",
        "fici",
        "inhibition",
        "inh",
        "log ic50",
        "logd",
        "logd7.4",
        "logp",
        "percent effect",
        "pic50",
        "pka",
        "pka_b1",
        "pki",
        "ratio",
        "ratio auc",
        "ratio ic50",
        "ratio ki",
        "ratio_papp",
    }
)

POSITIVE_PK_ENDPOINTS = frozenset(
    {
        "auc",
        "cl",
        "cl_renal",
        "clint",
        "cmax",
        "kcat",
        "kcat/km",
        "km",
        "mrt",
        "papp",
        "peff",
        "permeability",
        "solubility",
        "t1/2",
        "tmax",
        "vdss",
    }
)

UNITLESS_ENDPOINTS = frozenset(
    {
        "fici",
        "log ic50",
        "logd",
        "logd7.4",
        "logp",
        "pic50",
        "pka",
        "pka_b1",
        "pki",
        "ratio",
        "ratio auc",
        "ratio ic50",
        "ratio ki",
        "ratio_papp",
    }
)

# Unit matching uses ``_unit_token`` below, which folds case, whitespace and
# common micro symbols while retaining punctuation needed to distinguish rates.
DEFAULT_KNOWN_UNITS = frozenset(
    {
        "%",
        "/min",
        "/nm/min",
        "/s",
        "1/min",
        "1/s",
        "10'-4/nm/min",
        "10'-4/s",
        "10'-5/nm/min",
        "10'-6 cm/s",
        "10'-7/nm/min",
        "10'5/m/s",
        "10'5/s/m",
        "10^-6 cm/s",
        "cm/s",
        "day",
        "degrees c",
        "g/l",
        "h",
        "hr",
        "l.kg-1",
        "l/kg",
        "log10cfu/ml",
        "m",
        "mg/l",
        "mg/ml",
        "min",
        "ml.min-1.g-1",
        "ml.min-1.kg-1",
        "ml/min/g",
        "ml/min/kg",
        "mm",
        "mm3",
        "ms",
        "mv",
        "nanomolar",
        "ng.hr.ml-1",
        "ng/ml",
        "nm",
        "nm/s",
        "picomolar",
        "pm",
        "s",
        "s-1",
        "ucm",
        "ucm/s",
        "ug ml-1",
        "ug.ml-1",
        "ug/ml",
        "ul.min-1.(10^6cells)-1",
        "ul/min/10^6 cells",
        "um",
    }
)

CONCENTRATION_UNITS_TO_NM = {
    "m": 1_000_000_000.0,
    "millimolar": 1_000_000.0,
    "mm": 1_000_000.0,
    "micromolar": 1_000.0,
    "um": 1_000.0,
    "nanomolar": 1.0,
    "nm": 1.0,
    "picomolar": 0.001,
    "pm": 0.001,
}

MASS_CONCENTRATION_UNITS = frozenset({"g/l", "mg/l", "mg/ml", "ng/ml", "ug ml-1", "ug.ml-1", "ug/ml"})

RELATIONS = frozenset({"", "=", "<", "<=", ">", ">=", "~"})
AMBIGUOUS_ASSAY_TERMS = (
    "not specified",
    "unknown assay",
    "unknown origin",
    "unspecified",
)


@dataclass(frozen=True)
class QualityConfig:
    """Adjustable validation policy used by :func:`audit_table`."""

    known_units: frozenset[str] = DEFAULT_KNOWN_UNITS
    concentration_units_to_nm: Mapping[str, float] = field(
        default_factory=lambda: dict(CONCENTRATION_UNITS_TO_NM)
    )
    conflict_log10_threshold: float = 1.0
    p_value_min: float = 0.0
    p_value_max: float = 14.0
    earliest_document_year: int = 1800
    latest_document_year: int = field(default_factory=lambda: datetime.now().year + 1)
    require_assay_description: bool = True
    require_assay_type: bool = True
    expected_target_ids: Mapping[str, frozenset[str]] = field(
        default_factory=lambda: {
            "menin": frozenset({"CHEMBL1615381", "O00255"}),
            "herg": frozenset({"CHEMBL240", "Q12809"}),
        }
    )
    expected_target_terms: Mapping[str, frozenset[str]] = field(
        default_factory=lambda: {
            "menin": frozenset({"menin", "men1"}),
            "herg": frozenset({"herg", "kcnh2", "kv11.1"}),
        }
    )

    def __post_init__(self) -> None:
        if self.conflict_log10_threshold < 0:
            raise ValueError("conflict_log10_threshold must be non-negative")
        if self.p_value_min >= self.p_value_max:
            raise ValueError("p_value_min must be smaller than p_value_max")
        object.__setattr__(self, "known_units", frozenset(_unit_token(x) for x in self.known_units))
        object.__setattr__(
            self,
            "concentration_units_to_nm",
            {_unit_token(k): float(v) for k, v in self.concentration_units_to_nm.items()},
        )


@dataclass(frozen=True)
class QualityFinding:
    """One row-, group-, or table-level quality finding."""

    table: str
    code: str
    severity: Severity
    scope: Literal["row", "group", "table"]
    message: str
    row_number: int | None = None
    column: str | None = None
    identifier: str | None = None
    value: Any = None
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["value"] = _json_value(result["value"])
        result["context"] = {str(k): _json_value(v) for k, v in self.context.items()}
        return result


@dataclass
class QualityReport:
    """Audit result containing detailed findings and aggregate summaries."""

    table: str
    row_count: int
    findings: list[QualityFinding] = field(default_factory=list)
    columns: tuple[str, ...] = ()
    generated_at: str = field(default_factory=lambda: _utc_timestamp())

    @property
    def passed(self) -> bool:
        """Whether the report contains no error-severity findings."""

        return not any(finding.severity == "error" for finding in self.findings)

    @property
    def finding_count(self) -> int:
        return len(self.findings)

    def findings_frame(self) -> pd.DataFrame:
        columns = [
            "table",
            "code",
            "severity",
            "scope",
            "message",
            "row_number",
            "column",
            "identifier",
            "value",
            "context",
        ]
        records = []
        for finding in self.findings:
            record = finding.to_dict()
            record["context"] = json.dumps(record["context"], sort_keys=True, separators=(",", ":"))
            records.append(record)
        return pd.DataFrame(records, columns=columns)

    def summary_frame(self) -> pd.DataFrame:
        """Aggregate findings by severity/code/scope in stable order."""

        grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
        for finding in self.findings:
            key = (finding.severity, finding.code, finding.scope)
            item = grouped.setdefault(
                key,
                {
                    "table": self.table,
                    "severity": finding.severity,
                    "code": finding.code,
                    "scope": finding.scope,
                    "finding_count": 0,
                    "affected_row_count": 0,
                },
            )
            item["finding_count"] += 1
            if finding.row_number is not None:
                item.setdefault("_rows", set()).add(finding.row_number)
        records = []
        severity_order = {"error": 0, "warning": 1, "info": 2}
        for item in grouped.values():
            rows = item.pop("_rows", set())
            item["affected_row_count"] = len(rows)
            records.append(item)
        records.sort(key=lambda item: (severity_order[item["severity"]], item["code"], item["scope"]))
        return pd.DataFrame(
            records,
            columns=[
                "table",
                "severity",
                "code",
                "scope",
                "finding_count",
                "affected_row_count",
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        severity_counts = Counter(finding.severity for finding in self.findings)
        return {
            "quality_report_version": "1.0",
            "table": self.table,
            "generated_at": self.generated_at,
            "row_count": self.row_count,
            "columns": list(self.columns),
            "passed": self.passed,
            "finding_count": self.finding_count,
            "severity_counts": {
                severity: int(severity_counts.get(severity, 0)) for severity in ("error", "warning", "info")
            },
            "summary": self.summary_frame().to_dict(orient="records"),
            "findings": [finding.to_dict() for finding in self.findings],
        }

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return output

    def write_findings_csv(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.findings_frame().to_csv(output, index=False)
        return output

    def write_summary_csv(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.summary_frame().to_csv(output, index=False)
        return output


def audit_table(
    data: pd.DataFrame,
    table: TableKind,
    *,
    config: QualityConfig | None = None,
    generated_at: str | None = None,
) -> QualityReport:
    """Audit one Menin, hERG, or PK table without mutating it.

    Parameters
    ----------
    data:
        A long-form activity table (``menin``/``herg``) or PK observations.
    table:
        Selects the schema, target policy, and endpoint-aware range checks.
    config:
        Optional policy override for units and thresholds.
    generated_at:
        Optional ISO-8601 timestamp, useful for byte-reproducible reports.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if table not in {"menin", "herg", "pk"}:
        raise ValueError("table must be one of: menin, herg, pk")

    policy = config or QualityConfig()
    report = QualityReport(
        table=table,
        row_count=len(data),
        columns=tuple(str(column) for column in data.columns),
        generated_at=_utc_timestamp(generated_at),
    )
    required = ACTIVITY_REQUIRED_COLUMNS if table in {"menin", "herg"} else PK_REQUIRED_COLUMNS
    _check_required_columns(data, required, report)

    # Continue with every check whose input columns are present.  This yields a
    # complete repair list even when the table fails its schema gate.
    _check_required_values_and_types(data, table, report)
    _check_numeric_values_and_ranges(data, table, report, policy)
    _check_units(data, table, report, policy)
    _check_relations(data, table, report)
    _check_identifiers(data, table, report)
    _check_targets(data, table, report, policy)
    _check_assays(data, table, report, policy)
    _check_repeated_measurements(data, table, report, policy)
    _sort_findings(report)
    return report


def audit_menin_table(
    data: pd.DataFrame,
    *,
    config: QualityConfig | None = None,
    generated_at: str | None = None,
) -> QualityReport:
    return audit_table(data, "menin", config=config, generated_at=generated_at)


def audit_herg_table(
    data: pd.DataFrame,
    *,
    config: QualityConfig | None = None,
    generated_at: str | None = None,
) -> QualityReport:
    return audit_table(data, "herg", config=config, generated_at=generated_at)


def audit_pk_table(
    data: pd.DataFrame,
    *,
    config: QualityConfig | None = None,
    generated_at: str | None = None,
) -> QualityReport:
    return audit_table(data, "pk", config=config, generated_at=generated_at)


def audit_tables(
    *,
    menin: pd.DataFrame | None = None,
    herg: pd.DataFrame | None = None,
    pk: pd.DataFrame | None = None,
    config: QualityConfig | None = None,
    generated_at: str | None = None,
) -> dict[str, QualityReport]:
    """Audit any supplied project tables and return reports keyed by name."""

    supplied = {"menin": menin, "herg": herg, "pk": pk}
    return {
        name: audit_table(frame, name, config=config, generated_at=generated_at)  # type: ignore[arg-type]
        for name, frame in supplied.items()
        if frame is not None
    }


def write_quality_outputs(
    report: QualityReport,
    directory: str | Path,
    *,
    prefix: str | None = None,
) -> dict[str, Path]:
    """Write JSON, detailed CSV, and summary CSV artifacts for a report."""

    output_dir = Path(directory)
    stem = prefix or f"{report.table}_quality"
    return {
        "json": report.write_json(output_dir / f"{stem}.json"),
        "findings_csv": report.write_findings_csv(output_dir / f"{stem}_findings.csv"),
        "summary_csv": report.write_summary_csv(output_dir / f"{stem}_summary.csv"),
    }


def _check_required_columns(data: pd.DataFrame, required: Sequence[str], report: QualityReport) -> None:
    for column in required:
        if column not in data.columns:
            _add(
                report,
                code="missing_required_column",
                severity="error",
                scope="table",
                column=column,
                message=f"Required column {column!r} is absent.",
            )


def _check_required_values_and_types(data: pd.DataFrame, table: TableKind, report: QualityReport) -> None:
    if table in {"menin", "herg"}:
        required_values = ("source", "source_record_id", "compound_id", "endpoint")
        string_columns = (
            "source",
            "compound_id",
            "smiles",
            "target_name",
            "target_id",
            "endpoint",
            "relation",
            "standard_units",
            "assay_description",
            "assay_type",
        )
    else:
        required_values = ("source", "activity_id", "molecule_chembl_id", "standard_type")
        string_columns = (
            "source",
            "molecule_chembl_id",
            "smiles",
            "standard_type",
            "standard_relation",
            "standard_units",
            "assay_description",
            "assay_type",
            "target_chembl_id",
            "target_pref_name",
        )

    for column in required_values:
        if column not in data.columns:
            continue
        for row_number in np.flatnonzero(_missing_mask(data[column])):
            _add_row(
                report,
                data,
                int(row_number),
                code="missing_required_value",
                severity="error",
                column=column,
                message=f"Required value in {column!r} is missing.",
            )

    for column in string_columns:
        if column not in data.columns:
            continue
        invalid = data[column].map(
            lambda value: (
                not _is_missing(value) and not isinstance(value, (str, int, float, np.integer, np.floating))
            )
        )
        for row_number in np.flatnonzero(invalid):
            _add_row(
                report,
                data,
                int(row_number),
                code="invalid_column_type",
                severity="error",
                column=column,
                value=type(data.iloc[int(row_number)][column]).__name__,
                message=f"{column!r} must contain scalar text-compatible values.",
            )

    if "smiles" in data.columns:
        for row_number in np.flatnonzero(_missing_mask(data["smiles"])):
            _add_row(
                report,
                data,
                int(row_number),
                code="missing_structure",
                severity="error",
                column="smiles",
                message="No molecular structure is available for this observation.",
            )

    for column in ("is_exact", "is_core_endpoint"):
        if column not in data.columns:
            continue
        valid_values = {True, False, "true", "false", "True", "False", "0", "1"}
        invalid = data[column].map(
            lambda value, allowed=valid_values: (
                not _is_missing(value)
                and (not isinstance(value, (str, int, float, bool, np.number)) or value not in allowed)
            )
        )
        for row_number in np.flatnonzero(invalid):
            _add_row(
                report,
                data,
                int(row_number),
                code="invalid_column_type",
                severity="error",
                column=column,
                value=data.iloc[int(row_number)][column],
                message=f"{column!r} must be boolean-compatible.",
            )


def _check_numeric_values_and_ranges(
    data: pd.DataFrame,
    table: TableKind,
    report: QualityReport,
    policy: QualityConfig,
) -> None:
    value_column = "value_raw" if table in {"menin", "herg"} else "standard_value"
    endpoint_column = "endpoint" if table in {"menin", "herg"} else "standard_type"
    if value_column in data.columns:
        values = data[value_column].map(_parse_numeric)
        missing = _missing_mask(data[value_column])
        invalid = ~missing & ~np.isfinite(values)
        for row_number in np.flatnonzero(invalid):
            _add_row(
                report,
                data,
                int(row_number),
                code="invalid_numeric_value",
                severity="error",
                column=value_column,
                value=data.iloc[int(row_number)][value_column],
                message=f"{value_column!r} does not contain a finite numeric value.",
            )
        for row_number in np.flatnonzero(missing):
            _add_row(
                report,
                data,
                int(row_number),
                code="missing_numeric_value",
                severity="error",
                column=value_column,
                message=f"{value_column!r} is missing.",
            )

        if endpoint_column in data.columns:
            endpoints = data[endpoint_column].map(_endpoint_token)
            if table == "pk":
                requires_positive = endpoints.isin(POSITIVE_PK_ENDPOINTS | CONCENTRATION_ENDPOINTS)
            else:
                requires_positive = endpoints.isin(CONCENTRATION_ENDPOINTS)
            nonpositive = requires_positive & np.isfinite(values) & (values <= 0)
            for row_number in np.flatnonzero(nonpositive):
                _add_row(
                    report,
                    data,
                    int(row_number),
                    code="nonpositive_value",
                    severity="error",
                    column=value_column,
                    value=values.iloc[int(row_number)],
                    message="A concentration or positive-only endpoint has a nonpositive value.",
                )

    if "value_nm" in data.columns:
        normalized = pd.to_numeric(data["value_nm"], errors="coerce")
        invalid = ~_missing_mask(data["value_nm"]) & ~np.isfinite(normalized)
        nonpositive = np.isfinite(normalized) & (normalized <= 0)
        for row_number in np.flatnonzero(invalid):
            _add_row(
                report,
                data,
                int(row_number),
                code="invalid_numeric_value",
                severity="error",
                column="value_nm",
                value=data.iloc[int(row_number)]["value_nm"],
                message="Normalized nM value must be finite numeric data.",
            )
        for row_number in np.flatnonzero(nonpositive):
            _add_row(
                report,
                data,
                int(row_number),
                code="nonpositive_value",
                severity="error",
                column="value_nm",
                value=normalized.iloc[int(row_number)],
                message="Normalized nM value must be positive.",
            )

    if "p_value" in data.columns:
        p_values = pd.to_numeric(data["p_value"], errors="coerce")
        invalid = ~_missing_mask(data["p_value"]) & ~np.isfinite(p_values)
        outside = np.isfinite(p_values) & ((p_values < policy.p_value_min) | (p_values > policy.p_value_max))
        for row_number in np.flatnonzero(invalid):
            _add_row(
                report,
                data,
                int(row_number),
                code="invalid_numeric_value",
                severity="error",
                column="p_value",
                value=data.iloc[int(row_number)]["p_value"],
                message="p-value must be finite numeric data when present.",
            )
        for row_number in np.flatnonzero(outside):
            _add_row(
                report,
                data,
                int(row_number),
                code="value_out_of_range",
                severity="warning",
                column="p_value",
                value=p_values.iloc[int(row_number)],
                message=(
                    f"p-value is outside the configured range [{policy.p_value_min}, {policy.p_value_max}]."
                ),
            )

    if "document_year" in data.columns:
        years = pd.to_numeric(data["document_year"], errors="coerce")
        present = ~_missing_mask(data["document_year"])
        invalid = present & (~np.isfinite(years) | (years % 1 != 0))
        outside = np.isfinite(years) & (
            (years < policy.earliest_document_year) | (years > policy.latest_document_year)
        )
        for row_number in np.flatnonzero(invalid):
            _add_row(
                report,
                data,
                int(row_number),
                code="invalid_document_year",
                severity="warning",
                column="document_year",
                value=data.iloc[int(row_number)]["document_year"],
                message="Document year must be a whole number when present.",
            )
        for row_number in np.flatnonzero(outside & ~invalid):
            _add_row(
                report,
                data,
                int(row_number),
                code="value_out_of_range",
                severity="warning",
                column="document_year",
                value=years.iloc[int(row_number)],
                message="Document year is outside the configured plausible range.",
            )

    if table == "pk" and endpoint_column in data.columns and value_column in data.columns:
        values = data[value_column].map(_parse_numeric)
        endpoints = data[endpoint_column].map(_endpoint_token)
        for row_number, endpoint in enumerate(endpoints):
            value = values.iloc[row_number]
            if not np.isfinite(value):
                continue
            if endpoint in {"logd", "logd7.4", "logp"} and not -10 <= value <= 15:
                _add_row(
                    report,
                    data,
                    row_number,
                    code="value_out_of_range",
                    severity="warning",
                    column=value_column,
                    value=value,
                    message="LogP/LogD is outside the broad plausible range [-10, 15].",
                )
            if endpoint.startswith("pka") and not 0 <= value <= 14:
                _add_row(
                    report,
                    data,
                    row_number,
                    code="value_out_of_range",
                    severity="warning",
                    column=value_column,
                    value=value,
                    message="pKa is outside the broad aqueous range [0, 14].",
                )


def _check_units(
    data: pd.DataFrame,
    table: TableKind,
    report: QualityReport,
    policy: QualityConfig,
) -> None:
    if "standard_units" not in data.columns:
        return
    endpoint_column = "endpoint" if table in {"menin", "herg"} else "standard_type"
    endpoints = (
        data[endpoint_column].map(_endpoint_token)
        if endpoint_column in data.columns
        else pd.Series("", index=data.index)
    )
    missing = _missing_mask(data["standard_units"])
    unitless = endpoints.isin(UNITLESS_ENDPOINTS)
    for row_number in np.flatnonzero(missing & ~unitless):
        _add_row(
            report,
            data,
            int(row_number),
            code="missing_unit",
            severity="error" if endpoints.iloc[int(row_number)] in CONCENTRATION_ENDPOINTS else "warning",
            column="standard_units",
            message="A numeric assay result is missing its unit.",
        )

    tokens = data["standard_units"].map(_unit_token)
    unknown = ~missing & ~tokens.isin(policy.known_units)
    for row_number in np.flatnonzero(unknown):
        _add_row(
            report,
            data,
            int(row_number),
            code="unknown_unit",
            severity="error" if endpoints.iloc[int(row_number)] in CONCENTRATION_ENDPOINTS else "warning",
            column="standard_units",
            value=data.iloc[int(row_number)]["standard_units"],
            message="Unit is not in the configured controlled vocabulary.",
        )

    concentration_rows = endpoints.isin(CONCENTRATION_ENDPOINTS) & ~missing
    valid_mass_solubility = endpoints.eq("solubility") & tokens.isin(MASS_CONCENTRATION_UNITS)
    incompatible = (
        concentration_rows & ~tokens.isin(policy.concentration_units_to_nm) & ~valid_mass_solubility
    )
    for row_number in np.flatnonzero(incompatible):
        _add_row(
            report,
            data,
            int(row_number),
            code="incompatible_unit",
            severity="error",
            column="standard_units",
            value=data.iloc[int(row_number)]["standard_units"],
            message="Concentration endpoint does not use a recognized concentration unit.",
        )
    for row_number in np.flatnonzero(valid_mass_solubility):
        _add_row(
            report,
            data,
            int(row_number),
            code="mass_concentration_requires_molecular_weight",
            severity="info",
            column="standard_units",
            value=data.iloc[int(row_number)]["standard_units"],
            message=(
                "Mass-concentration solubility is valid but requires molecular weight before "
                "comparison with molar solubility."
            ),
        )


def _check_relations(data: pd.DataFrame, table: TableKind, report: QualityReport) -> None:
    column = "relation" if table in {"menin", "herg"} else "standard_relation"
    if column not in data.columns:
        return
    tokens = data[column].map(lambda value: "" if _is_missing(value) else str(value).strip())
    invalid = ~tokens.isin(RELATIONS)
    for row_number in np.flatnonzero(invalid):
        _add_row(
            report,
            data,
            int(row_number),
            code="unknown_relation",
            severity="warning",
            column=column,
            value=data.iloc[int(row_number)][column],
            message="Relation qualifier is not one of =, <, <=, >, >=, or ~.",
        )


def _check_identifiers(data: pd.DataFrame, table: TableKind, report: QualityReport) -> None:
    identifier_column = "source_record_id" if table in {"menin", "herg"} else "activity_id"
    if identifier_column not in data.columns:
        return
    source = data["source"].map(_text_token) if "source" in data.columns else pd.Series("", index=data.index)
    identifiers = data[identifier_column].map(_text_token)
    usable = identifiers != ""
    keys = pd.Series(list(zip(source, identifiers, strict=False)), index=data.index)
    duplicated = usable & keys.duplicated(keep=False)
    compare_columns = [
        column
        for column in (
            "compound_id",
            "molecule_chembl_id",
            "smiles",
            "endpoint",
            "standard_type",
            "value_raw",
            "standard_value",
            "standard_units",
            "target_id",
            "target_chembl_id",
        )
        if column in data.columns
    ]
    for row_number in np.flatnonzero(duplicated):
        _add_row(
            report,
            data,
            int(row_number),
            code="duplicate_identifier",
            severity="warning",
            column=identifier_column,
            value=data.iloc[int(row_number)][identifier_column],
            message="Source identifier occurs more than once within the same source.",
        )

    positions_by_key: dict[tuple[str, str], list[int]] = {}
    for position in np.flatnonzero(duplicated):
        positions_by_key.setdefault(keys.iloc[int(position)], []).append(int(position))
    for key, positions in positions_by_key.items():
        subset = data.iloc[positions]
        exact_duplicate = subset.astype(str).duplicated(keep=False)
        for offset in np.flatnonzero(exact_duplicate):
            row_number = positions[int(offset)]
            _add_row(
                report,
                data,
                row_number,
                code="exact_duplicate_row",
                severity="warning",
                column=identifier_column,
                message="The complete row is duplicated.",
            )
        conflicts = [
            column for column in compare_columns if subset[column].map(_text_token).nunique(dropna=False) > 1
        ]
        if conflicts:
            for row_number in positions:
                _add_row(
                    report,
                    data,
                    row_number,
                    code="conflicting_identifier",
                    severity="error",
                    column=identifier_column,
                    message="Repeated source identifier maps to conflicting measurement metadata.",
                    context={"source": key[0], "conflicting_columns": conflicts},
                )

    # A registered compound identifier may repeat across experiments, but it
    # should not silently resolve to different structures.
    compound_column = "compound_id" if table in {"menin", "herg"} else "molecule_chembl_id"
    if compound_column not in data.columns or "smiles" not in data.columns:
        return
    compounds = data[compound_column].map(_text_token)
    structures = data["smiles"].map(_text_token)
    positions_by_compound: dict[str, list[int]] = {}
    for position in np.flatnonzero((compounds != "") & (structures != "")):
        positions_by_compound.setdefault(compounds.iloc[int(position)], []).append(int(position))
    for compound, positions in positions_by_compound.items():
        distinct_structures = sorted(set(structures.iloc[positions]))
        if len(distinct_structures) <= 1:
            continue
        for row_number in positions:
            _add_row(
                report,
                data,
                row_number,
                code="compound_identity_conflict",
                severity="error",
                column=compound_column,
                value=compound,
                message="One compound identifier maps to multiple structure strings.",
                context={"distinct_structure_count": len(distinct_structures)},
            )


def _check_targets(
    data: pd.DataFrame,
    table: TableKind,
    report: QualityReport,
    policy: QualityConfig,
) -> None:
    id_column = "target_id" if table in {"menin", "herg"} else "target_chembl_id"
    name_column = "target_name" if table in {"menin", "herg"} else "target_pref_name"
    if id_column not in data.columns and name_column not in data.columns:
        return
    target_ids = (
        data[id_column].map(_text_token) if id_column in data.columns else pd.Series("", index=data.index)
    )
    target_names = (
        data[name_column].map(_text_token) if name_column in data.columns else pd.Series("", index=data.index)
    )
    missing_ids = target_ids == ""
    missing_names = target_names == ""
    explicit_relevance = (
        data["is_target_relevant"].map(lambda value: str(value).strip().casefold() in {"true", "1", "yes"})
        if "is_target_relevant" in data.columns
        else pd.Series(False, index=data.index)
    )
    for row_number in np.flatnonzero(missing_ids & missing_names & ~explicit_relevance):
        _add_row(
            report,
            data,
            int(row_number),
            code="missing_target",
            severity="error" if table in {"menin", "herg"} else "warning",
            message="Both target identifier and target name are missing.",
        )
    for row_number in np.flatnonzero(missing_ids & missing_names & explicit_relevance):
        _add_row(
            report,
            data,
            int(row_number),
            code="target_identity_derived",
            severity="warning",
            message=(
                "Target relevance is supported by recorded curation evidence, but the "
                "source row does not expose a direct target identifier or name."
            ),
            context={
                "target_relevance": data.iloc[int(row_number)].get("target_relevance", ""),
                "target_relevance_reason": data.iloc[int(row_number)].get("target_relevance_reason", ""),
            },
        )
    for row_number in np.flatnonzero(missing_ids ^ missing_names):
        _add_row(
            report,
            data,
            int(row_number),
            code="incomplete_target_identity",
            severity="warning",
            column=id_column if missing_ids.iloc[int(row_number)] else name_column,
            message="Only one of target identifier and target name is populated.",
        )

    ambiguous = target_ids.str.contains(r"(?:[,;|]|\bunknown\b)", regex=True, case=False)
    for row_number in np.flatnonzero(ambiguous):
        _add_row(
            report,
            data,
            int(row_number),
            code="ambiguous_target",
            severity="warning",
            column=id_column,
            value=data.iloc[int(row_number)][id_column],
            message="Target field contains multiple or explicitly unknown identifiers.",
        )

    if table not in {"menin", "herg"}:
        return
    expected_ids = {item.casefold() for item in policy.expected_target_ids.get(table, frozenset())}
    expected_terms = {item.casefold() for item in policy.expected_target_terms.get(table, frozenset())}
    for row_number, (target_id, target_name) in enumerate(zip(target_ids, target_names, strict=False)):
        if not target_id and not target_name:
            continue
        if bool(explicit_relevance.iloc[row_number]):
            continue
        id_parts = {part.strip().casefold() for part in re.split(r"[,;|]", target_id) if part.strip()}
        id_match = bool(id_parts & expected_ids)
        name_match = any(term in target_name.casefold() for term in expected_terms)
        if not id_match and not name_match:
            _add_row(
                report,
                data,
                row_number,
                code="unexpected_target",
                severity="error",
                column=id_column,
                value=data.iloc[row_number][id_column] if id_column in data.columns else target_id,
                message=f"Row does not identify the configured {table} target.",
            )


def _check_assays(
    data: pd.DataFrame,
    table: TableKind,
    report: QualityReport,
    policy: QualityConfig,
) -> None:
    description_column = "assay_description"
    type_column = "assay_type"
    descriptions = (
        data[description_column].map(_text_token)
        if description_column in data.columns
        else pd.Series("", index=data.index)
    )
    assay_types = (
        data[type_column].map(_text_token) if type_column in data.columns else pd.Series("", index=data.index)
    )
    if policy.require_assay_description and description_column in data.columns:
        for row_number in np.flatnonzero(descriptions == ""):
            _add_row(
                report,
                data,
                int(row_number),
                code="missing_assay_description",
                severity="warning",
                column=description_column,
                message="Assay description is missing, limiting comparability review.",
            )
    if policy.require_assay_type and type_column in data.columns:
        for row_number in np.flatnonzero(assay_types == ""):
            _add_row(
                report,
                data,
                int(row_number),
                code="missing_assay_type",
                severity="warning",
                column=type_column,
                message="Assay type is missing, limiting modality stratification.",
            )
    both_missing = (descriptions == "") & (assay_types == "")
    for row_number in np.flatnonzero(both_missing):
        _add_row(
            report,
            data,
            int(row_number),
            code="assay_ambiguity",
            severity="warning",
            message="No assay description or assay type identifies the experiment.",
        )

    ambiguous = descriptions.map(
        lambda value: any(term in value.casefold() for term in AMBIGUOUS_ASSAY_TERMS)
    )
    for row_number in np.flatnonzero(ambiguous):
        _add_row(
            report,
            data,
            int(row_number),
            code="ambiguous_assay_context",
            severity="warning",
            column=description_column,
            value=data.iloc[int(row_number)][description_column],
            message="Assay description explicitly leaves biological context unspecified.",
        )

    # Identical descriptions should not silently change modality or target.
    target_column = "target_id" if table in {"menin", "herg"} else "target_chembl_id"
    if description_column not in data.columns:
        return
    normalized_description = descriptions.str.casefold().str.replace(r"\s+", " ", regex=True)
    valid = normalized_description != ""
    groups: dict[str, list[int]] = {}
    for row_number in np.flatnonzero(valid):
        groups.setdefault(normalized_description.iloc[int(row_number)], []).append(int(row_number))
    for positions in groups.values():
        if len(positions) < 2:
            continue
        conflicts: list[str] = []
        for column in (type_column, target_column):
            if column in data.columns:
                values = data.iloc[positions][column].map(_text_token)
                values = values[values != ""]
                if values.nunique() > 1:
                    conflicts.append(column)
        if conflicts:
            for row_number in positions:
                _add_row(
                    report,
                    data,
                    row_number,
                    code="assay_metadata_conflict",
                    severity="warning",
                    column=description_column,
                    message="The same assay description maps to conflicting metadata.",
                    context={"conflicting_columns": conflicts},
                )


def _check_repeated_measurements(
    data: pd.DataFrame,
    table: TableKind,
    report: QualityReport,
    policy: QualityConfig,
) -> None:
    endpoint_column = "endpoint" if table in {"menin", "herg"} else "standard_type"
    value_column = "value_raw" if table in {"menin", "herg"} else "standard_value"
    relation_column = "relation" if table in {"menin", "herg"} else "standard_relation"
    compound_column = "compound_id" if table in {"menin", "herg"} else "molecule_chembl_id"
    target_column = "target_id" if table in {"menin", "herg"} else "target_chembl_id"
    required = (endpoint_column, value_column)
    if any(column not in data.columns for column in required):
        return

    compounds = _compound_keys(data, compound_column)
    endpoints = data[endpoint_column].map(_endpoint_token)
    targets = (
        data[target_column].map(_text_token)
        if target_column in data.columns
        else pd.Series("", index=data.index)
    )
    relations = (
        data[relation_column].map(lambda value: "" if _is_missing(value) else str(value).strip())
        if relation_column in data.columns
        else pd.Series("=", index=data.index)
    )
    exact = relations.isin({"", "="})
    raw_values = data[value_column].map(_parse_numeric)

    if table in {"menin", "herg"}:
        if "value_nm" in data.columns:
            values = pd.to_numeric(data["value_nm"], errors="coerce")
        else:
            units = (
                data["standard_units"].map(_unit_token)
                if "standard_units" in data.columns
                else pd.Series("", index=data.index)
            )
            factors = units.map(policy.concentration_units_to_nm)
            values = raw_values * factors
        unit_keys = pd.Series("nm", index=data.index)
    else:
        values = raw_values
        unit_keys = (
            data["standard_units"].map(_unit_token)
            if "standard_units" in data.columns
            else pd.Series("", index=data.index)
        )

    usable = (compounds != "") & (endpoints != "") & exact & np.isfinite(values) & (values > 0)
    group_positions: dict[tuple[str, str, str, str], list[int]] = {}
    for row_number in np.flatnonzero(usable):
        position = int(row_number)
        key = (
            compounds.iloc[position],
            endpoints.iloc[position],
            targets.iloc[position],
            unit_keys.iloc[position],
        )
        group_positions.setdefault(key, []).append(position)

    for key, positions in group_positions.items():
        if len(positions) < 2:
            continue
        group_values = values.iloc[positions].astype(float)
        log_spread = float(np.log10(group_values.max()) - np.log10(group_values.min()))
        if not np.isfinite(log_spread) or log_spread < policy.conflict_log10_threshold:
            continue
        assay_count = 0
        if "assay_description" in data.columns:
            descriptions = data.iloc[positions]["assay_description"].map(_text_token)
            assay_count = int(descriptions[descriptions != ""].nunique())
        context = {
            "compound_key": key[0],
            "endpoint": key[1],
            "target": key[2],
            "unit": key[3],
            "measurement_count": len(positions),
            "assay_count": assay_count,
            "minimum": float(group_values.min()),
            "maximum": float(group_values.max()),
            "log10_spread": log_spread,
            "threshold": policy.conflict_log10_threshold,
        }
        for row_number in positions:
            _add_row(
                report,
                data,
                row_number,
                code="conflicting_repeated_measurement",
                severity="warning",
                column=value_column,
                value=values.iloc[row_number],
                message="Repeated exact measurements differ by at least the configured log10 threshold.",
                context=context,
            )


def _compound_keys(data: pd.DataFrame, compound_column: str) -> pd.Series:
    if compound_column in data.columns:
        keys = data[compound_column].map(_text_token)
    else:
        keys = pd.Series("", index=data.index)
    if "smiles" in data.columns:
        smiles = data["smiles"].map(_text_token)
        keys = keys.where(keys != "", "SMILES:" + smiles)
    return keys


def _add_row(
    report: QualityReport,
    data: pd.DataFrame,
    row_number: int,
    *,
    code: str,
    severity: Severity,
    message: str,
    column: str | None = None,
    value: Any = None,
    context: Mapping[str, Any] | None = None,
) -> None:
    id_columns = (
        "source_record_id",
        "activity_id",
        "compound_id",
        "molecule_chembl_id",
    )
    identifier = None
    row = data.iloc[row_number]
    for id_column in id_columns:
        if id_column in data.columns and not _is_missing(row[id_column]):
            identifier = str(row[id_column])
            break
    _add(
        report,
        code=code,
        severity=severity,
        scope="row",
        message=message,
        row_number=row_number,
        column=column,
        identifier=identifier,
        value=value,
        context=context or {},
    )


def _add(
    report: QualityReport,
    *,
    code: str,
    severity: Severity,
    scope: Literal["row", "group", "table"],
    message: str,
    row_number: int | None = None,
    column: str | None = None,
    identifier: str | None = None,
    value: Any = None,
    context: Mapping[str, Any] | None = None,
) -> None:
    report.findings.append(
        QualityFinding(
            table=report.table,
            code=code,
            severity=severity,
            scope=scope,
            message=message,
            row_number=row_number,
            column=column,
            identifier=identifier,
            value=value,
            context=context or {},
        )
    )


def _sort_findings(report: QualityReport) -> None:
    severity_order = {"error": 0, "warning": 1, "info": 2}
    scope_order = {"table": 0, "group": 1, "row": 2}
    report.findings.sort(
        key=lambda finding: (
            severity_order[finding.severity],
            scope_order[finding.scope],
            finding.code,
            -1 if finding.row_number is None else finding.row_number,
            finding.column or "",
        )
    )


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        return False
    if isinstance(value, str):
        return value.strip().casefold() in {"", "na", "n/a", "nan", "none", "null"}
    return False


def _missing_mask(series: pd.Series) -> pd.Series:
    return series.map(_is_missing).astype(bool)


def _text_token(value: Any) -> str:
    return "" if _is_missing(value) else str(value).strip()


def _endpoint_token(value: Any) -> str:
    return re.sub(r"\s+", " ", _text_token(value)).casefold()


def _unit_token(value: Any) -> str:
    text = _text_token(value).replace("μ", "u").replace("µ", "u").replace("Μ", "u")
    return re.sub(r"\s+", " ", text).casefold()


def _parse_numeric(value: Any) -> float:
    if _is_missing(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        result = float(value)
        return result if math.isfinite(result) else np.nan
    match = re.search(
        r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        str(value),
    )
    if not match:
        return np.nan
    try:
        result = float(match.group(0).replace(",", ""))
    except ValueError:
        return np.nan
    return result if math.isfinite(result) else np.nan


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    return str(value)


def _utc_timestamp(value: str | datetime | None = None) -> str:
    if value is None:
        moment = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        moment = value
    else:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = [
    "ACTIVITY_REQUIRED_COLUMNS",
    "PK_REQUIRED_COLUMNS",
    "QualityConfig",
    "QualityFinding",
    "QualityReport",
    "audit_table",
    "audit_tables",
    "audit_menin_table",
    "audit_herg_table",
    "audit_pk_table",
    "write_quality_outputs",
]
