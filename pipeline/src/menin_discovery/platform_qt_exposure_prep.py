"""Prepare exposure-aware QT/QTc collection assets without inventing PK.

The builder is deliberately conservative.  It inventories source-reported
regimen and PK candidates for the frozen clinical QT/QTc context, but it does
not promote candidate values into adjudicated Cmax, fraction-unbound, hERG
IC50, safety-margin, clinical-safety, or training labels.

DailyMed evidence is linked only through an exact Drugs@FDA application-number
overlap already associated with a structure by the clinical-link release.
ChEMBL PK rows remain candidate inventory rows until analyte, matrix, dose,
timing, population, and endpoint semantics are manually reconciled.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "platform-qt-exposure-prep/1.0"
PARSER_VERSION = "platform_qt_exposure_prep/1.0"
MANIFEST_NAME = "qt_exposure_prep_manifest.json"
REPORT_NAME = "QT_EXPOSURE_PREP.md"
STRUCTURE_OUTPUT = "qt_exposure_structure_template.parquet"
STRUCTURE_TRIAL_OUTPUT = "qt_exposure_structure_trial_template.parquet"
GAP_OUTPUT = "qt_exposure_gap_priority_queue.parquet"
SOURCE_OUTPUT = "qt_exposure_source_adjudication_queue.parquet"
MARGIN_CONTRACT_OUTPUT = "ic50_unbound_cmax_margin_contract.json"
DAILYMED_REVIEW_CAP_PER_STRUCTURE = 10

NO_LABEL_SEMANTICS = "exposure_and_QT_context_collection_only_not_a_direct_hERG_clinical_safety_clinical_risk_or_training_label"


class QtExposurePrepError(RuntimeError):
    """Raised when an exposure-preparation invariant fails closed."""


STRUCTURE_SCHEMA = pa.schema(
    [
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("standardized_smiles", pa.large_string(), nullable=False),
        pa.field("standard_inchi_key", pa.large_string(), nullable=False),
        pa.field("molecular_weight", pa.float64()),
        pa.field("qt_endpoint_count", pa.int64(), nullable=False),
        pa.field("qt_trial_count", pa.int64(), nullable=False),
        pa.field("qt_phenotype_classes_json", pa.large_string(), nullable=False),
        pa.field("qt_correction_methods_json", pa.large_string(), nullable=False),
        pa.field("qt_correction_status", pa.large_string(), nullable=False),
        pa.field("clinicaltrials_intervention_record_count", pa.int64(), nullable=False),
        pa.field("clinicaltrials_dose_text_candidate_count", pa.int64(), nullable=False),
        pa.field("clinicaltrials_route_text_candidate_count", pa.int64(), nullable=False),
        pa.field("clinicaltrials_regimen_text_candidate_count", pa.int64(), nullable=False),
        pa.field("drugsfda_exact_application_count", pa.int64(), nullable=False),
        pa.field("drugsfda_product_record_count", pa.int64(), nullable=False),
        pa.field("drugsfda_route_or_form_candidate_count", pa.int64(), nullable=False),
        pa.field("drugsfda_strength_candidate_count", pa.int64(), nullable=False),
        pa.field("dailymed_exact_application_document_count", pa.int64(), nullable=False),
        pa.field("dailymed_pk_candidate_section_count", pa.int64(), nullable=False),
        pa.field("dailymed_cmax_candidate_section_count", pa.int64(), nullable=False),
        pa.field("dailymed_protein_binding_bounded_span_count", pa.int64(), nullable=False),
        pa.field("dailymed_metabolite_bounded_span_count", pa.int64(), nullable=False),
        pa.field("dailymed_unbound_bounded_span_count", pa.int64(), nullable=False),
        pa.field("chembl37_pk_candidate_row_count", pa.int64(), nullable=False),
        pa.field("chembl37_human_cmax_candidate_count", pa.int64(), nullable=False),
        pa.field("chembl37_human_fraction_unbound_candidate_count", pa.int64(), nullable=False),
        pa.field("chembl37_human_protein_binding_candidate_count", pa.int64(), nullable=False),
        pa.field("processed_pk_observation_count", pa.int64(), nullable=False),
        pa.field("adjudicated_human_cmax_count", pa.int64(), nullable=False),
        pa.field("adjudicated_fraction_unbound_count", pa.int64(), nullable=False),
        pa.field("margin_ready", pa.bool_(), nullable=False),
        pa.field("readiness_blockers_json", pa.large_string(), nullable=False),
        pa.field("evidence_semantics", pa.large_string(), nullable=False),
        pa.field("direct_herg_label", pa.bool_(), nullable=False),
        pa.field("qt_used_as_herg_label", pa.bool_(), nullable=False),
        pa.field("clinical_risk_label", pa.bool_(), nullable=False),
        pa.field("use_as_training_label", pa.bool_(), nullable=False),
    ]
)

STRUCTURE_TRIAL_SCHEMA = pa.schema(
    [
        pa.field("collection_id", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("standardized_smiles", pa.large_string(), nullable=False),
        pa.field("standard_inchi_key", pa.large_string(), nullable=False),
        pa.field("nct_id", pa.large_string(), nullable=False),
        pa.field("clinicaltrials_url", pa.large_string(), nullable=False),
        pa.field("study_brief_title", pa.large_string()),
        pa.field("study_phases_json", pa.large_string(), nullable=False),
        pa.field("study_overall_status", pa.large_string()),
        pa.field("qt_endpoint_count", pa.int64(), nullable=False),
        pa.field("qt_endpoint_candidate_ids_json", pa.large_string(), nullable=False),
        pa.field("qt_phenotype_classes_json", pa.large_string(), nullable=False),
        pa.field("qt_correction_methods_json", pa.large_string(), nullable=False),
        pa.field("qt_correction_status", pa.large_string(), nullable=False),
        pa.field("clinicaltrials_intervention_names_json", pa.large_string(), nullable=False),
        pa.field("dose_text_candidates_json", pa.large_string(), nullable=False),
        pa.field("route_text_candidates_json", pa.large_string(), nullable=False),
        pa.field("regimen_text_candidates_json", pa.large_string(), nullable=False),
        pa.field("clinicaltrials_source_records_json", pa.large_string(), nullable=False),
        pa.field("regulatory_evidence_summary_json", pa.large_string(), nullable=False),
        pa.field("pk_candidate_evidence_summary_json", pa.large_string(), nullable=False),
        pa.field("adjudicated_dose_value", pa.float64()),
        pa.field("adjudicated_dose_unit", pa.large_string()),
        pa.field("adjudicated_dose_relation", pa.large_string()),
        pa.field("adjudicated_route", pa.large_string()),
        pa.field("adjudicated_dose_schedule", pa.large_string()),
        pa.field("adjudicated_population", pa.large_string()),
        pa.field("adjudicated_cmax_value", pa.float64()),
        pa.field("adjudicated_cmax_unit", pa.large_string()),
        pa.field("adjudicated_cmax_relation", pa.large_string()),
        pa.field("adjudicated_cmax_matrix", pa.large_string()),
        pa.field("adjudicated_cmax_time_basis", pa.large_string()),
        pa.field("adjudicated_cmax_analyte", pa.large_string()),
        pa.field("adjudicated_cmax_source_id", pa.large_string()),
        pa.field("adjudicated_fraction_unbound_value", pa.float64()),
        pa.field("adjudicated_fraction_unbound_unit", pa.large_string()),
        pa.field("adjudicated_fraction_unbound_relation", pa.large_string()),
        pa.field("adjudicated_fraction_unbound_matrix", pa.large_string()),
        pa.field("adjudicated_fraction_unbound_analyte", pa.large_string()),
        pa.field("adjudicated_fraction_unbound_source_id", pa.large_string()),
        pa.field("adjudicated_active_metabolites_json", pa.large_string()),
        pa.field("adjudicated_herg_ic50_value", pa.float64()),
        pa.field("adjudicated_herg_ic50_unit", pa.large_string()),
        pa.field("adjudicated_herg_ic50_relation", pa.large_string()),
        pa.field("adjudicated_herg_ic50_source_observation_id", pa.large_string()),
        pa.field("margin_point", pa.float64()),
        pa.field("margin_lower_bound", pa.float64()),
        pa.field("margin_upper_bound", pa.float64()),
        pa.field("margin_status", pa.large_string(), nullable=False),
        pa.field("evidence_semantics", pa.large_string(), nullable=False),
        pa.field("direct_herg_label", pa.bool_(), nullable=False),
        pa.field("qt_used_as_herg_label", pa.bool_(), nullable=False),
        pa.field("clinical_risk_label", pa.bool_(), nullable=False),
        pa.field("use_as_training_label", pa.bool_(), nullable=False),
    ]
)

GAP_SCHEMA = pa.schema(
    [
        pa.field("priority_rank", pa.int64(), nullable=False),
        pa.field("gap_id", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("nct_id", pa.large_string(), nullable=False),
        pa.field("priority_tier", pa.large_string(), nullable=False),
        pa.field("priority_score", pa.float64(), nullable=False),
        pa.field("qt_endpoint_count", pa.int64(), nullable=False),
        pa.field("qt_correction_status", pa.large_string(), nullable=False),
        pa.field("candidate_evidence_domains_json", pa.large_string(), nullable=False),
        pa.field("required_gap_fields_json", pa.large_string(), nullable=False),
        pa.field("adjudication_sequence_json", pa.large_string(), nullable=False),
        pa.field("source_review_links_json", pa.large_string(), nullable=False),
        pa.field("margin_status", pa.large_string(), nullable=False),
        pa.field("direct_herg_label", pa.bool_(), nullable=False),
        pa.field("qt_used_as_herg_label", pa.bool_(), nullable=False),
        pa.field("clinical_risk_label", pa.bool_(), nullable=False),
        pa.field("use_as_training_label", pa.bool_(), nullable=False),
    ]
)

SOURCE_SCHEMA = pa.schema(
    [
        pa.field("priority_rank", pa.int64(), nullable=False),
        pa.field("source_review_id", pa.large_string(), nullable=False),
        pa.field("priority_tier", pa.large_string(), nullable=False),
        pa.field("priority_score", pa.float64(), nullable=False),
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("nct_id", pa.large_string()),
        pa.field("source_family", pa.large_string(), nullable=False),
        pa.field("source_record_id", pa.large_string(), nullable=False),
        pa.field("source_url", pa.large_string()),
        pa.field("evidence_domains_json", pa.large_string(), nullable=False),
        pa.field("source_locator_json", pa.large_string(), nullable=False),
        pa.field("candidate_evidence_json", pa.large_string(), nullable=False),
        pa.field("identity_linkage_basis", pa.large_string(), nullable=False),
        pa.field("context_limitations_json", pa.large_string(), nullable=False),
        pa.field("candidate_pool_count", pa.int64(), nullable=False),
        pa.field("selection_policy", pa.large_string(), nullable=False),
        pa.field("candidate_status", pa.large_string(), nullable=False),
        pa.field("direct_herg_label", pa.bool_(), nullable=False),
        pa.field("qt_used_as_herg_label", pa.bool_(), nullable=False),
        pa.field("clinical_risk_label", pa.bool_(), nullable=False),
        pa.field("use_as_training_label", pa.bool_(), nullable=False),
    ]
)

_DOSE_RE = re.compile(
    r"(?<!\w)\d+(?:\.\d+)?(?:\s*(?:-|to)\s*\d+(?:\.\d+)?)?\s*"
    r"(?:ng|mcg|μg|µg|ug|mg|g|mL|ml|IU|units?)"
    r"(?:\s*/\s*(?:kg|m2|m²|day|dose|week|month))?",
    re.IGNORECASE,
)
_ROUTE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("oral", re.compile(r"\b(?:oral(?:ly)?|by mouth|p\.?o\.?)\b", re.I)),
    ("intravenous", re.compile(r"\b(?:intravenous(?:ly)?|IV infusion|IV injection)\b", re.I)),
    ("subcutaneous", re.compile(r"\b(?:subcutaneous(?:ly)?|SC injection)\b", re.I)),
    ("intramuscular", re.compile(r"\b(?:intramuscular(?:ly)?|IM injection)\b", re.I)),
    ("inhaled", re.compile(r"\b(?:inhaled|inhalation)\b", re.I)),
    ("topical", re.compile(r"\btopical(?:ly)?\b", re.I)),
    ("transdermal", re.compile(r"\btransdermal(?:ly)?\b", re.I)),
    ("ophthalmic", re.compile(r"\bophthalmic(?:ally)?\b", re.I)),
    ("intranasal", re.compile(r"\b(?:intranasal(?:ly)?|nasal spray)\b", re.I)),
)
_REGIMEN_RE = re.compile(
    r"\b(?:once daily|twice daily|three times daily|qd|q\.d\.|bid|b\.i\.d\.|tid|"
    r"weekly|every \d+ (?:hours?|days?|weeks?)|day \d+|treatment cycle|single dose|multiple doses?)\b",
    re.IGNORECASE,
)
_DAILYMED_TERM_PATTERNS: dict[str, re.Pattern[str]] = {
    "protein_binding": re.compile(
        r"protein[- ]bound|protein binding|bound to (?:plasma|serum) proteins?", re.I
    ),
    "metabolite": re.compile(r"\bmetabolites?\b|\bmetaboli[sz]ed\b", re.I),
    "unbound": re.compile(r"\bunbound\b|\bfree fraction\b|fraction unbound", re.I),
}
_PK_TYPES = frozenset({"cmax", "fu", "ppb"})
_NULL_TEMPLATE_FIELDS = tuple(
    field.name
    for field in STRUCTURE_TRIAL_SCHEMA
    if field.name.startswith("adjudicated_")
    or (field.name.startswith("margin_") and field.name != "margin_status")
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join("" if part is None else str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode()).hexdigest()[:24].upper()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_hash(payload: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return hashlib.sha256(_canonical_json(unsigned).encode()).hexdigest()


def _checked_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise QtExposurePrepError(f"missing or unsafe regular-file input: {path}")
    return path


def _input_binding(role: str, path: Path) -> dict[str, Any]:
    checked = _checked_file(path).resolve()
    result = {
        "role": role,
        "path": str(checked),
        "bytes": checked.stat().st_size,
        "sha256": _sha256_file(checked),
    }
    if checked.suffix == ".parquet":
        parquet = pq.ParquetFile(checked)
        result["rows"] = parquet.metadata.num_rows
        result["arrow_schema_sha256"] = _schema_hash(parquet.schema_arrow)
    return result


def _schema_hash(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _artifact(path: Path, schema: pa.Schema | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if schema is not None:
        result["rows"] = pq.ParquetFile(path).metadata.num_rows
        result["arrow_schema_sha256"] = _schema_hash(schema)
    return result


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]], schema: pa.Schema) -> dict[str, Any]:
    table = pa.Table.from_pylist(list(rows), schema=schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=False,
        row_group_size=65_536,
        version="2.6",
    )
    return _artifact(path, schema)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    _checked_file(path)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QtExposurePrepError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise QtExposurePrepError(f"JSONL row must be an object at {path}:{line_number}")
            yield value


def _read_csv_gz(path: Path) -> list[dict[str, str]]:
    _checked_file(path)
    with gzip.open(path, mode="rt", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _application_number(value: object) -> str:
    digits = "".join(character for character in str(value or "") if character.isdigit())
    return digits.lstrip("0") or ("0" if digits else "")


def _json_list(value: object) -> list[Any]:
    if value in (None, ""):
        return []
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise QtExposurePrepError("expected a JSON list")
    return parsed


def _correction_status(methods: Iterable[str]) -> str:
    unique = set(methods)
    resolved = unique - {"unresolved"}
    if not resolved:
        return "all_endpoints_unresolved"
    if "unresolved" in unique:
        return "mixed_resolved_and_unresolved"
    return "all_endpoints_have_explicit_correction_method"


def _excerpt(text: str, start: int, end: int, radius: int = 90) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return " ".join(text[left:right].split())


def _text_evidence(row: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    name = str(row.get("intervention_name") or "")
    description = str(row.get("intervention_description") or "")
    text = " ".join(part for part in (name, description) if part).strip()
    common = {
        "intervention_candidate_id": str(row.get("intervention_candidate_id") or ""),
        "raw_json_pointer": str(row.get("raw_json_pointer") or ""),
        "source_page_path": str(row.get("source_page_path") or ""),
        "source_page_sha256": str(row.get("source_page_sha256") or ""),
    }
    result: dict[str, list[dict[str, str]]] = {"dose": [], "route": [], "regimen": []}
    for match in _DOSE_RE.finditer(text):
        result["dose"].append(
            common | {"matched_text": match.group(0), "verbatim_context": _excerpt(text, *match.span())}
        )
    for route, pattern in _ROUTE_PATTERNS:
        for match in pattern.finditer(text):
            result["route"].append(
                common
                | {
                    "normalized_route_candidate": route,
                    "matched_text": match.group(0),
                    "verbatim_context": _excerpt(text, *match.span()),
                }
            )
    for match in _REGIMEN_RE.finditer(text):
        result["regimen"].append(
            common | {"matched_text": match.group(0), "verbatim_context": _excerpt(text, *match.span())}
        )
    return result


def _bounded_daily_med_terms(section: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {name: [] for name in _DAILYMED_TERM_PATTERNS}
    for span in section.get("evidence_spans") or []:
        text = str(span.get("text") or "")
        for domain, pattern in _DAILYMED_TERM_PATTERNS.items():
            for match in pattern.finditer(text):
                result[domain].append(
                    {
                        "matched_text": match.group(0),
                        "verbatim_context": _excerpt(text, *match.span()),
                        "span_sha256": str(span.get("sha256") or ""),
                    }
                )
    return result


@dataclass(frozen=True)
class PositiveInterval:
    """A positive interval with ``None`` representing an open infinite bound."""

    lower: float | None
    upper: float | None
    lower_inclusive: bool = True
    upper_inclusive: bool = True

    def validate(self) -> None:
        if self.lower is not None and (not math.isfinite(self.lower) or self.lower <= 0):
            raise ValueError("interval lower bound must be positive and finite")
        if self.upper is not None and (not math.isfinite(self.upper) or self.upper <= 0):
            raise ValueError("interval upper bound must be positive and finite")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("interval bounds are reversed")


def relation_interval(value: float, relation: str) -> PositiveInterval:
    """Convert a positive point/relation into interval form."""

    if not math.isfinite(value) or value <= 0:
        raise ValueError("value must be positive and finite")
    normalized = relation.strip()
    if normalized in {"", "="}:
        return PositiveInterval(value, value)
    if normalized == ">":
        return PositiveInterval(value, None, lower_inclusive=False)
    if normalized == ">=":
        return PositiveInterval(value, None, lower_inclusive=True)
    if normalized == "<":
        return PositiveInterval(None, value, upper_inclusive=False)
    if normalized == "<=":
        return PositiveInterval(None, value, upper_inclusive=True)
    raise ValueError(f"unsupported relation: {relation}")


def concentration_to_micromolar(
    value: float, unit: str, *, analyte_molecular_weight_g_mol: float | None = None
) -> float:
    """Convert a supported concentration unit to micromolar.

    Mass-unit conversion requires an explicitly adjudicated analyte molecular
    weight; the structure-master parent molecular weight is never substituted
    automatically.
    """

    if not math.isfinite(value) or value <= 0:
        raise ValueError("concentration must be positive and finite")
    normalized = re.sub(r"[\s._-]+", "", unit.casefold()).replace("µ", "u").replace("μ", "u")
    molar_factors = {
        "m": 1_000_000.0,
        "mm": 1_000.0,
        "um": 1.0,
        "nm": 0.001,
        "pm": 0.000001,
        "mol/l": 1_000_000.0,
        "mmol/l": 1_000.0,
        "umol/l": 1.0,
        "nmol/l": 0.001,
    }
    if normalized in molar_factors:
        return value * molar_factors[normalized]
    if analyte_molecular_weight_g_mol is None or analyte_molecular_weight_g_mol <= 0:
        raise ValueError("mass concentration conversion requires adjudicated analyte molecular weight")
    mass_factors = {
        "ng/ml": 1.0,
        "ug/l": 1.0,
        "ug/ml": 1_000.0,
        "mg/l": 1_000.0,
        "mg/ml": 1_000_000.0,
        "ng/l": 0.001,
    }
    if normalized not in mass_factors:
        raise ValueError(f"unsupported concentration unit: {unit}")
    return value * mass_factors[normalized] / analyte_molecular_weight_g_mol


def fraction_unbound_interval(value: float, unit: str, relation: str = "=") -> PositiveInterval:
    """Normalize dimensionless or percent fraction-unbound evidence."""

    normalized = unit.strip().casefold()
    normalized_value = value / 100.0 if normalized in {"%", "percent", "percentage"} else value
    interval = relation_interval(normalized_value, relation)
    if (interval.lower is not None and interval.lower > 1) or (
        interval.upper is not None and interval.upper > 1
    ):
        raise ValueError("fraction unbound must be bounded within (0, 1]")
    bounded = PositiveInterval(
        interval.lower,
        1.0 if interval.upper is None else interval.upper,
        interval.lower_inclusive,
        True if interval.upper is None else interval.upper_inclusive,
    )
    if bounded.lower == bounded.upper and not (bounded.lower_inclusive and bounded.upper_inclusive):
        raise ValueError("fraction unbound interval is empty")
    return bounded


def multiply_positive_intervals(left: PositiveInterval, right: PositiveInterval) -> PositiveInterval:
    left.validate()
    right.validate()
    lower = None if left.lower is None or right.lower is None else left.lower * right.lower
    upper = None if left.upper is None or right.upper is None else left.upper * right.upper
    return PositiveInterval(
        lower,
        upper,
        left.lower_inclusive and right.lower_inclusive,
        left.upper_inclusive and right.upper_inclusive,
    )


def quotient_positive_intervals(
    numerator: PositiveInterval, denominator: PositiveInterval
) -> PositiveInterval:
    """Return the conservative quotient interval for positive operands."""

    numerator.validate()
    denominator.validate()
    lower = (
        None if numerator.lower is None or denominator.upper is None else numerator.lower / denominator.upper
    )
    upper = (
        None if numerator.upper is None or denominator.lower is None else numerator.upper / denominator.lower
    )
    return PositiveInterval(
        lower,
        upper,
        numerator.lower_inclusive and denominator.upper_inclusive,
        numerator.upper_inclusive and denominator.lower_inclusive,
    )


def pic50_interval_to_ic50_micromolar(
    pic50_lower: float | None, pic50_upper: float | None
) -> PositiveInterval:
    """Convert a pIC50 interval to an IC50 micromolar interval, reversing bounds."""

    if pic50_lower is not None and not math.isfinite(pic50_lower):
        raise ValueError("pIC50 lower bound must be finite")
    if pic50_upper is not None and not math.isfinite(pic50_upper):
        raise ValueError("pIC50 upper bound must be finite")
    if pic50_lower is not None and pic50_upper is not None and pic50_lower > pic50_upper:
        raise ValueError("pIC50 bounds are reversed")
    ic50_lower = None if pic50_upper is None else 10 ** (6 - pic50_upper)
    ic50_upper = None if pic50_lower is None else 10 ** (6 - pic50_lower)
    return PositiveInterval(ic50_lower, ic50_upper)


def _margin_contract() -> dict[str, Any]:
    return {
        "schema_version": "ic50-unbound-cmax-margin-contract/1.0",
        "output_name": "hERG_IC50_over_unbound_Cmax_margin",
        "output_unit": "dimensionless",
        "formula": {
            "direct_unbound_exposure": "margin = IC50_uM / Cmax_unbound_uM",
            "derived_unbound_exposure": "Cmax_unbound_uM = Cmax_total_uM * fraction_unbound; margin = IC50_uM / Cmax_unbound_uM",
        },
        "required_identity_context": [
            "exact parent_or_explicit_metabolite analyte identity for every input",
            "human exposure at a documented dose, route, schedule, population, and time basis",
            "plasma versus serum versus whole-blood matrix preserved and not silently converted",
            "hERG potency explicitly targets human wild-type KCNH2 Q12809; mutant variant or unresolved construct is ineligible",
            "hERG functional assay endpoint, modality, protocol, relation, and censoring retained",
            "fraction unbound applicable to the same analyte and biologic matrix",
        ],
        "accepted_relations": ["=", ">", ">=", "<", "<="],
        "canonical_concentration_unit": "uM",
        "molar_units": ["M", "mM", "uM", "nM", "pM", "mol/L", "mmol/L", "umol/L", "nmol/L"],
        "mass_units_requiring_adjudicated_analyte_molecular_weight": [
            "ng/mL",
            "ug/L",
            "ug/mL",
            "mg/L",
            "mg/mL",
            "ng/L",
        ],
        "molecular_weight_rule": (
            "Never substitute standardized-parent molecular weight for a measured salt, prodrug, or metabolite; "
            "mass-unit conversion is blocked until analyte identity and molecular weight are adjudicated."
        ),
        "fraction_unbound_rule": {
            "dimensionless_range": "0 < fu <= 1",
            "percent_conversion": "fu = reported_percent / 100",
            "protein_binding_conversion": (
                "fu = 1 - fraction_bound only when the source explicitly reports plasma/serum protein-bound "
                "fraction for the same analyte and context; censoring must be algebraically reversed."
            ),
        },
        "pic50_conversion": "IC50_uM = 10^(6 - pIC50); monotone decrease reverses lower and upper bounds",
        "censoring_interval_rules": {
            "exact": "[value,value]",
            "greater_than": "(value,+infinity)",
            "greater_than_or_equal": "[value,+infinity)",
            "less_than": "(0,value)",
            "less_than_or_equal": "(0,value]",
            "unbound_cmax_product": "[Cmax_lower*fu_lower, Cmax_upper*fu_upper] with open/infinite bounds preserved",
            "margin_quotient": "[IC50_lower/Cmax_upper, IC50_upper/Cmax_lower] with open/infinite bounds preserved",
        },
        "multiple_record_policy": (
            "Do not average or choose a maximum automatically. Preserve each dose, route, population, time basis, "
            "matrix, analyte, and source as a separate exposure context until a documented analysis estimand selects one."
        ),
        "metabolite_policy": (
            "Parent and active metabolites are separate analytes. A combined margin requires a prespecified, "
            "mechanistically justified aggregation method and is not defined by this contract."
        ),
        "hard_blocks": [
            "missing IC50 or compatible censoring bounds",
            "missing unbound Cmax or either component needed to derive it",
            "nonpositive or nonfinite numeric input",
            "unsupported or missing unit",
            "mass concentration without adjudicated analyte molecular weight",
            "analyte mismatch",
            "hERG target is mutant variant nonhuman or unresolved rather than explicit human wild-type KCNH2",
            "hERG endpoint is not a functional potency measurement with retained modality and relation",
            "unresolved plasma/serum/whole-blood or tissue matrix mismatch",
            "unresolved dose, route, schedule, population, or time-basis applicability",
            "candidate-only machine-detected DailyMed or ChEMBL row not manually adjudicated",
        ],
        "missing_input_output": {
            "margin_point": None,
            "margin_lower_bound": None,
            "margin_upper_bound": None,
            "margin_status": "not_computed_missing_or_unadjudicated_required_inputs",
        },
        "imputation_policy": "No PK, protein-binding, metabolite, or hERG potency imputation is permitted.",
        "label_policy": NO_LABEL_SEMANTICS,
    }


def _check_parquet(path: Path, columns: Sequence[str]) -> None:
    _checked_file(path)
    missing = sorted(set(columns) - set(pq.ParquetFile(path).schema_arrow.names))
    if missing:
        raise QtExposurePrepError(f"{path} is missing required columns: {missing}")


def _manifest_artifact_paths(manifest_path: Path, required_names: set[str] | None = None) -> list[Path]:
    _checked_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts") or manifest.get("chunks") or manifest.get("parts") or []
    selected: list[Path] = []
    seen_names: set[str] = set()
    for artifact in artifacts:
        relative = str(artifact.get("path") or "")
        if not relative:
            continue
        name = Path(relative).name
        if required_names is not None and name not in required_names:
            continue
        path = manifest_path.parent / relative
        _checked_file(path)
        declared_bytes = artifact.get("bytes", artifact.get("size_bytes"))
        if declared_bytes is None:
            raise QtExposurePrepError(f"manifest has no byte size for {path}")
        if path.stat().st_size != int(declared_bytes):
            raise QtExposurePrepError(f"manifest byte mismatch for {path}")
        if _sha256_file(path) != artifact["sha256"]:
            raise QtExposurePrepError(f"manifest sha256 mismatch for {path}")
        selected.append(path)
        seen_names.add(name)
    if required_names is not None and seen_names != required_names:
        raise QtExposurePrepError(
            f"manifest is missing required artifacts: {sorted(required_names - seen_names)}"
        )
    return sorted(selected)


def _report_text(
    *,
    cohort: Mapping[str, int],
    coverage: Mapping[str, Any],
    source_counts: Mapping[str, int],
    gap_counts: Mapping[str, int],
    pkdb_decision: Mapping[str, Any],
) -> str:
    return f"""# QT/QTc Exposure Preparation v1.5

## Outcome

This additive release creates collection and review assets for **{cohort["structures"]:,} structures**, **{cohort["structure_trials"]:,} structure-trial contexts**, and **{cohort["qt_endpoints"]:,} reported QT/QTc endpoints**. It computes no real safety margins and admits no QT, clinical, DailyMed, Drugs@FDA, or PK candidate as a direct hERG, clinical-risk, or training label.

All adjudicated dose, route, Cmax, fraction-unbound, metabolite, hERG IC50, and margin fields remain null. Candidate evidence is retained only to direct human review.

## Evidence actually present

| Evidence domain | Structures or contexts with candidate evidence | Interpretation |
|---|---:|---|
| Explicit QT correction method | {coverage["structures_with_any_resolved_qt_correction"]:,} structures | Source-reported method context; not a hERG label |
| ClinicalTrials dose/strength text | {coverage["structure_trials_with_dose_text"]:,} structure-trial contexts | Regex-bounded verbatim candidate; arm attribution still requires review |
| ClinicalTrials route text | {coverage["structure_trials_with_route_text"]:,} structure-trial contexts | Reported intervention text candidate |
| Drugs@FDA exact-name-linked application | {coverage["structures_with_drugsfda_applications"]:,} structures | Product metadata; formulation strength is not administered trial dose |
| Exact-application-linked DailyMed PK document | {coverage["structures_with_dailymed_documents"]:,} structures | Candidate label document, not asserted current/preferred label |
| DailyMed Cmax candidate section | {coverage["structures_with_dailymed_cmax_sections"]:,} structures | Machine-detected section only; no numeric value promoted |
| DailyMed bounded protein-binding term | {coverage["structures_with_dailymed_protein_binding_spans"]:,} structures | Presence in stored evidence spans; non-detection is unknown |
| DailyMed bounded metabolite term | {coverage["structures_with_dailymed_metabolite_spans"]:,} structures | Presence in stored evidence spans; identity/activity unresolved |
| ChEMBL human Cmax candidate | {coverage["structures_with_human_cmax_candidates"]:,} structures | Raw candidate awaiting dose/matrix/analyte adjudication |
| ChEMBL human FU/PPB candidate | {coverage["structures_with_human_binding_candidates"]:,} structures | Raw candidate; tissue/cell/plasma semantics remain source-native |
| Both human Cmax and FU/PPB candidates | {coverage["structures_with_both_human_candidate_types"]:,} structures | Highest-priority reconciliation set; still not margin-ready |
| Adjudicated human Cmax plus fraction unbound | 0 structures | Required before real margin computation |

The processed PK/ADME observation table overlaps {coverage["structures_with_processed_pk_observations"]:,} cohort structures. The broader ChEMBL 37 inventory overlaps {coverage["structures_with_any_chembl_pk_candidates"]:,} structures but remains candidate-level. The local PK-DB admission decision reports {int(pkdb_decision.get("candidate_observation_rows", 0)):,} candidate observation rows and canonical admission = `{str(bool(pkdb_decision.get("canonical_admission", False))).lower()}`.

## Review queues

The gap queue contains {sum(gap_counts.values()):,} structure-trial rows: {", ".join(f"{key}={value:,}" for key, value in sorted(gap_counts.items()))}. P0 identifies contexts whose structure has both human Cmax and human FU/PPB candidates, not completed margins.

The source queue contains {sum(source_counts.values()):,} review records: {", ".join(f"{key}={value:,}" for key, value in sorted(source_counts.items()))}. DailyMed review candidates are deterministically capped at {DAILYMED_REVIEW_CAP_PER_STRUCTURE} sections per structure; complete candidate-pool counts remain in the structure template.

## Margin contract

The machine-readable contract defines `IC50 / unbound Cmax` in micromolar units, analyte-specific molecular-weight conversion, fraction-unbound normalization, pIC50 bound reversal, and conservative interval arithmetic. It blocks calculation for missing or unadjudicated identity, unit, analyte, matrix, dose, route, population, schedule, time-basis, or source compatibility. No imputation is allowed.

## Boundaries

- ClinicalTrials intervention descriptions are trial-level candidates; endpoint-group-to-arm-to-dose attribution is not assumed.
- Drugs@FDA route/form and strength fields describe products, not necessarily the regimen used in a QT trial.
- DailyMed links use exact application-number overlap and preserve document/version/member hashes. They are candidate review links, not regulator-preferred-label assertions.
- DailyMed protein-binding, unbound, and metabolite flags are searched only within stored bounded PK evidence spans. Absence means not detected in those spans, never evidence of biological absence.
- ChEMBL PK inventory records preserve raw endpoint, relation, value, unit, organism, assay text, and document identifiers. They are not promoted into curated exposure inputs.
- Parent compound and active metabolites remain separate analytes.
- QT/QTc endpoints remain human phenotype context and are never converted to molecular hERG labels.
- Aggregate trial outcomes are not patient-level risk, causal attribution, a clinical safety classification, or medical guidance; this release makes no clinical-risk inference.

## Artifacts

- `{STRUCTURE_OUTPUT}`: one row per structure with source coverage and hard blockers.
- `{STRUCTURE_TRIAL_OUTPUT}`: one row per structure-trial with source text candidates and intentionally blank adjudication fields.
- `{GAP_OUTPUT}`: prioritized missing-input and reconciliation queue.
- `{SOURCE_OUTPUT}`: source-linked adjudication queue with provenance and limitations.
- `{MARGIN_CONTRACT_OUTPUT}`: units, identity, censoring, and interval-arithmetic contract.
- `{MANIFEST_NAME}`: input/output hashes, schemas, counts, and zero-label checks.
"""


def build_qt_exposure_prep(
    *,
    output_dir: Path,
    report_path: Path,
    qt_index_path: Path,
    clinical_context_master_path: Path,
    structure_master_path: Path,
    link_audit_path: Path,
    structure_annotations_path: Path,
    interventions_path: Path,
    studies_path: Path,
    drugsfda_products_path: Path,
    dailymed_manifest_path: Path,
    chembl_pk_manifest_path: Path,
    processed_pk_path: Path,
    pkdb_admission_path: Path,
) -> dict[str, Any]:
    """Build a deterministic, zero-label exposure-aware QT preparation release."""

    if output_dir.exists():
        raise QtExposurePrepError(f"refusing to overwrite existing output directory: {output_dir}")
    if report_path.exists():
        raise QtExposurePrepError(f"refusing to overwrite existing report: {report_path}")

    _check_parquet(
        qt_index_path,
        [
            "candidate_id",
            "structure_id",
            "nct_id",
            "endpoint_candidate_id",
            "qt_phenotype_class",
            "correction_methods_json",
            "herg_potency_derived",
            "qt_used_as_herg_label",
        ],
    )
    _check_parquet(
        clinical_context_master_path,
        [
            "context_class",
            "structure_id",
            "nct_id",
            "endpoint_candidate_id",
            "direct_herg_label",
            "use_as_training_label",
        ],
    )
    _check_parquet(
        structure_master_path,
        ["structure_id", "standardized_smiles", "standard_inchi_key", "molecular_weight"],
    )
    _check_parquet(
        link_audit_path,
        [
            "source_kind",
            "source_record_id",
            "nct_id",
            "raw_name",
            "linked_molecule_id",
            "link_is_exact_and_unique",
        ],
    )
    _check_parquet(
        structure_annotations_path,
        ["molecule_id", "drugsfda_application_numbers_json", "drugsfda_exact_name_link_count"],
    )
    _check_parquet(
        drugsfda_products_path,
        [
            "application_number",
            "product_number",
            "dosage_form_route_raw",
            "strength_raw",
            "drug_name_raw",
            "active_ingredient_raw",
            "source_archive",
            "source_member",
            "source_row_number_one_based",
            "source_field_map_sha256",
        ],
    )
    _checked_file(interventions_path)
    _checked_file(studies_path)
    _checked_file(processed_pk_path)
    _checked_file(pkdb_admission_path)

    dailymed_names = {"latest_available_candidate_documents.jsonl", "section_candidates.jsonl"}
    dailymed_paths = _manifest_artifact_paths(dailymed_manifest_path, dailymed_names)
    dailymed_by_name = {path.name: path for path in dailymed_paths}
    pk_paths = [
        path
        for path in _manifest_artifact_paths(chembl_pk_manifest_path)
        if path.suffix == ".parquet" and "pk_adme_candidates_" in path.name
    ]
    if not pk_paths:
        raise QtExposurePrepError("ChEMBL PK manifest contains no candidate parquet shards")

    qt_rows = pq.read_table(qt_index_path).to_pylist()
    if not qt_rows:
        raise QtExposurePrepError("QT index is empty")
    if any(row["herg_potency_derived"] or row["qt_used_as_herg_label"] for row in qt_rows):
        raise QtExposurePrepError("QT source violates the zero-hERG-label contract")
    qt_ids = [str(row["candidate_id"]) for row in qt_rows]
    if len(qt_ids) != len(set(qt_ids)):
        raise QtExposurePrepError("QT candidate IDs are not unique")
    target_structures = {str(row["structure_id"]) for row in qt_rows}
    target_pairs = {(str(row["structure_id"]), str(row["nct_id"])) for row in qt_rows}
    target_triples = {
        (str(row["structure_id"]), str(row["nct_id"]), str(row["endpoint_candidate_id"])) for row in qt_rows
    }

    context_rows = pq.read_table(clinical_context_master_path).to_pylist()
    qt_context_rows = [
        row for row in context_rows if row["context_class"] == "human_QT_QTc_reported_result_context"
    ]
    context_triples = {
        (str(row["structure_id"]), str(row["nct_id"]), str(row["endpoint_candidate_id"]))
        for row in qt_context_rows
    }
    if context_triples != target_triples:
        raise QtExposurePrepError("master clinical QT context does not exactly match the QT index")
    if any(row["direct_herg_label"] or row["use_as_training_label"] for row in qt_context_rows):
        raise QtExposurePrepError("master clinical context violates the zero-label contract")

    structures = {
        str(row["structure_id"]): row
        for row in pq.read_table(structure_master_path).to_pylist()
        if str(row["structure_id"]) in target_structures
    }
    if set(structures) != target_structures:
        raise QtExposurePrepError("structure master does not cover every QT structure")
    inchi_to_structure = {
        str(row["standard_inchi_key"]): structure_id for structure_id, row in structures.items()
    }
    if len(inchi_to_structure) != len(structures):
        raise QtExposurePrepError("target structures do not have unique standard InChIKeys")

    qt_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    qt_by_structure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in qt_rows:
        structure_id = str(row["structure_id"])
        pair = (structure_id, str(row["nct_id"]))
        qt_by_pair[pair].append(row)
        qt_by_structure[structure_id].append(row)

    linked_interventions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    link_id_to_pair: dict[str, tuple[str, str]] = {}
    link_name_by_id: dict[str, str] = {}
    for row in pq.read_table(link_audit_path).to_pylist():
        pair = (str(row.get("linked_molecule_id") or ""), str(row.get("nct_id") or ""))
        if (
            row["source_kind"] == "clinicaltrials_intervention"
            and row["link_is_exact_and_unique"]
            and pair in target_pairs
        ):
            record_id = str(row["source_record_id"])
            if record_id in link_id_to_pair and link_id_to_pair[record_id] != pair:
                raise QtExposurePrepError("one intervention audit ID maps to multiple target pairs")
            link_id_to_pair[record_id] = pair
            link_name_by_id[record_id] = str(row["raw_name"])
    intervention_rows = _read_csv_gz(interventions_path)
    intervention_by_id = {row["intervention_candidate_id"]: row for row in intervention_rows}
    if not set(link_id_to_pair).issubset(intervention_by_id):
        raise QtExposurePrepError("linked intervention rows are missing from ClinicalTrials inventory")
    for intervention_id, pair in link_id_to_pair.items():
        row = intervention_by_id[intervention_id]
        if row["nct_id"] != pair[1]:
            raise QtExposurePrepError("ClinicalTrials intervention NCT mismatch")
        enriched: dict[str, Any] = dict(row)
        enriched["linked_name"] = link_name_by_id[intervention_id]
        enriched["text_evidence"] = _text_evidence(row)
        linked_interventions[pair].append(enriched)
    if set(linked_interventions) != target_pairs:
        raise QtExposurePrepError("every structure-trial must retain at least one exact linked intervention")

    studies = {row["nct_id"]: row for row in _read_csv_gz(studies_path)}
    if not {pair[1] for pair in target_pairs}.issubset(studies):
        raise QtExposurePrepError("ClinicalTrials study inventory is incomplete for target trials")

    annotations = {
        str(row["molecule_id"]): row
        for row in pq.read_table(structure_annotations_path).to_pylist()
        if str(row["molecule_id"]) in target_structures
    }
    if set(annotations) != target_structures:
        raise QtExposurePrepError("structure-development annotations do not cover every QT structure")
    apps_by_structure: dict[str, set[str]] = {}
    structures_by_app: dict[str, set[str]] = defaultdict(set)
    for structure_id, row in annotations.items():
        apps = {_application_number(value) for value in _json_list(row["drugsfda_application_numbers_json"])}
        apps.discard("")
        apps_by_structure[structure_id] = apps
        for app in apps:
            structures_by_app[app].add(structure_id)

    products_by_app: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pq.read_table(drugsfda_products_path).to_pylist():
        app = _application_number(row["application_number"])
        if app in structures_by_app:
            products_by_app[app].append(row)

    dailymed_docs_by_structure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    matched_doc_key: dict[tuple[str, str, str], list[tuple[str, tuple[str, ...]]]] = defaultdict(list)
    for document in _read_jsonl(dailymed_by_name["latest_available_candidate_documents.jsonl"]):
        document_apps = {_application_number(value) for value in document.get("approval_ids") or []}
        matching_apps = sorted(
            (document_apps & structures_by_app.keys()), key=lambda value: (len(value), value)
        )
        if not matching_apps:
            continue
        document_key = (
            str(document["set_id"]),
            str(document["version_number"]),
            str(document["document_id"]),
        )
        for structure_id in sorted(set().union(*(structures_by_app[app] for app in matching_apps))):
            structure_apps = tuple(sorted(set(matching_apps) & apps_by_structure[structure_id]))
            if not structure_apps:
                continue
            record = dict(document)
            record["matching_application_numbers"] = list(structure_apps)
            dailymed_docs_by_structure[structure_id].append(record)
            matched_doc_key[document_key].append((structure_id, structure_apps))

    dailymed_sections_by_structure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for section in _read_jsonl(dailymed_by_name["section_candidates.jsonl"]):
        key = (str(section["set_id"]), str(section["version_number"]), str(section["document_id"]))
        for structure_id, section_matching_apps in matched_doc_key.get(key, []):
            record = dict(section)
            record["matching_application_numbers"] = list(section_matching_apps)
            record["bounded_terms"] = _bounded_daily_med_terms(section)
            dailymed_sections_by_structure[structure_id].append(record)

    pk_required_columns = [
        "activity_id",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "assay_description",
        "assay_organism",
        "molecule_chembl_id",
        "standard_inchi_key",
        "document_chembl_id",
        "document_title",
        "activity_source_name",
    ]
    pk_candidates_by_structure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in pk_paths:
        _check_parquet(path, pk_required_columns)
        for row in pq.read_table(path, columns=pk_required_columns).to_pylist():
            matched_structure_id = inchi_to_structure.get(str(row.get("standard_inchi_key") or ""))
            if matched_structure_id:
                record = dict(row)
                record["source_shard"] = path.name
                pk_candidates_by_structure[matched_structure_id].append(record)

    processed_pk_by_structure: dict[str, list[dict[str, str]]] = defaultdict(list)
    with processed_pk_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            matched_structure_id = inchi_to_structure.get(str(row.get("inchi_key") or ""))
            if matched_structure_id:
                processed_pk_by_structure[matched_structure_id].append(dict(row))

    pkdb_decision = json.loads(pkdb_admission_path.read_text(encoding="utf-8"))
    if pkdb_decision.get("canonical_admission") or int(pkdb_decision.get("training_label_rows", 0)):
        raise QtExposurePrepError("PK-DB inventory unexpectedly claims canonical/training admission")

    source_rows: list[dict[str, Any]] = []
    pair_text_evidence: dict[tuple[str, str], dict[str, list[dict[str, str]]]] = {}
    for pair, rows in sorted(linked_interventions.items()):
        aggregate: dict[str, list[dict[str, str]]] = {"dose": [], "route": [], "regimen": []}
        pool_count = len(rows)
        for row in sorted(rows, key=lambda value: value["intervention_candidate_id"]):
            evidence = row["text_evidence"]
            for domain in aggregate:
                aggregate[domain].extend(evidence[domain])
            domains = ["reported_intervention_identity"]
            domains.extend(domain + "_text_candidate" for domain in aggregate if evidence[domain])
            score = (
                25.0
                + 7.0 * bool(evidence["dose"])
                + 6.0 * bool(evidence["route"])
                + 4.0 * bool(evidence["regimen"])
            )
            source_rows.append(
                {
                    "source_review_id": _stable_id(
                        "QTSRC", "clinicaltrials", pair[0], pair[1], row["intervention_candidate_id"]
                    ),
                    "priority_score": score,
                    "structure_id": pair[0],
                    "nct_id": pair[1],
                    "source_family": "clinicaltrials_gov_intervention",
                    "source_record_id": row["intervention_candidate_id"],
                    "source_url": f"https://clinicaltrials.gov/study/{pair[1]}",
                    "evidence_domains_json": _canonical_json(sorted(set(domains))),
                    "source_locator_json": _canonical_json(
                        {
                            "raw_json_pointer": row["raw_json_pointer"],
                            "source_page_path": row["source_page_path"],
                            "source_page_sha256": row["source_page_sha256"],
                        }
                    ),
                    "candidate_evidence_json": _canonical_json(
                        {
                            "intervention_name": row["intervention_name"],
                            "intervention_description": row["intervention_description"],
                            "dose_hits": evidence["dose"],
                            "route_hits": evidence["route"],
                            "regimen_hits": evidence["regimen"],
                        }
                    ),
                    "identity_linkage_basis": "existing_exact_unique_normalized_intervention_name_structure_link",
                    "context_limitations_json": _canonical_json(
                        [
                            "reported text is not normalized dose or PK",
                            "endpoint group to protocol arm and regimen is not assumed",
                            "dose regex can capture formulation strength or administration fluid volume",
                        ]
                    ),
                    "candidate_pool_count": pool_count,
                    "selection_policy": "all_exact_linked_intervention_rows_for_structure_trial",
                    "candidate_status": "source_reported_text_candidate_pending_arm_and_regimen_adjudication",
                    "direct_herg_label": False,
                    "qt_used_as_herg_label": False,
                    "clinical_risk_label": False,
                    "use_as_training_label": False,
                }
            )
        pair_text_evidence[pair] = aggregate

    regulatory_summary: dict[str, dict[str, Any]] = {}
    for structure_id in sorted(target_structures):
        apps = apps_by_structure[structure_id]
        product_rows = [row for app in apps for row in products_by_app.get(app, [])]
        regulatory_summary[structure_id] = {
            "application_numbers": sorted(apps, key=lambda value: (len(value), value)),
            "product_record_count": len(product_rows),
            "dosage_form_routes": sorted(
                {str(row["dosage_form_route_raw"]) for row in product_rows if row["dosage_form_route_raw"]}
            ),
            "strengths": sorted({str(row["strength_raw"]) for row in product_rows if row["strength_raw"]}),
        }
        for app in sorted(apps, key=lambda value: (len(value), value)):
            rows = products_by_app.get(app, [])
            if not rows:
                continue
            domains = []
            if any(row["dosage_form_route_raw"] for row in rows):
                domains.append("regulatory_product_dosage_form_route")
            if any(row["strength_raw"] for row in rows):
                domains.append("regulatory_product_strength")
            source_rows.append(
                {
                    "source_review_id": _stable_id("QTSRC", "drugsfda", structure_id, app),
                    "priority_score": 30.0 + 4.0 * bool(domains),
                    "structure_id": structure_id,
                    "nct_id": None,
                    "source_family": "drugs_at_fda_product",
                    "source_record_id": f"FDAAPP-{app.zfill(6)}",
                    "source_url": (
                        "https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo="
                        f"{app.zfill(6)}"
                    ),
                    "evidence_domains_json": _canonical_json(domains),
                    "source_locator_json": _canonical_json(
                        [
                            {
                                "application_number": str(row["application_number"]),
                                "product_number": str(row["product_number"]),
                                "source_archive": row["source_archive"],
                                "source_member": row["source_member"],
                                "source_row_number_one_based": row["source_row_number_one_based"],
                                "source_field_map_sha256": row["source_field_map_sha256"],
                            }
                            for row in rows
                        ]
                    ),
                    "candidate_evidence_json": _canonical_json(
                        {
                            "dosage_form_routes": sorted(
                                {
                                    str(row["dosage_form_route_raw"])
                                    for row in rows
                                    if row["dosage_form_route_raw"]
                                }
                            ),
                            "strengths": sorted(
                                {str(row["strength_raw"]) for row in rows if row["strength_raw"]}
                            ),
                            "drug_names": sorted(
                                {str(row["drug_name_raw"]) for row in rows if row["drug_name_raw"]}
                            ),
                            "active_ingredients": sorted(
                                {
                                    str(row["active_ingredient_raw"])
                                    for row in rows
                                    if row["active_ingredient_raw"]
                                }
                            ),
                        }
                    ),
                    "identity_linkage_basis": "existing_structure_to_exact_normalized_drugsfda_ingredient_name_link",
                    "context_limitations_json": _canonical_json(
                        [
                            "product strength is not administered clinical-trial dose",
                            "product route/form is not automatically the QT-trial route",
                            "salt mixture parent and analyte identity require review",
                        ]
                    ),
                    "candidate_pool_count": len(rows),
                    "selection_policy": "all_product_rows_for_exact_linked_application",
                    "candidate_status": "regulatory_product_metadata_candidate_not_exposure_or_safety_label",
                    "direct_herg_label": False,
                    "qt_used_as_herg_label": False,
                    "clinical_risk_label": False,
                    "use_as_training_label": False,
                }
            )

    dailymed_summary: dict[str, dict[str, int]] = {}
    for structure_id in sorted(target_structures):
        documents = dailymed_docs_by_structure.get(structure_id, [])
        sections = dailymed_sections_by_structure.get(structure_id, [])
        dailymed_summary[structure_id] = {
            "document_count": len(
                {(row["set_id"], row["version_number"], row["document_id"]) for row in documents}
            ),
            "section_count": len(sections),
            "cmax_count": sum("cmax" in (row.get("endpoint_hits") or []) for row in sections),
            "protein_binding_count": sum(bool(row["bounded_terms"]["protein_binding"]) for row in sections),
            "metabolite_count": sum(bool(row["bounded_terms"]["metabolite"]) for row in sections),
            "unbound_count": sum(bool(row["bounded_terms"]["unbound"]) for row in sections),
        }
        scored_sections: list[tuple[float, dict[str, Any]]] = []
        for section in sections:
            terms = section["bounded_terms"]
            domains = ["dailymed_pk_candidate_section"]
            if "cmax" in (section.get("endpoint_hits") or []):
                domains.append("cmax_candidate")
            domains.extend(domain + "_bounded_span" for domain, hits in terms.items() if hits)
            score = (
                38.0
                + 22.0 * ("cmax_candidate" in domains)
                + 18.0 * bool(terms["unbound"])
                + 14.0 * bool(terms["protein_binding"])
                + 8.0 * bool(terms["metabolite"])
                + 4.0
                * bool(
                    (section.get("context_completeness_flags") or {}).get(
                        "all_core_context_flags_in_same_section"
                    )
                )
            )
            record = dict(section)
            record["review_domains"] = domains
            scored_sections.append((score, record))
        scored_sections.sort(
            key=lambda item: (
                -item[0],
                -int(str(item[1].get("effective_time") or "0") or "0"),
                str(item[1].get("candidate_id") or ""),
            )
        )
        for score, section in scored_sections[:DAILYMED_REVIEW_CAP_PER_STRUCTURE]:
            terms = section["bounded_terms"]
            source_rows.append(
                {
                    "source_review_id": _stable_id(
                        "QTSRC", "dailymed", structure_id, section["candidate_id"]
                    ),
                    "priority_score": score,
                    "structure_id": structure_id,
                    "nct_id": None,
                    "source_family": "dailymed_pk_candidate_section",
                    "source_record_id": str(section["candidate_id"]),
                    "source_url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={section['set_id']}",
                    "evidence_domains_json": _canonical_json(section["review_domains"]),
                    "source_locator_json": _canonical_json(
                        {
                            "matching_application_numbers": section["matching_application_numbers"],
                            "set_id": section["set_id"],
                            "document_id": section["document_id"],
                            "version_number": section["version_number"],
                            "effective_time": section.get("effective_time"),
                            "archive": section["archive"],
                            "outer_member": section["outer_member"],
                            "inner_xml_member": section["inner_xml_member"],
                            "inner_xml_sha256": section["inner_xml_sha256"],
                            "section_id": section["section_id"],
                            "section_code": section.get("section_code"),
                            "section_title": section.get("section_title"),
                        }
                    ),
                    "candidate_evidence_json": _canonical_json(
                        {
                            "endpoint_hits": section.get("endpoint_hits") or [],
                            "bounded_term_hits": terms,
                            "context_completeness_flags": section.get("context_completeness_flags") or {},
                            "selection_reason": section.get("selection_reason"),
                        }
                    ),
                    "identity_linkage_basis": "exact_drugsfda_application_number_overlap_to_existing_structure_annotation",
                    "context_limitations_json": _canonical_json(
                        [
                            "machine-selected candidate section not an adjudicated PK record",
                            "latest available frozen-corpus document is not asserted regulator-preferred or marketed",
                            "protein-binding metabolite and unbound searches are limited to bounded stored evidence spans",
                            "numeric values are not extracted or promoted",
                        ]
                    ),
                    "candidate_pool_count": len(scored_sections),
                    "selection_policy": f"top_{DAILYMED_REVIEW_CAP_PER_STRUCTURE}_per_structure_by_candidate_domain_score_then_effective_time",
                    "candidate_status": "exact_application_linked_dailymed_candidate_pending_identity_and_context_review",
                    "direct_herg_label": False,
                    "qt_used_as_herg_label": False,
                    "clinical_risk_label": False,
                    "use_as_training_label": False,
                }
            )

    chembl_summary: dict[str, dict[str, int]] = {}
    for structure_id in sorted(target_structures):
        rows = pk_candidates_by_structure.get(structure_id, [])
        human = [row for row in rows if row.get("assay_organism") == "Homo sapiens"]
        by_type = Counter(str(row.get("standard_type") or "").casefold() for row in human)
        chembl_summary[structure_id] = {
            "all_candidate_count": len(rows),
            "human_cmax_count": by_type["cmax"],
            "human_fu_count": by_type["fu"],
            "human_ppb_count": by_type["ppb"],
        }
        relevant = [row for row in human if str(row.get("standard_type") or "").casefold() in _PK_TYPES]
        type_pool = Counter(str(row.get("standard_type") or "").casefold() for row in relevant)
        for row in relevant:
            endpoint = str(row.get("standard_type") or "").casefold()
            domain = {
                "cmax": "human_cmax_candidate",
                "fu": "human_fraction_unbound_candidate",
                "ppb": "human_protein_binding_candidate",
            }[endpoint]
            score = 68.0 if endpoint == "cmax" else 64.0
            source_rows.append(
                {
                    "source_review_id": _stable_id("QTSRC", "chembl37_pk", structure_id, row["activity_id"]),
                    "priority_score": score,
                    "structure_id": structure_id,
                    "nct_id": None,
                    "source_family": "chembl37_pk_candidate",
                    "source_record_id": f"CHEMBL_ACTIVITY-{row['activity_id']}",
                    "source_url": (
                        f"https://www.ebi.ac.uk/chembl/explore/document/{row['document_chembl_id']}"
                        if row.get("document_chembl_id")
                        else None
                    ),
                    "evidence_domains_json": _canonical_json([domain]),
                    "source_locator_json": _canonical_json(
                        {
                            "source_shard": row["source_shard"],
                            "activity_id": row["activity_id"],
                            "molecule_chembl_id": row.get("molecule_chembl_id"),
                            "document_chembl_id": row.get("document_chembl_id"),
                            "activity_source_name": row.get("activity_source_name"),
                        }
                    ),
                    "candidate_evidence_json": _canonical_json(
                        {
                            "standard_type": row.get("standard_type"),
                            "standard_relation": row.get("standard_relation"),
                            "standard_value": row.get("standard_value"),
                            "standard_units": row.get("standard_units"),
                            "assay_organism": row.get("assay_organism"),
                            "assay_description": row.get("assay_description"),
                            "document_title": row.get("document_title"),
                        }
                    ),
                    "identity_linkage_basis": "exact_standard_inchi_key_overlap_with_structure_master",
                    "context_limitations_json": _canonical_json(
                        [
                            "specialized-view candidate row is not automatically analysis-ready",
                            "analyte matrix dose route schedule population and time basis require review",
                            "FU can describe tissue cell homogenate serum blood or plasma and is not automatically plasma fu",
                            "PPB-to-FU conversion requires explicit bound-fraction semantics and censoring reversal",
                            "candidate is not automatically applicable to any QT trial",
                        ]
                    ),
                    "candidate_pool_count": type_pool[endpoint],
                    "selection_policy": "all_homo_sapiens_Cmax_FU_or_PPB_candidate_rows_for_exact_InChIKey",
                    "candidate_status": "raw_chembl37_pk_candidate_pending_context_adjudication",
                    "direct_herg_label": False,
                    "qt_used_as_herg_label": False,
                    "clinical_risk_label": False,
                    "use_as_training_label": False,
                }
            )

    structure_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for structure_id in sorted(target_structures):
        qt_structure_rows = qt_by_structure[structure_id]
        methods = sorted(
            {
                str(method)
                for row in qt_structure_rows
                for method in _json_list(row["correction_methods_json"])
            }
        )
        phenotype_classes = sorted({str(row["qt_phenotype_class"]) for row in qt_structure_rows})
        pairs = sorted(pair for pair in target_pairs if pair[0] == structure_id)
        dose_count = sum(len(pair_text_evidence[pair]["dose"]) for pair in pairs)
        route_count = sum(len(pair_text_evidence[pair]["route"]) for pair in pairs)
        regimen_count = sum(len(pair_text_evidence[pair]["regimen"]) for pair in pairs)
        intervention_count = sum(len(linked_interventions[pair]) for pair in pairs)
        reg = regulatory_summary[structure_id]
        dm = dailymed_summary[structure_id]
        chembl = chembl_summary[structure_id]
        processed_count = len(processed_pk_by_structure.get(structure_id, []))
        blockers = [
            "adjudicated_hERG_IC50_missing_from_this_collection_release",
            "adjudicated_human_Cmax_missing",
            "adjudicated_fraction_unbound_missing",
            "analyte_matrix_dose_route_schedule_population_time_basis_compatibility_unresolved",
        ]
        structure_rows.append(
            {
                "structure_id": structure_id,
                "standardized_smiles": structures[structure_id]["standardized_smiles"],
                "standard_inchi_key": structures[structure_id]["standard_inchi_key"],
                "molecular_weight": structures[structure_id]["molecular_weight"],
                "qt_endpoint_count": len(qt_structure_rows),
                "qt_trial_count": len(pairs),
                "qt_phenotype_classes_json": _canonical_json(phenotype_classes),
                "qt_correction_methods_json": _canonical_json(methods),
                "qt_correction_status": _correction_status(methods),
                "clinicaltrials_intervention_record_count": intervention_count,
                "clinicaltrials_dose_text_candidate_count": dose_count,
                "clinicaltrials_route_text_candidate_count": route_count,
                "clinicaltrials_regimen_text_candidate_count": regimen_count,
                "drugsfda_exact_application_count": len(reg["application_numbers"]),
                "drugsfda_product_record_count": reg["product_record_count"],
                "drugsfda_route_or_form_candidate_count": len(reg["dosage_form_routes"]),
                "drugsfda_strength_candidate_count": len(reg["strengths"]),
                "dailymed_exact_application_document_count": dm["document_count"],
                "dailymed_pk_candidate_section_count": dm["section_count"],
                "dailymed_cmax_candidate_section_count": dm["cmax_count"],
                "dailymed_protein_binding_bounded_span_count": dm["protein_binding_count"],
                "dailymed_metabolite_bounded_span_count": dm["metabolite_count"],
                "dailymed_unbound_bounded_span_count": dm["unbound_count"],
                "chembl37_pk_candidate_row_count": chembl["all_candidate_count"],
                "chembl37_human_cmax_candidate_count": chembl["human_cmax_count"],
                "chembl37_human_fraction_unbound_candidate_count": chembl["human_fu_count"],
                "chembl37_human_protein_binding_candidate_count": chembl["human_ppb_count"],
                "processed_pk_observation_count": processed_count,
                "adjudicated_human_cmax_count": 0,
                "adjudicated_fraction_unbound_count": 0,
                "margin_ready": False,
                "readiness_blockers_json": _canonical_json(blockers),
                "evidence_semantics": NO_LABEL_SEMANTICS,
                "direct_herg_label": False,
                "qt_used_as_herg_label": False,
                "clinical_risk_label": False,
                "use_as_training_label": False,
            }
        )

        for pair in pairs:
            qt_pair_rows = qt_by_pair[pair]
            pair_methods = sorted(
                {str(method) for row in qt_pair_rows for method in _json_list(row["correction_methods_json"])}
            )
            evidence = pair_text_evidence[pair]
            study = studies[pair[1]]
            intervention_records = sorted(
                linked_interventions[pair], key=lambda row: row["intervention_candidate_id"]
            )
            pk_summary = {
                "dailymed": dm,
                "chembl37": chembl,
                "processed_pk_observation_count": processed_count,
                "interpretation": "candidate counts only; no value admitted into collection fields",
            }
            pair_rows.append(
                {
                    "collection_id": _stable_id("QTEXP", pair[0], pair[1]),
                    "structure_id": pair[0],
                    "standardized_smiles": structures[structure_id]["standardized_smiles"],
                    "standard_inchi_key": structures[structure_id]["standard_inchi_key"],
                    "nct_id": pair[1],
                    "clinicaltrials_url": f"https://clinicaltrials.gov/study/{pair[1]}",
                    "study_brief_title": study.get("brief_title") or None,
                    "study_phases_json": study.get("phases_json") or "[]",
                    "study_overall_status": study.get("overall_status") or None,
                    "qt_endpoint_count": len(qt_pair_rows),
                    "qt_endpoint_candidate_ids_json": _canonical_json(
                        sorted(str(row["endpoint_candidate_id"]) for row in qt_pair_rows)
                    ),
                    "qt_phenotype_classes_json": _canonical_json(
                        sorted({str(row["qt_phenotype_class"]) for row in qt_pair_rows})
                    ),
                    "qt_correction_methods_json": _canonical_json(pair_methods),
                    "qt_correction_status": _correction_status(pair_methods),
                    "clinicaltrials_intervention_names_json": _canonical_json(
                        sorted({str(row["intervention_name"]) for row in intervention_records})
                    ),
                    "dose_text_candidates_json": _canonical_json(evidence["dose"]),
                    "route_text_candidates_json": _canonical_json(evidence["route"]),
                    "regimen_text_candidates_json": _canonical_json(evidence["regimen"]),
                    "clinicaltrials_source_records_json": _canonical_json(
                        [
                            {
                                "intervention_candidate_id": row["intervention_candidate_id"],
                                "raw_json_pointer": row["raw_json_pointer"],
                                "source_page_path": row["source_page_path"],
                                "source_page_sha256": row["source_page_sha256"],
                            }
                            for row in intervention_records
                        ]
                    ),
                    "regulatory_evidence_summary_json": _canonical_json(reg),
                    "pk_candidate_evidence_summary_json": _canonical_json(pk_summary),
                    "adjudicated_dose_value": None,
                    "adjudicated_dose_unit": None,
                    "adjudicated_dose_relation": None,
                    "adjudicated_route": None,
                    "adjudicated_dose_schedule": None,
                    "adjudicated_population": None,
                    "adjudicated_cmax_value": None,
                    "adjudicated_cmax_unit": None,
                    "adjudicated_cmax_relation": None,
                    "adjudicated_cmax_matrix": None,
                    "adjudicated_cmax_time_basis": None,
                    "adjudicated_cmax_analyte": None,
                    "adjudicated_cmax_source_id": None,
                    "adjudicated_fraction_unbound_value": None,
                    "adjudicated_fraction_unbound_unit": None,
                    "adjudicated_fraction_unbound_relation": None,
                    "adjudicated_fraction_unbound_matrix": None,
                    "adjudicated_fraction_unbound_analyte": None,
                    "adjudicated_fraction_unbound_source_id": None,
                    "adjudicated_active_metabolites_json": None,
                    "adjudicated_herg_ic50_value": None,
                    "adjudicated_herg_ic50_unit": None,
                    "adjudicated_herg_ic50_relation": None,
                    "adjudicated_herg_ic50_source_observation_id": None,
                    "margin_point": None,
                    "margin_lower_bound": None,
                    "margin_upper_bound": None,
                    "margin_status": "not_computed_missing_or_unadjudicated_required_inputs",
                    "evidence_semantics": NO_LABEL_SEMANTICS,
                    "direct_herg_label": False,
                    "qt_used_as_herg_label": False,
                    "clinical_risk_label": False,
                    "use_as_training_label": False,
                }
            )

    structure_by_id = {row["structure_id"]: row for row in structure_rows}
    gap_rows: list[dict[str, Any]] = []
    for pair_row in pair_rows:
        structure = structure_by_id[pair_row["structure_id"]]
        has_cmax = structure["chembl37_human_cmax_candidate_count"] > 0
        has_binding = (
            structure["chembl37_human_fraction_unbound_candidate_count"]
            + structure["chembl37_human_protein_binding_candidate_count"]
            > 0
        )
        dose_present = bool(_json_list(pair_row["dose_text_candidates_json"]))
        route_present = bool(_json_list(pair_row["route_text_candidates_json"]))
        domains = []
        for present, domain in (
            (dose_present, "clinicaltrials_dose_text"),
            (route_present, "clinicaltrials_route_text"),
            (structure["drugsfda_product_record_count"] > 0, "drugsfda_product_metadata"),
            (structure["dailymed_cmax_candidate_section_count"] > 0, "dailymed_cmax_section"),
            (structure["dailymed_protein_binding_bounded_span_count"] > 0, "dailymed_protein_binding_span"),
            (structure["dailymed_metabolite_bounded_span_count"] > 0, "dailymed_metabolite_span"),
            (has_cmax, "chembl_human_cmax_candidate"),
            (has_binding, "chembl_human_fu_or_ppb_candidate"),
        ):
            if present:
                domains.append(domain)
        score = (
            min(10.0, float(pair_row["qt_endpoint_count"]))
            + 40.0 * has_cmax
            + 35.0 * has_binding
            + 25.0 * (has_cmax and has_binding)
            + 8.0 * (structure["dailymed_cmax_candidate_section_count"] > 0)
            + 6.0
            * (
                structure["dailymed_unbound_bounded_span_count"] > 0
                or structure["dailymed_protein_binding_bounded_span_count"] > 0
            )
            + 5.0 * dose_present
            + 5.0 * route_present
            + 4.0 * (pair_row["qt_correction_status"] != "all_endpoints_unresolved")
        )
        if has_cmax and has_binding:
            tier = "P0_candidate_pair_reconciliation"
        elif has_cmax or has_binding:
            tier = "P1_single_numeric_domain_plus_context"
        elif structure["dailymed_pk_candidate_section_count"] > 0 or dose_present or route_present:
            tier = "P2_source_enrichment_and_adjudication"
        else:
            tier = "P3_external_exposure_collection"
        source_links = [{"source": "ClinicalTrials.gov", "url": pair_row["clinicaltrials_url"]}]
        source_links.extend(
            {
                "source": "Drugs@FDA",
                "application_number": app,
                "url": (
                    "https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo="
                    f"{app.zfill(6)}"
                ),
            }
            for app in regulatory_summary[pair_row["structure_id"]]["application_numbers"][:20]
        )
        source_links.extend(
            {
                "source": "DailyMed",
                "set_id": document["set_id"],
                "url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={document['set_id']}",
            }
            for document in sorted(
                dailymed_docs_by_structure.get(pair_row["structure_id"], []),
                key=lambda row: (-int(str(row.get("effective_time") or "0") or "0"), row["set_id"]),
            )[:10]
        )
        gap_rows.append(
            {
                "gap_id": _stable_id("QTGAP", pair_row["structure_id"], pair_row["nct_id"]),
                "structure_id": pair_row["structure_id"],
                "nct_id": pair_row["nct_id"],
                "priority_tier": tier,
                "priority_score": score,
                "qt_endpoint_count": pair_row["qt_endpoint_count"],
                "qt_correction_status": pair_row["qt_correction_status"],
                "candidate_evidence_domains_json": _canonical_json(domains),
                "required_gap_fields_json": _canonical_json(
                    [
                        "endpoint_group_to_intervention_arm_mapping",
                        "adjudicated_dose_route_schedule_population",
                        "adjudicated_human_Cmax_analyte_matrix_time_basis_and_units",
                        "adjudicated_fraction_unbound_or_explicit_unbound_Cmax",
                        "adjudicated_active_metabolite_identity_and_exposure",
                        "adjudicated_hERG_IC50_with_relation_units_and_assay_context",
                        "cross_source_context_compatibility_decision",
                    ]
                ),
                "adjudication_sequence_json": _canonical_json(
                    [
                        "confirm exact parent_salt_prodrug_or_metabolite identity",
                        "map QT endpoint result groups to protocol arms and intervention regimen",
                        "adjudicate human Cmax dose route matrix population schedule and time basis",
                        "adjudicate fraction unbound in the compatible analyte and matrix",
                        "adjudicate hERG IC50 endpoint relation units and assay scope",
                        "apply interval margin contract only after all hard blocks clear",
                    ]
                ),
                "source_review_links_json": _canonical_json(source_links),
                "margin_status": "not_computed_missing_or_unadjudicated_required_inputs",
                "direct_herg_label": False,
                "qt_used_as_herg_label": False,
                "clinical_risk_label": False,
                "use_as_training_label": False,
            }
        )

    gap_rows.sort(key=lambda row: (-row["priority_score"], row["structure_id"], row["nct_id"]))
    for rank, row in enumerate(gap_rows, start=1):
        row["priority_rank"] = rank

    source_rows.sort(
        key=lambda row: (
            -row["priority_score"],
            row["structure_id"],
            row["nct_id"] or "",
            row["source_family"],
            row["source_record_id"],
        )
    )
    for rank, row in enumerate(source_rows, start=1):
        row["priority_rank"] = rank
        score = row["priority_score"]
        row["priority_tier"] = "P0" if score >= 90 else "P1" if score >= 60 else "P2" if score >= 35 else "P3"

    cohort = {
        "qt_endpoints": len(qt_rows),
        "structures": len(target_structures),
        "trials": len({row["nct_id"] for row in qt_rows}),
        "structure_trials": len(target_pairs),
    }
    coverage = {
        "structures_with_any_resolved_qt_correction": sum(
            row["qt_correction_status"] != "all_endpoints_unresolved" for row in structure_rows
        ),
        "structure_trials_with_dose_text": sum(
            bool(_json_list(row["dose_text_candidates_json"])) for row in pair_rows
        ),
        "structure_trials_with_route_text": sum(
            bool(_json_list(row["route_text_candidates_json"])) for row in pair_rows
        ),
        "structures_with_drugsfda_applications": sum(
            row["drugsfda_exact_application_count"] > 0 for row in structure_rows
        ),
        "structures_with_dailymed_documents": sum(
            row["dailymed_exact_application_document_count"] > 0 for row in structure_rows
        ),
        "structures_with_dailymed_cmax_sections": sum(
            row["dailymed_cmax_candidate_section_count"] > 0 for row in structure_rows
        ),
        "structures_with_dailymed_protein_binding_spans": sum(
            row["dailymed_protein_binding_bounded_span_count"] > 0 for row in structure_rows
        ),
        "structures_with_dailymed_metabolite_spans": sum(
            row["dailymed_metabolite_bounded_span_count"] > 0 for row in structure_rows
        ),
        "structures_with_human_cmax_candidates": sum(
            row["chembl37_human_cmax_candidate_count"] > 0 for row in structure_rows
        ),
        "structures_with_human_binding_candidates": sum(
            row["chembl37_human_fraction_unbound_candidate_count"]
            + row["chembl37_human_protein_binding_candidate_count"]
            > 0
            for row in structure_rows
        ),
        "structures_with_both_human_candidate_types": sum(
            row["chembl37_human_cmax_candidate_count"] > 0
            and row["chembl37_human_fraction_unbound_candidate_count"]
            + row["chembl37_human_protein_binding_candidate_count"]
            > 0
            for row in structure_rows
        ),
        "structures_with_processed_pk_observations": sum(
            row["processed_pk_observation_count"] > 0 for row in structure_rows
        ),
        "structures_with_any_chembl_pk_candidates": sum(
            row["chembl37_pk_candidate_row_count"] > 0 for row in structure_rows
        ),
    }
    source_counts = Counter(row["source_family"] for row in source_rows)
    gap_counts = Counter(row["priority_tier"] for row in gap_rows)

    output_dir.mkdir(parents=True, exist_ok=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = [
        _write_parquet(output_dir / STRUCTURE_OUTPUT, structure_rows, STRUCTURE_SCHEMA),
        _write_parquet(output_dir / STRUCTURE_TRIAL_OUTPUT, pair_rows, STRUCTURE_TRIAL_SCHEMA),
        _write_parquet(output_dir / GAP_OUTPUT, gap_rows, GAP_SCHEMA),
        _write_parquet(output_dir / SOURCE_OUTPUT, source_rows, SOURCE_SCHEMA),
    ]
    contract_path = output_dir / MARGIN_CONTRACT_OUTPUT
    contract_path.write_text(_canonical_json(_margin_contract()) + "\n", encoding="utf-8")
    artifacts.append(_artifact(contract_path))

    report_path.write_text(
        _report_text(
            cohort=cohort,
            coverage=coverage,
            source_counts=source_counts,
            gap_counts=gap_counts,
            pkdb_decision=pkdb_decision,
        ),
        encoding="utf-8",
    )

    input_paths = [
        ("builder_implementation", Path(__file__).resolve()),
        ("qt_clinical_phenotype_index", qt_index_path),
        ("clinical_context_master", clinical_context_master_path),
        ("structure_master", structure_master_path),
        ("exact_name_structure_link_audit", link_audit_path),
        ("structure_development_annotations", structure_annotations_path),
        ("clinicaltrials_interventions", interventions_path),
        ("clinicaltrials_studies", studies_path),
        ("drugsfda_products", drugsfda_products_path),
        ("dailymed_evidence_manifest", dailymed_manifest_path),
        ("dailymed_latest_documents", dailymed_by_name["latest_available_candidate_documents.jsonl"]),
        ("dailymed_section_candidates", dailymed_by_name["section_candidates.jsonl"]),
        ("chembl_pk_candidate_manifest", chembl_pk_manifest_path),
        ("processed_pk_observations", processed_pk_path),
        ("pkdb_admission_decision", pkdb_admission_path),
    ]
    input_paths.extend((f"chembl_pk_candidate_shard:{path.name}", path) for path in pk_paths)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "release_name": "v1_5_qt_exposure_prep",
        "cohort": cohort,
        "coverage": coverage,
        "source_queue_counts": dict(sorted(source_counts.items())),
        "gap_queue_counts": dict(sorted(gap_counts.items())),
        "daily_med_review_cap_per_structure": DAILYMED_REVIEW_CAP_PER_STRUCTURE,
        "input_bindings": [_input_binding(role, path) for role, path in input_paths],
        "artifacts": artifacts,
        "report": {
            "path": str(report_path),
            "bytes": report_path.stat().st_size,
            "sha256": _sha256_file(report_path),
        },
        "safety_contract": {
            "adjudicated_exposure_values": 0,
            "computed_margin_values": 0,
            "direct_herg_labels": 0,
            "qt_used_as_herg_labels": 0,
            "clinical_risk_labels": 0,
            "training_labels": 0,
            "candidate_evidence_only": True,
            "unknown_is_not_negative": True,
        },
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
    verify_qt_exposure_prep(output_dir, report_path=report_path, verify_inputs=True)
    return manifest


def verify_qt_exposure_prep(
    output_dir: Path, *, report_path: Path | None = None, verify_inputs: bool = True
) -> dict[str, Any]:
    """Verify hashes, schemas, cohort topology, null templates, and zero labels."""

    if output_dir.is_symlink() or not output_dir.is_dir():
        raise QtExposurePrepError(f"missing or unsafe output directory: {output_dir}")
    manifest_path = output_dir / MANIFEST_NAME
    _checked_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("parser_version") != PARSER_VERSION:
        raise QtExposurePrepError("manifest schema/parser version mismatch")
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise QtExposurePrepError("manifest self-hash mismatch")

    expected_output_names = {
        MANIFEST_NAME,
        STRUCTURE_OUTPUT,
        STRUCTURE_TRIAL_OUTPUT,
        GAP_OUTPUT,
        SOURCE_OUTPUT,
        MARGIN_CONTRACT_OUTPUT,
    }
    actual_output_names = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual_output_names != expected_output_names or any(
        path.is_symlink() or not path.is_file() for path in output_dir.iterdir()
    ):
        raise QtExposurePrepError("output directory has unexpected, missing, or unsafe members")

    expected_schemas = {
        STRUCTURE_OUTPUT: STRUCTURE_SCHEMA,
        STRUCTURE_TRIAL_OUTPUT: STRUCTURE_TRIAL_SCHEMA,
        GAP_OUTPUT: GAP_SCHEMA,
        SOURCE_OUTPUT: SOURCE_SCHEMA,
    }
    artifacts = {str(row["path"]): row for row in manifest.get("artifacts") or []}
    expected_artifacts = set(expected_schemas) | {MARGIN_CONTRACT_OUTPUT}
    if set(artifacts) != expected_artifacts:
        raise QtExposurePrepError("manifest artifact set mismatch")
    for name, entry in artifacts.items():
        path = output_dir / name
        _checked_file(path)
        if path.stat().st_size != int(entry["bytes"]) or _sha256_file(path) != entry["sha256"]:
            raise QtExposurePrepError(f"artifact byte/hash mismatch: {name}")
        if name in expected_schemas:
            schema = pq.ParquetFile(path).schema_arrow
            if not schema.equals(expected_schemas[name], check_metadata=True):
                raise QtExposurePrepError(f"artifact schema mismatch: {name}")
            if int(entry["rows"]) != pq.ParquetFile(path).metadata.num_rows:
                raise QtExposurePrepError(f"artifact row-count mismatch: {name}")
            if entry["arrow_schema_sha256"] != _schema_hash(expected_schemas[name]):
                raise QtExposurePrepError(f"artifact schema hash mismatch: {name}")

    contract = json.loads((output_dir / MARGIN_CONTRACT_OUTPUT).read_text(encoding="utf-8"))
    if contract != _margin_contract():
        raise QtExposurePrepError("margin contract content mismatch")
    if contract["missing_input_output"]["margin_point"] is not None:
        raise QtExposurePrepError("missing-input margin policy is not null")

    structure_rows = pq.read_table(output_dir / STRUCTURE_OUTPUT).to_pylist()
    pair_rows = pq.read_table(output_dir / STRUCTURE_TRIAL_OUTPUT).to_pylist()
    gap_rows = pq.read_table(output_dir / GAP_OUTPUT).to_pylist()
    source_rows = pq.read_table(output_dir / SOURCE_OUTPUT).to_pylist()
    cohort = manifest["cohort"]
    if len(structure_rows) != int(cohort["structures"]):
        raise QtExposurePrepError("structure cohort count mismatch")
    if len(pair_rows) != int(cohort["structure_trials"]):
        raise QtExposurePrepError("structure-trial cohort count mismatch")
    if sum(int(row["qt_endpoint_count"]) for row in pair_rows) != int(cohort["qt_endpoints"]):
        raise QtExposurePrepError("QT endpoint count does not reconcile through structure-trial rows")
    if len({row["structure_id"] for row in structure_rows}) != len(structure_rows):
        raise QtExposurePrepError("structure template is not one row per structure")
    pair_keys = {(row["structure_id"], row["nct_id"]) for row in pair_rows}
    if len(pair_keys) != len(pair_rows):
        raise QtExposurePrepError("structure-trial template keys are not unique")
    if {(row["structure_id"], row["nct_id"]) for row in gap_rows} != pair_keys:
        raise QtExposurePrepError("gap queue does not cover every structure-trial exactly")
    if len({row["gap_id"] for row in gap_rows}) != len(gap_rows):
        raise QtExposurePrepError("gap IDs are not unique")
    if len({row["source_review_id"] for row in source_rows}) != len(source_rows):
        raise QtExposurePrepError("source review IDs are not unique")
    if [row["priority_rank"] for row in gap_rows] != list(range(1, len(gap_rows) + 1)):
        raise QtExposurePrepError("gap ranks are not contiguous")
    if [row["priority_rank"] for row in source_rows] != list(range(1, len(source_rows) + 1)):
        raise QtExposurePrepError("source ranks are not contiguous")

    for row in structure_rows:
        if (
            row["margin_ready"]
            or row["adjudicated_human_cmax_count"]
            or row["adjudicated_fraction_unbound_count"]
        ):
            raise QtExposurePrepError("structure template unexpectedly contains adjudicated margin inputs")
        for field in (
            "qt_phenotype_classes_json",
            "qt_correction_methods_json",
            "readiness_blockers_json",
        ):
            _json_list(row[field])
    for row in pair_rows:
        if any(row[field] is not None for field in _NULL_TEMPLATE_FIELDS):
            raise QtExposurePrepError("collection template contains a populated adjudicated or margin field")
        if row["margin_status"] != "not_computed_missing_or_unadjudicated_required_inputs":
            raise QtExposurePrepError("collection template margin status is unsafe")
        for field in (
            "study_phases_json",
            "qt_endpoint_candidate_ids_json",
            "qt_phenotype_classes_json",
            "qt_correction_methods_json",
            "clinicaltrials_intervention_names_json",
            "dose_text_candidates_json",
            "route_text_candidates_json",
            "regimen_text_candidates_json",
            "clinicaltrials_source_records_json",
        ):
            _json_list(row[field])
        for field in ("regulatory_evidence_summary_json", "pk_candidate_evidence_summary_json"):
            if not isinstance(json.loads(row[field]), dict):
                raise QtExposurePrepError(f"expected JSON object in {field}")

    for rows in (structure_rows, pair_rows, gap_rows, source_rows):
        if any(
            row["direct_herg_label"]
            or row["qt_used_as_herg_label"]
            or row["clinical_risk_label"]
            or row["use_as_training_label"]
            for row in rows
        ):
            raise QtExposurePrepError("an output violates the zero-label contract")
    if any(
        row["structure_id"] not in {value["structure_id"] for value in structure_rows} for row in source_rows
    ):
        raise QtExposurePrepError("source queue contains a structure outside the cohort")

    actual_source_counts = dict(sorted(Counter(row["source_family"] for row in source_rows).items()))
    actual_gap_counts = dict(sorted(Counter(row["priority_tier"] for row in gap_rows).items()))
    if actual_source_counts != manifest["source_queue_counts"]:
        raise QtExposurePrepError("source queue counts do not match manifest")
    if actual_gap_counts != manifest["gap_queue_counts"]:
        raise QtExposurePrepError("gap queue counts do not match manifest")
    safety = manifest["safety_contract"]
    if any(
        int(safety[key]) != 0
        for key in (
            "adjudicated_exposure_values",
            "computed_margin_values",
            "direct_herg_labels",
            "qt_used_as_herg_labels",
            "clinical_risk_labels",
            "training_labels",
        )
    ):
        raise QtExposurePrepError("manifest safety contract is nonzero")

    bound_report = Path(manifest["report"]["path"])
    if report_path is not None and bound_report.resolve() != report_path.resolve():
        raise QtExposurePrepError("report path does not match manifest")
    _checked_file(bound_report)
    if (
        bound_report.stat().st_size != int(manifest["report"]["bytes"])
        or _sha256_file(bound_report) != manifest["report"]["sha256"]
    ):
        raise QtExposurePrepError("report byte/hash mismatch")

    if verify_inputs:
        for binding in manifest["input_bindings"]:
            path = Path(binding["path"])
            _checked_file(path)
            if path.stat().st_size != int(binding["bytes"]) or _sha256_file(path) != binding["sha256"]:
                raise QtExposurePrepError(f"input binding mismatch: {binding['role']}")
            if path.suffix == ".parquet":
                parquet = pq.ParquetFile(path)
                if (
                    parquet.metadata.num_rows != int(binding["rows"])
                    or _schema_hash(parquet.schema_arrow) != binding["arrow_schema_sha256"]
                ):
                    raise QtExposurePrepError(f"input Parquet row/schema mismatch: {binding['role']}")
        qt_binding = next(
            row for row in manifest["input_bindings"] if row["role"] == "qt_clinical_phenotype_index"
        )
        qt_source = pq.read_table(
            Path(qt_binding["path"]), columns=["candidate_id", "structure_id", "nct_id"]
        )
        if len(qt_source) != int(cohort["qt_endpoints"]):
            raise QtExposurePrepError("bound QT input row count changed")
        if len(set(qt_source.column("structure_id").to_pylist())) != int(cohort["structures"]):
            raise QtExposurePrepError("bound QT input structure count changed")

    return {
        "all_passed": True,
        "schema_version": SCHEMA_VERSION,
        "cohort": cohort,
        "artifact_count": len(artifacts),
        "source_queue_rows": len(source_rows),
        "gap_queue_rows": len(gap_rows),
        "computed_margin_values": 0,
        "training_labels": 0,
    }


def _default_paths(root: Path) -> dict[str, Path]:
    return {
        "output_dir": root / "research/data/platform/processed/herg_hierarchy/v1_5_qt_exposure_prep",
        "report_path": root
        / "research/reports/platform/herg_paper/qt_exposure_prep_v1_5/QT_EXPOSURE_PREP.md",
        "qt_index_path": root
        / "research/data/platform/processed/herg_hierarchy/v1_2_modality_qt/qt_clinical_phenotype_index.parquet",
        "clinical_context_master_path": root
        / "research/data/platform/processed/herg_hierarchy/v1_3_master/clinical_context_master.parquet",
        "structure_master_path": root
        / "research/data/platform/processed/herg_hierarchy/v1_3_master/structure_master.parquet",
        "link_audit_path": root
        / "research/data/platform/processed/herg_hierarchy/v1_clinical_links/exact_name_structure_link_audit.parquet",
        "structure_annotations_path": root
        / "research/data/platform/processed/herg_hierarchy/v1_clinical_links/structure_development_annotations.parquet",
        "interventions_path": root
        / "research/data/platform/interim/clinical_results_candidates/interventions.csv.gz",
        "studies_path": root / "research/data/platform/interim/clinical_results_candidates/studies.csv.gz",
        "drugsfda_products_path": root
        / "research/data/platform/interim/regulatory_record_candidates/drugs_at_fda_20260804/products.parquet",
        "dailymed_manifest_path": root
        / "research/data/platform/raw/external_public/pk_expansion/avicenna/dailymed_pk_candidate_evidence/manifest.json",
        "chembl_pk_manifest_path": root
        / "research/data/platform/interim/chembl_37_bulk/specialized_views/pk_adme_candidates_manifest.json",
        "processed_pk_path": root / "research/data/processed/pk_admet_observations.csv",
        "pkdb_admission_path": root
        / "research/data/platform/interim/pkdb_candidates/admission_decision.json",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--skip-input-hash-verification", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = _default_paths(args.root)
    if args.output_dir is not None:
        paths["output_dir"] = args.output_dir
    if args.report_path is not None:
        paths["report_path"] = args.report_path
    if args.command == "build":
        result = build_qt_exposure_prep(**paths)
    else:
        result = verify_qt_exposure_prep(
            paths["output_dir"],
            report_path=paths["report_path"],
            verify_inputs=not args.skip_input_hash_verification,
        )
    print(_canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
