"""Deterministic candidate extraction from frozen ClinicalTrials.gov results.

This module is intentionally fail-closed and pre-canonical.  It inventories the
four manifest-bound cardiac-safety cohort pages, preserves reported result
context, and flags textual QT/QTc or pharmacokinetic candidates for manual
review.  It never turns an absent result, a zero count, or cohort membership
into a negative label; never resolves intervention text to a molecule; and
never admits canonical observations or trains a model.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = "platform-clinical-results-candidates/1.0"
PARSER_VERSION = "platform_clinical_results/1.0"
SOURCE_SCHEMA_VERSION = "platform-external-acquisition/1.0"
SOURCE_ID = "clinicaltrials_gov_v2"
DEFAULT_SNAPSHOT_KEY = "api-2.0.5__data-2026-08-04T09-00-05"
MANIFEST_NAME = "clinicaltrials_gov_v2_manifest.json"
OUTPUT_MANIFEST_NAME = "clinical_results_candidates_manifest.json"
ABSENCE_SEMANTICS = "not_reported_or_not_present_is_unknown_not_negative"
ADMISSION_STATUS = "candidate_only_not_canonical_not_model_label"
PAGE_RE = re.compile(r"page_(?P<index>\d{6})\.json$")
NCT_RE = re.compile(r"^NCT\d{8}$")
csv.field_size_limit(1024 * 1024 * 1024)


class ClinicalResultsError(RuntimeError):
    """Raised when immutable inputs or candidate outputs fail reconciliation."""


@dataclass(frozen=True)
class SourcePage:
    relative_path: str
    physical_path: Path
    page_index: int
    bytes: int
    sha256: str


@dataclass(frozen=True)
class InputBinding:
    source_root: Path
    source_manifest_path: Path
    source_manifest_sha256: str
    source_manifest_bytes: int
    source_manifest_internal_sha256: str
    source_api_version: str
    source_data_timestamp: str
    source_snapshot_key: str
    concatenated_page_bytes_sha256: str
    reported_total_count: int
    pages: tuple[SourcePage, ...]

    def as_record(self) -> dict[str, Any]:
        return {
            "source_id": SOURCE_ID,
            "source_manifest_path": MANIFEST_NAME,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_manifest_bytes": self.source_manifest_bytes,
            "source_manifest_internal_sha256": self.source_manifest_internal_sha256,
            "source_api_version": self.source_api_version,
            "source_data_timestamp": self.source_data_timestamp,
            "source_snapshot_key": self.source_snapshot_key,
            "concatenated_page_bytes_sha256": self.concatenated_page_bytes_sha256,
            "reported_total_count": self.reported_total_count,
            "pages": [
                {
                    "relative_path": page.relative_path,
                    "page_index": page.page_index,
                    "bytes": page.bytes,
                    "sha256": page.sha256,
                }
                for page in self.pages
            ],
        }


@dataclass(frozen=True)
class CsvArtifact:
    path: str
    rows: int
    bytes: int
    sha256: str
    header_sha256: str

    def as_record(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "artifact_role": "deterministic_gzip_normalized_candidate_inventory_csv",
            "rows": self.rows,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "header_sha256": self.header_sha256,
        }


COMMON_FIELDS = (
    "source_id",
    "source_api_version",
    "source_data_timestamp",
    "source_snapshot_key",
    "source_manifest_sha256",
    "source_page_path",
    "source_page_sha256",
    "source_page_index",
    "study_index_within_page",
    "nct_id",
    "raw_json_pointer",
)

CSV_FIELDS: dict[str, tuple[str, ...]] = {
    "studies.csv.gz": COMMON_FIELDS
    + (
        "study_candidate_id",
        "study_raw_sha256",
        "has_results_reported",
        "study_type",
        "phases_json",
        "overall_status",
        "brief_title",
        "official_title",
        "lead_sponsor_name",
        "lead_sponsor_class",
        "conditions_json",
        "keywords_json",
        "start_date",
        "primary_completion_date",
        "completion_date",
        "study_first_post_date",
        "results_first_post_date",
        "last_update_post_date",
        "posted_outcome_measure_count",
        "posted_adverse_event_group_count",
        "target_endpoint_candidate_count",
        "unknown_or_missing_fields_json",
        "absence_semantics",
        "admission_status",
    ),
    "interventions.csv.gz": COMMON_FIELDS
    + (
        "intervention_candidate_id",
        "intervention_index",
        "intervention_type",
        "intervention_name",
        "intervention_description",
        "reported_arm_group_labels_json",
        "identity_resolution_status",
        "linkage_status",
        "unknown_or_missing_fields_json",
        "admission_status",
    ),
    "arms_groups.csv.gz": COMMON_FIELDS
    + (
        "group_candidate_id",
        "group_source_kind",
        "context_candidate_id",
        "context_index",
        "group_index",
        "reported_group_id",
        "group_title",
        "group_description",
        "linkage_scope",
        "unknown_or_missing_fields_json",
        "admission_status",
    ),
    "outcome_measures.csv.gz": COMMON_FIELDS
    + (
        "outcome_candidate_id",
        "outcome_index",
        "outcome_type",
        "title",
        "description",
        "population_description",
        "reporting_status",
        "parameter_type",
        "dispersion_type",
        "unit_of_measure",
        "time_frame",
        "group_ids_json",
        "denominator_records_json",
        "denominator_record_count",
        "measurement_record_count",
        "candidate_domains_json",
        "candidate_classifications_json",
        "evidence_phrases_json",
        "unknown_or_missing_fields_json",
        "absence_semantics",
        "admission_status",
    ),
    "adverse_event_groups.csv.gz": COMMON_FIELDS
    + (
        "adverse_event_group_candidate_id",
        "group_index",
        "reported_group_id",
        "group_title",
        "group_description",
        "module_time_frame",
        "frequency_threshold",
        "deaths_num_affected",
        "deaths_num_at_risk",
        "serious_num_affected",
        "serious_num_at_risk",
        "other_num_affected",
        "other_num_at_risk",
        "unknown_or_missing_fields_json",
        "zero_counts_are_reported_values_not_study_level_negatives",
        "admission_status",
    ),
    "endpoint_candidates.csv.gz": COMMON_FIELDS
    + (
        "endpoint_candidate_id",
        "record_kind",
        "parent_candidate_id",
        "target_domain",
        "candidate_classification",
        "genuine_endpoint_candidate",
        "manual_review_required",
        "title_or_term",
        "description_or_organ_system",
        "unit_of_measure",
        "time_frame",
        "denominator_records_json",
        "value_records_json",
        "evidence_phrases_json",
        "unknown_or_missing_fields_json",
        "absence_semantics",
        "zero_counts_are_reported_values_not_study_level_negatives",
        "identity_and_linkage_status",
        "admission_status",
    ),
}

QT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("qtc", re.compile(r"\bQTc(?:F|B|I)?\b", re.IGNORECASE)),
    (
        "qt_specific",
        re.compile(
            r"\bQT(?:\s|-)*(?:interval|duration|prolong(?:ation|ed)|change|dispersion)\b",
            re.IGNORECASE,
        ),
    ),
    ("corrected_qt", re.compile(r"\bcorrected\s+QT\b", re.IGNORECASE)),
    ("long_qt", re.compile(r"\blong\s+QT\b", re.IGNORECASE)),
    ("correction_method", re.compile(r"\b(?:Fridericia|Bazett)\b", re.IGNORECASE)),
    ("torsade", re.compile(r"\btorsades?\s+de\s+pointes\b|\bTdP\b", re.IGNORECASE)),
    ("repolarization", re.compile(r"\bcardiac\s+repolari[sz]ation\b", re.IGNORECASE)),
)

PK_DIRECT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("auc", re.compile(r"\bAUC(?:[_\-0-9A-Za-z]*)\b", re.IGNORECASE)),
    ("cmax", re.compile(r"\bC\s*max\b", re.IGNORECASE)),
    ("tmax", re.compile(r"\bT\s*max\b", re.IGNORECASE)),
    ("cmin", re.compile(r"\bC\s*(?:min|trough|avg|ss)\b", re.IGNORECASE)),
    ("half_life", re.compile(r"\b(?:terminal\s+)?half[- ]life\b|\bt\s*1/2\b", re.IGNORECASE)),
    ("volume_distribution", re.compile(r"\bvolume\s+of\s+distribution\b|\bVdss\b", re.IGNORECASE)),
    ("bioavailability", re.compile(r"\bbioavailability\b", re.IGNORECASE)),
    ("fraction_unbound", re.compile(r"\bfraction\s+unbound\b|\bplasma\s+protein\s+binding\b", re.IGNORECASE)),
    ("accumulation_ratio", re.compile(r"\baccumulation\s+ratio\b", re.IGNORECASE)),
    ("mean_residence_time", re.compile(r"\bmean\s+residence\s+time\b", re.IGNORECASE)),
)

PK_CONTEXT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pharmacokinetic", re.compile(r"\bpharmacokinetics?\b", re.IGNORECASE)),
    ("pk_abbreviation", re.compile(r"\bPK\b")),
    (
        "concentration",
        re.compile(
            r"\b(?:plasma|serum|blood|whole[- ]blood|drug)\s+concentrations?\b|"
            r"\bconcentration[- ]time\b|\bsteady[- ]state\s+concentrations?\b",
            re.IGNORECASE,
        ),
    ),
)

CLEARANCE_RE = re.compile(r"\bclearance\b", re.IGNORECASE)
CLEARANCE_EXCLUSION_RE = re.compile(
    r"\b(?:creatinine|viral|virus|bacterial|parasite|parasitic|tumou?r|mucociliary|"
    r"pathogen|infection)\s+clearance\b",
    re.IGNORECASE,
)
PARTICIPANT_UNIT_RE = re.compile(r"\b(?:participants?|subjects?|patients?|events?|cases?)\b", re.IGNORECASE)
QT_NUMERIC_UNIT_RE = re.compile(r"\b(?:ms|msec|milliseconds?|seconds?)\b", re.IGNORECASE)
PK_QUANTITATIVE_UNIT_RE = re.compile(
    r"(?:ng|pg|mcg|µg|ug|mg|g|nmol|pmol|umol|µmol|mol).*(?:/|\bper\b)|"
    r"(?:/\s*(?:ml|l)\b)|(?:\b(?:hours?|hrs?|days?|minutes?|min)\b)|(?:l|ml)\s*/\s*(?:h|hr|min)",
    re.IGNORECASE,
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_json_text(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def sha256_file(path: str | os.PathLike[str], *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_with_sha256(document: Mapping[str, Any]) -> dict[str, Any]:
    body = {str(key): value for key, value in document.items() if key != "manifest_sha256"}
    body["manifest_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return body


def verify_document_sha256(document: Mapping[str, Any]) -> bool:
    expected = document.get("manifest_sha256")
    if not isinstance(expected, str) or not expected:
        return False
    body = {str(key): value for key, value in document.items() if key != "manifest_sha256"}
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest() == expected


def _write_identified_json(path: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    identified = document_with_sha256(document)
    path.write_text(
        json.dumps(identified, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return identified


def _safe_relative_path(value: Any) -> PurePosixPath:
    relative = PurePosixPath(str(value))
    if not str(relative) or relative.is_absolute() or ".." in relative.parts:
        raise ClinicalResultsError(f"unsafe relative path: {relative}")
    return relative


def _load_json_object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ClinicalResultsError(f"unreadable {context}: {path}") from error
    if not isinstance(value, dict):
        raise ClinicalResultsError(f"{context} is not a JSON object: {path}")
    return value


def bind_inputs(source_root: Path, *, snapshot_key: str = DEFAULT_SNAPSHOT_KEY) -> InputBinding:
    """Bind exactly four cardiac cohort pages to the source acquisition manifest."""

    if source_root.is_symlink() or not source_root.is_dir():
        raise ClinicalResultsError(f"missing or symlinked source root: {source_root}")
    source_root = source_root.resolve()
    manifest_path = source_root / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ClinicalResultsError(f"missing or symlinked source manifest: {manifest_path}")
    manifest = _load_json_object(manifest_path, context="source manifest")
    if manifest.get("schema_version") != SOURCE_SCHEMA_VERSION or manifest.get("source_id") != SOURCE_ID:
        raise ClinicalResultsError("source schema or identity mismatch")
    if not verify_document_sha256(manifest):
        raise ClinicalResultsError("source manifest internal SHA-256 failed")
    bundle = manifest.get("bundle_inventory")
    if not isinstance(bundle, dict) or not isinstance(bundle.get("entries"), list):
        raise ClinicalResultsError("source manifest lacks bundle inventory")
    entries = bundle["entries"]
    if hashlib.sha256(canonical_json_bytes(entries)).hexdigest() != bundle.get("entries_sha256"):
        raise ClinicalResultsError("source bundle entry digest failed")
    if int(bundle.get("entry_count", -1)) != len(entries):
        raise ClinicalResultsError("source bundle entry count failed")

    cohort = manifest.get("cardiac_safety_heuristic_cohort")
    if not isinstance(cohort, dict) or cohort.get("cohort_snapshot_key") != snapshot_key:
        raise ClinicalResultsError("cardiac cohort snapshot mismatch")
    prefix = f"cardiac_safety_heuristic_cohorts/{snapshot_key}/pages/"
    selected: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ClinicalResultsError("non-object bundle entry")
        relative = _safe_relative_path(entry.get("path"))
        name = relative.name
        if relative.as_posix().startswith(prefix) and PAGE_RE.fullmatch(name):
            selected.append(entry)
    selected.sort(key=lambda item: str(item["path"]))
    if len(selected) != 4 or int(cohort.get("page_count", -1)) != 4:
        raise ClinicalResultsError(
            f"expected exactly four manifest-bound cohort pages, found {len(selected)}"
        )

    pages: list[SourcePage] = []
    concatenated = hashlib.sha256()
    for entry in selected:
        relative = _safe_relative_path(entry["path"])
        match = PAGE_RE.fullmatch(relative.name)
        assert match is not None
        physical = source_root / Path(*relative.parts)
        if physical.is_symlink() or not physical.is_file():
            raise ClinicalResultsError(f"missing or symlinked source page: {relative}")
        size = physical.stat().st_size
        if size != int(entry.get("bytes", -1)):
            raise ClinicalResultsError(f"source page byte count changed: {relative}")
        digest = hashlib.sha256()
        with physical.open("rb") as handle:
            for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
                concatenated.update(chunk)
        page_sha = digest.hexdigest()
        if page_sha != entry.get("sha256"):
            raise ClinicalResultsError(f"source page SHA-256 changed: {relative}")
        pages.append(
            SourcePage(
                relative_path=relative.as_posix(),
                physical_path=physical,
                page_index=int(match.group("index")),
                bytes=size,
                sha256=page_sha,
            )
        )
    if [page.page_index for page in pages] != [0, 1, 2, 3]:
        raise ClinicalResultsError("cohort page index sequence is not exactly 000000..000003")
    concat_sha = concatenated.hexdigest()
    if concat_sha != cohort.get("concatenated_raw_page_bytes_sha256"):
        raise ClinicalResultsError("concatenated source page SHA-256 failed")
    return InputBinding(
        source_root=source_root,
        source_manifest_path=manifest_path,
        source_manifest_sha256=sha256_file(manifest_path),
        source_manifest_bytes=manifest_path.stat().st_size,
        source_manifest_internal_sha256=str(manifest["manifest_sha256"]),
        source_api_version=str(manifest.get("api_version", "")),
        source_data_timestamp=str(manifest.get("dataTimestamp", "")),
        source_snapshot_key=snapshot_key,
        concatenated_page_bytes_sha256=concat_sha,
        reported_total_count=int(cohort.get("reported_total_count", -1)),
        pages=tuple(pages),
    )


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _json(value: Any) -> str:
    return canonical_json_text(value)


def _sha_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256(canonical_json_bytes([_text(part) for part in parts])).hexdigest()[:24]
    return f"{prefix}-{digest.upper()}"


def _pointer_escape(value: Any) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _missing(mapping: Mapping[str, Any], fields: Iterable[str]) -> list[str]:
    return [field for field in fields if field not in mapping or mapping.get(field) in (None, "", [])]


def _date(module: Mapping[str, Any], key: str) -> str:
    value = module.get(key)
    return _text(value.get("date")) if isinstance(value, dict) else ""


def _evidence_for_patterns(
    fields: Sequence[tuple[str, str]], patterns: Sequence[tuple[str, re.Pattern[str]]]
) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for field_name, text_value in fields:
        for pattern_name, pattern in patterns:
            for match in pattern.finditer(text_value):
                key = (field_name, pattern_name, match.group(0).casefold())
                if key not in seen:
                    seen.add(key)
                    evidence.append({"field": field_name, "pattern": pattern_name, "phrase": match.group(0)})
    return evidence


def classify_endpoint_text(
    *, title: str, description: str, unit: str, record_kind: str
) -> list[dict[str, Any]]:
    """Return conservative candidate classifications with exact matched phrases."""

    fields = (("title_or_term", title), ("description_or_organ_system", description))
    combined = f"{title}\n{description}"
    decisions: list[dict[str, Any]] = []
    qt_evidence = _evidence_for_patterns(fields, QT_PATTERNS)
    if qt_evidence:
        event_language = bool(
            re.search(
                r"\b(?:participants?|subjects?|patients?|events?|incidence|proportion|adverse|"
                r"prolong(?:ation|ed)|torsades?|TdP)\b",
                combined,
                re.IGNORECASE,
            )
        )
        if record_kind == "adverse_event_term" or event_language or PARTICIPANT_UNIT_RE.search(unit):
            classification = "qt_qtc_event_or_threshold_candidate"
        elif QT_NUMERIC_UNIT_RE.search(unit):
            classification = "qt_qtc_interval_measure_candidate"
        else:
            classification = "qt_qtc_context_review_candidate"
        decisions.append(
            {
                "target_domain": "qt_qtc",
                "candidate_classification": classification,
                "genuine_endpoint_candidate": classification
                in {"qt_qtc_interval_measure_candidate", "qt_qtc_event_or_threshold_candidate"},
                "evidence": qt_evidence,
            }
        )

    direct_pk = _evidence_for_patterns(fields, PK_DIRECT_PATTERNS)
    context_pk = _evidence_for_patterns(fields, PK_CONTEXT_PATTERNS)
    if CLEARANCE_RE.search(combined) and not CLEARANCE_EXCLUSION_RE.search(combined):
        clearance_context = bool(
            re.search(
                r"\b(?:drug|plasma|systemic|oral|apparent|total\s+body|renal|hepatic|PK|"
                r"pharmacokinetic)\b",
                combined,
                re.IGNORECASE,
            )
            or re.search(r"(?:l|ml)\s*/\s*(?:h|hr|min)", unit, re.IGNORECASE)
        )
        if clearance_context:
            match = CLEARANCE_RE.search(combined)
            assert match is not None
            direct_pk.append(
                {"field": "title_or_description", "pattern": "drug_clearance", "phrase": match.group(0)}
            )
    if direct_pk or context_pk:
        participant_unit = bool(PARTICIPANT_UNIT_RE.search(unit))
        quantitative = bool(PK_QUANTITATIVE_UNIT_RE.search(unit)) and not participant_unit
        has_concentration = any(item["pattern"] == "concentration" for item in context_pk)
        if direct_pk and not participant_unit:
            classification = "pk_genuine_metric_candidate"
            genuine = True
        elif has_concentration and quantitative:
            classification = "pk_genuine_concentration_candidate"
            genuine = True
        elif participant_unit:
            classification = "pk_context_or_safety_count_not_genuine_metric"
            genuine = False
        else:
            classification = "pk_context_review_candidate"
            genuine = False
        decisions.append(
            {
                "target_domain": "pk",
                "candidate_classification": classification,
                "genuine_endpoint_candidate": genuine,
                "evidence": direct_pk + context_pk,
            }
        )
    return decisions


def _denominator_records(outcome: Mapping[str, Any]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []

    def append_denoms(denoms: Any, *, scope: str, class_index: int | None, class_title: str) -> None:
        if not isinstance(denoms, list):
            return
        for denom_index, denom in enumerate(denoms):
            if not isinstance(denom, dict):
                continue
            for count_index, count in enumerate(denom.get("counts", [])):
                if not isinstance(count, dict):
                    continue
                records.append(
                    {
                        "scope": scope,
                        "class_index": "" if class_index is None else str(class_index),
                        "class_title": class_title,
                        "denominator_index": str(denom_index),
                        "count_index": str(count_index),
                        "units": _text(denom.get("units")),
                        "group_id": _text(count.get("groupId")),
                        "value": _text(count.get("value")),
                    }
                )

    append_denoms(outcome.get("denoms"), scope="outcome", class_index=None, class_title="")
    for class_index, outcome_class in enumerate(outcome.get("classes", [])):
        if isinstance(outcome_class, dict):
            append_denoms(
                outcome_class.get("denoms"),
                scope="class",
                class_index=class_index,
                class_title=_text(outcome_class.get("title")),
            )
    return records


def _measurement_records(outcome: Mapping[str, Any]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for class_index, outcome_class in enumerate(outcome.get("classes", [])):
        if not isinstance(outcome_class, dict):
            continue
        for category_index, category in enumerate(outcome_class.get("categories", [])):
            if not isinstance(category, dict):
                continue
            for measurement_index, measurement in enumerate(category.get("measurements", [])):
                if not isinstance(measurement, dict):
                    continue
                records.append(
                    {
                        "class_index": str(class_index),
                        "class_title": _text(outcome_class.get("title")),
                        "category_index": str(category_index),
                        "category_title": _text(category.get("title")),
                        "measurement_index": str(measurement_index),
                        "group_id": _text(measurement.get("groupId")),
                        "value": _text(measurement.get("value")),
                        "spread": _text(measurement.get("spread")),
                        "lower_limit": _text(measurement.get("lowerLimit")),
                        "upper_limit": _text(measurement.get("upperLimit")),
                    }
                )
    return records


def _common(
    binding: InputBinding, page: SourcePage, study_index: int, nct_id: str, pointer: str
) -> dict[str, Any]:
    return {
        "source_id": SOURCE_ID,
        "source_api_version": binding.source_api_version,
        "source_data_timestamp": binding.source_data_timestamp,
        "source_snapshot_key": binding.source_snapshot_key,
        "source_manifest_sha256": binding.source_manifest_sha256,
        "source_page_path": page.relative_path,
        "source_page_sha256": page.sha256,
        "source_page_index": page.page_index,
        "study_index_within_page": study_index,
        "nct_id": nct_id,
        "raw_json_pointer": pointer,
    }


def _write_candidate_outputs(binding: InputBinding, staging: Path) -> tuple[dict[str, int], Counter[str]]:
    counts: dict[str, int] = {name: 0 for name in CSV_FIELDS}
    classification_counts: Counter[str] = Counter()
    nct_seen: set[str] = set()
    study_count = 0
    output_paths = {name: staging / name for name in CSV_FIELDS}

    with ExitStack() as stack:
        writers: dict[str, csv.DictWriter[str]] = {}
        for name, fields in CSV_FIELDS.items():
            raw_handle = stack.enter_context(output_paths[name].open("wb"))
            gzip_handle = stack.enter_context(
                gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=raw_handle, mtime=0)
            )
            handle = stack.enter_context(
                io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="", write_through=True)
            )
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
            writer.writeheader()
            writers[name] = writer

        def emit(name: str, row: Mapping[str, Any]) -> None:
            writers[name].writerow(row)
            counts[name] += 1

        for page in binding.pages:
            page_document = _load_json_object(page.physical_path, context="source page")
            studies = page_document.get("studies")
            if not isinstance(studies, list):
                raise ClinicalResultsError(f"source page lacks studies array: {page.relative_path}")
            for study_index, study in enumerate(studies):
                if not isinstance(study, dict):
                    raise ClinicalResultsError(f"study is not an object: {page.relative_path}#{study_index}")
                study_count += 1
                protocol = study.get("protocolSection")
                if not isinstance(protocol, dict):
                    raise ClinicalResultsError(
                        f"study lacks protocolSection: {page.relative_path}#{study_index}"
                    )
                identification = protocol.get("identificationModule", {})
                if not isinstance(identification, dict):
                    raise ClinicalResultsError("identificationModule is not an object")
                nct_id = _text(identification.get("nctId"))
                if not NCT_RE.fullmatch(nct_id) or nct_id in nct_seen:
                    raise ClinicalResultsError(f"invalid or duplicate NCT ID: {nct_id!r}")
                nct_seen.add(nct_id)
                base_pointer = f"/studies/{study_index}"
                study_id = _sha_id("CTS", nct_id, page.sha256, study_index)
                status = protocol.get("statusModule", {})
                design = protocol.get("designModule", {})
                sponsor = protocol.get("sponsorCollaboratorsModule", {})
                conditions = protocol.get("conditionsModule", {})
                arms = protocol.get("armsInterventionsModule", {})
                results = study.get("resultsSection", {})
                for value, label in (
                    (status, "statusModule"),
                    (design, "designModule"),
                    (sponsor, "sponsorCollaboratorsModule"),
                    (conditions, "conditionsModule"),
                    (arms, "armsInterventionsModule"),
                    (results, "resultsSection"),
                ):
                    if not isinstance(value, dict):
                        raise ClinicalResultsError(f"{label} is not an object for {nct_id}")

                common_base = _common(binding, page, study_index, nct_id, base_pointer)
                for intervention_index, intervention in enumerate(arms.get("interventions", [])):
                    if not isinstance(intervention, dict):
                        continue
                    pointer = f"{base_pointer}/protocolSection/armsInterventionsModule/interventions/{intervention_index}"
                    row = _common(binding, page, study_index, nct_id, pointer)
                    labels = intervention.get("armGroupLabels", [])
                    row.update(
                        {
                            "intervention_candidate_id": _sha_id("CTI", nct_id, pointer),
                            "intervention_index": intervention_index,
                            "intervention_type": _text(intervention.get("type")),
                            "intervention_name": _text(intervention.get("name")),
                            "intervention_description": _text(intervention.get("description")),
                            "reported_arm_group_labels_json": _json(
                                labels if isinstance(labels, list) else []
                            ),
                            "identity_resolution_status": "reported_intervention_name_only_no_chemical_identity_inference",
                            "linkage_status": "reported_arm_labels_only"
                            if labels
                            else "no_arm_linkage_in_selected_fields",
                            "unknown_or_missing_fields_json": _json(_missing(intervention, ("type", "name"))),
                            "admission_status": ADMISSION_STATUS,
                        }
                    )
                    emit("interventions.csv.gz", row)

                for arm_index, arm in enumerate(arms.get("armGroups", [])):
                    if not isinstance(arm, dict):
                        continue
                    pointer = f"{base_pointer}/protocolSection/armsInterventionsModule/armGroups/{arm_index}"
                    row = _common(binding, page, study_index, nct_id, pointer)
                    row.update(
                        {
                            "group_candidate_id": _sha_id("CTG", nct_id, pointer),
                            "group_source_kind": "protocol_arm_group",
                            "context_candidate_id": study_id,
                            "context_index": "",
                            "group_index": arm_index,
                            "reported_group_id": "",
                            "group_title": _text(arm.get("label")),
                            "group_description": _text(arm.get("description")),
                            "linkage_scope": "reported_protocol_arm_only",
                            "unknown_or_missing_fields_json": _json(_missing(arm, ("label",))),
                            "admission_status": ADMISSION_STATUS,
                        }
                    )
                    emit("arms_groups.csv.gz", row)

                outcome_module = results.get("outcomeMeasuresModule", {})
                if not isinstance(outcome_module, dict):
                    outcome_module = {}
                outcomes = outcome_module.get("outcomeMeasures", [])
                if not isinstance(outcomes, list):
                    outcomes = []
                target_candidate_count = 0
                for outcome_index, outcome in enumerate(outcomes):
                    if not isinstance(outcome, dict):
                        continue
                    pointer = (
                        f"{base_pointer}/resultsSection/outcomeMeasuresModule/outcomeMeasures/{outcome_index}"
                    )
                    outcome_id = _sha_id("CTO", nct_id, pointer)
                    title = _text(outcome.get("title"))
                    description = _text(outcome.get("description"))
                    unit = _text(outcome.get("unitOfMeasure"))
                    timeframe = _text(outcome.get("timeFrame"))
                    decisions = classify_endpoint_text(
                        title=title, description=description, unit=unit, record_kind="outcome_measure"
                    )
                    denominators = _denominator_records(outcome)
                    measurements = _measurement_records(outcome)
                    evidence_by_domain = {
                        decision["target_domain"]: decision["evidence"] for decision in decisions
                    }
                    outcome_row = _common(binding, page, study_index, nct_id, pointer)
                    outcome_row.update(
                        {
                            "outcome_candidate_id": outcome_id,
                            "outcome_index": outcome_index,
                            "outcome_type": _text(outcome.get("type")),
                            "title": title,
                            "description": description,
                            "population_description": _text(outcome.get("populationDescription")),
                            "reporting_status": _text(outcome.get("reportingStatus")),
                            "parameter_type": _text(outcome.get("paramType")),
                            "dispersion_type": _text(outcome.get("dispersionType")),
                            "unit_of_measure": unit,
                            "time_frame": timeframe,
                            "group_ids_json": _json(
                                [
                                    _text(group.get("id"))
                                    for group in outcome.get("groups", [])
                                    if isinstance(group, dict)
                                ]
                            ),
                            "denominator_records_json": _json(denominators),
                            "denominator_record_count": len(denominators),
                            "measurement_record_count": len(measurements),
                            "candidate_domains_json": _json(
                                [decision["target_domain"] for decision in decisions]
                            ),
                            "candidate_classifications_json": _json(
                                [decision["candidate_classification"] for decision in decisions]
                            ),
                            "evidence_phrases_json": _json(evidence_by_domain),
                            "unknown_or_missing_fields_json": _json(
                                _missing(
                                    outcome,
                                    ("title", "unitOfMeasure", "timeFrame", "groups", "denoms", "classes"),
                                )
                            ),
                            "absence_semantics": ABSENCE_SEMANTICS,
                            "admission_status": ADMISSION_STATUS,
                        }
                    )
                    emit("outcome_measures.csv.gz", outcome_row)
                    for group_index, group in enumerate(outcome.get("groups", [])):
                        if not isinstance(group, dict):
                            continue
                        group_pointer = f"{pointer}/groups/{group_index}"
                        group_row = _common(binding, page, study_index, nct_id, group_pointer)
                        group_row.update(
                            {
                                "group_candidate_id": _sha_id("CTG", nct_id, group_pointer),
                                "group_source_kind": "reported_outcome_measure_group",
                                "context_candidate_id": outcome_id,
                                "context_index": outcome_index,
                                "group_index": group_index,
                                "reported_group_id": _text(group.get("id")),
                                "group_title": _text(group.get("title")),
                                "group_description": _text(group.get("description")),
                                "linkage_scope": "exact_within_reported_outcome_measure_only",
                                "unknown_or_missing_fields_json": _json(_missing(group, ("id", "title"))),
                                "admission_status": ADMISSION_STATUS,
                            }
                        )
                        emit("arms_groups.csv.gz", group_row)
                    for decision in decisions:
                        target_candidate_count += 1
                        classification_counts[decision["candidate_classification"]] += 1
                        endpoint_pointer = pointer
                        endpoint_row = _common(binding, page, study_index, nct_id, endpoint_pointer)
                        endpoint_row.update(
                            {
                                "endpoint_candidate_id": _sha_id(
                                    "CTE", nct_id, pointer, decision["target_domain"]
                                ),
                                "record_kind": "reported_outcome_measure",
                                "parent_candidate_id": outcome_id,
                                "target_domain": decision["target_domain"],
                                "candidate_classification": decision["candidate_classification"],
                                "genuine_endpoint_candidate": str(
                                    bool(decision["genuine_endpoint_candidate"])
                                ).lower(),
                                "manual_review_required": "true",
                                "title_or_term": title,
                                "description_or_organ_system": description,
                                "unit_of_measure": unit,
                                "time_frame": timeframe,
                                "denominator_records_json": _json(denominators),
                                "value_records_json": _json(measurements),
                                "evidence_phrases_json": _json(decision["evidence"]),
                                "unknown_or_missing_fields_json": outcome_row[
                                    "unknown_or_missing_fields_json"
                                ],
                                "absence_semantics": ABSENCE_SEMANTICS,
                                "zero_counts_are_reported_values_not_study_level_negatives": "true",
                                "identity_and_linkage_status": "reported_result_context_only_no_intervention_molecule_linkage",
                                "admission_status": ADMISSION_STATUS,
                            }
                        )
                        emit("endpoint_candidates.csv.gz", endpoint_row)

                adverse = results.get("adverseEventsModule", {})
                if not isinstance(adverse, dict):
                    adverse = {}
                event_groups = adverse.get("eventGroups", [])
                if not isinstance(event_groups, list):
                    event_groups = []
                for group_index, group in enumerate(event_groups):
                    if not isinstance(group, dict):
                        continue
                    pointer = f"{base_pointer}/resultsSection/adverseEventsModule/eventGroups/{group_index}"
                    group_row = _common(binding, page, study_index, nct_id, pointer)
                    group_row.update(
                        {
                            "adverse_event_group_candidate_id": _sha_id("CTA", nct_id, pointer),
                            "group_index": group_index,
                            "reported_group_id": _text(group.get("id")),
                            "group_title": _text(group.get("title")),
                            "group_description": _text(group.get("description")),
                            "module_time_frame": _text(adverse.get("timeFrame")),
                            "frequency_threshold": _text(adverse.get("frequencyThreshold")),
                            "deaths_num_affected": _text(group.get("deathsNumAffected")),
                            "deaths_num_at_risk": _text(group.get("deathsNumAtRisk")),
                            "serious_num_affected": _text(group.get("seriousNumAffected")),
                            "serious_num_at_risk": _text(group.get("seriousNumAtRisk")),
                            "other_num_affected": _text(group.get("otherNumAffected")),
                            "other_num_at_risk": _text(group.get("otherNumAtRisk")),
                            "unknown_or_missing_fields_json": _json(
                                _missing(group, ("id", "title", "seriousNumAtRisk", "otherNumAtRisk"))
                            ),
                            "zero_counts_are_reported_values_not_study_level_negatives": "true",
                            "admission_status": ADMISSION_STATUS,
                        }
                    )
                    emit("adverse_event_groups.csv.gz", group_row)
                for event_kind, events_key in (
                    ("reported_serious_adverse_event_term", "seriousEvents"),
                    ("reported_other_adverse_event_term", "otherEvents"),
                ):
                    events = adverse.get(events_key, [])
                    if not isinstance(events, list):
                        continue
                    for event_index, event in enumerate(events):
                        if not isinstance(event, dict):
                            continue
                        pointer = (
                            f"{base_pointer}/resultsSection/adverseEventsModule/{events_key}/{event_index}"
                        )
                        term = _text(event.get("term"))
                        organ_system = _text(event.get("organSystem"))
                        decisions = classify_endpoint_text(
                            title=term,
                            description=organ_system,
                            unit="participants affected / at risk",
                            record_kind="adverse_event_term",
                        )
                        for decision in decisions:
                            if decision["target_domain"] != "qt_qtc":
                                continue
                            target_candidate_count += 1
                            classification_counts[decision["candidate_classification"]] += 1
                            stats = [
                                {
                                    "group_id": _text(stat.get("groupId")),
                                    "num_affected": _text(stat.get("numAffected")),
                                    "num_at_risk": _text(stat.get("numAtRisk")),
                                }
                                for stat in event.get("stats", [])
                                if isinstance(stat, dict)
                            ]
                            endpoint_row = _common(binding, page, study_index, nct_id, pointer)
                            endpoint_row.update(
                                {
                                    "endpoint_candidate_id": _sha_id(
                                        "CTE", nct_id, pointer, decision["target_domain"]
                                    ),
                                    "record_kind": event_kind,
                                    "parent_candidate_id": study_id,
                                    "target_domain": "qt_qtc",
                                    "candidate_classification": decision["candidate_classification"],
                                    "genuine_endpoint_candidate": "true",
                                    "manual_review_required": "true",
                                    "title_or_term": term,
                                    "description_or_organ_system": organ_system,
                                    "unit_of_measure": "participants affected / at risk",
                                    "time_frame": _text(adverse.get("timeFrame")),
                                    "denominator_records_json": _json([]),
                                    "value_records_json": _json(stats),
                                    "evidence_phrases_json": _json(decision["evidence"]),
                                    "unknown_or_missing_fields_json": _json(
                                        _missing(event, ("term", "organSystem", "stats"))
                                    ),
                                    "absence_semantics": ABSENCE_SEMANTICS,
                                    "zero_counts_are_reported_values_not_study_level_negatives": "true",
                                    "identity_and_linkage_status": "exact_adverse_event_group_ids_within_study_only_no_arm_or_molecule_inference",
                                    "admission_status": ADMISSION_STATUS,
                                }
                            )
                            emit("endpoint_candidates.csv.gz", endpoint_row)

                lead = sponsor.get("leadSponsor", {})
                if not isinstance(lead, dict):
                    lead = {}
                study_unknowns = []
                if not bool(study.get("hasResults", False)):
                    study_unknowns.append("posted_results_not_present_in_selected_record")
                if not outcomes:
                    study_unknowns.append("posted_outcome_measures_not_present")
                if not event_groups:
                    study_unknowns.append("posted_adverse_event_groups_not_present")
                study_row = dict(common_base)
                study_row.update(
                    {
                        "study_candidate_id": study_id,
                        "study_raw_sha256": hashlib.sha256(canonical_json_bytes(study)).hexdigest(),
                        "has_results_reported": str(bool(study.get("hasResults", False))).lower(),
                        "study_type": _text(design.get("studyType")),
                        "phases_json": _json(design.get("phases", [])),
                        "overall_status": _text(status.get("overallStatus")),
                        "brief_title": _text(identification.get("briefTitle")),
                        "official_title": _text(identification.get("officialTitle")),
                        "lead_sponsor_name": _text(lead.get("name")),
                        "lead_sponsor_class": _text(lead.get("class")),
                        "conditions_json": _json(conditions.get("conditions", [])),
                        "keywords_json": _json(conditions.get("keywords", [])),
                        "start_date": _date(status, "startDateStruct"),
                        "primary_completion_date": _date(status, "primaryCompletionDateStruct"),
                        "completion_date": _date(status, "completionDateStruct"),
                        "study_first_post_date": _date(status, "studyFirstPostDateStruct"),
                        "results_first_post_date": _date(status, "resultsFirstPostDateStruct"),
                        "last_update_post_date": _date(status, "lastUpdatePostDateStruct"),
                        "posted_outcome_measure_count": len(outcomes),
                        "posted_adverse_event_group_count": len(event_groups),
                        "target_endpoint_candidate_count": target_candidate_count,
                        "unknown_or_missing_fields_json": _json(study_unknowns),
                        "absence_semantics": ABSENCE_SEMANTICS,
                        "admission_status": ADMISSION_STATUS,
                    }
                )
                emit("studies.csv.gz", study_row)

    if study_count != binding.reported_total_count:
        raise ClinicalResultsError(
            f"study count {study_count} does not match source manifest {binding.reported_total_count}"
        )
    return counts, classification_counts


def _artifact(path: Path, rows: int, fields: Sequence[str]) -> CsvArtifact:
    return CsvArtifact(
        path=path.name,
        rows=rows,
        bytes=path.stat().st_size,
        sha256=sha256_file(path),
        header_sha256=hashlib.sha256((",".join(fields) + "\n").encode("utf-8")).hexdigest(),
    )


def build_candidates(
    source_root: Path,
    output_root: Path,
    *,
    snapshot_key: str = DEFAULT_SNAPSHOT_KEY,
    code_path: Path | None = None,
) -> dict[str, Any]:
    """Build a new immutable candidate inventory transaction."""

    binding = bind_inputs(source_root, snapshot_key=snapshot_key)
    output_root = output_root.absolute()
    if output_root.exists():
        raise ClinicalResultsError(f"output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging.", dir=output_root.parent))
    try:
        counts, classification_counts = _write_candidate_outputs(binding, staging)
        artifacts = [_artifact(staging / name, counts[name], CSV_FIELDS[name]) for name in sorted(CSV_FIELDS)]
        artifact_records = [artifact.as_record() for artifact in artifacts]
        code_file = (code_path or Path(__file__)).resolve()
        if not code_file.is_file():
            raise ClinicalResultsError(f"missing parser code file: {code_file}")
        genuine_count = 0
        with gzip.open(
            staging / "endpoint_candidates.csv.gz", mode="rt", newline="", encoding="utf-8"
        ) as handle:
            for row in csv.DictReader(handle):
                genuine_count += row["genuine_endpoint_candidate"] == "true"
        manifest_body: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "parser_version": PARSER_VERSION,
            "dataset_id": "clinical_results_candidates",
            "input_binding": binding.as_record(),
            "code_binding": {
                "path": "pipeline/src/menin_discovery/platform_clinical_results.py",
                "bytes": code_file.stat().st_size,
                "sha256": sha256_file(code_file),
            },
            "output_inventory": {
                "entries": artifact_records,
                "entries_sha256": hashlib.sha256(canonical_json_bytes(artifact_records)).hexdigest(),
                "entry_count": len(artifact_records),
                "total_bytes": sum(item.bytes for item in artifacts),
                "excluded_paths": [OUTPUT_MANIFEST_NAME],
            },
            "row_counts": {name: counts[name] for name in sorted(counts)},
            "candidate_classification_counts": dict(sorted(classification_counts.items())),
            "genuine_endpoint_candidate_count": genuine_count,
            "candidate_only": True,
            "canonical_observations_admitted": 0,
            "model_labels_admitted": 0,
            "substantive_model_training_performed": False,
            "absence_semantics": ABSENCE_SEMANTICS,
            "intervention_identity_resolution": "not_attempted_reported_text_only",
            "cross_module_group_linkage": "not_attempted_exact_context_only",
            "raw_json_duplicated_wholesale": False,
            "limitations": [
                "Heuristic cohort membership has false positives and is not an endpoint label.",
                "Only posted result modules selected by the frozen query are inventoried.",
                "No absence or unreported result is interpreted as a negative outcome.",
                "Intervention names are unnormalized registry text and are not molecule identities.",
                "Endpoint classifications are conservative text candidates requiring manual review.",
                "Outcome values remain reported aggregate results and are not canonical observations.",
            ],
        }
        manifest = _write_identified_json(staging / OUTPUT_MANIFEST_NAME, manifest_body)
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    verify_candidates(output_root, source_root=source_root, code_path=code_path)
    return manifest


def _count_csv(path: Path, expected_fields: Sequence[str]) -> tuple[int, set[str], set[str]]:
    rows = 0
    ncts: set[str] = set()
    endpoint_ids: set[str] = set()
    with gzip.open(path, mode="rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(expected_fields):
            raise ClinicalResultsError(f"CSV header mismatch: {path.name}")
        for row in reader:
            rows += 1
            nct = row.get("nct_id", "")
            if not NCT_RE.fullmatch(nct):
                raise ClinicalResultsError(f"invalid NCT ID in {path.name}: {nct!r}")
            if row.get("admission_status") != ADMISSION_STATUS:
                raise ClinicalResultsError(f"unsafe admission state in {path.name}")
            if not row.get("source_page_sha256") or not row.get("raw_json_pointer", "").startswith(
                "/studies/"
            ):
                raise ClinicalResultsError(f"missing source binding in {path.name}")
            ncts.add(nct)
            if path.name == "endpoint_candidates.csv.gz":
                endpoint_ids.add(row["endpoint_candidate_id"])
                if row["manual_review_required"] != "true" or row["absence_semantics"] != ABSENCE_SEMANTICS:
                    raise ClinicalResultsError("endpoint candidate bypasses review/absence safety")
    return rows, ncts, endpoint_ids


def _closed_output_files(root: Path) -> set[str]:
    observed: set[str] = set()
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ClinicalResultsError(f"candidate output contains a symlink: {path}")
        if stat.S_ISREG(mode):
            if path.name != OUTPUT_MANIFEST_NAME:
                observed.add(path.relative_to(root).as_posix())
        elif not stat.S_ISDIR(mode):
            raise ClinicalResultsError(f"candidate output contains a special entry: {path}")
    return observed


def verify_candidates(
    output_root: Path,
    *,
    source_root: Path | None = None,
    code_path: Path | None = None,
) -> dict[str, Any]:
    """Verify physical, source, code, count, and safety invariants."""

    if output_root.is_symlink() or not output_root.is_dir():
        raise ClinicalResultsError(f"missing or symlinked candidate output root: {output_root}")
    output_root = output_root.resolve()
    manifest_path = output_root / OUTPUT_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ClinicalResultsError("candidate output manifest is missing or symlinked")
    manifest = _load_json_object(manifest_path, context="candidate manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION or not verify_document_sha256(manifest):
        raise ClinicalResultsError("candidate manifest schema or internal SHA-256 failed")
    if any(
        (
            manifest.get("canonical_observations_admitted") != 0,
            manifest.get("model_labels_admitted") != 0,
            manifest.get("substantive_model_training_performed") is not False,
            manifest.get("candidate_only") is not True,
        )
    ):
        raise ClinicalResultsError("candidate-only safety boundary failed")
    inventory = manifest.get("output_inventory")
    if not isinstance(inventory, dict) or not isinstance(inventory.get("entries"), list):
        raise ClinicalResultsError("missing output inventory")
    entries = inventory["entries"]
    if hashlib.sha256(canonical_json_bytes(entries)).hexdigest() != inventory.get("entries_sha256"):
        raise ClinicalResultsError("output inventory entry digest failed")
    declared = {str(entry.get("path")): entry for entry in entries if isinstance(entry, dict)}
    if set(declared) != set(CSV_FIELDS):
        raise ClinicalResultsError("output inventory file set mismatch")
    observed = _closed_output_files(output_root)
    if observed != set(declared):
        raise ClinicalResultsError("output directory membership changed")
    row_counts: dict[str, int] = {}
    all_ncts: dict[str, set[str]] = {}
    endpoint_ids: set[str] = set()
    for name, fields in CSV_FIELDS.items():
        path = output_root / name
        entry = declared[name]
        if path.is_symlink() or path.stat().st_size != int(entry.get("bytes", -1)):
            raise ClinicalResultsError(f"output byte count changed: {name}")
        if sha256_file(path) != entry.get("sha256"):
            raise ClinicalResultsError(f"output SHA-256 changed: {name}")
        expected_header_sha = hashlib.sha256((",".join(fields) + "\n").encode("utf-8")).hexdigest()
        if expected_header_sha != entry.get("header_sha256"):
            raise ClinicalResultsError(f"output header digest failed: {name}")
        rows, ncts, file_endpoint_ids = _count_csv(path, fields)
        if rows != int(entry.get("rows", -1)) or rows != int(manifest["row_counts"].get(name, -1)):
            raise ClinicalResultsError(f"output row count changed: {name}")
        row_counts[name] = rows
        all_ncts[name] = ncts
        endpoint_ids.update(file_endpoint_ids)
    if row_counts["studies.csv.gz"] != len(all_ncts["studies.csv.gz"]):
        raise ClinicalResultsError("study inventory contains duplicate NCT IDs")
    study_ncts = all_ncts["studies.csv.gz"]
    if any(not ncts <= study_ncts for name, ncts in all_ncts.items() if name != "studies.csv.gz"):
        raise ClinicalResultsError("child inventory contains unknown NCT ID")
    if len(endpoint_ids) != row_counts["endpoint_candidates.csv.gz"]:
        raise ClinicalResultsError("duplicate endpoint candidate ID")

    code = manifest.get("code_binding", {})
    code_file = (code_path or Path(__file__)).resolve()
    if code_file.stat().st_size != int(code.get("bytes", -1)) or sha256_file(code_file) != code.get("sha256"):
        raise ClinicalResultsError("parser code binding failed")
    if source_root is not None:
        rebound = bind_inputs(source_root, snapshot_key=str(manifest["input_binding"]["source_snapshot_key"]))
        if rebound.as_record() != manifest.get("input_binding"):
            raise ClinicalResultsError("source input binding changed")
    return {
        "verification_status": "pass",
        "manifest_sha256": manifest["manifest_sha256"],
        "physical_manifest_sha256": sha256_file(manifest_path),
        "row_counts": row_counts,
        "endpoint_candidate_count": row_counts["endpoint_candidates.csv.gz"],
        "genuine_endpoint_candidate_count": manifest["genuine_endpoint_candidate_count"],
        "canonical_observations_admitted": 0,
        "model_labels_admitted": 0,
    }


def _write_verification_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(report)
    candidate_manifest_sha256 = body.pop("manifest_sha256", None)
    body["candidate_manifest_internal_sha256"] = candidate_manifest_sha256
    _write_identified_json(path, body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build immutable clinical result candidates")
    build.add_argument("--source-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--snapshot-key", default=DEFAULT_SNAPSHOT_KEY)
    verify = subparsers.add_parser("verify", help="verify candidate inventory")
    verify.add_argument("--source-root", type=Path)
    verify.add_argument("--output-root", type=Path, required=True)
    verify.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            manifest = build_candidates(args.source_root, args.output_root, snapshot_key=args.snapshot_key)
            payload: Mapping[str, Any] = {
                "status": "built_and_verified",
                "manifest_sha256": manifest["manifest_sha256"],
                "row_counts": manifest["row_counts"],
                "candidate_classification_counts": manifest["candidate_classification_counts"],
                "genuine_endpoint_candidate_count": manifest["genuine_endpoint_candidate_count"],
            }
        else:
            payload = verify_candidates(args.output_root, source_root=args.source_root)
            if args.report:
                _write_verification_report(args.report, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except ClinicalResultsError as error:
        print(f"clinical-results error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
