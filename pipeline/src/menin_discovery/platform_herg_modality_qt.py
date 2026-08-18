"""Build a deterministic wild-type hERG method and QT-phenotype index.

This module is additive: it does not alter the assay-native hERG ledger or the
clinical-link artifacts.  It makes two distinctions that are essential for a
hERG paper:

* measurement technology, automation evidence, and dose design are separate
  axes; and
* clinical QT/QTc is a phenotype related to cardiac repolarization, not a hERG
  potency measurement.

Only rows explicitly marked ``mutant_or_variant`` are excluded.  Rows whose
target is ``wild_type_or_unspecified`` remain useful but are never presented as
confirmed wild type.  Method names are inferred only from explicit source or
assay metadata; otherwise the ontology records ``unresolved``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from menin_discovery.platform_herg_clinical_links import (
    MANIFEST_NAME as CLINICAL_MANIFEST_NAME,
)
from menin_discovery.platform_herg_clinical_links import (
    T3_OUTPUT,
    verify_herg_clinical_links,
)
from menin_discovery.platform_herg_hierarchy import validate_herg_hierarchy

SCHEMA_VERSION = "platform-herg-modality-qt/1.1"
MANIFEST_NAME = "herg_modality_qt_manifest.json"
MODALITY_OUTPUT = "herg_measurement_modality_index.parquet"
EXCLUSION_OUTPUT = "herg_variant_exclusions.parquet"
QT_OUTPUT = "qt_clinical_phenotype_index.parquet"
QC_OUTPUT = "modality_qt_qc_counts.parquet"

CONFIRMED_WT = "confirmed_wild_type"
UNSPECIFIED_WT = "wild_type_or_unspecified"
EXPLICIT_MUTANT = "explicit_mutant_or_variant"

MODALITIES = frozenset(
    {
        "high_throughput_thallium_flux",
        "patch_clamp_electrophysiology",
        "radioligand_binding",
        "binding_unspecified",
        "functional_ion_flux",
        "functional_electrophysiology",
        "functional_unspecified",
        "clinical_qt_in_vivo",
        "unresolved",
    }
)
AUTOMATION_CLASSES = frozenset({"automated", "manual", "unresolved", "not_applicable"})
CONFIDENCE_LEVELS = frozenset({"high", "medium", "low", "unresolved"})
DOSE_DESIGNS = frozenset(
    {
        "fixed_dose_categorical",
        "fixed_dose_quantitative",
        "concentration_response_summary",
        "kinetic_measurement",
        "unresolved",
    }
)
QT_PHENOTYPES = frozenset({"interval_measurement", "event_or_threshold", "context_unresolved"})
QT_PHENOTYPE_ASSAY_IDS = frozenset({"CHEMBL820994", "CHEMBL854887", "CHEMBL854888"})


class HergModalityQtError(RuntimeError):
    """Raised when method/QT indexing or validation fails closed."""


_MODALITY_SCHEMA = pa.schema(
    [
        pa.field("observation_id", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string()),
        pa.field("source_family", pa.large_string(), nullable=False),
        pa.field("source_record_id", pa.large_string(), nullable=False),
        pa.field("assay_id", pa.large_string()),
        pa.field("target_variant_original", pa.large_string(), nullable=False),
        pa.field("wild_type_evidence_scope", pa.large_string(), nullable=False),
        pa.field("native_endpoint", pa.large_string(), nullable=False),
        pa.field("assay_description", pa.large_string()),
        pa.field("measurement_modality", pa.large_string(), nullable=False),
        pa.field("method_detail", pa.large_string(), nullable=False),
        pa.field("modality_confidence", pa.large_string(), nullable=False),
        pa.field("modality_evidence", pa.large_string(), nullable=False),
        pa.field("automation_class", pa.large_string(), nullable=False),
        pa.field("automation_confidence", pa.large_string(), nullable=False),
        pa.field("automation_evidence", pa.large_string(), nullable=False),
        pa.field("dose_design", pa.large_string(), nullable=False),
        pa.field("dose_design_confidence", pa.large_string(), nullable=False),
        pa.field("dose_design_evidence", pa.large_string(), nullable=False),
        pa.field("clinical_qt_like_assay", pa.bool_(), nullable=False),
    ]
)

_EXCLUSION_SCHEMA = pa.schema(
    [
        pa.field("observation_id", pa.large_string(), nullable=False),
        pa.field("source_family", pa.large_string(), nullable=False),
        pa.field("source_record_id", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string()),
        pa.field("target_variant_original", pa.large_string(), nullable=False),
        pa.field("exclusion_reason", pa.large_string(), nullable=False),
    ]
)

_QT_SCHEMA = pa.schema(
    [
        pa.field("candidate_id", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("nct_id", pa.large_string(), nullable=False),
        pa.field("endpoint_candidate_id", pa.large_string(), nullable=False),
        pa.field("qt_phenotype_class", pa.large_string(), nullable=False),
        pa.field("qt_metric_semantics", pa.large_string(), nullable=False),
        pa.field("correction_methods_json", pa.large_string(), nullable=False),
        pa.field("title_or_term", pa.large_string()),
        pa.field("description_or_organ_system", pa.large_string()),
        pa.field("unit_of_measure", pa.large_string()),
        pa.field("time_frame", pa.large_string()),
        pa.field("result_record_count", pa.int64(), nullable=False),
        pa.field("numeric_result_count", pa.int64(), nullable=False),
        pa.field("denominator_record_count", pa.int64(), nullable=False),
        pa.field("value_records_json", pa.large_string(), nullable=False),
        pa.field("denominator_records_json", pa.large_string(), nullable=False),
        pa.field("trial_context", pa.large_string(), nullable=False),
        pa.field("source_page_path", pa.large_string()),
        pa.field("raw_json_pointer", pa.large_string()),
        pa.field("herg_potency_derived", pa.bool_(), nullable=False),
        pa.field("qt_used_as_herg_label", pa.bool_(), nullable=False),
    ]
)

_QC_SCHEMA = pa.schema(
    [
        pa.field("axis", pa.large_string(), nullable=False),
        pa.field("category", pa.large_string(), nullable=False),
        pa.field("record_count", pa.int64(), nullable=False),
        pa.field("unique_structures", pa.int64(), nullable=False),
        pa.field("unique_trials", pa.int64(), nullable=False),
        pa.field("interpretation", pa.large_string(), nullable=False),
    ]
)

_AUTOMATED_PATCH_PATTERNS = (
    ("ionworks", "named_platform:IonWorks"),
    ("qpatch", "named_platform:QPatch"),
    ("q-patch", "named_platform:Q-Patch"),
    ("patchliner", "named_platform:Patchliner"),
    ("syncropatch", "named_platform:SyncroPatch"),
    ("automated patch", "explicit_phrase:automated_patch"),
    ("automatic patch", "explicit_phrase:automatic_patch"),
    ("whole-cell plate-based electrophysiology", "explicit_phrase:plate_based_electrophysiology"),
    ("planar patch", "explicit_phrase:planar_patch"),
)
_PATCH_TERMS = (
    "patch clamp",
    "patch-clamp",
    "patchclamp",
    "ionworks",
    "qpatch",
    "q-patch",
    "patchliner",
    "syncropatch",
    "whole-cell plate-based electrophysiology",
    "planar patch",
)
_RADIOLIGAND_TERMS = (
    "radioligand",
    "[3h]",
    "[35s]",
    "3h-",
    "35s-",
    "dofetilide displacement",
    "displacement of dofetilide",
    "scintillation proximity",
    "scintillation counting",
    "scintillation spectrophotometric",
    "membrane filtration, radioactivity",
)
_THALLIUM_TERMS = ("thallium flux", "thallium-flux", "fluxor")
_FUNCTIONAL_FLUX_TERMS = ("rb+ efflux", "rubidium efflux", "potassium flux")
_ELECTROPHYSIOLOGY_TERMS = (
    "voltage clamp",
    "voltage-clamp",
    "tail current",
    "herg current",
    "ikr current",
    "potassium current",
    "current block",
    "channel current",
    "electrophysiology",
)
_POTENCY_ENDPOINT = re.compile(
    r"^(?:p?ic\d+|p?ec\d+|p?ac\d+|p?xc\d+|p?ki|p?kd|potency|inflection point|ed\d+)$",
    re.IGNORECASE,
)
_KINETIC_ENDPOINT = re.compile(r"^(?:kon|k_on|koff|k_off|time|residence time)$", re.IGNORECASE)
_FIXED_CONCENTRATION = re.compile(
    r"\b(?:at|tested at|exposed (?:at|to)|incubated (?:at|with)|after)\s*(?:approximately\s*)?"
    r"\d+(?:\.\d+)?\s*(?:nM|uM|µM|mM)\b",
    re.IGNORECASE,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _input_binding(role: str, path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HergModalityQtError(f"missing or unsafe {role} input: {path}")
    return {"role": role, "path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _artifact(path: Path, schema: pa.Schema, rows: int) -> dict[str, Any]:
    return {
        "path": path.name,
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "arrow_schema_sha256": _schema_sha256(schema),
    }


def _write(path: Path, rows: Sequence[Mapping[str, Any]], schema: pa.Schema) -> dict[str, Any]:
    table = pa.Table.from_pylist(list(rows), schema=schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
        version="2.6",
    )
    return _artifact(path, schema, table.num_rows)


def _json_object(value: Any, *, field: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as error:
        raise HergModalityQtError(f"invalid JSON object in {field}") from error
    if not isinstance(parsed, dict):
        raise HergModalityQtError(f"{field} is not a JSON object")
    return parsed


def _json_list(value: Any, *, field: str) -> list[Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as error:
        raise HergModalityQtError(f"invalid JSON list in {field}") from error
    if not isinstance(parsed, list):
        raise HergModalityQtError(f"{field} is not a JSON list")
    return parsed


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(str(value).strip()))
    except (TypeError, ValueError):
        return False


def _scope(target_variant: str) -> str:
    if target_variant == "wild_type":
        return CONFIRMED_WT
    if target_variant == "wild_type_or_unspecified":
        return UNSPECIFIED_WT
    if target_variant == "mutant_or_variant":
        return EXPLICIT_MUTANT
    raise HergModalityQtError(f"unknown target_variant: {target_variant!r}")


def _classify_modality(
    source_family: str,
    assay_family: str,
    description: str,
    assay_id: str = "",
    native_endpoint: str = "",
) -> tuple[str, str, str, str]:
    text = description.casefold()
    if source_family == "pubchem_aid720551":
        return (
            "high_throughput_thallium_flux",
            "FluxOR thallium-flux qHTS",
            "high",
            "source_contract:PubChem_AID_720551",
        )
    if source_family == "quantitative_pic50_release":
        return (
            "unresolved",
            "mixed-source pIC50 compilation",
            "unresolved",
            "source_metadata_does_not_report_measurement_method",
        )
    if any(term in text for term in _PATCH_TERMS):
        return (
            "patch_clamp_electrophysiology",
            "patch clamp",
            "high",
            "assay_description:explicit_patch_clamp_or_named_platform",
        )
    if any(term in text for term in _RADIOLIGAND_TERMS):
        return (
            "radioligand_binding",
            "radioligand/displacement binding",
            "high",
            "assay_description:explicit_radioligand_or_displacement_method",
        )
    if any(term in text for term in _THALLIUM_TERMS):
        return (
            "high_throughput_thallium_flux",
            "thallium-flux surrogate",
            "high",
            "assay_description:explicit_thallium_or_FluxOR",
        )
    if any(term in text for term in _FUNCTIONAL_FLUX_TERMS):
        return (
            "functional_ion_flux",
            "ion-efflux surrogate",
            "high",
            "assay_description:explicit_ion_efflux",
        )
    if assay_id in QT_PHENOTYPE_ASSAY_IDS or native_endpoint.casefold() in {
        "qt interval",
        "qtc interval",
    }:
        return (
            "clinical_qt_in_vivo",
            "QT/QTc phenotype assay",
            "high",
            "curated_assay_registry_or_native_QT_endpoint",
        )
    if any(term in text for term in _ELECTROPHYSIOLOGY_TERMS):
        return (
            "functional_electrophysiology",
            "current/voltage electrophysiology",
            "high",
            "assay_description:explicit_current_or_voltage_method",
        )
    if "binding" in text or assay_family == "binding":
        return (
            "binding_unspecified",
            "binding method not resolved",
            "medium" if "binding" in text else "low",
            "assay_description_or_assay_family:binding_without_explicit_technology",
        )
    if assay_family == "functional":
        return (
            "functional_unspecified",
            "functional method not resolved",
            "low",
            "assay_family:functional_without_explicit_technology",
        )
    return "unresolved", "method unresolved", "unresolved", "no_explicit_method_metadata"


def _classify_automation(source_family: str, modality: str, description: str) -> tuple[str, str, str]:
    text = description.casefold()
    if source_family == "pubchem_aid720551":
        return "automated", "high", "source_contract:qHTS_plate_reader_workflow"
    for term, evidence in _AUTOMATED_PATCH_PATTERNS:
        if term in text:
            return "automated", "high", evidence
    if "manual patch" in text or "manual whole-cell" in text:
        return "manual", "high", "assay_description:explicit_manual"
    if "conventional patch" in text:
        return "manual", "medium", "assay_description:conventional_patch_clamp"
    if "qhts" in text or "high-throughput" in text or "high throughput" in text:
        return "automated", "high", "assay_description:explicit_high_throughput"
    if modality == "high_throughput_thallium_flux" and ("fluxor" in text or "flipr" in text):
        return "automated", "medium", "assay_description:plate_based_flux_platform"
    if modality == "clinical_qt_in_vivo":
        return "not_applicable", "high", "clinical_phenotype_not_in_vitro_assay_automation"
    return "unresolved", "unresolved", "no_explicit_automation_metadata"


def _classify_dose_design(
    source_family: str, endpoint: str, description: str, native_label: Any
) -> tuple[str, str, str]:
    if source_family == "pubchem_aid720551":
        return (
            "fixed_dose_categorical",
            "high",
            "source_contract:categorical_outcome_at_two_fixed_concentrations",
        )
    if source_family == "quantitative_pic50_release":
        return (
            "concentration_response_summary",
            "medium",
            "reported_pIC50_summary_but_underlying_curve_metadata_unavailable",
        )
    if _KINETIC_ENDPOINT.match(endpoint.strip()):
        return "kinetic_measurement", "high", "native_endpoint:kinetic"
    if _POTENCY_ENDPOINT.match(endpoint.strip()):
        return "concentration_response_summary", "high", "native_endpoint:potency_or_affinity_summary"
    if _FIXED_CONCENTRATION.search(description):
        kind = "fixed_dose_categorical" if native_label is not None else "fixed_dose_quantitative"
        return kind, "high", "assay_description:explicit_test_concentration"
    return "unresolved", "unresolved", "no_explicit_dose_design_metadata"


def _method_row(row: Mapping[str, Any], aux: Mapping[str, Any]) -> dict[str, Any]:
    source = str(row["source_family"])
    assay_family = str(row["assay_family"])
    description = str(aux.get("assay_description") or aux.get("assay_name") or "").strip()
    modality, detail, modality_confidence, modality_evidence = _classify_modality(
        source,
        assay_family,
        description,
        str(row.get("assay_id") or ""),
        str(row["native_endpoint"]),
    )
    automation, automation_confidence, automation_evidence = _classify_automation(
        source, modality, description
    )
    design, design_confidence, design_evidence = _classify_dose_design(
        source,
        str(row["native_endpoint"]),
        description,
        row.get("native_label"),
    )
    clinical_qt = modality == "clinical_qt_in_vivo"
    return {
        "observation_id": str(row["observation_id"]),
        "structure_id": row.get("structure_id"),
        "source_family": source,
        "source_record_id": str(row["source_record_id"]),
        "assay_id": row.get("assay_id"),
        "target_variant_original": str(row["target_variant"]),
        "wild_type_evidence_scope": _scope(str(row["target_variant"])),
        "native_endpoint": str(row["native_endpoint"]),
        "assay_description": description or None,
        "measurement_modality": modality,
        "method_detail": detail,
        "modality_confidence": modality_confidence,
        "modality_evidence": modality_evidence,
        "automation_class": automation,
        "automation_confidence": automation_confidence,
        "automation_evidence": automation_evidence,
        "dose_design": design,
        "dose_design_confidence": design_confidence,
        "dose_design_evidence": design_evidence,
        "clinical_qt_like_assay": clinical_qt,
    }


def _correction_methods(text: str) -> list[str]:
    lower = text.casefold()
    methods: list[str] = []
    if "fridericia" in lower or "qtcf" in lower:
        methods.append("QTcF")
    if "bazett" in lower or "qtcb" in lower:
        methods.append("QTcB")
    if "individualized" in lower or "individual correction" in lower or "qtci" in lower:
        methods.append("QTcI")
    if not methods:
        methods.append("unresolved")
    return methods


def _qt_phenotype(classification: str) -> str:
    lower = classification.casefold()
    if "event" in lower or "threshold" in lower:
        return "event_or_threshold"
    if "interval" in lower or "measure" in lower:
        return "interval_measurement"
    return "context_unresolved"


def _qt_semantics(phenotype: str, unit: str, text: str) -> str:
    lower_unit = unit.casefold()
    lower = text.casefold()
    if "participant" in lower_unit or "subject" in lower_unit or phenotype == "event_or_threshold":
        return "participant_event_or_threshold_count"
    if any(token in lower_unit for token in ("ms", "msec", "millisecond")):
        if "change from baseline" in lower or "delta" in lower:
            return "qt_or_qtc_change_from_baseline"
        return "qt_or_qtc_interval_value"
    return "reported_qt_endpoint_value_semantics_require_review"


def _qt_row(row: Mapping[str, Any]) -> dict[str, Any]:
    values = _json_list(row["value_records_json"], field="value_records_json")
    denominators = _json_list(row["denominator_records_json"], field="denominator_records_json")
    if any(not isinstance(value, Mapping) for value in [*values, *denominators]):
        raise HergModalityQtError("QT value/denominator JSON contains a non-object record")
    title = str(row.get("title_or_term") or "")
    description = str(row.get("description_or_organ_system") or "")
    unit = str(row.get("unit_of_measure") or "")
    combined = " ".join((title, description, unit))
    phenotype = _qt_phenotype(str(row["candidate_classification"]))
    numeric_count = sum(_finite(value.get("value")) for value in values)
    if numeric_count != int(row["reported_numeric_value_count"]):
        raise HergModalityQtError("QT declared numeric-result count does not match native records")
    return {
        "candidate_id": str(row["candidate_id"]),
        "structure_id": str(row["molecule_id"]),
        "nct_id": str(row["nct_id"]),
        "endpoint_candidate_id": str(row["endpoint_candidate_id"]),
        "qt_phenotype_class": phenotype,
        "qt_metric_semantics": _qt_semantics(phenotype, unit, combined),
        "correction_methods_json": _canonical_json(_correction_methods(combined)),
        "title_or_term": row.get("title_or_term"),
        "description_or_organ_system": row.get("description_or_organ_system"),
        "unit_of_measure": row.get("unit_of_measure"),
        "time_frame": row.get("time_frame"),
        "result_record_count": len(values),
        "numeric_result_count": numeric_count,
        "denominator_record_count": len(denominators),
        "value_records_json": _canonical_json(values),
        "denominator_records_json": _canonical_json(denominators),
        "trial_context": "posted_ClinicalTrials.gov_result_exact_unique_structure_link",
        "source_page_path": row.get("source_page_path"),
        "raw_json_pointer": row.get("raw_json_pointer"),
        "herg_potency_derived": False,
        "qt_used_as_herg_label": False,
    }


def _qc_rows(
    modality_rows: Sequence[Mapping[str, Any]],
    exclusion_rows: Sequence[Mapping[str, Any]],
    qt_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    def add(
        axis: str,
        field: str,
        rows: Sequence[Mapping[str, Any]],
        *,
        interpretation: str,
    ) -> None:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row[field])].append(row)
        for category, group in sorted(grouped.items()):
            output.append(
                {
                    "axis": axis,
                    "category": category,
                    "record_count": len(group),
                    "unique_structures": len(
                        {str(row["structure_id"]) for row in group if row.get("structure_id")}
                    ),
                    "unique_trials": len({str(row["nct_id"]) for row in group if row.get("nct_id")}),
                    "interpretation": interpretation,
                }
            )

    add(
        "wild_type_evidence_scope",
        "wild_type_evidence_scope",
        modality_rows,
        interpretation="confirmed wild type and unspecified target scope remain distinct",
    )
    add(
        "measurement_modality",
        "measurement_modality",
        modality_rows,
        interpretation="technology inferred only from explicit source or assay metadata",
    )
    add(
        "automation_class",
        "automation_class",
        modality_rows,
        interpretation="unresolved means automation was not reported",
    )
    add(
        "dose_design",
        "dose_design",
        modality_rows,
        interpretation="dose design is not a measurement technology",
    )
    add(
        "variant_exclusion",
        "exclusion_reason",
        exclusion_rows,
        interpretation="explicit mutant or variant hERG is outside the wild-type project scope",
    )
    add(
        "qt_phenotype_class",
        "qt_phenotype_class",
        qt_rows,
        interpretation="clinical phenotype axis; never a hERG potency label",
    )
    return sorted(output, key=lambda row: (str(row["axis"]), str(row["category"])))


def build_herg_modality_qt(
    hierarchy_root: str | os.PathLike[str],
    clinical_links_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Build the modality and QT axes from verified upstream artifacts."""

    hierarchy = Path(hierarchy_root).resolve()
    clinical = Path(clinical_links_root).resolve()
    validate_herg_hierarchy(hierarchy)
    verify_herg_clinical_links(clinical)
    observation_path = hierarchy / "observation_ledger.parquet"
    qt_path = clinical / T3_OUTPUT
    output = Path(output_root)
    if output.exists() and (output.is_symlink() or not output.is_dir()):
        raise HergModalityQtError(f"output must be absent or a safe directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    observations = pq.read_table(observation_path).to_pylist()
    modality_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for row in observations:
        scope = _scope(str(row["target_variant"]))
        if scope == EXPLICIT_MUTANT:
            exclusions.append(
                {
                    "observation_id": str(row["observation_id"]),
                    "source_family": str(row["source_family"]),
                    "source_record_id": str(row["source_record_id"]),
                    "structure_id": row.get("structure_id"),
                    "target_variant_original": str(row["target_variant"]),
                    "exclusion_reason": "explicit_mutant_or_variant_outside_wild_type_scope",
                }
            )
            continue
        aux = _json_object(row["native_aux_json"], field="native_aux_json")
        modality_rows.append(_method_row(row, aux))
    modality_rows.sort(key=lambda row: str(row["observation_id"]))
    exclusions.sort(key=lambda row: str(row["observation_id"]))

    qt_rows: list[dict[str, Any]] = []
    for row in pq.read_table(qt_path).to_pylist():
        if (
            row["candidate_rule_passed"] is not True
            or row["clinical_herg_label_admitted"] is not False
            or row["model_label_admitted"] is not False
        ):
            raise HergModalityQtError("QT candidate is unqualified or was promoted into a hERG/model label")
        qt_rows.append(_qt_row(row))
    qt_rows.sort(key=lambda row: str(row["candidate_id"]))
    qc_rows = _qc_rows(modality_rows, exclusions, qt_rows)

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        artifacts = [
            _write(staging / MODALITY_OUTPUT, modality_rows, _MODALITY_SCHEMA),
            _write(staging / EXCLUSION_OUTPUT, exclusions, _EXCLUSION_SCHEMA),
            _write(staging / QT_OUTPUT, qt_rows, _QT_SCHEMA),
            _write(staging / QC_OUTPUT, qc_rows, _QC_SCHEMA),
        ]
        bindings = [
            _input_binding("hierarchy_manifest", hierarchy / "manifest.json"),
            _input_binding("hierarchy_observation_ledger", observation_path),
            _input_binding("clinical_links_manifest", clinical / CLINICAL_MANIFEST_NAME),
            _input_binding("clinical_posted_qt_candidates", qt_path),
        ]
        modality_counts = Counter(str(row["measurement_modality"]) for row in modality_rows)
        automation_counts = Counter(str(row["automation_class"]) for row in modality_rows)
        dose_counts = Counter(str(row["dose_design"]) for row in modality_rows)
        scope_counts = Counter(str(row["wild_type_evidence_scope"]) for row in modality_rows)
        qt_counts = Counter(str(row["qt_phenotype_class"]) for row in qt_rows)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": "wild_type_herg_measurement_modality_and_qt_axis",
            "scientific_contract": {
                "target": "wild-type human KCNH2/hERG",
                "explicit_mutant_rows_in_modality_index": 0,
                "unspecified_target_treated_as_confirmed_wild_type": False,
                "method_invented_when_metadata_absent": False,
                "measurement_modality_and_dose_design_are_separate": True,
                "qt_is_herg_potency": False,
                "qt_used_as_herg_model_label": False,
            },
            "counts": {
                "upstream_observations": len(observations),
                "wild_type_scope_observations": len(modality_rows),
                "explicit_mutant_exclusions": len(exclusions),
                "wild_type_evidence_scope": dict(sorted(scope_counts.items())),
                "measurement_modality": dict(sorted(modality_counts.items())),
                "automation_class": dict(sorted(automation_counts.items())),
                "dose_design": dict(sorted(dose_counts.items())),
                "qt_endpoint_candidates": len(qt_rows),
                "qt_phenotype_class": dict(sorted(qt_counts.items())),
                "qt_result_records": sum(int(row["result_record_count"]) for row in qt_rows),
                "qt_numeric_result_records": sum(int(row["numeric_result_count"]) for row in qt_rows),
                "qt_denominator_records": sum(int(row["denominator_record_count"]) for row in qt_rows),
            },
            "method_ontology": {
                "measurement_modalities": sorted(MODALITIES),
                "automation_classes": sorted(AUTOMATION_CLASSES),
                "dose_designs": sorted(DOSE_DESIGNS),
                "confidence_levels": sorted(CONFIDENCE_LEVELS),
            },
            "input_bindings": bindings,
            "input_set_sha256": hashlib.sha256(_canonical_json(bindings).encode()).hexdigest(),
            "artifacts": artifacts,
            "artifact_set_sha256": hashlib.sha256(_canonical_json(artifacts).encode()).hexdigest(),
        }
        manifest["manifest_sha256"] = hashlib.sha256(_canonical_json(manifest).encode()).hexdigest()
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        validate_herg_modality_qt(staging)
        if output.exists():
            if any(output.iterdir()):
                # An identical rerun is a successful no-op.  A changed build
                # must use a new versioned root so the prior artifact is never
                # silently overwritten or deleted.
                existing = validate_herg_modality_qt(output)
                members = {MANIFEST_NAME, MODALITY_OUTPUT, EXCLUSION_OUTPUT, QT_OUTPUT, QC_OUTPUT}
                if not all((output / name).read_bytes() == (staging / name).read_bytes() for name in members):
                    raise HergModalityQtError(
                        "existing output differs from rebuilt artifacts; choose a new versioned output root"
                    )
                shutil.rmtree(staging)
                return existing
            output.rmdir()
        os.replace(staging, output)
        return validate_herg_modality_qt(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_herg_modality_qt(output_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate hashes, schemas, counts, wild-type scope, and QT separation."""

    root = Path(output_root)
    manifest_path = root / MANIFEST_NAME
    if root.is_symlink() or not root.is_dir() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise HergModalityQtError(f"missing or unsafe modality/QT output: {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HergModalityQtError("unreadable modality/QT manifest") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise HergModalityQtError("unexpected modality/QT manifest schema")
    declared = manifest.get("manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if hashlib.sha256(_canonical_json(body).encode()).hexdigest() != declared:
        raise HergModalityQtError("modality/QT manifest digest mismatch")
    contract = manifest.get("scientific_contract", {})
    if (
        contract.get("explicit_mutant_rows_in_modality_index") != 0
        or contract.get("unspecified_target_treated_as_confirmed_wild_type") is not False
        or contract.get("method_invented_when_metadata_absent") is not False
        or contract.get("qt_is_herg_potency") is not False
        or contract.get("qt_used_as_herg_model_label") is not False
    ):
        raise HergModalityQtError("modality/QT scientific contract was weakened")

    schemas = {
        MODALITY_OUTPUT: _MODALITY_SCHEMA,
        EXCLUSION_OUTPUT: _EXCLUSION_SCHEMA,
        QT_OUTPUT: _QT_SCHEMA,
        QC_OUTPUT: _QC_SCHEMA,
    }
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or {item.get("path") for item in artifacts} != set(schemas):
        raise HergModalityQtError("modality/QT artifact membership mismatch")
    expected = {MANIFEST_NAME, *schemas}
    if {path.name for path in root.iterdir()} != expected or any(
        path.is_symlink() or not path.is_file() for path in root.iterdir()
    ):
        raise HergModalityQtError("modality/QT output contains unexpected or unsafe members")
    for binding in manifest.get("input_bindings", []):
        path = Path(str(binding.get("path", "")))
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(binding.get("bytes", -1))
            or _sha256_file(path) != binding.get("sha256")
        ):
            raise HergModalityQtError(f"modality/QT input binding mismatch: {path}")
    artifact_by_name = {str(item["path"]): item for item in artifacts}
    for name, schema in schemas.items():
        path = root / name
        artifact = artifact_by_name[name]
        parquet = pq.ParquetFile(path)
        if _sha256_file(path) != artifact.get("sha256"):
            raise HergModalityQtError(f"modality/QT artifact hash mismatch: {path}")
        if parquet.schema_arrow != schema:
            raise HergModalityQtError(f"modality/QT schema mismatch: {path}")
        if parquet.metadata is None or parquet.metadata.num_rows != int(artifact.get("rows", -1)):
            raise HergModalityQtError(f"modality/QT row count mismatch: {path}")

    methods = pq.read_table(root / MODALITY_OUTPUT).to_pylist()
    exclusions = pq.read_table(root / EXCLUSION_OUTPUT).to_pylist()
    qt = pq.read_table(root / QT_OUTPUT).to_pylist()
    if len({row["observation_id"] for row in methods}) != len(methods):
        raise HergModalityQtError("duplicate observation in modality index")
    if len({row["observation_id"] for row in exclusions}) != len(exclusions):
        raise HergModalityQtError("duplicate observation in variant exclusions")
    if {row["observation_id"] for row in methods} & {row["observation_id"] for row in exclusions}:
        raise HergModalityQtError("included and excluded observation sets overlap")
    if any(row["target_variant_original"] == "mutant_or_variant" for row in methods):
        raise HergModalityQtError("explicit mutant entered wild-type modality index")
    if any(
        row["wild_type_evidence_scope"] not in {CONFIRMED_WT, UNSPECIFIED_WT}
        or row["measurement_modality"] not in MODALITIES
        or row["automation_class"] not in AUTOMATION_CLASSES
        or row["modality_confidence"] not in CONFIDENCE_LEVELS
        or row["automation_confidence"] not in CONFIDENCE_LEVELS
        or row["dose_design"] not in DOSE_DESIGNS
        or row["dose_design_confidence"] not in CONFIDENCE_LEVELS
        for row in methods
    ):
        raise HergModalityQtError("modality index contains an unknown ontology value")
    if any(row["target_variant_original"] != "mutant_or_variant" for row in exclusions):
        raise HergModalityQtError("non-mutant observation entered variant exclusions")
    if len({row["candidate_id"] for row in qt}) != len(qt):
        raise HergModalityQtError("duplicate candidate in QT phenotype index")
    if any(
        row["qt_phenotype_class"] not in QT_PHENOTYPES
        or row["herg_potency_derived"] is not False
        or row["qt_used_as_herg_label"] is not False
        for row in qt
    ):
        raise HergModalityQtError("QT phenotype was misclassified or promoted into hERG")

    counts = manifest.get("counts", {})
    expected_counts: dict[str, Any] = {
        "wild_type_scope_observations": len(methods),
        "explicit_mutant_exclusions": len(exclusions),
        "wild_type_evidence_scope": dict(
            sorted(Counter(str(row["wild_type_evidence_scope"]) for row in methods).items())
        ),
        "measurement_modality": dict(
            sorted(Counter(str(row["measurement_modality"]) for row in methods).items())
        ),
        "automation_class": dict(sorted(Counter(str(row["automation_class"]) for row in methods).items())),
        "dose_design": dict(sorted(Counter(str(row["dose_design"]) for row in methods).items())),
        "qt_endpoint_candidates": len(qt),
        "qt_phenotype_class": dict(sorted(Counter(str(row["qt_phenotype_class"]) for row in qt).items())),
        "qt_result_records": sum(int(row["result_record_count"]) for row in qt),
        "qt_numeric_result_records": sum(int(row["numeric_result_count"]) for row in qt),
        "qt_denominator_records": sum(int(row["denominator_record_count"]) for row in qt),
    }
    if any(counts.get(key) != value for key, value in expected_counts.items()):
        raise HergModalityQtError("manifest counts do not match modality/QT artifacts")
    if counts.get("upstream_observations") != len(methods) + len(exclusions):
        raise HergModalityQtError("upstream observation partition is incomplete")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hierarchy-root", type=Path)
    parser.add_argument("--clinical-links-root", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.validate_only:
        validate_herg_modality_qt(args.output_root)
    else:
        if args.hierarchy_root is None or args.clinical_links_root is None:
            raise HergModalityQtError("build mode requires --hierarchy-root and --clinical-links-root")
        build_herg_modality_qt(args.hierarchy_root, args.clinical_links_root, args.output_root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
