"""Normalization of the internal PK/hERG workbook and Sun hERG supplement.

The internal ``Provenance`` sheet is authoritative for measurements.  The wide
``SMILES`` sheet supplies compound structures and submitted descriptors only;
its pre-averaged endpoint cells are never used as labels.  Study pairing is
performed only when the source/value correspondence is explicit.  Ambiguous
pairing remains in the output with an error record instead of being guessed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .chemistry import standardize_smiles
from .features import scaffold_key
from .research_contracts import (
    AssayProtocol,
    ChemicalState,
    Compound,
    Conformer,
    DerivedPKParameter,
    FeatureLineage,
    Measurement,
    PhysicsObservable,
    PhysicsRun,
    PKSample,
    PKStudy,
    records_to_frame,
    write_contract_parquet,
)

try:  # pragma: no cover - depends on the optional chemistry installation.
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError:  # pragma: no cover
    Chem = None  # type: ignore[assignment]
    Descriptors = None  # type: ignore[assignment]


_MISSING = frozenset({"", "na", "n/a", "nan", "none", "not", "not tested", "not determined", "nd", "-"})
_RELATION_ALIASES = {
    "": "=",
    "=": "=",
    "==": "=",
    "~": "~",
    "≈": "~",
    "<": "<",
    "≤": "<=",
    "<=": "<=",
    "=<": "<=",
    ">": ">",
    "≥": ">=",
    ">=": ">=",
    "=>": ">=",
}
_UNIT_ALIASES = {
    "nm": "nM",
    "um": "uM",
    "µm": "uM",
    "μm": "uM",
    "mm": "mM",
    "h": "h",
    "hr": "h",
    "hrs": "h",
    "hour": "h",
    "hours": "h",
    "min": "min",
    "mins": "min",
    "minute": "min",
    "minutes": "min",
    "ng/ml": "ng/mL",
    "ng*h/ml": "ng*h/mL",
    "ng·h/ml": "ng*h/mL",
    "ng h/ml": "ng*h/mL",
    "ml/kg/min": "mL/kg/min",
    "l/kg": "L/kg",
    "mg/kg": "mg/kg",
    "%": "%",
    "percent": "%",
    "fraction": "fraction",
    "unitless": "unitless",
    "binary": "binary",
    "pka": "pKa",
    "pic50": "pIC50",
}


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    payload = "\0".join(_clean_text(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()
    return f"{prefix}-{digest[:length]}"


def normalize_unit(unit: object) -> str | None:
    """Normalize a supported unit spelling without inferring a missing unit."""

    text = _clean_text(unit).replace("μ", "µ")
    if not text:
        return None
    compact = re.sub(r"\s+", " ", text).strip().casefold()
    return _UNIT_ALIASES.get(compact, text)


def normalize_relation(relation: object) -> Literal["=", "<", "<=", ">", ">=", "~", "not_reported"]:
    """Normalize a relation token or raise instead of silently changing it."""

    text = _clean_text(relation)
    if text.casefold() in _MISSING:
        return "not_reported"
    try:
        return _RELATION_ALIASES[text]  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError(f"unsupported relation token: {text!r}") from exc


@dataclass(frozen=True)
class ParsedQualifiedValue:
    """One numeric value plus its explicit qualification and parse status."""

    submitted_value: str
    value: float | None
    relation: Literal["=", "<", "<=", ">", ">=", "~", "not_reported"]
    censoring: Literal["none", "left", "right", "interval", "missing", "unknown"]
    lower_bound: float | None = None
    upper_bound: float | None = None
    annotation: str | None = None
    status: str = "parsed"


def parse_qualified_value(value: object) -> ParsedQualifiedValue:
    """Parse exact, one-sided, interval, missing, and annotated numeric values.

    Infinity is preserved as unbounded right censoring rather than converted to
    an arbitrary large number.  Commas are accepted only as thousands
    separators.  Free-text annotations are retained.
    """

    submitted = _clean_text(value)
    if submitted.casefold() in _MISSING:
        return ParsedQualifiedValue(submitted, None, "not_reported", "missing", status="missing")
    if submitted in {"∞", "+∞", "inf", "+inf", "Infinity", "+Infinity"}:
        return ParsedQualifiedValue(
            submitted,
            None,
            ">",
            "right",
            annotation="unbounded infinity reported by source",
            status="unbounded_right_censored",
        )

    text = submitted.replace("≤", "<=").replace("≥", ">=").replace("−", "-")
    relation_match = re.match(r"^\s*(<=|>=|<|>|~|≈|=)?\s*(.*)$", text)
    relation_token, remainder = relation_match.groups() if relation_match else (None, text)
    relation = normalize_relation(relation_token or "=")

    interval_match = re.fullmatch(
        r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(?:-|–|to)\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*",
        remainder,
        flags=re.IGNORECASE,
    )
    if interval_match and relation == "=":
        lower, upper = sorted(float(item) for item in interval_match.groups())
        return ParsedQualifiedValue(
            submitted,
            None,
            "~",
            "interval",
            lower_bound=lower,
            upper_bound=upper,
            status="interval",
        )

    number_match = re.match(
        r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*)$",
        remainder.replace(",", ""),
    )
    if not number_match:
        return ParsedQualifiedValue(
            submitted,
            None,
            "not_reported",
            "unknown",
            annotation=submitted,
            status="unparsed",
        )
    number_text, annotation = number_match.groups()
    parsed = float(number_text)
    if not math.isfinite(parsed):
        return ParsedQualifiedValue(submitted, None, "not_reported", "unknown", status="nonfinite")
    censoring: Literal["none", "left", "right", "interval", "missing", "unknown"] = (
        "left" if relation in {"<", "<="} else "right" if relation in {">", ">="} else "none"
    )
    return ParsedQualifiedValue(
        submitted,
        parsed,
        relation,
        censoring,
        annotation=annotation.strip() or None,
    )


@dataclass(frozen=True)
class ConcentrationResponse:
    response: ParsedQualifiedValue
    concentration_value: float
    concentration_unit: str
    submitted_segment: str


_PANEL_RE = re.compile(
    r"(?P<response>[<>≤≥~≈]?\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*%?\s*@\s*"
    r"(?P<concentration>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?P<unit>[numµμ]?M)",
    flags=re.IGNORECASE,
)


def parse_concentration_response_panel(value: object) -> list[ConcentrationResponse]:
    """Parse concentration-specific inhibition values into separate observations."""

    submitted = _clean_text(value)
    panel: list[ConcentrationResponse] = []
    for match in _PANEL_RE.finditer(submitted):
        response = parse_qualified_value(match.group("response"))
        unit = normalize_unit(match.group("unit"))
        if unit is None:
            continue
        panel.append(
            ConcentrationResponse(
                response=response,
                concentration_value=float(match.group("concentration")),
                concentration_unit=unit,
                submitted_segment=match.group(0),
            )
        )
    return panel


@dataclass(frozen=True)
class NormalizationResult:
    """Canonical tables plus review-only diagnostics and a deterministic summary."""

    tables: Mapping[str, pd.DataFrame]
    issues: pd.DataFrame
    quarantine: pd.DataFrame
    summary: Mapping[str, Any]
    review_tables: Mapping[str, pd.DataFrame] = field(default_factory=dict)

    def table(self, name: str) -> pd.DataFrame:
        return self.tables[name]


def _read_internal_sheets(
    source: str | os.PathLike[str] | Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    if isinstance(source, Mapping):
        return {name: frame.copy() for name, frame in source.items()}
    path = Path(source)
    required = ("SMILES", "Provenance", "pKa_detail")
    return {name: pd.read_excel(path, sheet_name=name) for name in required}


def _split_values(value: object) -> list[str]:
    text = _clean_text(value)
    return [part.strip() for part in text.split("|")] if text else [""]


def _split_sources(value: object) -> list[str]:
    text = _clean_text(value)
    return [part.strip() for part in re.split(r"\s*,\s*", text) if part.strip()]


@dataclass(frozen=True)
class _ExpandedEvidence:
    raw_value: str
    source_locator: str | None
    source_locator_raw: str | None
    source_ordinal: int
    pairing_status: Literal["resolved", "unresolved"]


def _expand_evidence(raw_value: object, source_value: object) -> list[_ExpandedEvidence]:
    values = _split_values(raw_value)
    sources = _split_sources(source_value)
    source_raw = _clean_text(source_value) or None
    if len(values) == len(sources) and sources:
        return [
            _ExpandedEvidence(item, sources[index], source_raw, index, "resolved")
            for index, item in enumerate(values)
        ]
    if len(values) == 1 and sources:
        # One result cited in several locations is one observation, not several
        # invented studies.  The complete evidence bundle remains joinable.
        return [_ExpandedEvidence(values[0], " | ".join(sources), source_raw, 0, "resolved")]
    if not sources and len(values) == 1:
        return [_ExpandedEvidence(values[0], None, None, 0, "unresolved")]
    return [
        _ExpandedEvidence(item, None, source_raw, index, "unresolved") for index, item in enumerate(values)
    ]


def _source_key(item: _ExpandedEvidence) -> str | None:
    return item.source_locator if item.pairing_status == "resolved" else None


_PK_PARAMETER_RE = re.compile(r"^(Rat|Mouse) PK:\s*(.+)$", flags=re.IGNORECASE)
_PK_ENDPOINTS: dict[str, tuple[str, str | None, str, str]] = {
    "T1/2 (PO) h": ("terminal_half_life", "PO", "h", "primary_observation"),
    "Tmax (PO) h": ("tmax", "PO", "h", "primary_observation"),
    "Cmax (PO) ng/mL": ("cmax", "PO", "ng/mL", "primary_observation"),
    "AUC0-t (PO)": ("auc_0_t", "PO", "ng*h/mL", "primary_observation"),
    "AUC0-inf (PO)": ("auc_0_inf", "PO", "ng*h/mL", "primary_observation"),
    "AUC0-t (IV)": ("auc_0_t", "IV", "ng*h/mL", "primary_observation"),
    "AUC0-inf (IV)": ("auc_0_inf", "IV", "ng*h/mL", "primary_observation"),
    "CL (IV) mL/kg/min": ("clearance", "IV", "mL/kg/min", "derived_from_exposure"),
    "Vdss (IV) L/kg": ("vdss", "IV", "L/kg", "reported_derived"),
    "%F": ("bioavailability", None, "%", "derived_from_exposure"),
}


def _parameter_spec(parameter: str) -> dict[str, Any] | None:
    pk_match = _PK_PARAMETER_RE.match(parameter)
    if pk_match:
        species, label = pk_match.groups()
        if label == "Dose (IV/PO) mg/kg":
            return {"kind": "pk_dose", "species": species.title()}
        if label in _PK_ENDPOINTS:
            endpoint, route, unit, leakage = _PK_ENDPOINTS[label]
            return {
                "kind": "pk_measurement",
                "species": species.title(),
                "endpoint": endpoint,
                "route": route,
                "unit": unit,
                "leakage_role": leakage,
            }
        return None
    if parameter == "hERG IC50 (µM)":
        return {
            "kind": "measurement",
            "endpoint": "herg_ic50",
            "unit": "uM",
            "assay_family": "herg_inhibition",
        }
    if parameter == "hERG % inhibition":
        return {
            "kind": "herg_panel",
            "endpoint": "herg_percent_inhibition",
            "unit": "%",
            "assay_family": "herg_inhibition",
        }

    match = re.match(r"^MetStab T1/2 \(min\):\s*(.+)$", parameter)
    if match:
        return {
            "kind": "measurement",
            "endpoint": "microsomal_stability_half_life",
            "unit": "min",
            "species": match.group(1),
            "matrix": "liver_microsomes",
            "assay_family": "metabolic_stability",
        }
    match = re.match(r"^Hepatic Extraction Eh:\s*(.+)$", parameter)
    if match:
        return {
            "kind": "measurement",
            "endpoint": "hepatic_extraction_ratio",
            "unit": "fraction",
            "species": match.group(1),
            "matrix": "liver",
            "assay_family": "hepatic_extraction",
        }
    match = re.match(r"^PPB %(Bound|Unbound):\s*(.+)$", parameter)
    if match:
        state, species = match.groups()
        return {
            "kind": "measurement",
            "endpoint": f"plasma_protein_{state.casefold()}_percent",
            "unit": "%",
            "species": species,
            "matrix": "plasma",
            "assay_family": "plasma_protein_binding",
        }
    match = re.match(r"^Plasma Stability T1/2 \(min\):\s*(.+)$", parameter)
    if match:
        return {
            "kind": "measurement",
            "endpoint": "plasma_stability_half_life",
            "unit": "min",
            "species": match.group(1),
            "matrix": "plasma",
            "assay_family": "plasma_stability",
        }
    return None


def _assay_protocol_record(
    endpoint: str,
    assay_family: str,
    *,
    species: str | None = None,
    matrix: str | None = None,
    route: str | None = None,
    concentration_value: float | None = None,
    concentration_unit: str | None = None,
    source_locator: str | None = None,
) -> dict[str, Any]:
    protocol_id = _stable_id(
        "AP",
        endpoint,
        assay_family,
        species,
        matrix,
        route,
        concentration_value,
        concentration_unit,
    )
    return AssayProtocol(
        assay_protocol_id=protocol_id,
        endpoint=endpoint,
        assay_family=assay_family,
        target_name="KCNH2/hERG" if assay_family == "herg_inhibition" else None,
        species=species,
        matrix=matrix,
        route=route,
        test_concentration_value=concentration_value,
        test_concentration_unit=concentration_unit,
        source="internal_workbook",
        source_locator=source_locator,
        context_note="Unreported protocol attributes remain unavailable.",
    ).model_dump(mode="python")


def _molecular_weight(smiles: str | None) -> float | None:
    if not smiles or Chem is None or Descriptors is None:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    return float(Descriptors.MolWt(molecule)) if molecule is not None else None


def _structure_grouping_fields(smiles: str | None) -> dict[str, str | None]:
    """Derive explicit stereo/scaffold metadata for the Compound contract."""

    if not smiles or Chem is None:
        return {
            "stereochemistry_status": "unknown",
            "series_id": None,
            "scaffold": None,
            "scaffold_method": None,
        }
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {
            "stereochemistry_status": "unknown",
            "series_id": None,
            "scaffold": None,
            "scaffold_method": None,
        }
    potential_stereo = list(Chem.FindPotentialStereo(molecule))
    stereo_labels = [str(item.specified).casefold() for item in potential_stereo]
    specified_count = sum(label == "specified" for label in stereo_labels)
    if not potential_stereo:
        stereo_status = "not_applicable"
    elif specified_count == len(potential_stereo):
        stereo_status = "specified"
    elif specified_count:
        stereo_status = "partially_specified"
    else:
        stereo_status = "unspecified"
    scaffold, method = scaffold_key(smiles)
    return {
        "stereochemistry_status": stereo_status,
        "series_id": _stable_id("SERIES", scaffold),
        "scaffold": scaffold,
        "scaffold_method": method,
    }


def _optional_float(value: object) -> float | None:
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    parsed = float(_clean_text(value))
    return parsed if math.isfinite(parsed) else None


def _empty_review(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def normalize_internal_workbook(
    source: str | os.PathLike[str] | Mapping[str, pd.DataFrame],
    *,
    closure_tolerance: float = 0.15,
) -> NormalizationResult:
    """Normalize internal PK/hERG evidence without cross-study averaging."""

    if closure_tolerance < 0:
        raise ValueError("closure_tolerance must be non-negative")
    sheets = _read_internal_sheets(source)
    missing = {"SMILES", "Provenance", "pKa_detail"} - set(sheets)
    if missing:
        raise ValueError(f"internal workbook is missing required sheets: {sorted(missing)}")
    wide = sheets["SMILES"].copy()
    provenance = sheets["Provenance"].copy()
    pka_detail = sheets["pKa_detail"].copy()
    for column in ("Compound", "Kekule Canonical SMILES"):
        if column not in wide:
            raise ValueError(f"SMILES sheet is missing {column!r}")
    for column in ("Compound", "Parameter", "Source slide(s)", "Raw value(s)"):
        if column not in provenance:
            raise ValueError(f"Provenance sheet is missing {column!r}")

    issues: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    compound_records_by_id: dict[str, dict[str, Any]] = {}
    state_records_by_id: dict[str, dict[str, Any]] = {}
    compound_aliases: list[dict[str, Any]] = []
    compound_ids: dict[str, str] = {}
    state_ids: dict[str, str] = {}

    for row_index, row in wide.iterrows():
        source_name = _clean_text(row.get("Compound"))
        if not source_name:
            issues.append(
                {"severity": "error", "code": "missing_compound_name", "source_row": int(row_index)}
            )
            continue
        submitted_smiles = _clean_text(row.get("Kekule Canonical SMILES"))
        structure = standardize_smiles(submitted_smiles)
        # Structure identity, not a source alias, is the canonical join key.
        # Invalid or absent structures remain source-specific so they cannot be
        # accidentally collapsed.
        identity_token = structure.structure_id or f"source-alias:{source_name}"
        compound_id = _stable_id("CMP", identity_token)
        compound_ids[source_name] = compound_id
        supplied_mw = parse_qualified_value(row.get("MW"))
        mw = (
            supplied_mw.value
            if supplied_mw.value is not None and supplied_mw.value > 0
            else _molecular_weight(structure.standardized_smiles)
        )
        validation_status = (
            "validated"
            if structure.structure_valid is True
            else "invalid"
            if structure.structure_valid is False
            else "unvalidated"
        )
        grouping_fields = _structure_grouping_fields(structure.standardized_smiles or None)
        compound_record = Compound(
            compound_id=compound_id,
            structure_id=structure.structure_id or None,
            submitted_smiles=submitted_smiles or None,
            canonical_smiles=structure.canonical_smiles or None,
            standardized_smiles=structure.standardized_smiles or None,
            standard_inchi_key=structure.standard_inchi_key or None,
            molecular_weight_g_mol=mw,
            formal_charge=structure.formal_charge,
            **grouping_fields,
            source="internal_workbook:SMILES",
            source_record_id=structure.structure_id or source_name,
            validation_status=validation_status,
            standardization_version=structure.structure_standardization_version,
            context_note=structure.structure_error or None,
        ).model_dump(mode="python")
        compound_records_by_id.setdefault(compound_id, compound_record)
        compound_aliases.append(
            {
                "source_compound_name": source_name,
                "compound_id": compound_id,
                "structure_id": structure.structure_id or None,
                "source_row": int(row_index),
                "submitted_smiles": submitted_smiles or None,
            }
        )
        state_id = _stable_id("STATE", compound_id, "submitted_parent")
        state_ids[source_name] = state_id
        state_record = ChemicalState(
            chemical_state_id=state_id,
            compound_id=compound_id,
            state_type="unknown",
            smiles=structure.standardized_smiles or submitted_smiles or None,
            formal_charge=structure.formal_charge,
            method="submitted parent structure; protonation microstate not assigned",
            source="internal_workbook:SMILES",
            context_note="The assay-relevant protonation, tautomer, and conformer states are unavailable.",
        ).model_dump(mode="python")
        state_records_by_id.setdefault(state_id, state_record)
        if structure.structure_valid is False:
            quarantine.append(
                {
                    "code": "invalid_structure",
                    "compound": source_name,
                    "source_row": int(row_index),
                    "detail": structure.structure_error,
                }
            )

    protocol_records: dict[str, dict[str, Any]] = {}
    study_records: dict[str, dict[str, Any]] = {}
    dose_pairs: dict[tuple[str, str, str], dict[str, Any]] = {}
    measurement_records: list[dict[str, Any]] = []
    measurement_aliases: list[dict[str, Any]] = []
    pk_observations: list[dict[str, Any]] = []

    def add_issue(severity: str, code: str, **context: Any) -> None:
        issues.append({"severity": severity, "code": code, **context})

    def register_protocol(**kwargs: Any) -> str:
        record = _assay_protocol_record(**kwargs)
        protocol_records.setdefault(record["assay_protocol_id"], record)
        return str(record["assay_protocol_id"])

    # Pass 1: explicit IV and PO dose events.
    for row_index, row in provenance.iterrows():
        parameter = _clean_text(row.get("Parameter"))
        spec = _parameter_spec(parameter)
        if not spec or spec["kind"] != "pk_dose":
            continue
        compound_name = _clean_text(row.get("Compound"))
        resolved_compound_id = compound_ids.get(compound_name)
        if not resolved_compound_id:
            add_issue("error", "unknown_compound", source_row=int(row_index), compound=compound_name)
            continue
        compound_id = resolved_compound_id
        for evidence in _expand_evidence(row.get("Raw value(s)"), row.get("Source slide(s)")):
            dose_match = re.fullmatch(
                r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*/\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*",
                evidence.raw_value,
            )
            if not dose_match:
                add_issue(
                    "error",
                    "unparsed_iv_po_dose",
                    source_row=int(row_index),
                    compound=compound_name,
                    submitted_value=evidence.raw_value,
                )
                continue
            iv_dose, po_dose = (float(item) for item in dose_match.groups())
            locator_token = evidence.source_locator or f"UNRESOLVED:{evidence.source_ordinal}"
            pair_id = _stable_id("PKPAIR", compound_id, spec["species"], locator_token)
            pair_key = (compound_id, spec["species"], locator_token)
            pair = {
                "event_pair_id": pair_id,
                "compound_id": compound_id,
                "species": spec["species"],
                "source_locator": evidence.source_locator,
                "source_locator_raw": evidence.source_locator_raw,
                "pairing_status": evidence.pairing_status,
                "source_row": int(row_index),
            }
            for route, dose in (("IV", iv_dose), ("PO", po_dose)):
                protocol_id = register_protocol(
                    endpoint="dose",
                    assay_family="in_vivo_pk",
                    species=spec["species"],
                    matrix=None,
                    route=route,
                    source_locator=evidence.source_locator,
                )
                study_id = _stable_id("PKSTUDY", pair_id, route)
                study_records[study_id] = PKStudy(
                    pk_study_id=study_id,
                    event_pair_id=pair_id,
                    compound_id=compound_id,
                    chemical_state_id=state_ids.get(compound_name),
                    assay_protocol_id=protocol_id,
                    species=spec["species"],
                    route=route,
                    dose_value=dose,
                    dose_unit="mg/kg",
                    source="internal_workbook:Provenance",
                    source_locator=evidence.source_locator or evidence.source_locator_raw,
                    source_record_id=f"Provenance:{row_index}",
                    pairing_status=evidence.pairing_status,
                    context_note=(
                        None
                        if evidence.pairing_status == "resolved"
                        else "The number of source locations did not match the number of raw values."
                    ),
                ).model_dump(mode="python")
                pair[route] = {"study_id": study_id, "dose": dose}
            dose_pairs[pair_key] = pair
            if evidence.pairing_status == "unresolved":
                add_issue(
                    "error",
                    "unresolved_study_pairing",
                    source_row=int(row_index),
                    compound=compound_name,
                    parameter=parameter,
                    source_locator_raw=evidence.source_locator_raw,
                    raw_value=evidence.raw_value,
                )

    # Pass 2: endpoint observations from the authoritative provenance rows.
    for row_index, row in provenance.iterrows():
        parameter = _clean_text(row.get("Parameter"))
        spec = _parameter_spec(parameter)
        if not spec or spec["kind"] == "pk_dose":
            continue
        compound_name = _clean_text(row.get("Compound"))
        resolved_compound_id = compound_ids.get(compound_name)
        if not resolved_compound_id:
            add_issue("error", "unknown_compound", source_row=int(row_index), compound=compound_name)
            continue
        compound_id = resolved_compound_id
        expanded = _expand_evidence(row.get("Raw value(s)"), row.get("Source slide(s)"))
        for evidence in expanded:
            if evidence.pairing_status == "unresolved" and spec["kind"] == "pk_measurement":
                add_issue(
                    "error",
                    "unresolved_study_pairing",
                    source_row=int(row_index),
                    compound=compound_name,
                    parameter=parameter,
                    source_locator_raw=evidence.source_locator_raw,
                    raw_value=evidence.raw_value,
                )
            observations: list[tuple[ParsedQualifiedValue, float | None, str | None, str]]
            if spec["kind"] == "herg_panel":
                panel = parse_concentration_response_panel(evidence.raw_value)
                if not panel:
                    add_issue(
                        "error",
                        "unparsed_concentration_panel",
                        source_row=int(row_index),
                        compound=compound_name,
                        raw_value=evidence.raw_value,
                    )
                    continue
                observations = [
                    (item.response, item.concentration_value, item.concentration_unit, item.submitted_segment)
                    for item in panel
                ]
            else:
                observations = [(parse_qualified_value(evidence.raw_value), None, None, evidence.raw_value)]
            for panel_index, (parsed, concentration, concentration_unit, submitted_segment) in enumerate(
                observations
            ):
                endpoint = str(spec["endpoint"])
                species_value = spec.get("species")
                observation_species = str(species_value) if species_value is not None else None
                matrix_value = spec.get("matrix")
                observation_matrix = str(matrix_value) if matrix_value is not None else None
                route_value = spec.get("route")
                observation_route = str(route_value) if route_value is not None else None
                assay_family = str(spec.get("assay_family", "in_vivo_pk"))
                protocol_id = register_protocol(
                    endpoint=endpoint,
                    assay_family=assay_family,
                    species=observation_species,
                    matrix=observation_matrix,
                    route=observation_route,
                    concentration_value=concentration,
                    concentration_unit=concentration_unit,
                    source_locator=evidence.source_locator,
                )
                observation_pk_study_id: str | None = None
                matched_pair_id: str | None = None
                effective_pairing: Literal["resolved", "unresolved", "not_applicable"] = "not_applicable"
                matched_pair: dict[str, Any] | None = None
                if spec["kind"] == "pk_measurement":
                    matched_locator = (
                        evidence.source_locator if evidence.pairing_status == "resolved" else None
                    )
                    matched_pair = (
                        dose_pairs.get((compound_id, observation_species, matched_locator))
                        if matched_locator and observation_species
                        else None
                    )
                    if matched_pair and (observation_route is None or observation_route in matched_pair):
                        matched_pair_id = str(matched_pair["event_pair_id"])
                        observation_pk_study_id = (
                            str(matched_pair[observation_route]["study_id"]) if observation_route else None
                        )
                        effective_pairing = "resolved"
                    else:
                        effective_pairing = "unresolved"
                        add_issue(
                            "error",
                            "missing_explicit_dose_pair",
                            source_row=int(row_index),
                            compound=compound_name,
                            parameter=parameter,
                            source_locator=evidence.source_locator,
                        )
                leakage = spec.get("leakage_role", "primary_observation")
                origin = (
                    "reported_derived"
                    if leakage in {"derived_from_exposure", "reported_derived"}
                    else "measured"
                )
                leakage_role = (
                    "derived_from_exposure" if leakage == "derived_from_exposure" else "primary_observation"
                )
                measurement_id = _stable_id(
                    "MEAS",
                    compound_id,
                    parameter,
                    row_index,
                    evidence.source_ordinal,
                    panel_index,
                    submitted_segment,
                )
                note_parts = [part for part in (parsed.annotation,) if part]
                note_parts.append(f"Source compound alias: {compound_name}.")
                if effective_pairing == "unresolved":
                    note_parts.append(
                        "No explicit one-to-one source/dose-event pairing was available; no study was guessed."
                    )
                record = Measurement(
                    measurement_id=measurement_id,
                    compound_id=compound_id,
                    chemical_state_id=state_ids.get(compound_name),
                    assay_protocol_id=protocol_id,
                    pk_study_id=observation_pk_study_id,
                    endpoint=endpoint,
                    value=parsed.value,
                    unit=normalize_unit(spec["unit"]),
                    relation=parsed.relation,
                    censoring=parsed.censoring,
                    lower_bound=parsed.lower_bound,
                    upper_bound=parsed.upper_bound,
                    test_concentration_value=concentration,
                    test_concentration_unit=concentration_unit,
                    species=observation_species,
                    matrix=observation_matrix,
                    route=observation_route,
                    origin=origin,
                    submitted_value=submitted_segment,
                    source="internal_workbook:Provenance",
                    source_locator=evidence.source_locator or evidence.source_locator_raw,
                    source_record_id=f"Provenance:{row_index}:{evidence.source_ordinal}:{panel_index}",
                    pairing_status=effective_pairing,
                    leakage_role=leakage_role,
                    model_eligible=leakage_role != "derived_from_exposure" and parsed.status != "unparsed",
                    context_note=" ".join(note_parts) or None,
                ).model_dump(mode="python")
                measurement_records.append(record)
                measurement_aliases.append(
                    {
                        "measurement_id": measurement_id,
                        "source_compound_name": compound_name,
                        "compound_id": compound_id,
                        "source_row": int(row_index),
                    }
                )
                if parsed.status == "unparsed":
                    add_issue(
                        "error",
                        "unparsed_measurement",
                        measurement_id=measurement_id,
                        compound=compound_name,
                        parameter=parameter,
                        raw_value=evidence.raw_value,
                    )
                if spec["kind"] == "pk_measurement":
                    pk_observations.append(
                        {
                            "compound_id": compound_id,
                            "species": observation_species,
                            "source_locator": evidence.source_locator,
                            "endpoint": endpoint,
                            "route": observation_route,
                            "value": parsed.value,
                            "unit": spec["unit"],
                            "measurement_id": measurement_id,
                            "event_pair_id": matched_pair_id,
                            "pk_study_id": observation_pk_study_id,
                            "pairing_status": effective_pairing,
                            "source_row": int(row_index),
                        }
                    )

    # pKa detail is a separate, explicit source of chemical-state evidence.
    if "Compound Name" not in pka_detail:
        add_issue("error", "missing_pka_compound_column")
    else:
        for row_index, row in pka_detail.iterrows():
            compound_name = _clean_text(row.get("Compound Name"))
            pka_compound_id = compound_ids.get(compound_name)
            if not pka_compound_id:
                continue
            fields = (
                ("All basic pKaH (desc)", "basic_pka", "descending"),
                ("Acidic pKa < 13", "acidic_pka_below_13", "reported if below 13"),
            )
            for column, endpoint, note in fields:
                values = [item.strip() for item in _clean_text(row.get(column)).split(",") if item.strip()]
                for rank, submitted in enumerate(values, start=1):
                    parsed = parse_qualified_value(submitted)
                    protocol_id = register_protocol(
                        endpoint=endpoint,
                        assay_family="calculated_or_reported_ionization",
                        source_locator=f"pKa_detail:{row_index}",
                    )
                    measurement_records.append(
                        Measurement(
                            measurement_id=_stable_id("MEAS", pka_compound_id, endpoint, row_index, rank),
                            compound_id=pka_compound_id,
                            chemical_state_id=state_ids.get(compound_name),
                            assay_protocol_id=protocol_id,
                            endpoint=endpoint,
                            value=parsed.value,
                            unit="pKa",
                            relation=parsed.relation,
                            censoring=parsed.censoring,
                            submitted_value=submitted,
                            source="internal_workbook:pKa_detail",
                            source_locator=f"pKa_detail:{row_index}",
                            source_record_id=f"pKa_detail:{row_index}:{column}:{rank}",
                            pairing_status="not_applicable",
                            leakage_role="descriptor",
                            model_eligible=parsed.value is not None,
                            context_note=f"Source ordering: {note}; rank={rank}. Chemical microstate assignment unavailable.",
                        ).model_dump(mode="python")
                    )
                    measurement_aliases.append(
                        {
                            "measurement_id": _stable_id("MEAS", pka_compound_id, endpoint, row_index, rank),
                            "source_compound_name": compound_name,
                            "compound_id": pka_compound_id,
                            "source_row": int(row_index),
                        }
                    )

    # Mechanistic closure: reported CL and F are checked against their exposure inputs.
    derived_records: list[dict[str, Any]] = []
    observations_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in pk_observations:
        if observation["pairing_status"] == "resolved" and observation["event_pair_id"]:
            observations_by_pair[str(observation["event_pair_id"])].append(observation)

    def pick(rows: Sequence[dict[str, Any]], endpoint: str, route: str | None) -> dict[str, Any] | None:
        candidates = [
            row
            for row in rows
            if row["endpoint"] == endpoint and row["route"] == route and row["value"] is not None
        ]
        return candidates[0] if len(candidates) == 1 else None

    pairs_by_id = {
        pair["event_pair_id"]: pair for pair in dose_pairs.values() if pair["pairing_status"] == "resolved"
    }
    for pair_id, pair in pairs_by_id.items():
        rows = observations_by_pair.get(pair_id, [])
        iv_auc = pick(rows, "auc_0_inf", "IV") or pick(rows, "auc_0_t", "IV")
        po_auc = pick(rows, "auc_0_inf", "PO") or pick(rows, "auc_0_t", "PO")
        reported_cl = pick(rows, "clearance", "IV")
        reported_f = pick(rows, "bioavailability", None)
        if iv_auc and iv_auc["value"] and pair["IV"]["dose"]:
            recomputed_cl = 1_000_000.0 * pair["IV"]["dose"] / (60.0 * iv_auc["value"])
            error = (
                abs(reported_cl["value"] - recomputed_cl) / max(abs(reported_cl["value"]), 1e-12)
                if reported_cl and reported_cl["value"] is not None
                else None
            )
            derived_records.append(
                DerivedPKParameter(
                    derived_pk_parameter_id=_stable_id("DPK", pair_id, "clearance_recomputed"),
                    compound_id=pair["compound_id"],
                    pk_study_id=pair["IV"]["study_id"],
                    event_pair_id=pair_id,
                    endpoint="clearance",
                    value=recomputed_cl,
                    unit="mL/kg/min",
                    origin="recomputed",
                    method="dose/AUC closure",
                    formula="CL[mL/kg/min] = dose[mg/kg] * 1e6 / (AUC[ng*h/mL] * 60)",
                    input_ids=(iv_auc["measurement_id"], pair["IV"]["study_id"]),
                    reported_value=reported_cl["value"] if reported_cl else None,
                    recomputed_value=recomputed_cl,
                    closure_relative_error=error,
                    closure_status="not_tested"
                    if error is None
                    else "pass"
                    if error <= closure_tolerance
                    else "fail",
                    leakage_role="derived_from_exposure",
                    model_eligible=False,
                    source="internal_workbook:recomputed",
                    source_locator=pair["source_locator"],
                    context_note="AUC0-inf was preferred; AUC0-t was used only when AUC0-inf was unavailable.",
                ).model_dump(mode="python")
            )
        if iv_auc and po_auc and iv_auc["value"] and po_auc["value"]:
            recomputed_f = (
                100.0 * (po_auc["value"] / pair["PO"]["dose"]) / (iv_auc["value"] / pair["IV"]["dose"])
            )
            error = (
                abs(reported_f["value"] - recomputed_f) / max(abs(reported_f["value"]), 1e-12)
                if reported_f and reported_f["value"] is not None
                else None
            )
            derived_records.append(
                DerivedPKParameter(
                    derived_pk_parameter_id=_stable_id("DPK", pair_id, "bioavailability_recomputed"),
                    compound_id=pair["compound_id"],
                    event_pair_id=pair_id,
                    endpoint="bioavailability",
                    value=recomputed_f,
                    unit="%",
                    origin="recomputed",
                    method="dose-normalized AUC closure",
                    formula="F[%] = 100 * (AUC_PO/dose_PO) / (AUC_IV/dose_IV)",
                    input_ids=(
                        po_auc["measurement_id"],
                        iv_auc["measurement_id"],
                        pair["PO"]["study_id"],
                        pair["IV"]["study_id"],
                    ),
                    reported_value=reported_f["value"] if reported_f else None,
                    recomputed_value=recomputed_f,
                    closure_relative_error=error,
                    closure_status="not_tested"
                    if error is None
                    else "pass"
                    if error <= closure_tolerance
                    else "fail",
                    leakage_role="derived_from_exposure",
                    model_eligible=False,
                    source="internal_workbook:recomputed",
                    source_locator=pair["source_locator"],
                    context_note="Matched PO and IV evidence only; no cross-source pairing was inferred.",
                ).model_dump(mode="python")
            )

    # Alias-level disagreements are preserved and flagged, never averaged.
    measurement_alias_frame = pd.DataFrame(measurement_aliases)
    if measurement_records and not measurement_alias_frame.empty:
        conflict_source = pd.DataFrame(measurement_records).merge(
            measurement_alias_frame[["measurement_id", "source_compound_name"]],
            on="measurement_id",
            how="left",
        )
        conflict_keys = [
            "compound_id",
            "endpoint",
            "species",
            "matrix",
            "route",
            "test_concentration_value",
            "test_concentration_unit",
        ]
        for key, group in conflict_source.groupby(conflict_keys, dropna=False):
            aliases = sorted(set(group["source_compound_name"].dropna().astype(str)))
            conflict_values = sorted(set(float(item) for item in group["value"].dropna()))
            if len(aliases) > 1 and len(conflict_values) > 1:
                quarantine.append(
                    {
                        "code": "aliased_structure_measurement_conflict",
                        "compound": " | ".join(aliases),
                        "source_row": None,
                        "detail": (
                            f"key={key!r}; values={conflict_values!r}; "
                            f"measurements={group['measurement_id'].tolist()!r}"
                        ),
                    }
                )

    compound_records = list(compound_records_by_id.values())
    state_records = list(state_records_by_id.values())
    tables = {
        "compounds": records_to_frame(Compound, compound_records),
        "chemical_states": records_to_frame(ChemicalState, state_records),
        "assay_protocols": records_to_frame(AssayProtocol, protocol_records.values()),
        "measurements": records_to_frame(Measurement, measurement_records),
        "pk_studies": records_to_frame(PKStudy, study_records.values()),
        "derived_pk_parameters": records_to_frame(DerivedPKParameter, derived_records),
    }
    issues_frame = pd.DataFrame(issues)
    quarantine_frame = pd.DataFrame(quarantine)
    endpoint_counts = Counter(tables["measurements"]["endpoint"].astype(str))
    summary = {
        "source": "internal_workbook",
        "provenance_authoritative": True,
        "wide_endpoint_values_used": False,
        "compounds": len(tables["compounds"]),
        "chemical_states": len(tables["chemical_states"]),
        "pk_studies": len(tables["pk_studies"]),
        "measurements": len(tables["measurements"]),
        "derived_pk_parameters": len(tables["derived_pk_parameters"]),
        "endpoint_counts": dict(sorted(endpoint_counts.items())),
        "unresolved_pairing_errors": int(
            (issues_frame.get("code", pd.Series(dtype=str)) == "unresolved_study_pairing").sum()
        ),
        "missing_explicit_dose_pair_errors": int(
            (issues_frame.get("code", pd.Series(dtype=str)) == "missing_explicit_dose_pair").sum()
        ),
        "closure_pass": int(
            (tables["derived_pk_parameters"].get("closure_status", pd.Series(dtype=str)) == "pass").sum()
        ),
        "closure_fail": int(
            (tables["derived_pk_parameters"].get("closure_status", pd.Series(dtype=str)) == "fail").sum()
        ),
        "closure_tolerance": closure_tolerance,
        "source_compound_aliases": len(compound_aliases),
        "unique_canonical_compounds": len(tables["compounds"]),
        "aliased_structure_conflicts": sum(
            1 for row in quarantine if row.get("code") == "aliased_structure_measurement_conflict"
        ),
    }
    return NormalizationResult(
        tables,
        issues_frame,
        quarantine_frame,
        summary,
        review_tables={
            "compound_aliases": pd.DataFrame(compound_aliases),
            "measurement_alias_lineage": measurement_alias_frame,
        },
    )


def _read_sun_sheets(
    source: str | os.PathLike[str] | Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    if isinstance(source, Mapping):
        return {name: frame.copy() for name, frame in source.items()}
    path = Path(source)
    return {
        "Classification": pd.read_excel(path, sheet_name="Classification", header=3),
        "Regression": pd.read_excel(path, sheet_name="Regression"),
        "Validation": pd.read_excel(path, sheet_name="Validation"),
    }


def _pic50_from_ic50(value_nm: float | None, relation: str) -> tuple[float | None, str]:
    if value_nm is None or value_nm <= 0:
        return None, "not_reported"
    inverted = {"=": "=", "~": "~", ">": "<", ">=": "<=", "<": ">", "<=": ">="}
    return 9.0 - math.log10(value_nm), inverted[relation]


def normalize_sun_herg_workbook(
    source: str | os.PathLike[str] | Mapping[str, pd.DataFrame],
    *,
    blocker_threshold_nm: float = 10_000.0,
    domain_max_mw_g_mol: float = 600.0,
    conflict_tolerance_pic50: float = 0.30,
) -> NormalizationResult:
    """Normalize and QC the Sun hERG classification/regression supplement."""

    if blocker_threshold_nm <= 0 or domain_max_mw_g_mol <= 0 or conflict_tolerance_pic50 < 0:
        raise ValueError("thresholds must be positive and conflict tolerance non-negative")
    sheets = _read_sun_sheets(source)
    required = {"Classification", "Regression", "Validation"}
    if required - set(sheets):
        raise ValueError(f"Sun workbook is missing sheets: {sorted(required - set(sheets))}")
    classification = sheets["Classification"].copy()
    regression = sheets["Regression"].copy()
    validation = sheets["Validation"].copy()
    required_columns = {
        "Classification": {"Smiles", "hERG Class", "IC50(nM)"},
        "Regression": {"Smiles", "IC50(nM)", "hERG"},
        "Validation": {"Smiles", "hERG (nM)"},
    }
    for name, columns in required_columns.items():
        missing = columns - set(sheets[name].columns)
        if missing:
            raise ValueError(f"{name} sheet is missing columns: {sorted(missing)}")

    all_smiles = pd.concat(
        [classification["Smiles"], regression["Smiles"], validation["Smiles"]],
        ignore_index=True,
    ).map(_clean_text)
    unique_smiles = sorted(set(all_smiles))
    standardized = {smiles: standardize_smiles(smiles) for smiles in unique_smiles}
    activity_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []

    def add_activity(
        role: str,
        source_row: int,
        smiles: object,
        *,
        source_class: int | None = None,
        ic50: ParsedQualifiedValue | None = None,
        stored_log10_nm: float | None = None,
        source_herg_auxiliary: float | None = None,
    ) -> None:
        raw_smiles = _clean_text(smiles)
        structure = standardized[raw_smiles]
        structure_id = structure.structure_id or _stable_id("RAWSTR", raw_smiles)
        compound_id = _stable_id("SUNCMP", structure_id)
        mw = _molecular_weight(structure.standardized_smiles or raw_smiles)
        canonical_blocker = 1 - source_class if source_class in {0, 1} else None
        ic50_value: float | None
        pic50_value: float | None
        ic50_relation: str
        pic50_relation: str
        if stored_log10_nm is not None and math.isfinite(stored_log10_nm):
            ic50_value = 10.0**stored_log10_nm
            ic50_relation = "="
            pic50_value = 9.0 - stored_log10_nm
            pic50_relation = "="
            ic50_submitted = str(stored_log10_nm)
            parse_status = "parsed_log10_nm"
        else:
            parsed = ic50 or ParsedQualifiedValue("", None, "not_reported", "missing", status="missing")
            ic50_value = parsed.value
            ic50_relation = parsed.relation
            pic50_value, pic50_relation = _pic50_from_ic50(ic50_value, ic50_relation)
            ic50_submitted = parsed.submitted_value
            parse_status = parsed.status
        activity_rows.append(
            {
                "dataset_role": role,
                "source_row": int(source_row),
                "raw_smiles": raw_smiles,
                "structure_id": structure_id,
                "compound_id": compound_id,
                "structure_valid": structure.structure_valid,
                "source_class": source_class,
                "canonical_blocker_class": canonical_blocker,
                "ic50_nm_value": ic50_value,
                "ic50_relation": ic50_relation,
                "pic50_value": pic50_value,
                "pic50_relation": pic50_relation,
                "stored_log10_nm": stored_log10_nm,
                "source_herg_auxiliary": source_herg_auxiliary,
                "submitted_value": ic50_submitted,
                "parse_status": parse_status,
                "computed_mw_g_mol": mw,
            }
        )
        if structure.structure_valid is False:
            issues.append(
                {
                    "severity": "error",
                    "code": "invalid_structure",
                    "dataset_role": role,
                    "source_row": int(source_row),
                }
            )
            quarantine.append(
                {
                    "code": "invalid_structure",
                    "dataset_role": role,
                    "source_row": int(source_row),
                    "structure_id": structure_id,
                }
            )

    for row_index, row in classification.iterrows():
        source_class = pd.to_numeric(pd.Series([row.get("hERG Class")]), errors="coerce").iloc[0]
        class_value = int(source_class) if pd.notna(source_class) and int(source_class) in {0, 1} else None
        if class_value is None:
            issues.append(
                {
                    "severity": "error",
                    "code": "invalid_source_class",
                    "dataset_role": "classification",
                    "source_row": int(row_index),
                }
            )
        add_activity(
            "classification",
            int(row_index),
            row.get("Smiles"),
            source_class=class_value,
            ic50=parse_qualified_value(row.get("IC50(nM)")),
        )
    for row_index, row in regression.iterrows():
        stored = pd.to_numeric(pd.Series([row.get("IC50(nM)")]), errors="coerce").iloc[0]
        auxiliary = pd.to_numeric(pd.Series([row.get("hERG")]), errors="coerce").iloc[0]
        add_activity(
            "regression",
            int(row_index),
            row.get("Smiles"),
            stored_log10_nm=float(stored) if pd.notna(stored) else None,
            source_herg_auxiliary=float(auxiliary) if pd.notna(auxiliary) else None,
        )
    for row_index, row in validation.iterrows():
        add_activity(
            "validation",
            int(row_index),
            row.get("Smiles"),
            ic50=parse_qualified_value(row.get("hERG (nM)")),
        )

    activity = pd.DataFrame(activity_rows)

    # Source class 0 means blocker; canonical class 1 means blocker.
    determinate_expected: list[float | None] = []
    for row in activity.to_dict("records"):
        expected: float | None = None
        if row["dataset_role"] == "classification" and row["ic50_nm_value"] is not None:
            value = float(row["ic50_nm_value"])
            relation = row["ic50_relation"]
            if relation in {"=", "~"}:
                expected = float(value < blocker_threshold_nm)
            elif relation in {">", ">="} and value >= blocker_threshold_nm:
                expected = 0.0
            elif relation in {"<", "<="} and value <= blocker_threshold_nm:
                expected = 1.0
        determinate_expected.append(expected)
    activity["threshold_expected_blocker_class"] = determinate_expected
    disagreements = activity[
        activity["threshold_expected_blocker_class"].notna()
        & activity["canonical_blocker_class"].notna()
        & (activity["threshold_expected_blocker_class"] != activity["canonical_blocker_class"])
    ]
    for row in disagreements.to_dict("records"):
        quarantine.append(
            {
                "code": "class_threshold_disagreement",
                "dataset_role": row["dataset_role"],
                "source_row": row["source_row"],
                "structure_id": row["structure_id"],
                "detail": f"canonical_class={row['canonical_blocker_class']}; threshold_class={row['threshold_expected_blocker_class']}",
            }
        )

    # Preserve every measurement but quarantine structure-level conflicts.
    for structure_id, group in activity.groupby("structure_id", dropna=False):
        classes = set(group["canonical_blocker_class"].dropna().astype(int))
        if len(classes) > 1:
            quarantine.append(
                {
                    "code": "conflicting_source_classes",
                    "dataset_role": "classification",
                    "source_row": None,
                    "structure_id": structure_id,
                    "detail": f"classes={sorted(classes)}; rows={group.index.tolist()}",
                }
            )
        exact = group[group["pic50_relation"].isin(["=", "~"]) & group["pic50_value"].notna()]
        if (
            len(exact) > 1
            and float(exact["pic50_value"].max() - exact["pic50_value"].min()) > conflict_tolerance_pic50
        ):
            roles = sorted(set(exact["dataset_role"]))
            code = (
                "validation_measurement_disagreement"
                if "validation" in roles and len(roles) > 1
                else "conflicting_measurements"
            )
            quarantine.append(
                {
                    "code": code,
                    "dataset_role": "|".join(roles),
                    "source_row": None,
                    "structure_id": structure_id,
                    "detail": f"pIC50_range={exact['pic50_value'].min():.6g}..{exact['pic50_value'].max():.6g}",
                }
            )

    training_structures = set(
        activity.loc[activity["dataset_role"].isin(["classification", "regression"]), "structure_id"]
    )
    validation_structures = set(activity.loc[activity["dataset_role"] == "validation", "structure_id"])
    overlaps = sorted(training_structures & validation_structures)
    for structure_id in overlaps:
        quarantine.append(
            {
                "code": "train_validation_structure_overlap",
                "dataset_role": "training|validation",
                "source_row": None,
                "structure_id": structure_id,
                "detail": "Standardized structure occurs in both model-development and validation sheets.",
            }
        )

    domain_contradictions = activity[
        activity["computed_mw_g_mol"].notna() & (activity["computed_mw_g_mol"] > domain_max_mw_g_mol)
    ].copy()
    domain_contradictions["domain_limit_g_mol"] = domain_max_mw_g_mol
    domain_contradictions["contradiction"] = "computed_mw_above_reported_domain_limit"

    # Deduplicate compounds by standardized structure while preserving all activity rows.
    compound_records: list[dict[str, Any]] = []
    for structure_id, group in activity.groupby("structure_id", sort=True):
        first = group.iloc[0]
        raw_smiles = str(first["raw_smiles"])
        structure = standardized[raw_smiles]
        grouping_fields = _structure_grouping_fields(structure.standardized_smiles or None)
        compound_records.append(
            Compound(
                compound_id=str(first["compound_id"]),
                structure_id=str(structure_id),
                submitted_smiles=raw_smiles,
                canonical_smiles=structure.canonical_smiles or None,
                standardized_smiles=structure.standardized_smiles or None,
                standard_inchi_key=structure.standard_inchi_key or None,
                molecular_weight_g_mol=float(first["computed_mw_g_mol"])
                if pd.notna(first["computed_mw_g_mol"])
                else None,
                formal_charge=structure.formal_charge,
                **grouping_fields,
                source="Sun hERG supplementary workbook",
                source_record_id=str(structure_id),
                validation_status="validated"
                if structure.structure_valid is True
                else "invalid"
                if structure.structure_valid is False
                else "unvalidated",
                standardization_version=structure.structure_standardization_version,
                context_note=f"{len(group)} source rows retained for this deduplicated structure.",
            ).model_dump(mode="python")
        )

    protocol_records = {
        "classification": _assay_protocol_record(
            endpoint="herg_blocker_class", assay_family="herg_compilation"
        ),
        "classification_ic50": _assay_protocol_record(endpoint="herg_ic50", assay_family="herg_compilation"),
        "regression": _assay_protocol_record(endpoint="herg_pic50", assay_family="herg_compilation"),
        "validation": _assay_protocol_record(endpoint="herg_ic50", assay_family="herg_external_validation"),
    }
    protocols_by_id = {record["assay_protocol_id"]: record for record in protocol_records.values()}
    measurement_records: list[dict[str, Any]] = []
    for row in activity.to_dict("records"):
        role = str(row["dataset_role"])
        source_row = int(row["source_row"])
        overlap = row["structure_id"] in overlaps
        if role == "classification":
            class_protocol = protocol_records["classification"]
            if pd.notna(row["canonical_blocker_class"]):
                measurement_records.append(
                    Measurement(
                        measurement_id=_stable_id("SUNMEAS", role, source_row, "class"),
                        compound_id=row["compound_id"],
                        assay_protocol_id=class_protocol["assay_protocol_id"],
                        endpoint="herg_blocker_class",
                        value=_optional_float(row["canonical_blocker_class"]),
                        unit="binary",
                        relation="=",
                        censoring="none",
                        origin="measured",
                        submitted_value=str(row["source_class"]),
                        source="Sun hERG:Classification",
                        source_locator=f"Classification:{source_row}",
                        source_record_id=f"Classification:{source_row}:class",
                        pairing_status="not_applicable",
                        leakage_role="primary_observation",
                        model_eligible=not overlap,
                        context_note="Source class was inverted: source 0=blocker, source 1=nonblocker; canonical 1=blocker.",
                    ).model_dump(mode="python")
                )
            ic50_protocol = protocol_records["classification_ic50"]
            relation = row["ic50_relation"]
            censoring = (
                "left"
                if relation in {"<", "<="}
                else "right"
                if relation in {">", ">="}
                else "missing"
                if relation == "not_reported"
                else "none"
            )
            measurement_records.append(
                Measurement(
                    measurement_id=_stable_id("SUNMEAS", role, source_row, "ic50"),
                    compound_id=row["compound_id"],
                    assay_protocol_id=ic50_protocol["assay_protocol_id"],
                    endpoint="herg_ic50",
                    value=_optional_float(row["ic50_nm_value"]),
                    unit="nM",
                    relation=relation,
                    censoring=censoring,
                    origin="measured",
                    submitted_value=row["submitted_value"],
                    source="Sun hERG:Classification",
                    source_locator=f"Classification:{source_row}",
                    source_record_id=f"Classification:{source_row}:ic50",
                    pairing_status="not_applicable",
                    leakage_role="primary_observation",
                    model_eligible=not overlap and row["parse_status"] != "unparsed",
                    context_note="Censoring was retained; no midpoint or cap imputation was applied.",
                ).model_dump(mode="python")
            )
        elif role == "regression":
            protocol = protocol_records["regression"]
            measurement_records.append(
                Measurement(
                    measurement_id=_stable_id("SUNMEAS", role, source_row, "pic50"),
                    compound_id=row["compound_id"],
                    assay_protocol_id=protocol["assay_protocol_id"],
                    endpoint="herg_pic50",
                    value=_optional_float(row["pic50_value"]),
                    unit="pIC50",
                    relation="=",
                    censoring="none",
                    origin="recomputed",
                    submitted_value=json.dumps(
                        {
                            "stored_log10_nM": _optional_float(row["stored_log10_nm"]),
                            "source_hERG_auxiliary": _optional_float(row["source_herg_auxiliary"]),
                        },
                        sort_keys=True,
                    ),
                    source="Sun hERG:Regression",
                    source_locator=f"Regression:{source_row}",
                    source_record_id=f"Regression:{source_row}:pic50",
                    pairing_status="not_applicable",
                    leakage_role="primary_observation",
                    model_eligible=not overlap,
                    context_note="pIC50 = 9 - stored log10(IC50 nM). The separate source hERG column is retained as auxiliary context; its semantics were not guessed.",
                ).model_dump(mode="python")
            )
        else:
            protocol = protocol_records["validation"]
            relation = row["ic50_relation"]
            measurement_records.append(
                Measurement(
                    measurement_id=_stable_id("SUNMEAS", role, source_row, "ic50"),
                    compound_id=row["compound_id"],
                    assay_protocol_id=protocol["assay_protocol_id"],
                    endpoint="herg_ic50",
                    value=_optional_float(row["ic50_nm_value"]),
                    unit="nM",
                    relation=relation,
                    censoring="none"
                    if relation == "="
                    else "missing"
                    if relation == "not_reported"
                    else "unknown",
                    origin="measured",
                    submitted_value=row["submitted_value"],
                    source="Sun hERG:Validation",
                    source_locator=f"Validation:{source_row}",
                    source_record_id=f"Validation:{source_row}:ic50",
                    pairing_status="not_applicable",
                    leakage_role="primary_observation",
                    model_eligible=not overlap,
                    context_note="External validation membership retained explicitly.",
                ).model_dump(mode="python")
            )

    tables = {
        "compounds": records_to_frame(Compound, compound_records),
        "assay_protocols": records_to_frame(AssayProtocol, protocols_by_id.values()),
        "measurements": records_to_frame(Measurement, measurement_records),
    }
    quarantine_frame = (
        pd.DataFrame(quarantine).drop_duplicates().reset_index(drop=True)
        if quarantine
        else _empty_review(["code", "dataset_role", "source_row", "structure_id", "detail"])
    )
    issues_frame = pd.DataFrame(issues)
    raw_duplicate_counts = {
        "classification": int(classification.duplicated("Smiles", keep=False).sum()),
        "regression": int(regression.duplicated("Smiles", keep=False).sum()),
        "validation": int(validation.duplicated("Smiles", keep=False).sum()),
    }
    summary = {
        "source": "Sun hERG supplementary workbook",
        "classification_rows": len(classification),
        "regression_rows": len(regression),
        "validation_rows": len(validation),
        "measurements": len(tables["measurements"]),
        "unique_standardized_structures": len(tables["compounds"]),
        "raw_duplicate_rows_by_sheet": raw_duplicate_counts,
        "source_class_zero_count": int((activity["source_class"] == 0).sum()),
        "source_class_one_count": int((activity["source_class"] == 1).sum()),
        "canonical_blocker_count": int((activity["canonical_blocker_class"] == 1).sum()),
        "canonical_nonblocker_count": int((activity["canonical_blocker_class"] == 0).sum()),
        "class_threshold_disagreements": len(disagreements),
        "train_validation_structure_overlaps": len(overlaps),
        "quarantine_records": len(quarantine_frame),
        "computed_mw_min_g_mol": float(activity["computed_mw_g_mol"].min())
        if activity["computed_mw_g_mol"].notna().any()
        else None,
        "computed_mw_max_g_mol": float(activity["computed_mw_g_mol"].max())
        if activity["computed_mw_g_mol"].notna().any()
        else None,
        "computed_mw_above_domain_limit_rows": len(domain_contradictions),
        "domain_max_mw_g_mol": domain_max_mw_g_mol,
        "blocker_threshold_nm": blocker_threshold_nm,
        "regression_transform": "pIC50 = 9 - stored_log10_nM",
        "source_class_transform": "canonical_blocker_class = 1 - source_class",
    }
    return NormalizationResult(
        tables=tables,
        issues=issues_frame,
        quarantine=quarantine_frame,
        summary=summary,
        review_tables={"activity": activity, "domain_contradictions": domain_contradictions},
    )


def _write_review_parquet(frame: pd.DataFrame, path: Path) -> Path:
    """Atomically write a typed diagnostic/review table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}-",
            suffix=".parquet",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        frame.convert_dtypes().to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return path


def _issue_table(result: NormalizationResult, source_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record_type, frame in (("issue", result.issues), ("quarantine", result.quarantine)):
        for record in frame.to_dict("records"):
            code = _clean_text(record.pop("code", "unspecified")) or "unspecified"
            severity = _clean_text(record.pop("severity", "review")) or "review"
            clean_context = {
                key: (None if bool(pd.isna(value)) else value)
                for key, value in record.items()
                if not isinstance(value, (list, tuple, dict))
            }
            rows.append(
                {
                    "source": source_name,
                    "record_type": record_type,
                    "severity": severity,
                    "code": code,
                    "context_json": json.dumps(clean_context, sort_keys=True, default=str),
                }
            )
    return pd.DataFrame(rows, columns=["source", "record_type", "severity", "code", "context_json"])


def normalize_research_data(
    internal_workbook: Path,
    public_herg_workbook: Path | None,
    output_dir: Path,
) -> dict[str, Path | int]:
    """Normalize all currently available PK/hERG evidence and write Parquet.

    The returned mapping contains every output path plus record counts.  Empty
    process tables (for example PK samples when only reported summary PK is
    available) are still emitted with their typed schema; their emptiness is
    an explicit statement of unavailable context, not an inferred dataset.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    internal = normalize_internal_workbook(Path(internal_workbook))
    public = normalize_sun_herg_workbook(Path(public_herg_workbook)) if public_herg_workbook else None

    contract_frames: dict[str, pd.DataFrame] = {
        "compounds": internal.tables["compounds"],
        "chemical_states": internal.tables["chemical_states"],
        "conformers": records_to_frame(Conformer, []),
        "assay_protocols": internal.tables["assay_protocols"],
        "measurements": internal.tables["measurements"],
        "pk_studies": internal.tables["pk_studies"],
        "pk_samples": records_to_frame(PKSample, []),
        "derived_pk_parameters": internal.tables["derived_pk_parameters"],
        "physics_runs": records_to_frame(PhysicsRun, []),
        "physics_observables": records_to_frame(PhysicsObservable, []),
        "feature_lineage": records_to_frame(FeatureLineage, []),
    }
    if public is not None:
        for name in ("compounds", "assay_protocols", "measurements"):
            contract_frames[name] = pd.concat(
                [contract_frames[name], public.tables[name]],
                ignore_index=True,
            )

    outputs: dict[str, Path | int] = {}
    for name, frame in contract_frames.items():
        destination = output_dir / f"{name}.parquet"
        outputs[name] = write_contract_parquet(name, frame, destination)
        outputs[f"{name}_rows"] = len(frame)

    issue_frames = [_issue_table(internal, "internal_workbook")]
    if public is not None:
        issue_frames.append(_issue_table(public, "public_herg_workbook"))
    issues = pd.concat(issue_frames, ignore_index=True)
    outputs["validation_issues"] = _write_review_parquet(issues, output_dir / "validation_issues.parquet")
    outputs["validation_issues_rows"] = len(issues)
    outputs["compound_aliases"] = _write_review_parquet(
        internal.review_tables["compound_aliases"],
        output_dir / "compound_aliases.parquet",
    )
    outputs["compound_aliases_rows"] = len(internal.review_tables["compound_aliases"])

    if public is not None:
        activity = public.review_tables["activity"]
        domain = public.review_tables["domain_contradictions"]
        outputs["public_herg_normalized"] = _write_review_parquet(
            activity,
            output_dir / "public_herg_normalized.parquet",
        )
        outputs["public_herg_normalized_rows"] = len(activity)
        outputs["public_herg_quarantine"] = _write_review_parquet(
            public.quarantine,
            output_dir / "public_herg_quarantine.parquet",
        )
        outputs["public_herg_quarantine_rows"] = len(public.quarantine)
        outputs["public_herg_domain_contradictions"] = _write_review_parquet(
            domain,
            output_dir / "public_herg_domain_contradictions.parquet",
        )
        outputs["public_herg_domain_contradictions_rows"] = len(domain)
    return outputs


__all__ = [
    "ConcentrationResponse",
    "NormalizationResult",
    "ParsedQualifiedValue",
    "normalize_internal_workbook",
    "normalize_research_data",
    "normalize_relation",
    "normalize_sun_herg_workbook",
    "normalize_unit",
    "parse_concentration_response_panel",
    "parse_qualified_value",
]
