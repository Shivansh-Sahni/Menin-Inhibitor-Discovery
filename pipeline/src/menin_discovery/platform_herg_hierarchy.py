"""Build a deterministic, assay-native hERG evidence hierarchy.

The builder deliberately does not create a universal hERG potency.  PubChem
activity outcomes, a reported quantitative pIC50 compilation, optional HERGAI
provenance, and optional ChEMBL hERG events remain distinct observations.
Only exact, positive, functional ChEMBL IC50 values in nM or uM are converted
to pIC50.  Binary and fixed-concentration observations never are.

All source paths are explicit.  EPA and the existing strict canonical hERG
views are intentionally unsupported inputs so that they cannot be counted a
second time.  T1 is always emitted as a *candidate* annotation, never as an
admission decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

SCHEMA_VERSION = "platform-herg-hierarchy/1.0"
STANDARDIZATION_VERSION = "rdkit-cleanup-fragment-parent-uncharge-isomeric-v1"
PUBCHEM_AID = 720551
KCNH2_GENE_ID = "3757"
DEFAULT_BLOCKER_MAX_UM = 10.0
DEFAULT_NONBLOCKER_MIN_UM = 30.0

_OBSERVATION_SCHEMA = pa.schema(
    [
        pa.field("observation_id", pa.large_string(), nullable=False),
        pa.field("source_family", pa.large_string(), nullable=False),
        pa.field("source_priority", pa.large_string(), nullable=False),
        pa.field("source_record_id", pa.large_string(), nullable=False),
        pa.field("source_row_number", pa.int64(), nullable=False),
        pa.field("source_aid", pa.int64()),
        pa.field("source_sid", pa.large_string()),
        pa.field("source_cid", pa.large_string()),
        pa.field("raw_smiles", pa.large_string()),
        pa.field("standardized_smiles", pa.large_string()),
        pa.field("standard_inchi_key", pa.large_string()),
        pa.field("structure_id", pa.large_string()),
        pa.field("structure_valid", pa.bool_(), nullable=False),
        pa.field("target_id", pa.large_string(), nullable=False),
        pa.field("target_variant", pa.large_string(), nullable=False),
        pa.field("assay_id", pa.large_string()),
        pa.field("assay_family", pa.large_string(), nullable=False),
        pa.field("native_endpoint", pa.large_string(), nullable=False),
        pa.field("native_relation", pa.large_string()),
        pa.field("native_value", pa.float64()),
        pa.field("native_unit", pa.large_string()),
        pa.field("native_label", pa.large_string()),
        pa.field("pic50_value", pa.float64()),
        pa.field("pic50_origin", pa.large_string()),
        pa.field("derived_binary_label", pa.int8()),
        pa.field("derived_label_policy", pa.large_string()),
        pa.field("source_split", pa.large_string()),
        pa.field("native_aux_json", pa.large_string(), nullable=False),
        pa.field("reported_evidence_tier", pa.large_string(), nullable=False),
        pa.field("t1_candidate", pa.bool_(), nullable=False),
        pa.field("t1_candidate_reason", pa.large_string(), nullable=False),
        pa.field("quality_flags", pa.large_string(), nullable=False),
    ]
)

_BINARY_SCHEMA = pa.schema(
    [
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("standardized_smiles", pa.large_string(), nullable=False),
        pa.field("standard_inchi_key", pa.large_string(), nullable=False),
        pa.field("herg_blocker_label", pa.int8(), nullable=False),
    ]
)

_QUANTITATIVE_SCHEMA = pa.schema(
    [
        pa.field("observation_id", pa.large_string(), nullable=False),
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("standardized_smiles", pa.large_string(), nullable=False),
        pa.field("standard_inchi_key", pa.large_string(), nullable=False),
        pa.field("pic50_value", pa.float64(), nullable=False),
        pa.field("pic50_origin", pa.large_string(), nullable=False),
        pa.field("source_family", pa.large_string(), nullable=False),
        pa.field("source_record_id", pa.large_string(), nullable=False),
        pa.field("source_split", pa.large_string()),
        pa.field("derived_binary_label", pa.int8()),
        pa.field("reported_evidence_tier", pa.large_string(), nullable=False),
        pa.field("t1_candidate", pa.bool_(), nullable=False),
    ]
)

_HIERARCHY_SCHEMA = pa.schema(
    [
        pa.field("structure_id", pa.large_string(), nullable=False),
        pa.field("standardized_smiles", pa.large_string(), nullable=False),
        pa.field("standard_inchi_key", pa.large_string(), nullable=False),
        pa.field("reported_observation_count", pa.int64(), nullable=False),
        pa.field("source_families_json", pa.large_string(), nullable=False),
        pa.field("t1_candidate_observation_count", pa.int64(), nullable=False),
        pa.field("primary_decisive_labels_json", pa.large_string(), nullable=False),
        pa.field("secondary_decisive_labels_json", pa.large_string(), nullable=False),
        pa.field("primary_conflict", pa.bool_(), nullable=False),
        pa.field("secondary_conflict", pa.bool_(), nullable=False),
        pa.field("consensus_binary_label", pa.int8()),
        pa.field("consensus_status", pa.large_string(), nullable=False),
        pa.field("reported_evidence_tier", pa.large_string(), nullable=False),
        pa.field("highest_candidate_tier", pa.large_string(), nullable=False),
        pa.field("t1_candidate", pa.bool_(), nullable=False),
        pa.field("t1_candidate_reason", pa.large_string(), nullable=False),
    ]
)


class HergHierarchyError(RuntimeError):
    """Raised when inputs or generated hierarchy artifacts fail closed."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _canonical_json(value: Any) -> str:
    return _canonical_json_bytes(value).decode("utf-8")


def _stable_id(prefix: str, *parts: object) -> str:
    body = "\x1f".join("" if part is None else str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(body.encode('utf-8')).hexdigest()[:24].upper()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _clean(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float(value: object, *, context: str, required: bool = False) -> float | None:
    text = _clean(value)
    if text is None:
        if required:
            raise HergHierarchyError(f"Missing numeric value for {context}")
        return None
    try:
        number = float(text)
    except ValueError as error:
        raise HergHierarchyError(f"Invalid numeric value for {context}: {text!r}") from error
    if not math.isfinite(number):
        raise HergHierarchyError(f"Non-finite numeric value for {context}: {text!r}")
    return number


def _resolve_header(fieldnames: Sequence[str] | None, aliases: Sequence[str], *, context: str) -> str:
    if not fieldnames or len(set(fieldnames)) != len(fieldnames):
        raise HergHierarchyError(f"Missing or duplicate CSV headers in {context}")
    normalized = {name.strip().casefold(): name for name in fieldnames}
    for alias in aliases:
        if alias.casefold() in normalized:
            return normalized[alias.casefold()]
    raise HergHierarchyError(f"Missing required column {aliases[0]!r} in {context}")


def _optional_header(fieldnames: Sequence[str] | None, aliases: Sequence[str]) -> str | None:
    normalized = {name.strip().casefold(): name for name in (fieldnames or [])}
    return next((normalized[item.casefold()] for item in aliases if item.casefold() in normalized), None)


def _checked_input(path: Path, *, role: str, suffixes: frozenset[str]) -> Path:
    if path.is_symlink() or not path.is_file():
        raise HergHierarchyError(f"Missing or symlinked {role} input: {path}")
    if path.suffix.casefold() not in suffixes:
        raise HergHierarchyError(f"Unexpected {role} input extension: {path}")
    return path.resolve()


def _standardize_smiles(raw_smiles: object) -> tuple[str | None, str | None, str | None, str]:
    text = _clean(raw_smiles)
    if text is None:
        return None, None, None, "missing_smiles"
    try:
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(text)
            if molecule is None:
                return None, None, None, "invalid_smiles"
            molecule = rdMolStandardize.Cleanup(molecule)
            molecule = rdMolStandardize.FragmentParent(molecule)
            molecule = rdMolStandardize.Uncharger().uncharge(molecule)
            Chem.SanitizeMol(molecule)
            smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
            key = Chem.MolToInchiKey(molecule)
    except Exception:  # RDKit exposes several C++ exception types.
        return None, None, None, "standardization_failed"
    if not smiles or not key:
        return None, None, None, "standardization_failed"
    return smiles, key, _stable_id("HSTR", key), "standardized"


def _threshold_label(pic50: float | None, *, blocker_max_um: float, nonblocker_min_um: float) -> int | None:
    if pic50 is None:
        return None
    blocker_boundary = 6.0 - math.log10(blocker_max_um)
    nonblocker_boundary = 6.0 - math.log10(nonblocker_min_um)
    if pic50 >= blocker_boundary:
        return 1
    if pic50 <= nonblocker_boundary:
        return 0
    return None


def _record(
    *,
    source_family: str,
    source_priority: str,
    source_record_id: str,
    source_row_number: int,
    structure: tuple[str | None, str | None, str | None, str],
    raw_smiles: str | None,
    native_endpoint: str,
    native_relation: str | None,
    native_value: float | None,
    native_unit: str | None,
    native_label: str | None,
    pic50_value: float | None,
    pic50_origin: str | None,
    derived_binary_label: int | None,
    derived_label_policy: str | None,
    source_split: str | None,
    target_variant: str,
    assay_id: str | None,
    assay_family: str,
    aux: Mapping[str, Any],
    t1_candidate: bool,
    t1_candidate_reason: str,
    quality_flags: Iterable[str] = (),
    source_aid: int | None = None,
    source_sid: str | None = None,
    source_cid: str | None = None,
) -> dict[str, Any]:
    standardized_smiles, key, structure_id, structure_status = structure
    flags = sorted(
        {flag for flag in quality_flags if flag}
        | ({structure_status} if structure_status != "standardized" else set())
    )
    observation_id = _stable_id(
        "HOBS",
        source_family,
        source_record_id,
        source_row_number,
        native_endpoint,
        native_relation,
        native_value,
        native_label,
    )
    return {
        "observation_id": observation_id,
        "source_family": source_family,
        "source_priority": source_priority,
        "source_record_id": source_record_id,
        "source_row_number": source_row_number,
        "source_aid": source_aid,
        "source_sid": source_sid,
        "source_cid": source_cid,
        "raw_smiles": raw_smiles,
        "standardized_smiles": standardized_smiles,
        "standard_inchi_key": key,
        "structure_id": structure_id,
        "structure_valid": structure_id is not None,
        "target_id": "KCNH2",
        "target_variant": target_variant,
        "assay_id": assay_id,
        "assay_family": assay_family,
        "native_endpoint": native_endpoint,
        "native_relation": native_relation,
        "native_value": native_value,
        "native_unit": native_unit,
        "native_label": native_label,
        "pic50_value": pic50_value,
        "pic50_origin": pic50_origin,
        "derived_binary_label": derived_binary_label,
        "derived_label_policy": derived_label_policy,
        "source_split": source_split,
        "native_aux_json": _canonical_json(dict(aux)),
        "reported_evidence_tier": "T0_reported",
        "t1_candidate": bool(t1_candidate),
        "t1_candidate_reason": t1_candidate_reason,
        "quality_flags": ";".join(flags),
    }


def _read_pubchem_structures(
    path: Path,
) -> tuple[dict[str, tuple[str, tuple[str | None, str | None, str | None, str]]], int]:
    result: dict[str, tuple[str, tuple[str | None, str | None, str | None, str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        cid_column = _resolve_header(reader.fieldnames, ("CID", "PubChem CID"), context=str(path))
        smiles_column = _resolve_header(
            reader.fieldnames,
            ("ConnectivitySMILES", "CanonicalSMILES", "IsomericSMILES", "SMILES", "Canonical SMILES"),
            context=str(path),
        )
        rows = 0
        cache: dict[str, tuple[str | None, str | None, str | None, str]] = {}
        for rows, row in enumerate(reader, start=1):
            cid = _clean(row.get(cid_column))
            smiles = _clean(row.get(smiles_column))
            if cid is None or smiles is None:
                raise HergHierarchyError(f"Blank CID/SMILES in {path} row {rows + 1}")
            structure = cache.setdefault(smiles, _standardize_smiles(smiles))
            if structure[2] is None:
                raise HergHierarchyError(f"Invalid PubChem structure in {path} row {rows + 1}")
            prior = result.get(cid)
            if prior is not None and prior[1][2] != structure[2]:
                raise HergHierarchyError(f"CID {cid} maps to conflicting standardized structures")
            result[cid] = (smiles, structure)
    if not result:
        raise HergHierarchyError(f"No PubChem structures in {path}")
    return result, rows


def _read_pubchem_outcomes(
    path: Path,
    structures: Mapping[str, tuple[str, tuple[str | None, str | None, str | None, str]]],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        aid_column = _resolve_header(reader.fieldnames, ("AID",), context=str(path))
        sid_column = _resolve_header(reader.fieldnames, ("SID",), context=str(path))
        cid_column = _resolve_header(reader.fieldnames, ("CID",), context=str(path))
        outcome_column = _resolve_header(reader.fieldnames, ("Activity Outcome",), context=str(path))
        gene_column = _optional_header(reader.fieldnames, ("Target GeneID", "GeneID"))
        assay_name_column = _optional_header(reader.fieldnames, ("Assay Name",))
        assay_type_column = _optional_header(reader.fieldnames, ("Assay Type",))
        for row_number, row in enumerate(reader, start=1):
            aid = _clean(row.get(aid_column))
            if aid != str(PUBCHEM_AID):
                raise HergHierarchyError(f"Non-{PUBCHEM_AID} row in {path} row {row_number + 1}: {aid}")
            if gene_column and _clean(row.get(gene_column)) not in {None, KCNH2_GENE_ID}:
                raise HergHierarchyError(f"Non-KCNH2 row in {path} row {row_number + 1}")
            sid = _clean(row.get(sid_column))
            cid = _clean(row.get(cid_column))
            outcome = _clean(row.get(outcome_column))
            if sid is None or cid is None or outcome is None:
                raise HergHierarchyError(f"Blank PubChem identity/outcome in {path} row {row_number + 1}")
            if cid not in structures:
                raise HergHierarchyError(f"PubChem outcome CID {cid} has no supplied structure")
            normalized_outcome = outcome.casefold()
            if normalized_outcome not in {"active", "inactive", "inconclusive", "unspecified", "probe"}:
                raise HergHierarchyError(f"Unsupported PubChem outcome {outcome!r}")
            label = (
                1
                if normalized_outcome in {"active", "probe"}
                else 0
                if normalized_outcome == "inactive"
                else None
            )
            raw_smiles, structure = structures[cid]
            aux = {
                "activity_outcome": outcome,
                "assay_name": _clean(row.get(assay_name_column)) if assay_name_column else None,
                "assay_type": _clean(row.get(assay_type_column)) if assay_type_column else None,
            }
            observations.append(
                _record(
                    source_family="pubchem_aid720551",
                    source_priority="source_grade_primary",
                    source_record_id=f"SID:{sid}",
                    source_row_number=row_number,
                    source_aid=PUBCHEM_AID,
                    source_sid=sid,
                    source_cid=cid,
                    structure=structure,
                    raw_smiles=raw_smiles,
                    native_endpoint="activity_outcome",
                    native_relation=None,
                    native_value=None,
                    native_unit=None,
                    native_label=outcome,
                    pic50_value=None,
                    pic50_origin=None,
                    derived_binary_label=label,
                    derived_label_policy="source_reported_activity_outcome" if label is not None else None,
                    source_split=None,
                    target_variant="wild_type",
                    assay_id="PUBCHEM_AID_720551",
                    assay_family="source_reported_qhts",
                    aux=aux,
                    t1_candidate=label is not None,
                    t1_candidate_reason=(
                        "source_grade_wild_type_decisive_outcome"
                        if label is not None
                        else "nondecisive_source_outcome"
                    ),
                )
            )
    if not observations:
        raise HergHierarchyError(f"No PubChem observations in {path}")
    return observations


def _read_quantitative(
    paths: Sequence[Path], *, blocker_max_um: float, nonblocker_min_um: float
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    cache: dict[str, tuple[str | None, str | None, str | None, str]] = {}
    for path_index, path in enumerate(paths):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            smiles_column = _resolve_header(reader.fieldnames, ("SMILES",), context=str(path))
            pic50_column = _resolve_header(reader.fieldnames, ("pIC50",), context=str(path))
            source_column = _resolve_header(reader.fieldnames, ("Source",), context=str(path))
            key_column = _optional_header(reader.fieldnames, ("InChI Key", "InChl Key", "InChIKey"))
            split_column = _optional_header(reader.fieldnames, ("USED_AS", "split", "partition"))
            chembl_column = _optional_header(reader.fieldnames, ("ChEMBL ID",))
            cid_column = _optional_header(reader.fieldnames, ("PubChem CID", "CID"))
            for row_number, row in enumerate(reader, start=1):
                smiles = _clean(row.get(smiles_column))
                source = _clean(row.get(source_column))
                pic50 = _float(
                    row.get(pic50_column), context=f"{path} pIC50 row {row_number + 1}", required=True
                )
                if smiles is None or source is None or pic50 is None or not (0.0 < pic50 <= 14.0):
                    raise HergHierarchyError(f"Invalid quantitative row in {path} row {row_number + 1}")
                structure = cache.setdefault(smiles, _standardize_smiles(smiles))
                provided_key = _clean(row.get(key_column)) if key_column else None
                flags: list[str] = []
                if provided_key and structure[1] and provided_key != structure[1]:
                    flags.append("reported_inchi_key_differs_after_standardization")
                label = _threshold_label(
                    pic50, blocker_max_um=blocker_max_um, nonblocker_min_um=nonblocker_min_um
                )
                source_record_id = (
                    (_clean(row.get(chembl_column)) if chembl_column else None)
                    or (_clean(row.get(cid_column)) if cid_column else None)
                    or f"{path_index + 1}:{row_number}"
                )
                observations.append(
                    _record(
                        source_family="quantitative_pic50_release",
                        source_priority="secondary_quantitative_compilation",
                        source_record_id=source_record_id,
                        source_row_number=row_number,
                        source_cid=_clean(row.get(cid_column)) if cid_column else None,
                        structure=structure,
                        raw_smiles=smiles,
                        native_endpoint="pIC50",
                        native_relation="=",
                        native_value=pic50,
                        native_unit="pIC50",
                        native_label=None,
                        pic50_value=pic50,
                        pic50_origin="source_reported_pIC50",
                        derived_binary_label=label,
                        derived_label_policy=(
                            "pIC50_from_reported_exact_value__10uM_blocker__30uM_nonblocker_gap"
                            if label is not None
                            else None
                        ),
                        source_split=_clean(row.get(split_column)) if split_column else None,
                        target_variant="wild_type_or_unspecified",
                        assay_id=None,
                        assay_family="mixed_unresolved_compilation",
                        aux={"reported_source": source, "provided_inchi_key": provided_key},
                        t1_candidate=False,
                        t1_candidate_reason="assay_level_provenance_not_supplied",
                        quality_flags=flags,
                    )
                )
    if not observations:
        raise HergHierarchyError("At least one quantitative pIC50 observation is required")
    return observations


def _read_hergai(paths: Sequence[Path]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    cache: dict[str, tuple[str | None, str | None, str | None, str]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            sid_column = _resolve_header(reader.fieldnames, ("SID",), context=str(path))
            smiles_column = _resolve_header(reader.fieldnames, ("smiles", "SMILES"), context=str(path))
            activity_column = _resolve_header(reader.fieldnames, ("activity",), context=str(path))
            partition_column = _optional_header(reader.fieldnames, ("partition", "split"))
            potency_column = _optional_header(reader.fieldnames, ("potency",))
            for row_number, row in enumerate(reader, start=1):
                sid = _clean(row.get(sid_column))
                smiles = _clean(row.get(smiles_column))
                activity = _clean(row.get(activity_column))
                if sid is None or smiles is None or activity is None:
                    raise HergHierarchyError(f"Blank HERGAI identity in {path} row {row_number + 1}")
                normalized = activity.casefold()
                label = 1 if normalized == "active" else 0 if normalized == "inactive" else None
                structure = cache.setdefault(smiles, _standardize_smiles(smiles))
                observations.append(
                    _record(
                        source_family="hergai_secondary",
                        source_priority="secondary_provenance_only",
                        source_record_id=f"SID:{sid}",
                        source_row_number=row_number,
                        source_sid=sid,
                        structure=structure,
                        raw_smiles=smiles,
                        native_endpoint="secondary_binary_activity",
                        native_relation=None,
                        native_value=None,
                        native_unit=None,
                        native_label=activity,
                        pic50_value=None,
                        pic50_origin=None,
                        derived_binary_label=label,
                        derived_label_policy="secondary_source_activity_only" if label is not None else None,
                        source_split=_clean(row.get(partition_column)) if partition_column else None,
                        target_variant="wild_type_or_unspecified",
                        assay_id=None,
                        assay_family="unresolved_secondary_compilation",
                        aux={
                            "reported_potency_uninterpreted": _clean(row.get(potency_column))
                            if potency_column
                            else None
                        },
                        t1_candidate=False,
                        t1_candidate_reason="secondary_inherited_provenance_not_admissible",
                    )
                )
    return observations


def _read_chembl(
    paths: Sequence[Path], *, blocker_max_um: float, nonblocker_min_um: float
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    cache: dict[str, tuple[str | None, str | None, str | None, str]] = {}
    required = {
        "activity_id",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "assay_type",
        "canonical_smiles",
        "target_chembl_id",
    }
    row_offset = 0
    for path in paths:
        table = pq.read_table(path)
        missing = required - set(table.column_names)
        if missing:
            raise HergHierarchyError(f"Missing ChEMBL columns in {path}: {sorted(missing)}")
        for local_index, row in enumerate(table.to_pylist(), start=1):
            row_number = row_offset + local_index
            if _clean(row.get("target_chembl_id")) != "CHEMBL240":
                raise HergHierarchyError(f"Non-CHEMBL240 row in {path} row {local_index}")
            smiles = _clean(row.get("canonical_smiles"))
            structure = cache.setdefault(smiles or "", _standardize_smiles(smiles))
            endpoint = _clean(row.get("standard_type")) or _clean(row.get("type")) or "unspecified"
            relation = _clean(row.get("standard_relation")) or _clean(row.get("relation"))
            unit = _clean(row.get("standard_units"))
            value = _float(row.get("standard_value"), context=f"{path} row {local_index}")
            assay_type = (_clean(row.get("assay_type")) or "U").upper()
            variant = "mutant_or_variant" if row.get("variant_id") is not None else "wild_type_or_unspecified"
            normalized_unit = (unit or "").replace("µ", "u").replace("μ", "u").casefold()
            convertible = (
                endpoint.casefold() == "ic50"
                and relation == "="
                and value is not None
                and value > 0
                and normalized_unit in {"nm", "um"}
                and assay_type == "F"
                and structure[2] is not None
                and variant != "mutant_or_variant"
            )
            pic50 = None
            if convertible and value is not None:
                pic50 = (9.0 if normalized_unit == "nm" else 6.0) - math.log10(value)
            label = _threshold_label(
                pic50, blocker_max_um=blocker_max_um, nonblocker_min_um=nonblocker_min_um
            )
            standard_flag = row.get("standard_flag")
            potential_duplicate = row.get("potential_duplicate")
            validity = _clean(row.get("data_validity_comment"))
            confidence = row.get("confidence_score")
            t1_candidate = bool(
                convertible
                and label is not None
                and standard_flag in {1, "1"}
                and potential_duplicate in {0, None, "0"}
                and validity is None
                and (confidence is None or int(confidence) >= 8)
            )
            if not convertible:
                reason = "not_exact_positive_wildtype_functional_ic50_in_nm_or_um"
            elif label is None:
                reason = "exact_pic50_in_prespecified_10_to_30uM_gap"
            elif not t1_candidate:
                reason = "chembl_quality_gate_failed"
            else:
                reason = "curated_exact_functional_ic50_quality_gate_passed"
            observations.append(
                _record(
                    source_family="chembl_herg_specialized_view",
                    source_priority="source_grade_curated",
                    source_record_id=f"ACTIVITY:{row.get('activity_id')}",
                    source_row_number=row_number,
                    structure=structure,
                    raw_smiles=smiles,
                    native_endpoint=endpoint,
                    native_relation=relation,
                    native_value=value,
                    native_unit=unit,
                    native_label=_clean(row.get("standard_text_value")) or _clean(row.get("text_value")),
                    pic50_value=pic50,
                    pic50_origin="converted_from_exact_functional_IC50" if pic50 is not None else None,
                    derived_binary_label=label,
                    derived_label_policy=(
                        "exact_functional_IC50__10uM_blocker__30uM_nonblocker_gap"
                        if label is not None
                        else None
                    ),
                    source_split=None,
                    target_variant=variant,
                    assay_id=_clean(row.get("assay_chembl_id")),
                    assay_family="functional"
                    if assay_type == "F"
                    else "binding"
                    if assay_type == "B"
                    else "other",
                    aux={
                        "assay_description": _clean(row.get("assay_description")),
                        "assay_type": assay_type,
                        "data_validity_comment": validity,
                        "potential_duplicate": potential_duplicate,
                        "standard_flag": standard_flag,
                        "confidence_score": confidence,
                        "document_chembl_id": _clean(row.get("document_chembl_id")),
                    },
                    t1_candidate=t1_candidate,
                    t1_candidate_reason=reason,
                )
            )
        row_offset += table.num_rows
    return observations


def _sort_observations(observations: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        observations,
        key=lambda row: (
            row["source_family"],
            row["source_record_id"],
            row["source_row_number"],
            row["observation_id"],
        ),
    )


def _views(
    observations: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_structure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    quantitative: list[dict[str, Any]] = []
    for row in observations:
        if row["structure_valid"]:
            by_structure[str(row["structure_id"])].append(row)
        if row["pic50_value"] is not None and row["structure_valid"]:
            quantitative.append(
                {
                    "observation_id": row["observation_id"],
                    "structure_id": row["structure_id"],
                    "standardized_smiles": row["standardized_smiles"],
                    "standard_inchi_key": row["standard_inchi_key"],
                    "pic50_value": row["pic50_value"],
                    "pic50_origin": row["pic50_origin"],
                    "source_family": row["source_family"],
                    "source_record_id": row["source_record_id"],
                    "source_split": row["source_split"],
                    "derived_binary_label": row["derived_binary_label"],
                    "reported_evidence_tier": "T0_reported",
                    "t1_candidate": row["t1_candidate"],
                }
            )

    binary: list[dict[str, Any]] = []
    hierarchy: list[dict[str, Any]] = []
    for structure_id, rows in sorted(by_structure.items()):
        representative = min(rows, key=lambda row: (row["standardized_smiles"], row["observation_id"]))
        # The barebones high-volume binary backbone is intentionally homogeneous:
        # only the source-reported AID 720551 wild-type qHTS calls may determine it.
        # ChEMBL IC50, quantitative compilation, and HERGAI evidence are separate
        # annotations and may flag disagreement, but can neither create nor resolve
        # the PubChem consensus label.
        primary_labels = sorted(
            {
                int(row["derived_binary_label"])
                for row in rows
                if row["source_family"] == "pubchem_aid720551"
                and row["t1_candidate"]
                and row["derived_binary_label"] is not None
            }
        )
        secondary_labels = sorted(
            {
                int(row["derived_binary_label"])
                for row in rows
                if row["source_family"] != "pubchem_aid720551" and row["derived_binary_label"] is not None
            }
        )
        primary_conflict = len(primary_labels) > 1
        consensus = primary_labels[0] if len(primary_labels) == 1 else None
        secondary_conflict = consensus is not None and any(label != consensus for label in secondary_labels)
        if primary_conflict:
            status = "primary_conflict_excluded"
            reason = "conflicting_AID720551_source_reported_outcomes"
        elif consensus is None:
            status = "no_decisive_T1_candidate_evidence"
            reason = "no_decisive_AID720551_backbone_outcome"
        else:
            status = "one_label_candidate"
            reason = (
                "consistent_primary_evidence_secondary_conflict_flagged"
                if secondary_conflict
                else "consistent_AID720551_source_reported_outcomes"
            )
            binary.append(
                {
                    "structure_id": structure_id,
                    "standardized_smiles": representative["standardized_smiles"],
                    "standard_inchi_key": representative["standard_inchi_key"],
                    "herg_blocker_label": consensus,
                }
            )
        hierarchy.append(
            {
                "structure_id": structure_id,
                "standardized_smiles": representative["standardized_smiles"],
                "standard_inchi_key": representative["standard_inchi_key"],
                "reported_observation_count": len(rows),
                "source_families_json": _canonical_json(sorted({row["source_family"] for row in rows})),
                "t1_candidate_observation_count": sum(bool(row["t1_candidate"]) for row in rows),
                "primary_decisive_labels_json": _canonical_json(primary_labels),
                "secondary_decisive_labels_json": _canonical_json(secondary_labels),
                "primary_conflict": primary_conflict,
                "secondary_conflict": secondary_conflict,
                "consensus_binary_label": consensus,
                "consensus_status": status,
                "reported_evidence_tier": "T0_reported",
                "highest_candidate_tier": "T1_candidate" if consensus is not None else "T0_reported",
                "t1_candidate": consensus is not None,
                "t1_candidate_reason": reason,
            }
        )
    binary.sort(key=lambda row: row["structure_id"])
    quantitative.sort(key=lambda row: (row["structure_id"], row["source_family"], row["observation_id"]))
    return binary, quantitative, hierarchy


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]], schema: pa.Schema) -> dict[str, Any]:
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
    return {
        "path": path.name,
        "rows": table.num_rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "arrow_schema_sha256": _schema_sha256(schema),
    }


def _manifest_with_digest(body: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["manifest_sha256"] = hashlib.sha256(_canonical_json_bytes(result)).hexdigest()
    return result


def _input_entry(role: str, path: Path) -> dict[str, Any]:
    return {"role": role, "path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def build_herg_hierarchy(
    *,
    pubchem_outcomes_path: str | os.PathLike[str],
    pubchem_structures_path: str | os.PathLike[str],
    quantitative_pic50_paths: Sequence[str | os.PathLike[str]],
    output_root: str | os.PathLike[str],
    hergai_paths: Sequence[str | os.PathLike[str]] = (),
    chembl_parquet_paths: Sequence[str | os.PathLike[str]] = (),
    blocker_max_um: float = DEFAULT_BLOCKER_MAX_UM,
    nonblocker_min_um: float = DEFAULT_NONBLOCKER_MIN_UM,
) -> dict[str, Any]:
    """Build and verify a hierarchy from explicit, non-discovered input paths."""

    if blocker_max_um <= 0 or nonblocker_min_um <= blocker_max_um:
        raise HergHierarchyError("Thresholds require 0 < blocker_max_um < nonblocker_min_um")
    if not quantitative_pic50_paths:
        raise HergHierarchyError("At least one explicit quantitative pIC50 CSV path is required")
    pubchem_outcomes = _checked_input(
        Path(pubchem_outcomes_path), role="PubChem outcomes", suffixes=frozenset({".csv"})
    )
    pubchem_structures = _checked_input(
        Path(pubchem_structures_path), role="PubChem structures", suffixes=frozenset({".csv"})
    )
    quantitative_paths = [
        _checked_input(Path(path), role="quantitative pIC50", suffixes=frozenset({".csv"}))
        for path in quantitative_pic50_paths
    ]
    hergai_checked = [
        _checked_input(Path(path), role="HERGAI", suffixes=frozenset({".csv"})) for path in hergai_paths
    ]
    chembl_checked = [
        _checked_input(Path(path), role="ChEMBL hERG", suffixes=frozenset({".parquet"}))
        for path in chembl_parquet_paths
    ]
    all_inputs = [pubchem_outcomes, pubchem_structures, *quantitative_paths, *hergai_checked, *chembl_checked]
    if len(set(all_inputs)) != len(all_inputs):
        raise HergHierarchyError("One physical file cannot serve multiple hierarchy input roles")

    output = Path(output_root)
    if output.exists():
        raise HergHierarchyError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        structures, structure_source_rows = _read_pubchem_structures(pubchem_structures)
        observations = _read_pubchem_outcomes(pubchem_outcomes, structures)
        observations.extend(
            _read_quantitative(
                quantitative_paths,
                blocker_max_um=blocker_max_um,
                nonblocker_min_um=nonblocker_min_um,
            )
        )
        observations.extend(_read_hergai(hergai_checked))
        observations.extend(
            _read_chembl(
                chembl_checked,
                blocker_max_um=blocker_max_um,
                nonblocker_min_um=nonblocker_min_um,
            )
        )
        observations = _sort_observations(observations)
        binary, quantitative, hierarchy = _views(observations)

        artifacts = [
            _write_parquet(staging / "observation_ledger.parquet", observations, _OBSERVATION_SCHEMA),
            _write_parquet(staging / "structure_consensus_binary.parquet", binary, _BINARY_SCHEMA),
            _write_parquet(staging / "quantitative_pic50.parquet", quantitative, _QUANTITATIVE_SCHEMA),
            _write_parquet(staging / "hierarchy_annotations.parquet", hierarchy, _HIERARCHY_SCHEMA),
        ]
        inputs = [
            _input_entry("pubchem_aid720551_outcomes", pubchem_outcomes),
            _input_entry("pubchem_cid_structures", pubchem_structures),
            *[_input_entry("quantitative_pic50", path) for path in quantitative_paths],
            *[_input_entry("hergai_secondary_provenance", path) for path in hergai_checked],
            *[_input_entry("chembl_herg_specialized_view", path) for path in chembl_checked],
        ]
        manifest = _manifest_with_digest(
            {
                "schema_version": SCHEMA_VERSION,
                "standardization_version": STANDARDIZATION_VERSION,
                "rdkit_version": rdBase.rdkitVersion,
                "dataset_id": "herg_assay_native_hierarchy",
                "inputs": inputs,
                "input_set_sha256": hashlib.sha256(_canonical_json_bytes(inputs)).hexdigest(),
                "threshold_policy": {
                    "blocker_max_um": blocker_max_um,
                    "nonblocker_min_um": nonblocker_min_um,
                    "blocker_min_pic50": 6.0 - math.log10(blocker_max_um),
                    "nonblocker_max_pic50": 6.0 - math.log10(nonblocker_min_um),
                    "gap_policy": "unlabeled",
                },
                "scientific_contract": {
                    "assay_native_observations_preserved": True,
                    "binary_to_pic50_conversion_performed": False,
                    "fixed_dose_to_pic50_conversion_performed": False,
                    "binary_consensus_source": "PubChem AID 720551 source-reported outcomes only",
                    "cross_assay_binary_pooling_performed": False,
                    "chembl_pic50_conversion": "exact positive wild-type functional IC50 in nM/uM only",
                    "ki_binding_inhibition_pooled_into_pic50": False,
                    "hergai_role": "optional secondary provenance only",
                    "hierarchy_status": "T0 reported with conservative T1 candidate annotations; no admission",
                    "explicitly_excluded_inputs": ["EPA hERG", "strict canonical hERG task views"],
                    "clinical_claim": "none; hERG assay evidence is not QT/QTc or torsades risk",
                },
                "counts": {
                    "pubchem_structure_source_rows": structure_source_rows,
                    "observations": len(observations),
                    "valid_structure_observations": sum(bool(row["structure_valid"]) for row in observations),
                    "T0_reported_observations": len(observations),
                    "T1_candidate_observations": sum(bool(row["t1_candidate"]) for row in observations),
                    "binary_consensus_structures": len(binary),
                    "quantitative_pic50_observations": len(quantitative),
                    "hierarchy_structures": len(hierarchy),
                    "primary_conflict_structures": sum(bool(row["primary_conflict"]) for row in hierarchy),
                },
                "artifacts": artifacts,
                "artifact_set_sha256": hashlib.sha256(_canonical_json_bytes(artifacts)).hexdigest(),
            }
        )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        validate_herg_hierarchy(staging)
        os.replace(staging, output)
        return validate_herg_hierarchy(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_herg_hierarchy(output_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Fail closed on manifest, hash, schema, count, and scientific-contract drift."""

    root = Path(output_root)
    manifest_path = root / "manifest.json"
    if root.is_symlink() or not root.is_dir() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise HergHierarchyError(f"Missing or unsafe hierarchy output: {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HergHierarchyError(f"Unreadable hierarchy manifest: {manifest_path}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise HergHierarchyError("Unexpected hierarchy manifest schema")
    expected_manifest_digest = manifest.get("manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if hashlib.sha256(_canonical_json_bytes(body)).hexdigest() != expected_manifest_digest:
        raise HergHierarchyError("Hierarchy manifest digest mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 4:
        raise HergHierarchyError("Hierarchy manifest must describe exactly four Parquet artifacts")
    schemas = {
        "observation_ledger.parquet": _OBSERVATION_SCHEMA,
        "structure_consensus_binary.parquet": _BINARY_SCHEMA,
        "quantitative_pic50.parquet": _QUANTITATIVE_SCHEMA,
        "hierarchy_annotations.parquet": _HIERARCHY_SCHEMA,
    }
    if {item.get("path") for item in artifacts if isinstance(item, dict)} != set(schemas):
        raise HergHierarchyError("Hierarchy artifact membership mismatch")
    expected_members = {"manifest.json", *schemas}
    actual_members = {path.name for path in root.iterdir()}
    if actual_members != expected_members or any(
        path.is_symlink() or not path.is_file() for path in root.iterdir()
    ):
        raise HergHierarchyError("Hierarchy output contains unexpected or unsafe members")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise HergHierarchyError("Hierarchy manifest has no bound inputs")
    for item in inputs:
        if not isinstance(item, dict):
            raise HergHierarchyError("Hierarchy input binding is malformed")
        path = Path(str(item.get("path", "")))
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(item.get("bytes", -1))
            or _sha256_file(path) != item.get("sha256")
        ):
            raise HergHierarchyError(f"Hierarchy input binding mismatch: {path}")
    for item in artifacts:
        path = root / str(item["path"])
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != item.get("sha256"):
            raise HergHierarchyError(f"Hierarchy artifact hash mismatch: {path}")
        parquet = pq.ParquetFile(path)
        if parquet.schema_arrow != schemas[path.name]:
            raise HergHierarchyError(f"Hierarchy artifact schema mismatch: {path}")
        if parquet.metadata is None or parquet.metadata.num_rows != int(item.get("rows", -1)):
            raise HergHierarchyError(f"Hierarchy artifact row-count mismatch: {path}")

    observations = pq.read_table(root / "observation_ledger.parquet")
    observation_ids = observations.column("observation_id").to_pylist()
    if len(observation_ids) != len(set(observation_ids)):
        raise HergHierarchyError("Hierarchy observation IDs are not unique")
    source_family = observations.column("source_family").to_pylist()
    pic50 = observations.column("pic50_value").to_pylist()
    for family, value in zip(source_family, pic50, strict=True):
        if family in {"pubchem_aid720551", "hergai_secondary"} and value is not None:
            raise HergHierarchyError("Binary source was impermissibly converted to pIC50")
    binary = pq.read_table(root / "structure_consensus_binary.parquet")
    ids = binary.column("structure_id").to_pylist()
    labels = binary.column("herg_blocker_label").to_pylist()
    if len(ids) != len(set(ids)) or any(label not in {0, 1} for label in labels):
        raise HergHierarchyError("Binary consensus is not a one-label-per-structure view")
    pubchem_labels: dict[str, set[int]] = defaultdict(set)
    observation_rows = observations.select(
        ["source_family", "structure_id", "derived_binary_label", "t1_candidate"]
    ).to_pylist()
    for row in observation_rows:
        if (
            row["source_family"] == "pubchem_aid720551"
            and row["structure_id"] is not None
            and row["derived_binary_label"] is not None
            and row["t1_candidate"]
        ):
            pubchem_labels[str(row["structure_id"])].add(int(row["derived_binary_label"]))
    for structure_id, label in zip(ids, labels, strict=True):
        if pubchem_labels.get(str(structure_id)) != {int(label)}:
            raise HergHierarchyError("Binary consensus contains non-AID720551 or conflicting evidence")
    quantitative = pq.read_table(root / "quantitative_pic50.parquet")
    if any(
        value is None or not math.isfinite(float(value))
        for value in quantitative.column("pic50_value").to_pylist()
    ):
        raise HergHierarchyError("Quantitative view contains an invalid pIC50")
    hierarchy = pq.read_table(root / "hierarchy_annotations.parquet")
    if any(value != "T0_reported" for value in hierarchy.column("reported_evidence_tier").to_pylist()):
        raise HergHierarchyError("Hierarchy lost the T0 reported evidence annotation")
    counts = manifest.get("counts", {})
    expected_counts = {
        "observations": observations.num_rows,
        "valid_structure_observations": sum(observations.column("structure_valid").to_pylist()),
        "T0_reported_observations": observations.num_rows,
        "T1_candidate_observations": sum(observations.column("t1_candidate").to_pylist()),
        "binary_consensus_structures": binary.num_rows,
        "quantitative_pic50_observations": quantitative.num_rows,
        "hierarchy_structures": hierarchy.num_rows,
        "primary_conflict_structures": sum(hierarchy.column("primary_conflict").to_pylist()),
    }
    if any(counts.get(name) != value for name, value in expected_counts.items()):
        raise HergHierarchyError("Manifest hierarchy counts do not match physical artifacts")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pubchem-outcomes", type=Path)
    parser.add_argument("--pubchem-structures", type=Path)
    parser.add_argument("--quantitative-pic50", action="append", default=[], type=Path)
    parser.add_argument("--hergai", action="append", default=[], type=Path)
    parser.add_argument("--chembl-herg-parquet", action="append", default=[], type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--blocker-max-um", type=float, default=DEFAULT_BLOCKER_MAX_UM)
    parser.add_argument("--nonblocker-min-um", type=float, default=DEFAULT_NONBLOCKER_MIN_UM)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.validate_only:
        validate_herg_hierarchy(args.output_root)
    else:
        if args.pubchem_outcomes is None or args.pubchem_structures is None or not args.quantitative_pic50:
            parser.error(
                "build mode requires --pubchem-outcomes, --pubchem-structures, "
                "and at least one --quantitative-pic50"
            )
        build_herg_hierarchy(
            pubchem_outcomes_path=args.pubchem_outcomes,
            pubchem_structures_path=args.pubchem_structures,
            quantitative_pic50_paths=args.quantitative_pic50,
            hergai_paths=args.hergai,
            chembl_parquet_paths=args.chembl_herg_parquet,
            output_root=args.output_root,
            blocker_max_um=args.blocker_max_um,
            nonblocker_min_um=args.nonblocker_min_um,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
