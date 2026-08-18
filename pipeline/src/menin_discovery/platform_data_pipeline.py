"""Rights-gated ChEMBL raw-to-canonical/task materialization.

The executed integration build consumes only the byte-preserved ChEMBL_37
REST snapshots. Conditional BindingDB, PubChem, supplementary, and internal
artifacts are never opened here. Full-release source assertions and specialized
inventories are exported separately by :mod:`platform_data_bulk`; their
partitioned canonical materialization uses the explicit handoff contract
returned by
:func:`menin_discovery.platform_data_bulk_canonical.bulk_canonicalization_contract`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .chemistry import standardize_smiles
from .platform_data_schema import (
    SCHEMA_VERSION,
    TABLE_REQUIRED_COLUMNS,
    canonical_json,
    canonical_relation,
    clean_text,
    concentration_interval_to_nm,
    concentration_to_nm,
    data_dictionary_frame,
    interval_bounds,
    normalize_unit,
    p_activity_from_nm,
    parse_interval,
    parse_numeric,
    relation_from_value,
    schema_document,
    stable_id,
    validate_table,
)
from .platform_data_sources import sha256_file, source_file_inventory, source_registry

PUBLIC_ACCESS_CLASS = "public_redistributable"
CHEMBL_SOURCE_ID = stable_id("SRC", "ChEMBL", "ChEMBL_37")
GAS_CONSTANT_KCAL_MOL_K = 0.00198720425864083
REFERENCE_TEMPERATURE_K = 298.15

_ENDPOINT_NAMES = {
    "kd": "Kd",
    "ki": "Ki",
    "ic20": "IC20",
    "ic25": "IC25",
    "ic50": "IC50",
    "ic90": "IC90",
    "ec50": "EC50",
    "ac50": "AC50",
    "gi50": "GI50",
    "km": "Km",
    "auc": "AUC",
    "aucinf": "AUCinf",
    "auclast": "AUClast",
    "cmax": "Cmax",
    "tmax": "Tmax",
    "t12": "T1/2",
    "halflife": "T1/2",
    "cl": "CL",
    "clint": "CLint",
    "vd": "Vd",
    "vdss": "Vdss",
    "papp": "Papp",
    "peff": "Peff",
    "ppb": "PPB",
    "qt": "QT",
    "qtc": "QTc",
    "qtcf": "QTcF",
    "qtcb": "QTcB",
    "apd": "APD",
    "apd50": "APD50",
    "apd90": "APD90",
}
_PK_ENDPOINTS = frozenset(
    {
        "auc",
        "aucinf",
        "auclast",
        "cl",
        "clint",
        "clrenal",
        "cmax",
        "f",
        "fu",
        "halflife",
        "papp",
        "peff",
        "ppb",
        "solubility",
        "t12",
        "tmax",
        "vd",
        "vdss",
        "logd",
        "logp",
    }
)
_QT_ENDPOINTS = frozenset({"qt", "qtc", "qtcf", "qtcb", "apd", "apd50", "apd90"})
_PK_ALLOWED_UNITS: dict[str, frozenset[str]] = {
    "auc": frozenset({"ng*h/ml", "ug*h/ml", "nm*h", "um*h"}),
    "aucinf": frozenset({"ng*h/ml", "ug*h/ml", "nm*h", "um*h"}),
    "auclast": frozenset({"ng*h/ml", "ug*h/ml", "nm*h", "um*h"}),
    "cmax": frozenset({"ng/ml", "ug/ml", "nm", "um"}),
    "tmax": frozenset({"h", "min"}),
    "t12": frozenset({"h", "min"}),
    "halflife": frozenset({"h", "min"}),
    "cl": frozenset({"ml/min/kg", "l/h/kg", "ml/min", "l/h"}),
    "clint": frozenset({"ml/min/kg", "ul/min/mg", "ml/min/mg"}),
    "vd": frozenset({"l/kg", "l"}),
    "vdss": frozenset({"l/kg", "l"}),
    "papp": frozenset({"cm/s", "nm/s"}),
    "peff": frozenset({"cm/s", "nm/s"}),
    "ppb": frozenset({"%"}),
    "f": frozenset({"%"}),
    "fu": frozenset({"%"}),
    "logd": frozenset({"", "unitless"}),
    "logp": frozenset({"", "unitless"}),
    "solubility": frozenset({"nm", "um", "mm", "m", "ug/ml", "mg/ml"}),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_frame(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False, compression="zstd")
    elif path.suffix == ".csv":
        frame.to_csv(temporary, index=False)
    else:
        raise ValueError(f"Unsupported table format: {path}")
    os.replace(temporary, path)
    return {
        "path": path.name,
        "rows": len(frame),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _identifier_text(value: object) -> str:
    text = clean_text(value)
    return text[:-2] if re.fullmatch(r"\d+\.0", text) else text


def _first(row: pd.Series, *columns: str) -> str:
    for column in columns:
        if column in row.index:
            value = clean_text(row[column])
            if value:
                return value
    return ""


def _endpoint_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean_text(value).casefold())


def _normalize_endpoint(value: object) -> str:
    raw = clean_text(value)
    return _ENDPOINT_NAMES.get(_endpoint_key(raw), raw)


def _temperature_c(description: object) -> float:
    text = clean_text(description)
    match = re.search(r"(?<!\d)(-?\d+(?:\.\d+)?)\s*(?:\u00b0\s*)?[Cc]\b", text)
    if match is None:
        return math.nan
    value = float(match.group(1))
    return value if value >= -273.15 else math.nan


def _target_metadata(raw_root: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for path in sorted((raw_root / "chembl_37_panel" / "targets").glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        target_id = clean_text(document.get("target_chembl_id"))
        components = document.get("target_components", [])
        accessions = sorted(
            {clean_text(item.get("accession")) for item in components if clean_text(item.get("accession"))}
        )
        genes: set[str] = set()
        for component in components:
            for synonym in component.get("target_component_synonyms", []):
                if clean_text(synonym.get("syn_type")) == "GENE_SYMBOL":
                    genes.add(clean_text(synonym.get("component_synonym")))
        metadata[target_id] = {
            "target_type": clean_text(document.get("target_type")),
            "target_name": clean_text(document.get("pref_name")),
            "species": clean_text(document.get("organism")),
            "accessions": accessions,
            "genes": sorted(genes),
            "target_raw_file": path.relative_to(raw_root).as_posix(),
        }
    return metadata


def load_chembl_integration_rows(
    raw_root: str | os.PathLike[str],
    *,
    registry: pd.DataFrame | None = None,
    file_inventory: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read only platform-owned ChEMBL snapshots and collapse activity-ID mirrors."""

    root = Path(raw_root).resolve()
    registry = source_registry(root) if registry is None else registry
    files = source_file_inventory(root, registry) if file_inventory is None else file_inventory
    file_lookup = files.set_index("relative_path").to_dict("index") if not files.empty else {}
    local_sources = registry[registry["source_record_scope"].str.startswith("targeted Menin/hERG", na=False)]
    panel_sources = registry[
        registry["source_record_scope"].str.startswith("bounded heterogeneous", na=False)
    ]
    local_snapshot = clean_text(local_sources.iloc[0]["snapshot_id"]) if not local_sources.empty else ""
    panel_snapshot = clean_text(panel_sources.iloc[0]["snapshot_id"]) if not panel_sources.empty else ""
    frames: list[pd.DataFrame] = []
    local_root = root / "local_legacy" / "chembl"
    for path in sorted(local_root.glob("*.csv")):
        relative = path.relative_to(root).as_posix()
        frame = pd.read_csv(path, low_memory=False, dtype=object)
        frame["_raw_relative_path"] = relative
        frame["_raw_file_id"] = file_lookup.get(relative, {}).get("source_file_id", "")
        frame["_snapshot_id"] = local_snapshot
        frames.append(frame)
    acquisition_path = root / "chembl_37_panel" / "acquisition.json"
    if acquisition_path.exists():
        acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
        for query in acquisition.get("queries", []):
            for page in query.get("pages", []):
                path = root / "chembl_37_panel" / str(page["file"])
                relative = path.relative_to(root).as_posix()
                document = json.loads(path.read_text(encoding="utf-8"))
                frame = pd.DataFrame(document.get("activities", []), dtype=object)
                if frame.empty:
                    continue
                frame["_raw_relative_path"] = relative
                frame["_raw_file_id"] = file_lookup.get(relative, {}).get("source_file_id", "")
                frame["_snapshot_id"] = panel_snapshot
                frames.append(frame)
    if not frames:
        raise FileNotFoundError("No rights-verified ChEMBL activity snapshots are available")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["_activity_key"] = combined["activity_id"].map(_identifier_text)
    missing_key = combined["_activity_key"].eq("")
    if missing_key.any():
        combined.loc[missing_key, "_activity_key"] = [
            stable_id(
                "RAWROW",
                row.get("_raw_file_id", ""),
                row.get("record_id", ""),
                row.get("molecule_chembl_id", ""),
                row.get("assay_chembl_id", ""),
                row.get("standard_type", row.get("type", "")),
                row.get("standard_value", row.get("value", "")),
            )
            for _, row in combined.loc[missing_key].iterrows()
        ]
    substantive = [
        column
        for column in combined.columns
        if not column.startswith("_") and column not in {"pchembl_value", "ligand_efficiency"}
    ]
    rows: list[pd.Series] = []
    conflict_groups = 0
    for _, group in combined.sort_values(["_activity_key", "_raw_relative_path"], kind="stable").groupby(
        "_activity_key", sort=False, dropna=False
    ):
        selected = group.iloc[0].copy()
        selected["_raw_file_ids"] = ";".join(
            sorted({clean_text(value) for value in group["_raw_file_id"] if clean_text(value)})
        )
        selected["_snapshot_ids"] = ";".join(
            sorted({clean_text(value) for value in group["_snapshot_id"] if clean_text(value)})
        )
        selected["_snapshot_id_primary"] = clean_text(selected.get("_snapshot_id", ""))
        conflict = False
        if len(group) > 1:
            for column in substantive:
                values = {clean_text(value) for value in group[column] if clean_text(value)}
                if len(values) > 1:
                    conflict = True
                    break
        selected["_raw_duplicate_conflict"] = conflict
        selected["_raw_mirror_count"] = len(group)
        conflict_groups += int(conflict)
        rows.append(selected)
    collapsed = pd.DataFrame(rows).reset_index(drop=True)
    collapsed["_source_compound_key"] = [
        _identifier_text(row.get("molecule_chembl_id", ""))
        or stable_id("CMPD", "missing", row.get("_activity_key", ""))
        for _, row in collapsed.iterrows()
    ]
    collapsed["_source_target_key"] = [
        _identifier_text(row.get("target_chembl_id", ""))
        or stable_id("TGT", "missing", row.get("_activity_key", ""))
        for _, row in collapsed.iterrows()
    ]
    collapsed["_source_assay_key"] = [
        _identifier_text(row.get("assay_chembl_id", ""))
        or stable_id("SRCASSAY", "missing", row.get("_activity_key", ""))
        for _, row in collapsed.iterrows()
    ]
    stats = {
        "physical_input_rows": len(combined),
        "unique_source_records": len(collapsed),
        "exact_source_record_mirrors_removed": len(combined) - len(collapsed),
        "conflicting_source_record_groups": conflict_groups,
        "input_file_count": int(combined["_raw_relative_path"].nunique()),
        "ignored_source_fields": ["pchembl_value", "ligand_efficiency"],
    }
    return collapsed, stats


def _protein_entities(
    rows: pd.DataFrame, target_metadata: dict[str, dict[str, Any]]
) -> tuple[pd.DataFrame, dict[str, str]]:
    proteins: list[dict[str, Any]] = []
    target_to_protein: dict[str, str] = {}
    for target_id, group in rows.groupby("_source_target_key", sort=True):
        first = group.iloc[0]
        metadata = target_metadata.get(target_id, {})
        accessions = list(metadata.get("accessions", [])) or sorted(
            {item for item in _first(first, "component_accessions").split(";") if item}
        )
        target_type = clean_text(metadata.get("target_type")) or _first(first, "target_type")
        target_name = clean_text(metadata.get("target_name")) or _first(first, "target_pref_name")
        lowered = f"{target_type} {target_name}".casefold()
        if target_type.casefold() == "single protein":
            entity_type = "single_protein"
        elif "complex" in lowered or "/" in target_name:
            entity_type = "protein_complex"
        else:
            entity_type = "source_target_assertion"
        canonical_target = (
            accessions[0] if len(accessions) == 1 and entity_type == "single_protein" else target_id
        )
        protein_id = stable_id("PROT", "ChEMBL_37", target_id, canonical_target, ";".join(accessions))
        target_to_protein[target_id] = protein_id
        component_sequences = [item for item in _first(first, "component_sequences").split(";") if item]
        sequence = (
            component_sequences[0]
            if entity_type == "single_protein" and len(component_sequences) == 1
            else ""
        )
        proteins.append(
            {
                "protein_id": protein_id,
                "entity_type": entity_type,
                "canonical_target_id": canonical_target,
                "target_name": target_name,
                "gene_symbol": ";".join(metadata.get("genes", [])),
                "uniprot_accession": ";".join(accessions),
                "sequence": sequence,
                "sequence_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest() if sequence else "",
                "isoform": "",
                "species": clean_text(metadata.get("species")) or _first(first, "target_organism"),
                "component_protein_ids": "",
                "identity_resolution_status": "resolved" if accessions else "source_assertion",
            }
        )
    return pd.DataFrame(proteins), target_to_protein


def _molecule_entities(
    rows: pd.DataFrame,
    *,
    require_rdkit: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], set[str]]:
    molecules_by_id: dict[str, dict[str, Any]] = {}
    aliases: list[dict[str, Any]] = []
    compound_to_molecule: dict[str, str] = {}
    conflicting_compounds: set[str] = set()
    grouped = rows.groupby("_source_compound_key", sort=True, dropna=False)
    for compound_id, group in grouped:
        submitted_values = sorted(
            {
                clean_text(value)
                for value in group.get("canonical_smiles", pd.Series(dtype=object))
                if clean_text(value)
            }
        )
        submitted = submitted_values[0] if submitted_values else ""
        source_inchi_keys = sorted(
            {
                clean_text(value)
                for value in group.get("standard_inchi_key", pd.Series(dtype=object))
                if clean_text(value)
            }
        )
        if require_rdkit:
            standardized = standardize_smiles(submitted, require_rdkit=True)
            standardized_keys = {
                standardize_smiles(value, require_rdkit=True).standard_inchi_key or value
                for value in submitted_values
            }
            standardized_inchi = standardized.standard_inchi_key
            canonical_smiles = standardized.canonical_smiles
            standardized_smiles = standardized.standardized_smiles
            structure_identifier = standardized.structure_id
            full_structure_id = standardized.full_structure_id
            standardization_version = standardized.structure_standardization_version
            standardization_status = standardized.structure_standardization_status
            structure_valid = bool(standardized.structure_valid)
            formal_charge = standardized.formal_charge
            fragment_count = standardized.fragment_count
        else:
            standardized_keys = set(source_inchi_keys or submitted_values)
            standardized_inchi = source_inchi_keys[0] if source_inchi_keys else ""
            canonical_smiles = submitted
            standardized_smiles = submitted
            structure_identifier = (
                stable_id("STR", "ChEMBL_37.compound_structures", submitted) if submitted else ""
            )
            full_structure_id = structure_identifier
            standardization_version = "ChEMBL_37.compound_structures-source-standardization-v1"
            standardization_status = (
                "source_standard_inchi_key" if standardized_inchi else "source_smiles_unresolved"
            )
            structure_valid = bool(submitted and standardized_inchi)
            formal_charge = None
            fragment_count = submitted.count(".") + 1 if submitted else None
        conflict = len(standardized_keys) > 1
        if conflict:
            conflicting_compounds.add(compound_id)
        if standardized_inchi:
            molecule_id = stable_id("MOL", "standard-inchi-key", standardized_inchi)
        elif structure_identifier:
            molecule_id = stable_id("MOL", structure_identifier)
        else:
            molecule_id = stable_id("MOL", "ChEMBL_37", compound_id)
        compound_to_molecule[compound_id] = molecule_id
        status = standardization_status
        resolution = "conflicting" if conflict else "resolved" if structure_valid else "unresolved"
        candidate = {
            "molecule_id": molecule_id,
            "structure_id": structure_identifier,
            "full_structure_id": full_structure_id,
            "submitted_smiles": submitted,
            "canonical_smiles": canonical_smiles,
            "standardized_smiles": standardized_smiles,
            "standard_inchi_key": standardized_inchi,
            "standardization_version": standardization_version,
            "standardization_status": status,
            "identity_resolution_status": resolution,
            "formal_charge": formal_charge,
            "fragment_count": fragment_count,
            "source_count": 1,
        }
        if molecule_id not in molecules_by_id:
            molecules_by_id[molecule_id] = candidate
        snapshots = clean_text(group.iloc[0].get("_snapshot_id_primary", ""))
        aliases.append(
            {
                "molecule_alias_id": stable_id("MALIAS", CHEMBL_SOURCE_ID, compound_id),
                "molecule_id": molecule_id,
                "source_id": CHEMBL_SOURCE_ID,
                "snapshot_id": snapshots,
                "source_compound_id": compound_id,
                "source_record_id": f"ChEMBL:molecule:{compound_id}",
                "compound_name": _first(group.iloc[0], "molecule_pref_name"),
                "submitted_smiles": submitted,
                "submitted_inchi_key": standardized_inchi,
                "resolution_method": (
                    "RDKit standard InChIKey exact"
                    if require_rdkit and standardized_inchi
                    else "ChEMBL_37 source standard InChIKey exact"
                    if standardized_inchi
                    else "source-scoped fallback"
                ),
                "resolution_status": resolution,
            }
        )
    return (
        pd.DataFrame(molecules_by_id.values())
        .sort_values("molecule_id", kind="stable")
        .reset_index(drop=True),
        pd.DataFrame(aliases).sort_values("molecule_alias_id", kind="stable").reset_index(drop=True),
        compound_to_molecule,
        conflicting_compounds,
    )


def _assay_family(endpoint: str, target_id: str, assay_type: str, description: str) -> str:
    key = _endpoint_key(endpoint)
    text = f"{endpoint} {description}".casefold()
    if key in _QT_ENDPOINTS or re.search(r"\b(?:qtc?[fb]?|apd(?:50|90)?)\b", text):
        return "cardiac_electrophysiology_qt_apd"
    if target_id == "CHEMBL240":
        return {
            "b": "herg_binding",
            "f": "herg_functional",
        }.get(assay_type.casefold(), "herg_other")
    if key in _PK_ENDPOINTS or any(
        token in text
        for token in ("microsom", "hepatocyte", "permeab", "bioavailability", "plasma protein binding")
    ):
        return "pk_adme"
    if key in {"kd", "ki"} or assay_type.casefold() == "b":
        return "binding"
    return "other_bioactivity"


def _assay_context_family(target_id: str, assay_type: str, description: str) -> str:
    """Classify a source assay from assay/target context, independent of endpoint rows."""

    text = description.casefold()
    if re.search(r"\b(?:qtc?[fb]?|apd(?:50|90)?)\b", text):
        return "cardiac_electrophysiology_qt_apd"
    if target_id == "CHEMBL240":
        return {
            "b": "herg_binding",
            "f": "herg_functional",
        }.get(assay_type.casefold(), "herg_other")
    if any(
        token in text
        for token in (
            "microsom",
            "hepatocyte",
            "permeab",
            "bioavailability",
            "plasma protein binding",
            "clearance",
            "pharmacokinetic",
        )
    ):
        return "pk_adme"
    if assay_type.casefold() == "b":
        return "binding"
    return "other_bioactivity"


def _evidence_domain(endpoint: str, target_id: str, assay_type: str, description: str) -> str:
    family = _assay_family(endpoint, target_id, assay_type, description)
    return {
        "cardiac_electrophysiology_qt_apd": "qt",
        "pk_adme": "pk_adme",
        "binding": "binding",
    }.get(family, "herg" if family.startswith("herg_") else "other")


def _evidence_stage(description: str) -> object:
    text = description.casefold()
    if re.search(r"\bex[ -]?vivo\b", text):
        return "preclinical_ex_vivo"
    if re.search(r"\bin[ -]?vivo\b", text):
        return "preclinical_in_vivo"
    if any(
        token in text
        for token in (
            "cell-based",
            "whole-cell",
            "cell line",
            "recombinant",
            "purified protein",
            "isolated enzyme",
            "membrane preparation",
        )
    ):
        return "preclinical_in_vitro"
    return pd.NA


def _assay_entities(
    rows: pd.DataFrame,
    target_to_protein: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    assays: list[dict[str, Any]] = []
    assay_to_id: dict[str, str] = {}
    for source_assay_id, group in rows.groupby("_source_assay_key", sort=True, dropna=False):
        first = group.iloc[0]
        target_id = clean_text(first.get("_source_target_key", ""))
        protein_id = target_to_protein[target_id]
        description = _first(first, "assay_description")
        assay_type = _first(first, "assay_type")
        assay_format = _first(first, "bao_label", "bao_format")
        organism = _first(first, "assay_organism")
        cell_system = _first(first, "assay_cell_type")
        tissue = _first(first, "assay_tissue")
        strain = _first(first, "assay_strain")
        subcellular_fraction = _first(first, "assay_subcellular_fraction")
        core = {
            "description": description,
            "assay_type": assay_type,
            "assay_format": assay_format,
            "organism": organism,
            "protein": protein_id,
        }
        missing = sorted(name for name, value in core.items() if not clean_text(value))
        assay_id = stable_id("ASSAY", "ChEMBL_37", source_assay_id)
        assay_to_id[source_assay_id] = assay_id
        assays.append(
            {
                "assay_id": assay_id,
                "source_id": CHEMBL_SOURCE_ID,
                "snapshot_id": clean_text(group.iloc[0].get("_snapshot_id_primary", "")),
                "source_assay_id": source_assay_id,
                "protein_id": protein_id,
                "construct_id": "",
                "assay_type": assay_type,
                "assay_family": _assay_context_family(target_id, assay_type, description),
                "assay_format": assay_format,
                "description": description,
                "organism": organism,
                "cell_system": cell_system,
                "tissue": tissue,
                "strain": strain,
                "subcellular_fraction": subcellular_fraction,
                "matrix": "",
                "route": "",
                "assay_test_type": _first(first, "assay_test_type"),
                "assay_category": _first(first, "assay_category"),
                "assay_tax_id": _identifier_text(first.get("assay_tax_id", "")),
                "relationship_type": _first(first, "relationship_type"),
                "source_assay_external_id": _first(first, "src_assay_id"),
                "cell_id": _identifier_text(first.get("cell_id", "")),
                "tissue_id": _identifier_text(first.get("tissue_id", "")),
                "variant_id": _identifier_text(first.get("variant_id", "")),
                "assay_group": _first(first, "assay_group"),
                "bao_format_id": _first(first, "bao_format"),
                "confidence_score": pd.to_numeric(
                    pd.Series([first.get("confidence_score", pd.NA)]), errors="coerce"
                ).iloc[0],
                "temperature_c": _temperature_c(description),
                "ph": math.nan,
                "protocol_completeness": (len(core) - len(missing)) / len(core),
                "protocol_missing_fields": ";".join(missing),
            }
        )
    return pd.DataFrame(assays), assay_to_id


def _endpoint_family(endpoint: str, domain: str) -> str:
    key = _endpoint_key(endpoint)
    if key in {"kd", "ki"}:
        return "equilibrium_affinity"
    if key.startswith("ic") or key.startswith("ec") or key in {"ac50", "gi50", "potency"}:
        return "potency_activity"
    if domain == "pk_adme":
        return "pk_adme"
    if domain in {"herg", "qt"}:
        return "cardiac_electrophysiology"
    return "other"


def _measurement(row: pd.Series, endpoint: str) -> dict[str, Any]:
    lower = _first(row, "standard_value", "value")
    upper = _first(row, "standard_upper_value", "upper_value")
    unit = _first(row, "standard_units", "units")
    explicit_relation = _first(row, "standard_relation", "relation")
    value_raw = lower
    if lower and upper:
        value_raw = f"{lower}-{upper}"
        relation = "interval" if parse_interval(value_raw) is not None else "not_reported"
    else:
        relation = relation_from_value(lower, explicit_relation)
    if relation == "interval":
        lower_bound, upper_bound, canonical_unit, status = concentration_interval_to_nm(
            endpoint, value_raw, unit
        )
        canonical_value = math.nan
        value_numeric = math.nan
    else:
        canonical_value, canonical_unit, status = concentration_to_nm(endpoint, lower, unit)
        value_numeric = parse_numeric(lower)
        lower_bound, upper_bound = interval_bounds(canonical_value, relation)
    label_text = _first(row, "standard_text_value", "text_value", "activity_comment")
    p_activity = (
        p_activity_from_nm(canonical_value)
        if canonical_unit == "nM" and math.isfinite(canonical_value)
        else math.nan
    )
    inverse = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}
    p_relation = inverse.get(relation, relation if relation in {"=", "~"} else "not_reported")
    return {
        "relation": canonical_relation(relation),
        "value_raw": value_raw,
        "label_text": label_text,
        "value_numeric": value_numeric,
        "original_unit": unit,
        "canonical_value": canonical_value,
        "canonical_unit": canonical_unit,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "p_activity": p_activity,
        "p_activity_relation": p_relation,
        "unit_conversion_status": status,
    }


def _observations(
    rows: pd.DataFrame,
    compound_to_molecule: dict[str, str],
    conflicting_compounds: set[str],
    target_to_protein: dict[str, str],
    assay_to_id: dict[str, str],
    assays: pd.DataFrame,
    *,
    require_rdkit: bool = True,
) -> pd.DataFrame:
    assay_metadata = assays.set_index("assay_id").to_dict("index")
    observations: list[dict[str, Any]] = []
    for _, row in rows.sort_values("_activity_key", kind="stable").iterrows():
        activity_key = _identifier_text(row.get("_activity_key", ""))
        compound_id = clean_text(row.get("_source_compound_key", ""))
        target_id = clean_text(row.get("_source_target_key", ""))
        source_assay_id = clean_text(row.get("_source_assay_key", ""))
        endpoint = _normalize_endpoint(_first(row, "standard_type", "type")) or "not_reported"
        assay_id = assay_to_id[source_assay_id]
        protein_id = target_to_protein[target_id]
        molecule_id = compound_to_molecule[compound_id]
        description = _first(row, "assay_description")
        assay_type = _first(row, "assay_type")
        domain = _evidence_domain(endpoint, target_id, assay_type, description)
        measurement = _measurement(row, endpoint)
        reasons: list[str] = []
        reviews: list[str] = []
        if not _identifier_text(row.get("molecule_chembl_id", "")):
            reasons.append("missing_source_molecule_id")
        if not _identifier_text(row.get("target_chembl_id", "")):
            reasons.append("missing_source_target_id")
        if not _identifier_text(row.get("assay_chembl_id", "")):
            reasons.append("missing_source_assay_id")
        if endpoint == "not_reported":
            reasons.append("missing_endpoint")
        if compound_id in conflicting_compounds:
            reasons.append("conflicting_structure_for_source_compound")
        molecule_smiles = clean_text(row.get("canonical_smiles", ""))
        structure_valid = (
            bool(standardize_smiles(molecule_smiles, require_rdkit=True).structure_valid)
            if require_rdkit
            else bool(molecule_smiles and _first(row, "standard_inchi_key"))
        )
        if not structure_valid:
            reasons.append("missing_or_invalid_structure")
        if bool(row.get("_raw_duplicate_conflict", False)):
            reasons.append("conflicting_same_activity_id_across_raw_files")
        if measurement["relation"] == "not_reported" and (
            clean_text(measurement["value_raw"]) or clean_text(measurement["label_text"])
        ):
            reasons.append("unsupported_or_missing_relation")
        numeric_present = (
            math.isfinite(float(measurement["value_numeric"]))
            or math.isfinite(float(measurement["lower_bound"]))
            or math.isfinite(float(measurement["upper_bound"]))
        )
        if not numeric_present and not clean_text(measurement["label_text"]):
            reasons.append("missing_numeric_and_categorical_result")
        if numeric_present and not clean_text(measurement["canonical_unit"]):
            reasons.append("missing_or_unsupported_unit")
        if measurement["canonical_unit"] == "nM":
            numeric_candidates = [
                value
                for value in (
                    measurement["canonical_value"],
                    measurement["lower_bound"],
                    measurement["upper_bound"],
                )
                if math.isfinite(float(value))
            ]
            if any(float(value) <= 0 for value in numeric_candidates):
                reasons.append("nonpositive_concentration")
        endpoint_key = _endpoint_key(endpoint)
        if domain == "pk_adme" and numeric_present:
            allowed = _PK_ALLOWED_UNITS.get(endpoint_key)
            normalized_unit = normalize_unit(measurement["original_unit"]).casefold()
            if allowed is None or normalized_unit not in allowed:
                reasons.append("pk_endpoint_unit_not_allowlisted")
            assay_context = assay_metadata[assay_id]
            if not any(
                clean_text(assay_context.get(field, ""))
                for field in ("matrix", "route", "cell_system", "tissue", "subcellular_fraction")
            ):
                reasons.append("pk_required_context_missing")
        if domain == "qt" and numeric_present:
            if endpoint_key not in _QT_ENDPOINTS or normalize_unit(
                measurement["original_unit"]
            ).casefold() not in {
                "ms",
                "s",
                "%",
            }:
                reasons.append("qt_apd_context_or_unit_insufficient")
        standard_flag = _identifier_text(row.get("standard_flag", ""))
        potential_duplicate = _identifier_text(row.get("potential_duplicate", "")) == "1"
        validity_comment = _first(row, "data_validity_comment")
        if standard_flag != "1":
            reviews.append(
                "chembl_standard_flag_missing" if not standard_flag else "chembl_nonstandard_activity"
            )
        if potential_duplicate:
            reviews.append("chembl_potential_duplicate")
        if validity_comment:
            reviews.append("chembl_data_validity_comment")
        if measurement["relation"] == "~":
            reviews.append("approximate_relation")
        protocol = float(assay_metadata[assay_id]["protocol_completeness"])
        if reasons:
            quality = "quarantined"
            inclusion = "quarantined"
        elif reviews:
            quality = "identity_resolved" if structure_valid else "parsable"
            inclusion = "review"
        else:
            quality = "protocol_sufficient" if protocol >= 0.8 else "identity_resolved"
            inclusion = "included"
        source_record_id = f"ChEMBL:activity:{activity_key}"
        observation_id = stable_id("OBS", "ChEMBL_37", activity_key, endpoint)
        record = {
            "observation_id": observation_id,
            "source_id": CHEMBL_SOURCE_ID,
            "snapshot_id": clean_text(row.get("_snapshot_id_primary", "")),
            "source_record_id": source_record_id,
            "raw_file_ids": clean_text(row.get("_raw_file_ids", "")),
            "molecule_id": molecule_id,
            "protein_id": protein_id,
            "construct_id": "",
            "assay_id": assay_id,
            "evidence_domain": domain,
            "endpoint": endpoint,
            "endpoint_family": _endpoint_family(endpoint, domain),
            **measurement,
            "observation_kind": "experimental_summary",
            "value_provenance": "ChEMBL_37.activities.standard_value/relation/units; pchembl_value ignored",
            "chembl_standard_flag": standard_flag,
            "chembl_potential_duplicate": potential_duplicate,
            "chembl_data_validity_comment": validity_comment,
            "document_id": _first(row, "document_chembl_id"),
            "document_year": int(float(_first(row, "document_year")))
            if _first(row, "document_year")
            else pd.NA,
            "document_doi": _first(row, "document_doi"),
            "document_pubmed_id": _identifier_text(row.get("pubmed_id", "")),
            "document_patent_id": _first(row, "patent_id"),
            "document_title": _first(row, "document_title"),
            "document_type": _first(row, "document_type"),
            "document_chembl_release_id": _identifier_text(row.get("document_chembl_release_id", "")),
            "activity_origin_id": _identifier_text(row.get("src_id", "")),
            "activity_origin_name": _first(row, "activity_source_name"),
            "activity_origin_description": _first(row, "activity_source_description"),
            "evidence_stage": _evidence_stage(description),
            "development_stage": "unknown",
            "result_status": "reported",
            "quality_grade": quality,
            "access_class": PUBLIC_ACCESS_CLASS,
            "inclusion_status": inclusion,
            "exclusion_reason": ";".join(sorted([*reasons, *reviews])),
            "dedup_group_id": stable_id(
                "DEDUP",
                molecule_id,
                protein_id,
                assay_id,
                endpoint,
                measurement["relation"],
                measurement["canonical_value"],
                measurement["lower_bound"],
                measurement["upper_bound"],
                measurement["canonical_unit"],
            ),
            "conflict_group_id": "",
            "cross_source_mirror": False,
            "potential_leakage": False,
        }
        observations.append(record)
    frame = pd.DataFrame(observations)
    group_columns = ["molecule_id", "protein_id", "assay_id", "endpoint", "canonical_unit"]
    for key, group in frame.groupby(group_columns, dropna=False, sort=False):
        exact = group[
            (group["relation"] == "=") & pd.to_numeric(group["canonical_value"], errors="coerce").notna()
        ]
        values = pd.to_numeric(exact["canonical_value"], errors="coerce")
        positive = values[values > 0]
        if len(positive) >= 2 and float(positive.max() / positive.min()) >= 10.0:
            frame.loc[group.index, "conflict_group_id"] = stable_id("CONFLICT", *key)
    return frame.sort_values("observation_id", kind="stable").reset_index(drop=True)


def binding_free_energy_view(
    observations: pd.DataFrame,
    assays: pd.DataFrame,
    proteins: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive standard-state ΔG° only from exact positive Kd measurements."""

    assay_lookup = assays.set_index("assay_id").to_dict("index")
    single_proteins = set(proteins.loc[proteins["entity_type"] == "single_protein", "protein_id"])
    eligible = observations[
        (observations["endpoint"].map(_endpoint_key) == "kd")
        & (observations["relation"] == "=")
        & (observations["canonical_unit"] == "nM")
        & (observations["inclusion_status"] == "included")
        & observations["protein_id"].isin(single_proteins)
        & (pd.to_numeric(observations["canonical_value"], errors="coerce") > 0)
        & observations["observation_kind"].isin({"experimental_raw", "experimental_summary"})
    ]
    derivations: list[dict[str, Any]] = []
    derived_observations: list[dict[str, Any]] = []
    for _, source in eligible.iterrows():
        assay_temperature = assay_lookup.get(source["assay_id"], {}).get("temperature_c", math.nan)
        if pd.notna(assay_temperature) and math.isfinite(float(assay_temperature)):
            temperature_k = float(assay_temperature) + 273.15
            temperature_source = "source_reported_assay_temperature"
        else:
            temperature_k = REFERENCE_TEMPERATURE_K
            temperature_source = "reference_temperature_approximation"
        kd_molar = float(source["canonical_value"]) * 1e-9
        delta_g = GAS_CONSTANT_KCAL_MOL_K * temperature_k * math.log(kd_molar / 1.0)
        roundtrip = math.exp(delta_g / (GAS_CONSTANT_KCAL_MOL_K * temperature_k))
        relative_error = abs(roundtrip - kd_molar) / kd_molar
        derived_id = stable_id(
            "DOBS",
            source["observation_id"],
            "delta_g_standard",
            temperature_k,
            temperature_source,
            "1 mol/L",
        )
        lineage_payload = {
            "source_observation_id": source["observation_id"],
            "source_record_id": source["source_record_id"],
            "source_snapshot_id": source["snapshot_id"],
            "source_relation": source["relation"],
            "source_kd_value": float(source["canonical_value"]),
            "source_kd_unit": source["canonical_unit"],
            "formula": "delta_g_kcal_mol=R_kcal_mol_K*T_K*ln(Kd_M/1_M)",
            "R": GAS_CONSTANT_KCAL_MOL_K,
            "temperature_k": temperature_k,
            "temperature_source": temperature_source,
            "standard_state": "1 mol/L",
        }
        lineage = hashlib.sha256(canonical_json(lineage_payload).encode("utf-8")).hexdigest()
        derivations.append(
            {
                "observation_id": derived_id,
                "source_observation_id": source["observation_id"],
                "source_record_id": source["source_record_id"],
                "source_snapshot_id": source["snapshot_id"],
                "source_relation": source["relation"],
                "source_kd_value": float(source["canonical_value"]),
                "source_kd_unit": source["canonical_unit"],
                "molecule_id": source["molecule_id"],
                "protein_id": source["protein_id"],
                "assay_id": source["assay_id"],
                "kd_nM": float(source["canonical_value"]),
                "kd_molar": kd_molar,
                "temperature_k": temperature_k,
                "temperature_source": temperature_source,
                "standard_state": "1 mol/L",
                "gas_constant_kcal_mol_k": GAS_CONSTANT_KCAL_MOL_K,
                "formula": "delta_g_kcal_mol=R_kcal_mol_K*T_K*ln(Kd_M/1_M)",
                "delta_g_kcal_mol": delta_g,
                "roundtrip_kd_molar": roundtrip,
                "roundtrip_relative_error": relative_error,
                "label_lineage_digest": lineage,
                "observation_kind": "derived",
                "access_class": PUBLIC_ACCESS_CLASS,
            }
        )
        derived = source.to_dict()
        derived.update(
            {
                "observation_id": derived_id,
                "source_record_id": f"{source['source_record_id']}::derived-delta-g-standard",
                "endpoint": "standard_binding_free_energy",
                "endpoint_family": "thermodynamic_derivation",
                "relation": "=",
                "value_raw": "",
                "label_text": "",
                "value_numeric": delta_g,
                "original_unit": "kcal/mol",
                "canonical_value": delta_g,
                "canonical_unit": "kcal/mol",
                "lower_bound": delta_g,
                "upper_bound": delta_g,
                "p_activity": math.nan,
                "p_activity_relation": "not_reported",
                "unit_conversion_status": "derived_identity",
                "observation_kind": "derived",
                "value_provenance": f"standard-state Kd transform; lineage_sha256={lineage}",
                "dedup_group_id": stable_id("DEDUP", derived_id),
                "conflict_group_id": "",
                "potential_leakage": True,
            }
        )
        derived_observations.append(derived)
    return pd.DataFrame(derivations), pd.DataFrame(derived_observations)


def _observation_lineage(
    rows: pd.DataFrame,
    source_files: pd.DataFrame,
    derivations: pd.DataFrame,
) -> pd.DataFrame:
    """Create normalized observation-to-file edges with singular snapshots."""

    file_snapshot = (
        source_files.set_index("source_file_id")["snapshot_id"].to_dict() if not source_files.empty else {}
    )
    edges: list[dict[str, Any]] = []
    source_to_edges: dict[str, list[dict[str, Any]]] = {}
    for _, row in rows.iterrows():
        endpoint = _normalize_endpoint(_first(row, "standard_type", "type")) or "not_reported"
        observation_id = stable_id("OBS", "ChEMBL_37", row["_activity_key"], endpoint)
        primary_file = clean_text(row.get("_raw_file_id", ""))
        records: list[dict[str, Any]] = []
        for source_file_id in clean_text(row.get("_raw_file_ids", "")).split(";"):
            source_file_id = source_file_id.strip()
            if not source_file_id:
                continue
            snapshot_id = clean_text(file_snapshot.get(source_file_id, ""))
            record = {
                "lineage_id": stable_id("LINEAGE", observation_id, source_file_id),
                "observation_id": observation_id,
                "source_id": CHEMBL_SOURCE_ID,
                "snapshot_id": snapshot_id,
                "source_file_id": source_file_id,
                "lineage_role": "primary" if source_file_id == primary_file else "mirrored_assertion",
            }
            edges.append(record)
            records.append(record)
        source_to_edges[observation_id] = records
    if not derivations.empty:
        for _, derivation in derivations.iterrows():
            for source_edge in source_to_edges.get(clean_text(derivation["source_observation_id"]), []):
                derived_id = clean_text(derivation["observation_id"])
                edges.append(
                    {
                        "lineage_id": stable_id("LINEAGE", derived_id, source_edge["source_file_id"]),
                        "observation_id": derived_id,
                        "source_id": CHEMBL_SOURCE_ID,
                        "snapshot_id": source_edge["snapshot_id"],
                        "source_file_id": source_edge["source_file_id"],
                        "lineage_role": "derived_support",
                    }
                )
    return (
        pd.DataFrame(edges)
        .sort_values(["observation_id", "source_file_id"], kind="stable")
        .reset_index(drop=True)
    )


def _validate_observation_lineage(
    observations: pd.DataFrame,
    lineage: pd.DataFrame,
    sources: pd.DataFrame,
    source_files: pd.DataFrame,
) -> None:
    valid_snapshots = set(zip(sources["source_id"], sources["snapshot_id"], strict=False))
    invalid_snapshot_edges = lineage[
        ~pd.Series(
            list(zip(lineage["source_id"], lineage["snapshot_id"], strict=False)),
            index=lineage.index,
        ).isin(valid_snapshots)
    ]
    if not invalid_snapshot_edges.empty:
        raise RuntimeError("Observation lineage contains orphan source/snapshot edges")
    if not set(lineage["source_file_id"]).issubset(set(source_files["source_file_id"])):
        raise RuntimeError("Observation lineage contains orphan source-file edges")
    if not set(lineage["observation_id"]).issubset(set(observations["observation_id"])):
        raise RuntimeError("Observation lineage contains orphan observation edges")
    counts = lineage.groupby("observation_id")["lineage_id"].size()
    if not set(observations["observation_id"]).issubset(set(counts.index)):
        raise RuntimeError("Every canonical observation requires at least one normalized lineage edge")
    primary = lineage[lineage["lineage_role"] == "primary"].groupby("observation_id").size()
    experimental = observations[observations["observation_kind"] != "derived"]
    if any(int(primary.get(observation_id, 0)) != 1 for observation_id in experimental["observation_id"]):
        raise RuntimeError("Every non-derived observation requires exactly one primary lineage edge")
    primary_snapshot = (
        lineage[lineage["lineage_role"] == "primary"].set_index("observation_id")["snapshot_id"].to_dict()
    )
    if any(
        clean_text(row["snapshot_id"]) != clean_text(primary_snapshot.get(row["observation_id"], ""))
        for _, row in experimental.iterrows()
    ):
        raise RuntimeError("Canonical observation snapshot does not match its primary lineage edge")


def _task_registry(tasks: pd.DataFrame) -> pd.DataFrame:
    signature_columns = [
        "task_type",
        "evidence_domain",
        "endpoint",
        "assay_family",
        "label_kind",
        "label_unit",
        "observation_kind",
        "default_task_eligible",
        "sensitivity_task_eligible",
        "required_modalities",
    ]
    rows: list[dict[str, Any]] = []
    signature_to_task: dict[tuple[str, ...], str] = {}
    for task_id, group in tasks.groupby("task_id", sort=True):
        for column in signature_columns:
            if group[column].nunique(dropna=False) != 1:
                raise RuntimeError(f"Task {task_id} pools incompatible {column} values")
        first = group.iloc[0]
        signature = tuple(clean_text(first[column]) for column in signature_columns)
        prior = signature_to_task.setdefault(signature, clean_text(task_id))
        if prior != clean_text(task_id):
            raise RuntimeError("One task signature maps to multiple task IDs")
        rows.append(
            {
                "task_id": task_id,
                "task_type": first["task_type"],
                "evidence_domain": first["evidence_domain"],
                "endpoint": first["endpoint"],
                "assay_family": first["assay_family"],
                "label_kind": first["label_kind"],
                "label_unit": first["label_unit"],
                "observation_kind": first["observation_kind"],
                "default_task_eligible": bool(first["default_task_eligible"]),
                "sensitivity_task_eligible": bool(first["sensitivity_task_eligible"]),
                "required_modalities": first["required_modalities"],
                "policy_version": "platform-task-contract-v1",
                "row_count": len(group),
                "relation_counts_json": canonical_json(
                    {
                        str(key): int(value)
                        for key, value in group["label_relation"].value_counts().sort_index().items()
                    }
                ),
                "intended_use": "endpoint-specific supervised modeling under frozen group/temporal splits",
                "prohibited_claim": "not clinical efficacy, QT/TdP risk, or cross-endpoint equivalence",
            }
        )
    return pd.DataFrame(rows).sort_values("task_id", kind="stable").reset_index(drop=True)


def _task_view(
    observations: pd.DataFrame,
    assays: pd.DataFrame,
    *,
    derived_sensitivity: bool = False,
) -> pd.DataFrame:
    if derived_sensitivity:
        policy_mask = observations["observation_kind"].eq("derived") & observations["endpoint"].eq(
            "standard_binding_free_energy"
        )
    else:
        endpoint_keys = observations["endpoint"].map(_endpoint_key)
        assay_family_lookup = assays.set_index("assay_id")["assay_family"].to_dict()
        assay_families = observations["assay_id"].map(assay_family_lookup).fillna("")
        allowed_standard = endpoint_keys.isin({"kd", "ki", "ic50", "ec50"}) & observations[
            "canonical_unit"
        ].eq("nM")
        allowed_herg = (
            observations["evidence_domain"].eq("herg")
            & endpoint_keys.eq("ic50")
            & assay_families.eq("herg_functional")
        )
        allowed_nonherg = ~observations["evidence_domain"].eq("herg") & allowed_standard
        policy_mask = (
            observations["observation_kind"].isin(
                {"experimental_raw", "experimental_summary", "curated_assertion"}
            )
            & ~observations["evidence_domain"].isin({"qt", "pk_adme"})
            & ~observations["potential_leakage"].astype(bool)
            & (allowed_herg | allowed_nonherg)
        )
    admitted = observations[
        policy_mask
        & (observations["inclusion_status"] == "included")
        & (observations["access_class"] == PUBLIC_ACCESS_CLASS)
        & observations["relation"].isin({"=", "<", "<=", ">", ">=", "interval"})
        & (
            pd.to_numeric(observations["canonical_value"], errors="coerce").notna()
            | pd.to_numeric(observations["lower_bound"], errors="coerce").notna()
            | pd.to_numeric(observations["upper_bound"], errors="coerce").notna()
        )
    ].copy()
    admitted["label_kind"] = np.where(admitted["relation"] == "=", "continuous_exact", "continuous_censored")
    assay_family = assays.set_index("assay_id")["assay_family"].to_dict()
    admitted["assay_family"] = admitted["assay_id"].map(assay_family).fillna("unresolved_assay_family")
    admitted["task_type"] = admitted.apply(
        lambda row: "__".join(
            [
                "sensitivity" if derived_sensitivity else "default",
                re.sub(r"[^a-z0-9]+", "_", clean_text(row["evidence_domain"]).casefold()).strip("_"),
                re.sub(r"[^a-z0-9]+", "_", clean_text(row["endpoint"]).casefold()).strip("_"),
                re.sub(r"[^a-z0-9]+", "_", clean_text(row["assay_family"]).casefold()).strip("_"),
                re.sub(r"[^a-z0-9]+", "_", clean_text(row["canonical_unit"]).casefold()).strip("_")
                or "unitless",
                clean_text(row["label_kind"]),
            ]
        ),
        axis=1,
    )
    admitted["task_id"] = [
        stable_id("TASK", "platform-task-contract-v1", task_type) for task_type in admitted["task_type"]
    ]
    admitted["label_value"] = np.where(admitted["relation"] == "=", admitted["canonical_value"], np.nan)
    admitted["label_text"] = ""
    admitted["label_relation"] = admitted["relation"]
    admitted["label_lower_bound"] = admitted["lower_bound"]
    admitted["label_upper_bound"] = admitted["upper_bound"]
    admitted["label_unit"] = admitted["canonical_unit"]
    admitted["default_task_eligible"] = not derived_sensitivity
    admitted["sensitivity_task_eligible"] = derived_sensitivity
    admitted["threshold_low_nM"] = np.nan
    admitted["threshold_high_nM"] = np.nan
    admitted["threshold_source_value_nM"] = np.nan
    admitted["threshold_policy"] = ""
    admitted["required_modalities"] = "small_molecule_structure;protein_sequence"
    if not derived_sensitivity:
        exact_values = pd.to_numeric(admitted["canonical_value"], errors="coerce")
        binary = admitted[
            admitted["evidence_domain"].eq("herg")
            & admitted["endpoint"].map(_endpoint_key).eq("ic50")
            & admitted["assay_family"].eq("herg_functional")
            & admitted["relation"].eq("=")
            & ((exact_values <= 10_000.0) | (exact_values >= 30_000.0))
        ].copy()
        if not binary.empty:
            binary_values = pd.to_numeric(binary["canonical_value"], errors="raise")
            binary["task_type"] = (
                "default__herg__ic50__herg_functional__binary_exact__"
                "blocker_le_10000nm__nonblocker_ge_30000nm"
            )
            binary["task_id"] = stable_id(
                "TASK",
                "platform-task-contract-v1",
                binary["task_type"].iloc[0],
            )
            binary["label_kind"] = "categorical"
            binary["label_value"] = np.where(binary_values <= 10_000.0, 1.0, 0.0)
            binary["label_text"] = np.where(
                binary_values <= 10_000.0,
                "blocker",
                "nonblocker",
            )
            binary["label_relation"] = "="
            binary["label_lower_bound"] = np.nan
            binary["label_upper_bound"] = np.nan
            binary["label_unit"] = "class"
            binary["threshold_low_nM"] = 10_000.0
            binary["threshold_high_nM"] = 30_000.0
            binary["threshold_source_value_nM"] = binary_values.to_numpy()
            binary["threshold_policy"] = (
                "exact IC50 only: blocker<=10000 nM; nonblocker>=30000 nM; "
                "intermediate and censored rows excluded"
            )
            admitted = pd.concat([admitted, binary], ignore_index=True, sort=False)
    columns = list(TABLE_REQUIRED_COLUMNS["tasks"])
    extras = [
        "evidence_domain",
        "endpoint",
        "endpoint_family",
        "document_year",
        "quality_grade",
        "label_text",
        "value_provenance",
        "assay_family",
        "inclusion_status",
        "sensitivity_task_eligible",
        "activity_origin_id",
        "activity_origin_name",
        "document_id",
        "document_doi",
        "document_pubmed_id",
        "document_patent_id",
        "threshold_low_nM",
        "threshold_high_nM",
        "threshold_source_value_nM",
        "threshold_policy",
    ]
    for column in extras:
        if column not in admitted.columns:
            admitted[column] = ""
    return (
        admitted[columns + extras]
        .sort_values(["task_type", "observation_id"], kind="stable")
        .reset_index(drop=True)
    )


def _join_model_inputs(
    tasks: pd.DataFrame,
    molecules: pd.DataFrame,
    proteins: pd.DataFrame,
    assays: pd.DataFrame,
    derivations: pd.DataFrame,
) -> pd.DataFrame:
    joined = (
        tasks.merge(
            molecules[
                [
                    "molecule_id",
                    "standardized_smiles",
                    "canonical_smiles",
                    "standard_inchi_key",
                    "structure_id",
                ]
            ],
            on="molecule_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            proteins[["protein_id", "canonical_target_id", "target_name", "sequence", "species"]],
            on="protein_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            assays[["assay_id", "description", "matrix", "route"]],
            on="assay_id",
            how="left",
            validate="many_to_one",
        )
    )
    if not derivations.empty:
        joined = joined.merge(
            derivations[["observation_id", "label_lineage_digest"]],
            on="observation_id",
            how="left",
            validate="many_to_one",
        )
    else:
        joined["label_lineage_digest"] = ""
    joined["label_lineage_digest"] = joined["label_lineage_digest"].fillna("")
    return joined


def build_public_platform_dataset(
    project_root: str | os.PathLike[str],
    platform_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Materialize the public ChEMBL integration corpus and endpoint tasks."""

    _ = Path(project_root).resolve()  # Kept in the API for deterministic project-root provenance.
    platform = Path(platform_root).resolve()
    raw_root = platform / "raw"
    transient_bulk_files = [
        path
        for path in (raw_root / "chembl_37_bulk").rglob("*")
        if path.is_file() and (path.name.startswith(".") or path.suffix == ".part")
    ]
    if transient_bulk_files:
        raise RuntimeError("A ChEMBL bulk write/extraction is in progress; canonical build refused")
    canonical_root = platform / "canonical"
    reports_root = platform.parent.parent / "reports" / "platform"
    reports_root.mkdir(parents=True, exist_ok=True)
    registry_all = source_registry(raw_root)
    sources = registry_all[
        (registry_all["source_name"] == "ChEMBL") & (registry_all["access_class"] == PUBLIC_ACCESS_CLASS)
    ].reset_index(drop=True)
    if sources.empty or set(sources["access_class"]) != {PUBLIC_ACCESS_CLASS}:
        raise RuntimeError("Public canonical build requires rights-verified ChEMBL source snapshots")
    source_files = source_file_inventory(raw_root, registry_all)
    source_files = source_files[source_files["source_id"] == CHEMBL_SOURCE_ID].reset_index(drop=True)
    rows, dedup_stats = load_chembl_integration_rows(
        raw_root,
        registry=registry_all,
        file_inventory=source_files,
    )
    target_metadata = _target_metadata(raw_root)
    proteins, target_to_protein = _protein_entities(rows, target_metadata)
    molecules, aliases, compound_to_molecule, conflicting_compounds = _molecule_entities(rows)
    assays, assay_to_id = _assay_entities(rows, target_to_protein)
    observations = _observations(
        rows,
        compound_to_molecule,
        conflicting_compounds,
        target_to_protein,
        assay_to_id,
        assays,
    )
    derivations, derived_observations = binding_free_energy_view(observations, assays, proteins)
    if not derived_observations.empty:
        observations = pd.concat([observations, derived_observations], ignore_index=True, sort=False)
    observations = observations.sort_values("observation_id", kind="stable").reset_index(drop=True)
    observation_lineage = _observation_lineage(rows, source_files, derivations)
    if observations["observation_kind"].map(clean_text).eq("").any():
        raise RuntimeError("Blank observation_kind is prohibited")
    if observations["observation_kind"].eq("prediction").any():
        raise RuntimeError("Prediction rows are prohibited in the public canonical build")
    if set(observations["access_class"]) != {PUBLIC_ACCESS_CLASS}:
        raise RuntimeError("A non-public access class reached the public canonical build")
    constructs = pd.DataFrame(
        columns=list(TABLE_REQUIRED_COLUMNS["protein_constructs"])
        + ["construct_description", "quality_status"]
    )
    tasks = _task_view(observations, assays)
    derived_tasks = _task_view(observations, assays, derived_sensitivity=True)
    model_tasks = _join_model_inputs(tasks, molecules, proteins, assays, derivations)
    derived_model_tasks = _join_model_inputs(
        derived_tasks,
        molecules,
        proteins,
        assays,
        derivations,
    )
    _validate_observation_lineage(observations, observation_lineage, sources, source_files)
    task_registry = _task_registry(pd.concat([tasks, derived_tasks], ignore_index=True, sort=False))

    tables = {
        "sources": sources,
        "source_files": source_files,
        "observation_lineage": observation_lineage,
        "molecules": molecules,
        "molecule_aliases": aliases,
        "proteins": proteins,
        "protein_constructs": constructs,
        "assays": assays,
        "observations": observations,
        "tasks": tasks,
    }
    contract_issues = {name: validate_table(name, frame) for name, frame in tables.items()}
    contract_issues = {name: issues for name, issues in contract_issues.items() if issues}
    if contract_issues:
        raise RuntimeError(f"Canonical schema validation failed: {canonical_json(contract_issues)}")

    artifacts: dict[str, dict[str, Any]] = {}
    for name, frame in tables.items():
        artifacts[name] = _atomic_frame(frame, canonical_root / f"{name}.parquet")
    artifacts["model_task_view"] = _atomic_frame(
        model_tasks, canonical_root / "tasks" / "public_model_tasks.parquet"
    )
    artifacts["derived_sensitivity_task_view"] = _atomic_frame(
        derived_model_tasks,
        canonical_root / "tasks" / "binding_free_energy_sensitivity.parquet",
    )
    artifacts["task_registry"] = _atomic_frame(
        task_registry,
        canonical_root / "task_registry.parquet",
    )
    _atomic_frame(task_registry, canonical_root / "task_registry.csv")
    artifacts["binding_free_energy"] = _atomic_frame(
        derivations,
        canonical_root / "views" / "binding_free_energy_standard.parquet",
    )
    _atomic_frame(data_dictionary_frame(), canonical_root / "data_dictionary.csv")
    _atomic_json(canonical_root / "schema.json", schema_document())

    views = {
        "public_all": observations,
        "preclinical_reported": observations[
            observations["evidence_stage"].astype("string").str.startswith("preclinical_", na=False)
            & observations["result_status"].eq("reported")
        ],
        "clinical_reported": observations[
            observations["evidence_stage"].eq("clinical_results")
            & observations["result_status"].eq("reported")
        ],
        "herg": observations[observations["evidence_domain"].eq("herg")],
        "qt_apd_explicit": observations[observations["evidence_domain"].eq("qt")],
        "pk_adme": observations[observations["evidence_domain"].eq("pk_adme")],
        "binding": observations[observations["evidence_domain"].eq("binding")],
    }
    view_records: dict[str, Any] = {}
    for name, frame in views.items():
        record = _atomic_frame(frame, canonical_root / "views" / f"{name}.parquet")
        record["definition"] = {
            "clinical_reported": "explicit clinical_results evidence stage plus reported status; no trial-absence inference",
            "preclinical_reported": "explicitly classified preclinical evidence stage plus a reported result",
            "herg": "hERG-domain observations only; not QT/TdP/cardiotoxicity",
            "qt_apd_explicit": "explicit QT/QTc/APD naming only; never inferred from hERG",
        }.get(name, "lossless domain/access projection from canonical observations")
        view_records[name] = record
    task_records: dict[str, Any] = {}
    for task_type, frame in model_tasks.groupby("task_type", sort=True):
        slug = re.sub(r"[^a-z0-9]+", "_", task_type.casefold()).strip("_")
        task_records[task_type] = _atomic_frame(frame, canonical_root / "tasks" / f"{slug}.parquet")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "build_type": "public_chembl37_heterogeneous_integration_corpus",
        "built_at_utc": _utc_now(),
        "rights_gate": "ChEMBL-only; BindingDB/PubChem/Sun/internal artifacts excluded fail-closed",
        "completeness_boundary": (
            "Executed canonical artifacts use targeted plus bounded heterogeneous REST snapshots; "
            "the separately verified official bulk archive/export manifest governs full-release readiness."
        ),
        "source_id": CHEMBL_SOURCE_ID,
        "deduplication": dedup_stats,
        "table_artifacts": artifacts,
        "view_artifacts": view_records,
        "task_artifacts": task_records,
        "counts": {name: len(frame) for name, frame in tables.items()},
        "observation_kind_counts": observations["observation_kind"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict(),
        "access_class_counts": observations["access_class"].value_counts(dropna=False).sort_index().to_dict(),
        "domain_counts": observations["evidence_domain"].value_counts(dropna=False).sort_index().to_dict(),
        "inclusion_counts": observations["inclusion_status"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict(),
        "quality_counts": observations["quality_grade"].value_counts(dropna=False).sort_index().to_dict(),
        "binding_free_energy_policy": {
            "input_endpoint": "exact positive Kd only",
            "excluded_endpoints": ["Ki", "IC50", "EC50"],
            "formula": "delta_g_kcal_mol=R_kcal_mol_K*T_K*ln(Kd_M/1_M)",
            "standard_state": "1 mol/L",
            "reference_temperature_k_when_unreported": REFERENCE_TEMPERATURE_K,
            "reference_temperature_rows_are_labeled_approximations": True,
            "roundtrip_max_relative_error": float(derivations["roundtrip_relative_error"].max())
            if not derivations.empty
            else None,
        },
    }
    _atomic_json(canonical_root / "build_manifest.json", manifest)
    _atomic_json(reports_root / "data_build_manifest.json", manifest)
    _atomic_json(reports_root / "dedup_summary.json", dedup_stats)
    _atomic_frame(registry_all, reports_root / "source_registry.csv")
    return manifest


def load_public_model_task_view(platform_root: str | os.PathLike[str]) -> pd.DataFrame:
    """Load and recheck the stable joined task contract consumed by MODEL."""

    path = Path(platform_root).resolve() / "canonical" / "tasks" / "public_model_tasks.parquet"
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError("Public model task view is empty")
    if set(frame["access_class"]) != {PUBLIC_ACCESS_CLASS}:
        raise ValueError("Public task view contains a non-public access class")
    if (
        frame["observation_kind"].map(clean_text).eq("").any()
        or frame["observation_kind"].eq("prediction").any()
    ):
        raise ValueError("Public task view has blank/prediction observation kinds")
    eligible = frame.get("default_task_eligible", pd.Series(False, index=frame.index))
    if not eligible.map(lambda value: isinstance(value, (bool, np.bool_)) and bool(value)).all():
        raise ValueError("Public task view contains a default-ineligible row")
    return frame


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the rights-gated public platform data corpus")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--platform-root", type=Path, default=Path("research/data/platform"))
    arguments = parser.parse_args(argv)
    manifest = build_public_platform_dataset(arguments.project_root, arguments.platform_root)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CHEMBL_SOURCE_ID",
    "GAS_CONSTANT_KCAL_MOL_K",
    "PUBLIC_ACCESS_CLASS",
    "REFERENCE_TEMPERATURE_K",
    "binding_free_energy_view",
    "build_public_platform_dataset",
    "load_chembl_integration_rows",
    "load_public_model_task_view",
    "main",
]
