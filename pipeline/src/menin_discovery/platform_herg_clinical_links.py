"""Conservative molecule links for clinical cardiac-evidence candidates.

This module joins a structure-resolved hERG molecule inventory to ChEMBL
development metadata and to the pre-canonical ClinicalTrials.gov result
inventories.  It is deliberately fail-closed:

* ChEMBL ``max_phase`` and Drugs@FDA application membership are development
  annotations, never cardiac-safety labels.
* ClinicalTrials.gov absence, a registered endpoint, or a study phase is never
  interpreted as a negative result.
* A trial candidate is emitted only when a genuine posted QT/QTc result has a
  numeric reported value and the study has exactly one non-placebo drug
  intervention, resolved by a unique exact normalized name to one molecule.
* Every output remains a review candidate.  The module assigns no clinical
  hERG, QT, torsade, efficacy, or safety label and admits no model target.

Name normalization is intentionally narrow (Unicode NFKC, case-folding, and
whitespace collapse).  It does not strip salts, punctuation, stereochemistry,
or formulation language.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
import sqlite3
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem

SCHEMA_VERSION = "platform-herg-clinical-links/1.0"
PARSER_VERSION = "platform_herg_clinical_links/1.0"
ABSENCE_SEMANTICS = "not_reported_or_not_present_is_unknown_not_negative"
EVIDENCE_SEMANTICS = "candidate_clinical_cardiac_evidence_not_herg_assay_or_safety_label"
ADMISSION_STATUS = "candidate_only_manual_review_required_not_canonical_not_model_label"
MANIFEST_NAME = "herg_clinical_links_manifest.json"

STRUCTURE_OUTPUT = "structure_development_annotations.parquet"
LINK_AUDIT_OUTPUT = "exact_name_structure_link_audit.parquet"
T2_OUTPUT = "t2_clinical_cardiac_evidence_candidates.parquet"
T3_OUTPUT = "t3_posted_qt_trial_result_candidates.parquet"

csv.field_size_limit(1024 * 1024 * 1024)


class HergClinicalLinkError(RuntimeError):
    """Raised when an input or fail-closed linkage contract is violated."""


STRUCTURE_SCHEMA = pa.schema(
    [
        pa.field("molecule_id", pa.string(), nullable=False),
        pa.field("standard_inchi_key", pa.string()),
        pa.field("canonical_smiles", pa.string()),
        pa.field("chembl_molecule_ids_json", pa.string(), nullable=False),
        pa.field("chembl_exact_structure_match_count", pa.int64(), nullable=False),
        pa.field("chembl_max_phase", pa.float64()),
        pa.field("chembl_first_approval", pa.int64()),
        pa.field("chembl_therapeutic_flag", pa.bool_(), nullable=False),
        pa.field("chembl_dosed_ingredient", pa.bool_(), nullable=False),
        pa.field("chembl_withdrawn_flag", pa.bool_(), nullable=False),
        pa.field("drugsfda_application_numbers_json", pa.string(), nullable=False),
        pa.field("drugsfda_exact_name_link_count", pa.int64(), nullable=False),
        pa.field("clinical_development_annotation", pa.bool_(), nullable=False),
        pa.field("development_annotation_semantics", pa.string(), nullable=False),
        pa.field("clinical_cardiac_label_admitted", pa.bool_(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
    ]
)

LINK_AUDIT_SCHEMA = pa.schema(
    [
        pa.field("source_kind", pa.string(), nullable=False),
        pa.field("source_record_id", pa.string(), nullable=False),
        pa.field("nct_id", pa.string()),
        pa.field("application_number", pa.string()),
        pa.field("product_number", pa.string()),
        pa.field("raw_name", pa.string(), nullable=False),
        pa.field("normalized_name", pa.string(), nullable=False),
        pa.field("intervention_type", pa.string()),
        pa.field("combination_or_non_drug_name", pa.bool_(), nullable=False),
        pa.field("candidate_molecule_ids_json", pa.string(), nullable=False),
        pa.field("candidate_molecule_count", pa.int64(), nullable=False),
        pa.field("linked_molecule_id", pa.string()),
        pa.field("link_method", pa.string(), nullable=False),
        pa.field("link_state", pa.string(), nullable=False),
        pa.field("link_is_exact_and_unique", pa.bool_(), nullable=False),
        pa.field("linkage_semantics", pa.string(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
    ]
)

CLINICAL_SCHEMA = pa.schema(
    [
        pa.field("candidate_id", pa.string(), nullable=False),
        pa.field("molecule_id", pa.string(), nullable=False),
        pa.field("nct_id", pa.string(), nullable=False),
        pa.field("endpoint_candidate_id", pa.string(), nullable=False),
        pa.field("parent_candidate_id", pa.string()),
        pa.field("record_kind", pa.string(), nullable=False),
        pa.field("candidate_classification", pa.string(), nullable=False),
        pa.field("title_or_term", pa.string()),
        pa.field("description_or_organ_system", pa.string()),
        pa.field("unit_of_measure", pa.string()),
        pa.field("time_frame", pa.string()),
        pa.field("denominator_records_json", pa.string(), nullable=False),
        pa.field("value_records_json", pa.string(), nullable=False),
        pa.field("evidence_phrases_json", pa.string(), nullable=False),
        pa.field("reported_numeric_value_count", pa.int64(), nullable=False),
        pa.field("study_has_posted_results", pa.bool_(), nullable=False),
        pa.field("unique_nonplacebo_drug_name_count", pa.int64(), nullable=False),
        pa.field("exact_unique_molecule_link", pa.bool_(), nullable=False),
        pa.field("actual_qt_result_present", pa.bool_(), nullable=False),
        pa.field("candidate_rule_passed", pa.bool_(), nullable=False),
        pa.field("tier_assignment_status", pa.string(), nullable=False),
        pa.field("absence_semantics", pa.string(), nullable=False),
        pa.field("evidence_semantics", pa.string(), nullable=False),
        pa.field("clinical_herg_label_admitted", pa.bool_(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
        pa.field("source_page_path", pa.string()),
        pa.field("source_page_sha256", pa.string()),
        pa.field("raw_json_pointer", pa.string()),
    ]
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonempty(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_exact_name(value: str) -> str:
    """Return a deliberately conservative name-normalization key."""

    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


_COMBINATION_RE = re.compile(r"(?:\s*/\s*|\s*\+\s*|\s+and\s+|\s+with\s+)", re.IGNORECASE)
_NON_DRUG_NAMES = frozenset({"placebo", "vehicle", "standard of care", "usual care", "no treatment"})


def _combination_or_non_drug_name(raw_name: str) -> bool:
    normalized = normalize_exact_name(raw_name)
    return normalized in _NON_DRUG_NAMES or bool(_COMBINATION_RE.search(raw_name))


def _first_field(row: Mapping[str, Any], names: Sequence[str]) -> str | None:
    for name in names:
        value = _nonempty(row.get(name))
        if value is not None:
            return value
    return None


def _structure_key_from_smiles(smiles: str) -> tuple[str, str] | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    try:
        key = Chem.MolToInchiKey(molecule)
    except Exception:  # pragma: no cover - RDKit build-specific InChI failure
        return None
    if not key:
        return None
    return key, Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _read_tabular(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise HergClinicalLinkError(f"input must be a regular non-symlink file: {path}")
    suffixes = path.suffixes
    if path.suffix == ".parquet":
        return pq.read_table(path).to_pylist()
    if suffixes[-2:] == [".csv", ".gz"]:
        with gzip.open(path, mode="rt", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if path.suffix == ".csv":
        with path.open(mode="rt", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise HergClinicalLinkError(f"unsupported tabular input: {path}")


def _parse_aliases(row: Mapping[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for field in ("preferred_name", "pref_name", "compound_name", "molecule_name", "name"):
        value = _nonempty(row.get(field))
        if value:
            aliases.add(value)
    encoded = row.get("aliases_json")
    if encoded:
        try:
            parsed = json.loads(str(encoded)) if isinstance(encoded, str) else encoded
        except json.JSONDecodeError as exc:
            raise HergClinicalLinkError("invalid aliases_json in structure consensus") from exc
        values: Iterable[Any]
        if isinstance(parsed, Mapping):
            values = parsed.values()
        elif isinstance(parsed, list):
            values = parsed
        else:
            raise HergClinicalLinkError("aliases_json must contain a list or object")
        for value in values:
            if isinstance(value, list):
                aliases.update(str(item) for item in value if _nonempty(item))
            elif _nonempty(value):
                aliases.add(str(value))
    return aliases


def _load_consensus(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(_read_tabular(path), start=1):
        molecule_id = _first_field(row, ("molecule_id", "parent_molecule_id", "compound_id", "structure_id"))
        if molecule_id is None:
            raise HergClinicalLinkError(f"structure consensus row {row_number} has no molecule identifier")
        key = _first_field(row, ("parent_inchi_key", "standard_inchi_key", "inchi_key"))
        smiles = _first_field(
            row,
            ("parent_smiles", "canonical_smiles", "standardized_smiles", "smiles", "raw_smiles"),
        )
        canonical_smiles = smiles
        if key is None and smiles is not None:
            derived = _structure_key_from_smiles(smiles)
            if derived:
                key, canonical_smiles = derived
        if key is None and smiles is None:
            raise HergClinicalLinkError(f"structure consensus row {row_number} has no structure")
        current = records.get(molecule_id)
        aliases = _parse_aliases(row)
        if current is None:
            records[molecule_id] = {
                "molecule_id": molecule_id,
                "standard_inchi_key": key,
                "canonical_smiles": canonical_smiles,
                "aliases": aliases,
            }
            continue
        if current["standard_inchi_key"] and key and current["standard_inchi_key"] != key:
            raise HergClinicalLinkError(f"molecule {molecule_id} has conflicting InChIKeys")
        current["standard_inchi_key"] = current["standard_inchi_key"] or key
        current["canonical_smiles"] = current["canonical_smiles"] or canonical_smiles
        current["aliases"].update(aliases)
    if not records:
        raise HergClinicalLinkError("structure consensus is empty")
    return records


def _chembl_metadata(
    database: Path, molecules: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, set[str]]]:
    if not database.is_file() or database.is_symlink():
        raise HergClinicalLinkError(f"ChEMBL database must be a regular non-symlink file: {database}")
    key_to_ids: dict[str, list[str]] = defaultdict(list)
    for molecule_id, row in molecules.items():
        key = _nonempty(row.get("standard_inchi_key"))
        if key:
            key_to_ids[key].append(molecule_id)

    metadata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    aliases: dict[str, set[str]] = defaultdict(set)
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        all_keys = list(key_to_ids)
        for offset in range(0, len(all_keys), 800):
            keys = all_keys[offset : offset + 800]
            placeholders = ",".join("?" for _ in keys)
            query = f"""
                SELECT cs.standard_inchi_key, md.molregno, md.chembl_id, md.pref_name,
                       md.max_phase, md.first_approval, md.therapeutic_flag,
                       md.dosed_ingredient, md.withdrawn_flag
                FROM compound_structures cs
                JOIN molecule_dictionary md ON md.molregno = cs.molregno
                WHERE cs.standard_inchi_key IN ({placeholders})
                ORDER BY cs.standard_inchi_key, md.chembl_id
            """
            for result in connection.execute(query, keys):
                key, molregno, chembl_id, pref_name, *values = result
                item = {
                    "molregno": int(molregno),
                    "chembl_id": str(chembl_id),
                    "max_phase": values[0],
                    "first_approval": values[1],
                    "therapeutic_flag": values[2],
                    "dosed_ingredient": values[3],
                    "withdrawn_flag": values[4],
                }
                for molecule_id in key_to_ids[str(key)]:
                    metadata[molecule_id].append(item)
                    if _nonempty(pref_name):
                        aliases[molecule_id].add(str(pref_name))

        molregno_to_ids: dict[int, list[str]] = defaultdict(list)
        for molecule_id, rows in metadata.items():
            for row in rows:
                molregno_to_ids[int(row["molregno"])].append(molecule_id)
        all_molregnos = list(molregno_to_ids)
        for offset in range(0, len(all_molregnos), 800):
            molregnos = all_molregnos[offset : offset + 800]
            placeholders = ",".join("?" for _ in molregnos)
            query = f"""
                SELECT molregno, synonyms
                FROM molecule_synonyms
                WHERE molregno IN ({placeholders}) AND synonyms IS NOT NULL
                ORDER BY molregno, synonyms
            """
            for molregno, synonym in connection.execute(query, molregnos):
                if _nonempty(synonym):
                    for molecule_id in molregno_to_ids[int(molregno)]:
                        aliases[molecule_id].add(str(synonym))
    except sqlite3.DatabaseError as exc:
        raise HergClinicalLinkError(f"invalid or incompatible ChEMBL SQLite database: {exc}") from exc
    finally:
        connection.close()
    return metadata, aliases


def _unique_name_index(
    molecules: Mapping[str, Mapping[str, Any]], chembl_aliases: Mapping[str, set[str]]
) -> dict[str, tuple[str, ...]]:
    index: dict[str, set[str]] = defaultdict(set)
    for molecule_id, row in molecules.items():
        names = set(row["aliases"])
        names.update(chembl_aliases.get(molecule_id, set()))
        for name in names:
            normalized = normalize_exact_name(name)
            if normalized:
                index[normalized].add(molecule_id)
    return {key: tuple(sorted(values)) for key, values in sorted(index.items())}


def _link_record(
    *,
    source_kind: str,
    source_record_id: str,
    raw_name: str,
    name_index: Mapping[str, tuple[str, ...]],
    nct_id: str | None = None,
    application_number: str | None = None,
    product_number: str | None = None,
    intervention_type: str | None = None,
    combination_override: bool = False,
) -> dict[str, Any]:
    normalized = normalize_exact_name(raw_name)
    blocked = combination_override or _combination_or_non_drug_name(raw_name)
    candidates = () if blocked else name_index.get(normalized, ())
    if blocked:
        state = "rejected_combination_or_non_drug_name"
    elif not candidates:
        state = "unresolved_no_exact_name_match"
    elif len(candidates) > 1:
        state = "rejected_ambiguous_exact_name"
    else:
        state = "linked_unique_exact_normalized_name"
    linked = candidates[0] if len(candidates) == 1 and not blocked else None
    return {
        "source_kind": source_kind,
        "source_record_id": source_record_id,
        "nct_id": nct_id,
        "application_number": application_number,
        "product_number": product_number,
        "raw_name": raw_name,
        "normalized_name": normalized,
        "intervention_type": intervention_type,
        "combination_or_non_drug_name": blocked,
        "candidate_molecule_ids_json": _canonical_json(list(candidates)),
        "candidate_molecule_count": len(candidates),
        "linked_molecule_id": linked,
        "link_method": "unicode_nfkc_casefold_whitespace_exact_unique",
        "link_state": state,
        "link_is_exact_and_unique": linked is not None,
        "linkage_semantics": "identity_candidate_only_no_outcome_or_causality_inference",
        "model_label_admitted": False,
    }


def _read_gzip_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.is_symlink():
        raise HergClinicalLinkError(f"required ClinicalTrials table missing or unsafe: {path}")
    with gzip.open(path, mode="rt", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _json_list(value: str, *, field: str) -> list[Any]:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError as exc:
        raise HergClinicalLinkError(f"invalid JSON in {field}") from exc
    if not isinstance(parsed, list):
        raise HergClinicalLinkError(f"{field} must encode a list")
    return parsed


def _numeric_value_count(encoded: str) -> int:
    count = 0
    for record in _json_list(encoded, field="value_records_json"):
        if not isinstance(record, Mapping):
            continue
        raw = record.get("value")
        try:
            value = float(str(raw).strip())
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            count += 1
    return count


def _true(value: Any) -> bool:
    return str(value).strip().casefold() == "true"


def _write_parquet(path: Path, schema: pa.Schema, rows: Sequence[Mapping[str, Any]]) -> None:
    table = pa.Table.from_pylist(list(rows), schema=schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
    )


def _artifact(path: Path, rows: int) -> dict[str, Any]:
    return {"path": path.name, "rows": rows, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def build_herg_clinical_links(
    structure_consensus: str | os.PathLike[str],
    chembl_sqlite: str | os.PathLike[str],
    clinical_results_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    drugsfda_ingredients: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Build deterministic, candidate-only development and QT link inventories."""

    consensus_path = Path(structure_consensus).resolve()
    chembl_path = Path(chembl_sqlite).resolve()
    clinical_root = Path(clinical_results_root).resolve()
    output = Path(output_root).resolve()
    if output.exists() and (output.is_symlink() or not output.is_dir() or any(output.iterdir())):
        raise HergClinicalLinkError("output directory must be absent or empty and may not be a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)

    molecules = _load_consensus(consensus_path)
    chembl_metadata, chembl_aliases = _chembl_metadata(chembl_path, molecules)
    name_index = _unique_name_index(molecules, chembl_aliases)

    studies = _read_gzip_csv(clinical_root / "studies.csv.gz")
    interventions = _read_gzip_csv(clinical_root / "interventions.csv.gz")
    endpoints = _read_gzip_csv(clinical_root / "endpoint_candidates.csv.gz")
    study_has_results = {row["nct_id"]: _true(row.get("has_results_reported")) for row in studies}

    link_rows: list[dict[str, Any]] = []
    trial_links: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interventions:
        raw_name = _nonempty(row.get("intervention_name")) or ""
        record_id = (
            _nonempty(row.get("intervention_candidate_id"))
            or hashlib.sha256(_canonical_json(row).encode("utf-8")).hexdigest()
        )
        link = _link_record(
            source_kind="clinicaltrials_intervention",
            source_record_id=record_id,
            raw_name=raw_name,
            name_index=name_index,
            nct_id=_nonempty(row.get("nct_id")),
            intervention_type=_nonempty(row.get("intervention_type")),
            combination_override=_nonempty(row.get("intervention_type")) != "DRUG",
        )
        link_rows.append(link)
        trial_links[row["nct_id"]].append(link)

    fda_applications: dict[str, set[str]] = defaultdict(set)
    ingredient_path: Path | None = None
    if drugsfda_ingredients is not None:
        ingredient_path = Path(drugsfda_ingredients).resolve()
        ingredients = _read_tabular(ingredient_path)
        product_components: dict[tuple[str, str], int] = defaultdict(int)
        for row in ingredients:
            product_components[
                (str(row.get("application_number", "")), str(row.get("product_number", "")))
            ] += 1
        for row in ingredients:
            application = _nonempty(row.get("application_number")) or ""
            product = _nonempty(row.get("product_number")) or ""
            raw_name = _first_field(row, ("ingredient_component_exact", "active_ingredient_raw")) or ""
            record_id = (
                _first_field(row, ("ingredient_candidate_key", "source_field_map_sha256"))
                or hashlib.sha256(_canonical_json(row).encode("utf-8")).hexdigest()
            )
            link = _link_record(
                source_kind="drugsfda_ingredient",
                source_record_id=f"{application}:{product}:{record_id}",
                raw_name=raw_name,
                name_index=name_index,
                application_number=application,
                product_number=product,
                combination_override=product_components[(application, product)] > 1,
            )
            link_rows.append(link)
            if link["linked_molecule_id"]:
                fda_applications[str(link["linked_molecule_id"])].add(application)

    link_rows.sort(
        key=lambda row: (
            str(row["source_kind"]),
            str(row.get("nct_id") or ""),
            str(row.get("application_number") or ""),
            str(row.get("product_number") or ""),
            str(row["source_record_id"]),
        )
    )

    structure_rows: list[dict[str, Any]] = []
    for molecule_id in sorted(molecules):
        molecule = molecules[molecule_id]
        matches = chembl_metadata.get(molecule_id, [])
        phases = [float(row["max_phase"]) for row in matches if row["max_phase"] is not None]
        approvals = [int(row["first_approval"]) for row in matches if row["first_approval"] is not None]
        applications = sorted(fda_applications.get(molecule_id, set()))
        structure_rows.append(
            {
                "molecule_id": molecule_id,
                "standard_inchi_key": molecule["standard_inchi_key"],
                "canonical_smiles": molecule["canonical_smiles"],
                "chembl_molecule_ids_json": _canonical_json(
                    sorted({str(row["chembl_id"]) for row in matches})
                ),
                "chembl_exact_structure_match_count": len(matches),
                "chembl_max_phase": max(phases) if phases else None,
                "chembl_first_approval": min(approvals) if approvals else None,
                "chembl_therapeutic_flag": any(int(row["therapeutic_flag"] or 0) == 1 for row in matches),
                "chembl_dosed_ingredient": any(int(row["dosed_ingredient"] or 0) == 1 for row in matches),
                "chembl_withdrawn_flag": any(int(row["withdrawn_flag"] or 0) == 1 for row in matches),
                "drugsfda_application_numbers_json": _canonical_json(applications),
                "drugsfda_exact_name_link_count": len(applications),
                "clinical_development_annotation": bool(phases or approvals or applications),
                "development_annotation_semantics": (
                    "development_or_regulatory_metadata_not_cardiac_validation_or_safety_label"
                ),
                "clinical_cardiac_label_admitted": False,
                "model_label_admitted": False,
            }
        )

    clinical_rows: list[dict[str, Any]] = []
    for endpoint in endpoints:
        if endpoint.get("target_domain") != "qt_qtc" or not _true(endpoint.get("genuine_endpoint_candidate")):
            continue
        nct_id = endpoint["nct_id"]
        links = trial_links.get(nct_id, [])
        drug_links = [link for link in links if link.get("intervention_type") == "DRUG"]
        active_drug_links = [
            link for link in drug_links if str(link["normalized_name"]) not in _NON_DRUG_NAMES
        ]
        nonplacebo_names = {str(link["normalized_name"]) for link in active_drug_links}
        all_drugs_exact = bool(active_drug_links) and all(
            bool(link["link_is_exact_and_unique"]) for link in active_drug_links
        )
        linked_molecules = {
            str(link["linked_molecule_id"])
            for link in active_drug_links
            if link["linked_molecule_id"] is not None
        }
        exact_unique = (
            all_drugs_exact
            and len(nonplacebo_names) == 1
            and len(linked_molecules) == 1
            and not any(bool(link["combination_or_non_drug_name"]) for link in active_drug_links)
        )
        numeric_count = _numeric_value_count(endpoint.get("value_records_json", "[]"))
        posted = bool(study_has_results.get(nct_id, False))
        actual_result = numeric_count > 0
        if not (exact_unique and posted and actual_result):
            continue
        molecule_id = next(iter(linked_molecules))
        endpoint_id = endpoint["endpoint_candidate_id"]
        candidate_id = hashlib.sha256(f"{molecule_id}\0{nct_id}\0{endpoint_id}".encode()).hexdigest()
        clinical_rows.append(
            {
                "candidate_id": candidate_id,
                "molecule_id": molecule_id,
                "nct_id": nct_id,
                "endpoint_candidate_id": endpoint_id,
                "parent_candidate_id": _nonempty(endpoint.get("parent_candidate_id")),
                "record_kind": endpoint.get("record_kind", ""),
                "candidate_classification": endpoint.get("candidate_classification", ""),
                "title_or_term": _nonempty(endpoint.get("title_or_term")),
                "description_or_organ_system": _nonempty(endpoint.get("description_or_organ_system")),
                "unit_of_measure": _nonempty(endpoint.get("unit_of_measure")),
                "time_frame": _nonempty(endpoint.get("time_frame")),
                "denominator_records_json": _canonical_json(
                    _json_list(
                        endpoint.get("denominator_records_json", "[]"), field="denominator_records_json"
                    )
                ),
                "value_records_json": _canonical_json(
                    _json_list(endpoint.get("value_records_json", "[]"), field="value_records_json")
                ),
                "evidence_phrases_json": _canonical_json(
                    _json_list(endpoint.get("evidence_phrases_json", "[]"), field="evidence_phrases_json")
                ),
                "reported_numeric_value_count": numeric_count,
                "study_has_posted_results": posted,
                "unique_nonplacebo_drug_name_count": len(nonplacebo_names),
                "exact_unique_molecule_link": exact_unique,
                "actual_qt_result_present": actual_result,
                "candidate_rule_passed": True,
                "tier_assignment_status": "candidate_only_not_assigned_pending_scientific_review",
                "absence_semantics": ABSENCE_SEMANTICS,
                "evidence_semantics": EVIDENCE_SEMANTICS,
                "clinical_herg_label_admitted": False,
                "model_label_admitted": False,
                "source_page_path": _nonempty(endpoint.get("source_page_path")),
                "source_page_sha256": _nonempty(endpoint.get("source_page_sha256")),
                "raw_json_pointer": _nonempty(endpoint.get("raw_json_pointer")),
            }
        )
    clinical_rows.sort(
        key=lambda row: (str(row["molecule_id"]), str(row["nct_id"]), str(row["candidate_id"]))
    )

    output.mkdir(parents=True, exist_ok=True)
    _write_parquet(output / STRUCTURE_OUTPUT, STRUCTURE_SCHEMA, structure_rows)
    _write_parquet(output / LINK_AUDIT_OUTPUT, LINK_AUDIT_SCHEMA, link_rows)
    _write_parquet(output / T2_OUTPUT, CLINICAL_SCHEMA, clinical_rows)
    _write_parquet(output / T3_OUTPUT, CLINICAL_SCHEMA, clinical_rows)

    artifacts = [
        _artifact(output / STRUCTURE_OUTPUT, len(structure_rows)),
        _artifact(output / LINK_AUDIT_OUTPUT, len(link_rows)),
        _artifact(output / T2_OUTPUT, len(clinical_rows)),
        _artifact(output / T3_OUTPUT, len(clinical_rows)),
    ]
    source_inputs = [
        consensus_path,
        chembl_path,
        *(
            clinical_root / name
            for name in ("studies.csv.gz", "interventions.csv.gz", "endpoint_candidates.csv.gz")
        ),
    ]
    if ingredient_path is not None:
        source_inputs.append(ingredient_path)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "input_bindings": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in source_inputs
        ],
        "row_counts": {
            STRUCTURE_OUTPUT: len(structure_rows),
            LINK_AUDIT_OUTPUT: len(link_rows),
            T2_OUTPUT: len(clinical_rows),
            T3_OUTPUT: len(clinical_rows),
        },
        "artifacts": artifacts,
        "rules": {
            "identity": "exact structure for ChEMBL; unique NFKC-casefold-whitespace exact name for trial/regulatory text",
            "study_attribution": "exactly one non-placebo DRUG intervention name and exactly one linked molecule",
            "cardiac_result": "genuine QT/QTc candidate, posted results study, and at least one finite numeric reported value",
            "combination_policy": "reject combination-like trial names and multi-component Drugs@FDA products",
            "absence_semantics": ABSENCE_SEMANTICS,
        },
        "clinical_cardiac_labels_admitted": 0,
        "model_labels_admitted": 0,
        "tier_assignments_made": 0,
        "candidate_only": True,
        "admission_status": ADMISSION_STATUS,
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
    (output / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def verify_herg_clinical_links(output_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Verify hashes, schemas, counts, and zero-label contracts."""

    output = Path(output_root).resolve()
    manifest_path = output / MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise HergClinicalLinkError("manifest missing or unsafe")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.pop("manifest_sha256", None)
    actual = hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
    if declared != actual:
        raise HergClinicalLinkError("manifest internal SHA-256 mismatch")
    expected_schemas = {
        STRUCTURE_OUTPUT: STRUCTURE_SCHEMA,
        LINK_AUDIT_OUTPUT: LINK_AUDIT_SCHEMA,
        T2_OUTPUT: CLINICAL_SCHEMA,
        T3_OUTPUT: CLINICAL_SCHEMA,
    }
    expected_members = {MANIFEST_NAME, *expected_schemas}
    actual_members = {path.name for path in output.iterdir()}
    if actual_members != expected_members or any(
        path.is_symlink() or not path.is_file() for path in output.iterdir()
    ):
        raise HergClinicalLinkError("clinical-link output contains unexpected or unsafe members")
    bindings = manifest.get("input_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise HergClinicalLinkError("clinical-link manifest has no input bindings")
    for binding in bindings:
        if not isinstance(binding, Mapping):
            raise HergClinicalLinkError("malformed clinical-link input binding")
        path = Path(str(binding.get("path", "")))
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(binding.get("bytes", -1))
            or _sha256_file(path) != binding.get("sha256")
        ):
            raise HergClinicalLinkError(f"clinical-link input binding mismatch: {path}")
    for artifact in manifest["artifacts"]:
        path = output / artifact["path"]
        if not path.is_file() or path.is_symlink() or _sha256_file(path) != artifact["sha256"]:
            raise HergClinicalLinkError(f"artifact hash or path verification failed: {path.name}")
        table = pq.read_table(path)
        if table.schema != expected_schemas[path.name] or table.num_rows != artifact["rows"]:
            raise HergClinicalLinkError(f"artifact schema or row count mismatch: {path.name}")
        if manifest.get("row_counts", {}).get(path.name) != table.num_rows:
            raise HergClinicalLinkError(f"manifest row count mismatch: {path.name}")
        if "model_label_admitted" in table.column_names and any(table["model_label_admitted"].to_pylist()):
            raise HergClinicalLinkError(f"model label unexpectedly admitted: {path.name}")
        if "clinical_cardiac_label_admitted" in table.column_names and any(
            table["clinical_cardiac_label_admitted"].to_pylist()
        ):
            raise HergClinicalLinkError(f"clinical cardiac label unexpectedly admitted: {path.name}")
    for name in (T2_OUTPUT, T3_OUTPUT):
        table = pq.read_table(output / name)
        for required in (
            "exact_unique_molecule_link",
            "actual_qt_result_present",
            "study_has_posted_results",
        ):
            if not all(table[required].to_pylist()):
                raise HergClinicalLinkError(f"fail-closed candidate rule violated in {name}: {required}")
        if any(table["clinical_herg_label_admitted"].to_pylist()):
            raise HergClinicalLinkError(f"clinical hERG label unexpectedly admitted: {name}")
    return {
        "verification_status": "pass",
        "manifest_internal_sha256": declared,
        "artifact_count": len(manifest["artifacts"]),
        "model_labels_admitted": 0,
        "tier_assignments_made": 0,
    }
