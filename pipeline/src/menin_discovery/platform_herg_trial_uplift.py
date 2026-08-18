"""Auditable, label-free ClinicalTrials.gov intervention structure uplift.

The builder resolves reported intervention names to ChEMBL structures in four
increasingly permissive rule tiers.  It standardizes ChEMBL hierarchy parents
with the same RDKit policy used by the hERG hierarchy, records whether each
candidate structure already has local reported hERG evidence, and deliberately
assigns no hERG, QT, clinical-safety, or model label.

Only ChEMBL preferred names and curated molecule synonyms are used.  Publication
``compound_records`` names and fuzzy/edit-distance matching are excluded because
they produced substantial ambiguity in the source-profile analysis.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
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
from rdkit.Chem.MolStandardize import rdMolStandardize

SCHEMA_VERSION = "platform-herg-trial-uplift/1.0"
PARSER_VERSION = "platform_herg_trial_uplift/1.0"
MANIFEST_NAME = "herg_trial_uplift_manifest.json"
CANDIDATE_OUTPUT = "trial_structure_link_candidates.parquet"
AUDIT_OUTPUT = "trial_structure_link_audit.parquet"
IDENTITY_SEMANTICS = "candidate_structure_identity_only_no_hERG_QT_clinical_safety_or_causality_inference"

csv.field_size_limit(1024 * 1024 * 1024)


class HergTrialUpliftError(RuntimeError):
    """Raised when an input, output, or zero-label contract is violated."""


CANDIDATE_SCHEMA = pa.schema(
    [
        pa.field("intervention_candidate_id", pa.string(), nullable=False),
        pa.field("nct_id", pa.string(), nullable=False),
        pa.field("raw_intervention_name", pa.string(), nullable=False),
        pa.field("normalized_intervention_name", pa.string(), nullable=False),
        pa.field("intervention_type", pa.string(), nullable=False),
        pa.field("study_record_present", pa.bool_(), nullable=False),
        pa.field("study_has_posted_results", pa.bool_()),
        pa.field("first_resolved_rule_tier", pa.int64(), nullable=False),
        pa.field("first_resolved_rule_name", pa.string(), nullable=False),
        pa.field("automatic_identity_link", pa.bool_(), nullable=False),
        pa.field("candidate_status", pa.string(), nullable=False),
        pa.field("matched_forms_json", pa.string(), nullable=False),
        pa.field("component_names_json", pa.string(), nullable=False),
        pa.field("candidate_parent_structures_json", pa.string(), nullable=False),
        pa.field("candidate_parent_structure_keys_json", pa.string(), nullable=False),
        pa.field("candidate_parent_structure_count", pa.int64(), nullable=False),
        pa.field("candidate_chembl_ids_json", pa.string(), nullable=False),
        pa.field("candidate_molregnos_json", pa.string(), nullable=False),
        pa.field("local_herg_structure_keys_json", pa.string(), nullable=False),
        pa.field("local_herg_structure_count", pa.int64(), nullable=False),
        pa.field("has_any_local_reported_herg_evidence", pa.bool_(), nullable=False),
        pa.field("ambiguity_preserved", pa.bool_(), nullable=False),
        pa.field("is_component_set_candidate", pa.bool_(), nullable=False),
        pa.field("cumulative_tier_resolution_json", pa.string(), nullable=False),
        pa.field("identity_semantics", pa.string(), nullable=False),
        pa.field("herg_label_admitted", pa.bool_(), nullable=False),
        pa.field("clinical_label_admitted", pa.bool_(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
    ]
)

AUDIT_SCHEMA = pa.schema(
    [
        pa.field("intervention_candidate_id", pa.string(), nullable=False),
        pa.field("nct_id", pa.string(), nullable=False),
        pa.field("raw_intervention_name", pa.string(), nullable=False),
        pa.field("normalized_intervention_name", pa.string(), nullable=False),
        pa.field("intervention_type", pa.string(), nullable=False),
        pa.field("admitted_to_candidate_output", pa.bool_(), nullable=False),
        pa.field("audit_state", pa.string(), nullable=False),
        pa.field("exclusion_reason", pa.string()),
        pa.field("first_resolved_rule_name", pa.string()),
        pa.field("candidate_parent_structure_count", pa.int64(), nullable=False),
        pa.field("local_herg_structure_count", pa.int64(), nullable=False),
        pa.field("ambiguity_preserved", pa.bool_(), nullable=False),
        pa.field("identity_semantics", pa.string(), nullable=False),
        pa.field("herg_label_admitted", pa.bool_(), nullable=False),
        pa.field("clinical_label_admitted", pa.bool_(), nullable=False),
        pa.field("model_label_admitted", pa.bool_(), nullable=False),
    ]
)

RULES: tuple[tuple[int, str, tuple[str, ...]], ...] = (
    (0, "exact_parent_standardized", ("exact",)),
    (1, "punctuation_normalized", ("exact", "punctuation")),
    (
        2,
        "cleaned_dose_formulation_parenthetical_or_sponsor_code",
        ("exact", "punctuation", "cleaned", "parenthetical_outer", "parenthetical_inner", "sponsor_code"),
    ),
    (
        3,
        "component_set",
        (
            "exact",
            "punctuation",
            "cleaned",
            "parenthetical_outer",
            "parenthetical_inner",
            "sponsor_code",
            "component",
        ),
    ),
)

_NON_DRUG_EXACT = frozenset(
    {
        "placebo",
        "placebos",
        "vehicle",
        "standard of care",
        "usual care",
        "no treatment",
        "rescue medication",
        "normal saline",
        "saline",
        "sodium chloride",
        "nacl",
        "sterile water",
        "water",
        "magnesium stearate",
        "microcrystalline cellulose",
        "cellulose",
        "lactose",
        "dextrose",
        "glucose",
        "excipient",
    }
)
_NON_DRUG_TEXT_RE = re.compile(
    r"\b(?:matching\s+)?placebos?\b|\bvehicle\b|\b(?:normal\s+)?saline\b|"
    r"\b(?:0[.,]9\s*%\s*)?(?:sodium\s+chloride|nacl)\b|\bmagnesium\s+stearate\b|"
    r"\bmicrocrystalline\s+cellulose\b|\bsterile\s+water\b|\bexcipient\b",
    re.IGNORECASE,
)
_DOSE_RE = re.compile(
    r"(?<![a-z0-9])\d+(?:[.,]\d+)?(?:\s*(?:-|to)\s*\d+(?:[.,]\d+)?)?\s*"
    r"(?:mg|mcg|ug|g|kg|ml|iu|units?|mmol|mci|mbq|%)(?:/\w+)?",
    re.IGNORECASE,
)
_FORMULATION_RE = re.compile(
    r"\b(?:oral|intravenous|iv|po|subcutaneous|sc|intramuscular|im|tablet(?:s)?|"
    r"capsule(?:s)?|injection|injectable|infusion|solution|suspension|powder|formulation|"
    r"dose|dosing|film coated|extended release|immediate release|modified release|"
    r"slow release|fast release|target release|once daily|twice daily|qd|bid|tid|fed|"
    r"fasted|single dose|multi dose|dose escalation|dose expansion|rp2d|cohort|arm|group|"
    r"treatment|administered|regimen|active)\b",
    re.IGNORECASE,
)
_ISOTOPE_PREFIX_RE = re.compile(r"^(?:\[?\^?\d{1,3}[a-z]{1,2}\]?[- ]*|\d{1,3}[a-z]{1,2}[- ]+)", re.I)
_SPONSOR_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Z]{1,8}[- ]?\d[A-Z0-9-]{2,}|\d{1,3}[A-Z]-[A-Z0-9-]{2,})(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_COMPONENT_RE = re.compile(r"\s*(?:\+|/|\band\b|\bor\b)\s*", re.IGNORECASE)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_exact_name(value: str) -> str:
    """NFKC, case-fold, trim, and collapse whitespace."""

    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def normalize_punctuation_name(value: str) -> str:
    """Normalize punctuation without fuzzy or edit-distance matching."""

    normalized = normalize_exact_name(value).replace("μ", "u").replace("µ", "u")
    normalized = normalized.replace("β", "beta").replace("α", "alpha")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", normalized).split())


def _clean_reported_name(value: str) -> str:
    normalized = normalize_exact_name(value)
    normalized = _ISOTOPE_PREFIX_RE.sub("", normalized)
    normalized = _DOSE_RE.sub(" ", normalized)
    normalized = _FORMULATION_RE.sub(" ", normalized)
    normalized = re.sub(r"\b(?:for|on|of|as|at|per|day|daily|the)\b", " ", normalized)
    normalized = re.sub(r"\([^)]*(?:group|arm|cohort|formulation|dose|fed|fasted)[^)]*\)", " ", normalized)
    return normalize_punctuation_name(normalized)


def _valid_form(value: str) -> bool:
    return len(value) >= 3 and value not in _NON_DRUG_EXACT and not _NON_DRUG_TEXT_RE.search(value)


def _name_forms(value: str) -> dict[str, tuple[str, ...]]:
    forms: dict[str, set[str]] = defaultdict(set)
    forms["exact"].add(normalize_exact_name(value))
    forms["punctuation"].add(normalize_punctuation_name(value))
    forms["cleaned"].add(_clean_reported_name(value))
    outer = re.sub(r"\([^)]*\)", " ", value)
    forms["parenthetical_outer"].add(_clean_reported_name(outer))
    for inner in re.findall(r"\(([^)]{2,100})\)", value):
        forms["parenthetical_inner"].add(_clean_reported_name(inner))
    for code in _SPONSOR_CODE_RE.findall(value):
        forms["sponsor_code"].add(normalize_punctuation_name(code))
    components = _COMPONENT_RE.split(value)
    if len(components) > 1:
        for component in components:
            forms["component"].add(_clean_reported_name(component))
    return {
        kind: tuple(sorted(form for form in values if _valid_form(form)))
        for kind, values in sorted(forms.items())
    }


def _exclusion_reason(intervention_type: str, raw_name: str) -> str | None:
    if intervention_type.upper() != "DRUG":
        return "non_drug_intervention_type"
    exact = normalize_exact_name(raw_name)
    if exact in _NON_DRUG_EXACT or _NON_DRUG_TEXT_RE.search(exact):
        return "placebo_vehicle_or_excipient"
    return None


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise HergTrialUpliftError(f"input must be a regular non-symlink file: {path}")
    if path.suffix == ".parquet":
        return pq.read_table(path).to_pylist()
    if path.suffixes[-2:] == [".csv", ".gz"]:
        with gzip.open(path, mode="rt", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if path.suffix == ".csv":
        with path.open(mode="rt", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise HergTrialUpliftError(f"unsupported tabular input: {path}")


def _bool_or_none(value: Any) -> bool | None:
    normalized = normalize_exact_name(_text(value))
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def _standardize_parent_smiles(smiles: str) -> tuple[str, str] | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    try:
        molecule = rdMolStandardize.Cleanup(molecule)
        molecule = rdMolStandardize.FragmentParent(molecule)
        molecule = rdMolStandardize.Uncharger().uncharge(molecule)
        standardized_smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        key = Chem.MolToInchiKey(molecule)
    except Exception:  # pragma: no cover - RDKit/InChI build-specific failure
        return None
    return (standardized_smiles, key) if standardized_smiles and key else None


def _chunks(values: Sequence[int], size: int = 800) -> Iterable[Sequence[int]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        is not None
    )


def _load_chembl_candidates(
    database: Path, forms: Sequence[Mapping[str, tuple[str, ...]]]
) -> tuple[dict[tuple[str, str], set[int]], dict[int, dict[str, Any]]]:
    if not database.is_file() or database.is_symlink():
        raise HergTrialUpliftError(f"ChEMBL input must be a regular non-symlink file: {database}")
    exact_keys = {form for item in forms for form in item.get("exact", ())}
    punctuation_keys = {
        form for item in forms for kind, values in item.items() if kind != "exact" for form in values
    }
    alias_index: dict[tuple[str, str], set[int]] = defaultdict(set)
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        required = {"compound_structures", "molecule_dictionary", "molecule_synonyms"}
        if not all(_table_exists(connection, table) for table in required):
            raise HergTrialUpliftError("ChEMBL database lacks required structure/name tables")
        alias_queries = (
            "SELECT molregno, pref_name FROM molecule_dictionary WHERE pref_name IS NOT NULL",
            "SELECT molregno, synonyms FROM molecule_synonyms WHERE synonyms IS NOT NULL",
        )
        for query in alias_queries:
            cursor = connection.execute(query)
            while True:
                rows = cursor.fetchmany(100_000)
                if not rows:
                    break
                for raw_molregno, raw_alias in rows:
                    alias = str(raw_alias)
                    exact = normalize_exact_name(alias)
                    punctuation = normalize_punctuation_name(alias)
                    if exact in exact_keys:
                        alias_index[("exact", exact)].add(int(raw_molregno))
                    if punctuation in punctuation_keys:
                        alias_index[("punctuation", punctuation)].add(int(raw_molregno))

        matched = sorted({molregno for values in alias_index.values() for molregno in values})
        parent_by_molregno = {molregno: molregno for molregno in matched}
        if matched and _table_exists(connection, "molecule_hierarchy"):
            for batch in _chunks(matched):
                placeholders = ",".join("?" for _ in batch)
                query = f"SELECT molregno, parent_molregno FROM molecule_hierarchy WHERE molregno IN ({placeholders})"
                for molregno, parent in connection.execute(query, batch):
                    parent_by_molregno[int(molregno)] = int(parent) if parent is not None else int(molregno)
        required_molregnos = sorted(set(matched) | set(parent_by_molregno.values()))
        structure_by_molregno: dict[int, tuple[str, str | None]] = {}
        dictionary_by_molregno: dict[int, str] = {}
        for batch in _chunks(required_molregnos):
            placeholders = ",".join("?" for _ in batch)
            for molregno, smiles, raw_key in connection.execute(
                f"SELECT molregno, canonical_smiles, standard_inchi_key FROM compound_structures "
                f"WHERE molregno IN ({placeholders}) AND canonical_smiles IS NOT NULL",
                batch,
            ):
                structure_by_molregno[int(molregno)] = (str(smiles), _text(raw_key) or None)
            for molregno, chembl_id in connection.execute(
                f"SELECT molregno, chembl_id FROM molecule_dictionary WHERE molregno IN ({placeholders})",
                batch,
            ):
                dictionary_by_molregno[int(molregno)] = str(chembl_id)

        identities: dict[int, dict[str, Any]] = {}
        standardization_cache: dict[int, tuple[str, str] | None] = {}
        for matched_molregno in matched:
            parent = parent_by_molregno.get(matched_molregno, matched_molregno)
            source_molregno = parent if parent in structure_by_molregno else matched_molregno
            if source_molregno not in standardization_cache:
                structure = structure_by_molregno.get(source_molregno)
                standardization_cache[source_molregno] = (
                    _standardize_parent_smiles(structure[0]) if structure is not None else None
                )
            standardized = standardization_cache[source_molregno]
            if standardized is None:
                continue
            parent_smiles, parent_key = standardized
            identities[matched_molregno] = {
                "matched_molregno": matched_molregno,
                "parent_molregno": parent,
                "structure_source_molregno": source_molregno,
                "chembl_id": dictionary_by_molregno.get(parent)
                or dictionary_by_molregno.get(matched_molregno),
                "parent_standardized_smiles": parent_smiles,
                "parent_standard_inchi_key": parent_key,
                "source_standard_inchi_key": (structure_by_molregno.get(source_molregno, ("", None))[1]),
            }
    except sqlite3.DatabaseError as exc:
        raise HergTrialUpliftError(f"invalid or incompatible ChEMBL database: {exc}") from exc
    finally:
        connection.close()
    return alias_index, identities


def _resolve_form_values(
    values: Iterable[str],
    *,
    exact: bool,
    alias_index: Mapping[tuple[str, str], set[int]],
    identities: Mapping[int, Mapping[str, Any]],
) -> tuple[set[int], dict[str, set[int]]]:
    molregnos: set[int] = set()
    matched: dict[str, set[int]] = {}
    namespace = "exact" if exact else "punctuation"
    for value in values:
        candidates = set(alias_index.get((namespace, value), set())) & set(identities)
        if candidates:
            matched[value] = candidates
            molregnos.update(candidates)
    return molregnos, matched


def _resolved_for_kinds(
    form_map: Mapping[str, tuple[str, ...]],
    kinds: Sequence[str],
    alias_index: Mapping[tuple[str, str], set[int]],
    identities: Mapping[int, Mapping[str, Any]],
) -> tuple[set[int], dict[str, list[int]]]:
    molregnos: set[int] = set()
    matched: dict[str, list[int]] = {}
    for kind in kinds:
        values = form_map.get(kind, ())
        resolved, resolved_forms = _resolve_form_values(
            values,
            exact=kind == "exact",
            alias_index=alias_index,
            identities=identities,
        )
        molregnos.update(resolved)
        for value, candidates in resolved_forms.items():
            matched[f"{kind}:{value}"] = sorted(candidates)
    return molregnos, dict(sorted(matched.items()))


def _deduplicated_structures(
    molregnos: Iterable[int], identities: Mapping[int, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for molregno in sorted(molregnos):
        identity = identities[molregno]
        key = str(identity["parent_standard_inchi_key"])
        current = by_key.setdefault(
            key,
            {
                "parent_standard_inchi_key": key,
                "parent_standardized_smiles": identity["parent_standardized_smiles"],
                "parent_molregnos": set(),
                "matched_molregnos": set(),
                "chembl_ids": set(),
                "source_standard_inchi_keys": set(),
            },
        )
        current["parent_molregnos"].add(int(identity["parent_molregno"]))
        current["matched_molregnos"].add(int(molregno))
        if identity.get("chembl_id"):
            current["chembl_ids"].add(str(identity["chembl_id"]))
        if identity.get("source_standard_inchi_key"):
            current["source_standard_inchi_keys"].add(str(identity["source_standard_inchi_key"]))
    output: list[dict[str, Any]] = []
    for key in sorted(by_key):
        row = by_key[key]
        output.append(
            {
                "parent_standard_inchi_key": key,
                "parent_standardized_smiles": row["parent_standardized_smiles"],
                "parent_molregnos": sorted(row["parent_molregnos"]),
                "matched_molregnos": sorted(row["matched_molregnos"]),
                "chembl_ids": sorted(row["chembl_ids"]),
                "source_standard_inchi_keys": sorted(row["source_standard_inchi_keys"]),
            }
        )
    return output


def _hierarchy_keys(path: Path) -> set[str]:
    rows = _read_rows(path)
    if not rows or "standard_inchi_key" not in rows[0]:
        raise HergTrialUpliftError("hERG hierarchy must contain standard_inchi_key")
    return {_text(row.get("standard_inchi_key")) for row in rows if _text(row.get("standard_inchi_key"))}


def _write_parquet(path: Path, schema: pa.Schema, rows: list[dict[str, Any]]) -> None:
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )


def _artifact(path: Path, rows: int) -> dict[str, Any]:
    return {"path": path.name, "rows": rows, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _headline(tier_statistics: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    ordered = [rule_name for _, rule_name, _ in RULES]
    selected = next(
        (name for name in ordered if tier_statistics[name]["unique_parent_structure_count"] >= 1000),
        max(ordered, key=lambda name: tier_statistics[name]["unique_parent_structure_count"]),
    )
    stats = tier_statistics[selected]
    return {
        "selected_rule_tier": selected,
        "practical_over_1000_structure_threshold_met": stats["unique_parent_structure_count"] >= 1000,
        "all_chembl_unique_parent_structures": stats["unique_parent_structure_count"],
        "all_chembl_linked_intervention_records": stats["linked_intervention_record_count"],
        "all_chembl_linked_nct_count": stats["linked_nct_count"],
        "current_local_herg_intersection_unique_structures": stats["local_herg_unique_structure_count"],
        "current_local_herg_intersection_records": stats["local_herg_linked_record_count"],
        "disclosure": (
            "all-ChEMBL structure coverage is not equivalent to local reported hERG evidence; "
            "the local hERG intersection is disclosed separately"
        ),
    }


def build_herg_trial_uplift(
    interventions_input: str | os.PathLike[str],
    studies_input: str | os.PathLike[str],
    chembl_sqlite: str | os.PathLike[str],
    hierarchy_annotations: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Build deterministic, auditable intervention-to-structure candidates."""

    intervention_path = Path(interventions_input).resolve()
    study_path = Path(studies_input).resolve()
    chembl_path = Path(chembl_sqlite).resolve()
    hierarchy_path = Path(hierarchy_annotations).resolve()
    output = Path(output_root).resolve()
    if output.exists() and (output.is_symlink() or not output.is_dir() or any(output.iterdir())):
        raise HergTrialUpliftError("output directory must be absent or empty")

    intervention_rows = _read_rows(intervention_path)
    study_rows = _read_rows(study_path)
    required_intervention = {"nct_id", "intervention_type", "intervention_name"}
    if not intervention_rows or not required_intervention.issubset(intervention_rows[0]):
        raise HergTrialUpliftError("interventions input lacks required fields")
    if not study_rows or "nct_id" not in study_rows[0]:
        raise HergTrialUpliftError("studies input lacks nct_id")

    study_results: dict[str, bool | None] = {}
    for row in study_rows:
        nct_id = _text(row.get("nct_id"))
        if nct_id:
            study_results[nct_id] = _bool_or_none(row.get("has_results_reported", row.get("has_results")))
    prepared: list[dict[str, Any]] = []
    for index, row in enumerate(intervention_rows):
        nct_id = _text(row.get("nct_id"))
        raw_name = _text(row.get("intervention_name"))
        intervention_type = _text(row.get("intervention_type")) or "UNKNOWN"
        candidate_id = _text(row.get("intervention_candidate_id")) or f"ROW-{index:09d}"
        if not nct_id or not raw_name:
            raise HergTrialUpliftError("intervention rows require nonempty nct_id and intervention_name")
        prepared.append(
            {
                "nct_id": nct_id,
                "candidate_id": candidate_id,
                "raw_name": raw_name,
                "intervention_type": intervention_type,
                "forms": _name_forms(raw_name),
                "exclusion_reason": _exclusion_reason(intervention_type, raw_name),
            }
        )
    prepared.sort(key=lambda row: (row["nct_id"], row["candidate_id"], row["raw_name"]))

    alias_index, identities = _load_chembl_candidates(
        chembl_path, [row["forms"] for row in prepared if row["exclusion_reason"] is None]
    )
    local_herg_keys = _hierarchy_keys(hierarchy_path)
    candidates: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    cumulative_statistics: dict[str, dict[str, Any]] = {
        name: {"records": set(), "ncts": set(), "keys": set(), "local_records": set(), "local_keys": set()}
        for _, name, _ in RULES
    }

    for row in prepared:
        exclusion = row["exclusion_reason"]
        tier_resolutions: dict[str, dict[str, Any]] = {}
        first: tuple[int, str, list[dict[str, Any]], dict[str, list[int]]] | None = None
        if exclusion is None:
            for tier, rule_name, kinds in RULES:
                molregnos, matched = _resolved_for_kinds(row["forms"], kinds, alias_index, identities)
                structures = _deduplicated_structures(molregnos, identities)
                structure_keys = [str(item["parent_standard_inchi_key"]) for item in structures]
                local_keys = sorted(set(structure_keys) & local_herg_keys)
                tier_resolutions[rule_name] = {
                    "parent_structure_keys": structure_keys,
                    "local_herg_structure_keys": local_keys,
                    "matched_forms": matched,
                }
                if structures:
                    stats = cumulative_statistics[rule_name]
                    stats["records"].add(row["candidate_id"])
                    stats["ncts"].add(row["nct_id"])
                    stats["keys"].update(structure_keys)
                    if local_keys:
                        stats["local_records"].add(row["candidate_id"])
                        stats["local_keys"].update(local_keys)
                    if first is None:
                        first = (tier, rule_name, structures, matched)

        candidate_structure_count = 0
        local_count = 0
        ambiguity = False
        first_name: str | None = None
        if first is not None:
            tier, first_name, structures, matched = first
            structure_keys = [str(item["parent_standard_inchi_key"]) for item in structures]
            local_keys = sorted(set(structure_keys) & local_herg_keys)
            candidate_structure_count = len(structures)
            local_count = len(local_keys)
            ambiguity = candidate_structure_count > 1
            all_chembl_ids = sorted({value for item in structures for value in item["chembl_ids"]})
            all_molregnos = sorted({value for item in structures for value in item["matched_molregnos"]})
            component_names = list(row["forms"].get("component", ()))
            automatic = tier == 0 and not ambiguity
            candidates.append(
                {
                    "intervention_candidate_id": row["candidate_id"],
                    "nct_id": row["nct_id"],
                    "raw_intervention_name": row["raw_name"],
                    "normalized_intervention_name": normalize_exact_name(row["raw_name"]),
                    "intervention_type": row["intervention_type"],
                    "study_record_present": row["nct_id"] in study_results,
                    "study_has_posted_results": study_results.get(row["nct_id"]),
                    "first_resolved_rule_tier": tier,
                    "first_resolved_rule_name": first_name,
                    "automatic_identity_link": automatic,
                    "candidate_status": (
                        "automatic_unique_identity_candidate"
                        if automatic
                        else "candidate_only_review_required_ambiguity_preserved"
                        if ambiguity
                        else "candidate_only_review_required"
                    ),
                    "matched_forms_json": _canonical_json(matched),
                    "component_names_json": _canonical_json(component_names),
                    "candidate_parent_structures_json": _canonical_json(structures),
                    "candidate_parent_structure_keys_json": _canonical_json(structure_keys),
                    "candidate_parent_structure_count": candidate_structure_count,
                    "candidate_chembl_ids_json": _canonical_json(all_chembl_ids),
                    "candidate_molregnos_json": _canonical_json(all_molregnos),
                    "local_herg_structure_keys_json": _canonical_json(local_keys),
                    "local_herg_structure_count": local_count,
                    "has_any_local_reported_herg_evidence": bool(local_keys),
                    "ambiguity_preserved": ambiguity,
                    "is_component_set_candidate": tier == 3,
                    "cumulative_tier_resolution_json": _canonical_json(tier_resolutions),
                    "identity_semantics": IDENTITY_SEMANTICS,
                    "herg_label_admitted": False,
                    "clinical_label_admitted": False,
                    "model_label_admitted": False,
                }
            )

        audit_state = (
            "excluded"
            if exclusion is not None
            else "unresolved"
            if first is None
            else "resolved_candidate_ambiguity_preserved"
            if ambiguity
            else "resolved_candidate"
        )
        audits.append(
            {
                "intervention_candidate_id": row["candidate_id"],
                "nct_id": row["nct_id"],
                "raw_intervention_name": row["raw_name"],
                "normalized_intervention_name": normalize_exact_name(row["raw_name"]),
                "intervention_type": row["intervention_type"],
                "admitted_to_candidate_output": first is not None,
                "audit_state": audit_state,
                "exclusion_reason": exclusion,
                "first_resolved_rule_name": first_name,
                "candidate_parent_structure_count": candidate_structure_count,
                "local_herg_structure_count": local_count,
                "ambiguity_preserved": ambiguity,
                "identity_semantics": IDENTITY_SEMANTICS,
                "herg_label_admitted": False,
                "clinical_label_admitted": False,
                "model_label_admitted": False,
            }
        )

    candidates.sort(key=lambda row: (row["nct_id"], row["intervention_candidate_id"]))
    audits.sort(key=lambda row: (row["nct_id"], row["intervention_candidate_id"]))
    output.mkdir(parents=True, exist_ok=True)
    _write_parquet(output / CANDIDATE_OUTPUT, CANDIDATE_SCHEMA, candidates)
    _write_parquet(output / AUDIT_OUTPUT, AUDIT_SCHEMA, audits)

    tier_statistics: dict[str, dict[str, int]] = {}
    for _, name, _ in RULES:
        stats = cumulative_statistics[name]
        tier_statistics[name] = {
            "linked_intervention_record_count": len(stats["records"]),
            "linked_nct_count": len(stats["ncts"]),
            "unique_parent_structure_count": len(stats["keys"]),
            "local_herg_linked_record_count": len(stats["local_records"]),
            "local_herg_unique_structure_count": len(stats["local_keys"]),
        }
    artifacts = [
        _artifact(output / CANDIDATE_OUTPUT, len(candidates)),
        _artifact(output / AUDIT_OUTPUT, len(audits)),
    ]
    inputs = [intervention_path, study_path, chembl_path, hierarchy_path]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "input_bindings": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)} for path in inputs
        ],
        "artifacts": artifacts,
        "row_counts": {CANDIDATE_OUTPUT: len(candidates), AUDIT_OUTPUT: len(audits)},
        "tier_statistics": tier_statistics,
        "headline": _headline(tier_statistics),
        "rules": {
            "alias_sources": "ChEMBL preferred names and molecule_synonyms only",
            "structure_identity": "ChEMBL hierarchy parent then RDKit Cleanup/FragmentParent/Uncharger",
            "excluded": "non-DRUG interventions plus placebo, vehicle, saline, and named excipients",
            "ambiguity": "preserved as candidate molecule sets; never collapsed arbitrarily",
            "absence_semantics": "no local hERG intersection means unknown, never hERG-negative",
        },
        "herg_labels_admitted": 0,
        "clinical_labels_admitted": 0,
        "model_labels_admitted": 0,
        "candidate_only": True,
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical_json(manifest).encode()).hexdigest()
    (output / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    verify_herg_trial_uplift(output)
    return manifest


def verify_herg_trial_uplift(output_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Verify artifact integrity, schemas, counts, headline, and zero labels."""

    output = Path(output_root).resolve()
    manifest_path = output / MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise HergTrialUpliftError("manifest missing or unsafe")
    manifest = json.loads(manifest_path.read_text())
    declared = manifest.pop("manifest_sha256", None)
    actual = hashlib.sha256(_canonical_json(manifest).encode()).hexdigest()
    if declared != actual:
        raise HergTrialUpliftError("manifest internal SHA-256 mismatch")
    schemas = {CANDIDATE_OUTPUT: CANDIDATE_SCHEMA, AUDIT_OUTPUT: AUDIT_SCHEMA}
    expected_members = {MANIFEST_NAME, *schemas}
    if {path.name for path in output.iterdir()} != expected_members or any(
        path.is_symlink() or not path.is_file() for path in output.iterdir()
    ):
        raise HergTrialUpliftError("output contains unexpected or unsafe members")
    bindings = manifest.get("input_bindings")
    if not isinstance(bindings, list) or len(bindings) != 4:
        raise HergTrialUpliftError("input bindings are missing or incomplete")
    for binding in bindings:
        path = Path(str(binding.get("path", "")))
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(binding.get("bytes", -1))
            or _sha256_file(path) != binding.get("sha256")
        ):
            raise HergTrialUpliftError(f"input binding mismatch: {path}")
    for artifact in manifest["artifacts"]:
        path = output / str(artifact["path"])
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != int(artifact.get("bytes", -1))
            or _sha256_file(path) != artifact["sha256"]
        ):
            raise HergTrialUpliftError(f"artifact verification failed: {path.name}")
        table = pq.read_table(path)
        if table.schema != schemas[path.name] or table.num_rows != artifact["rows"]:
            raise HergTrialUpliftError(f"artifact schema/count mismatch: {path.name}")
        if manifest["row_counts"][path.name] != table.num_rows:
            raise HergTrialUpliftError(f"manifest row count mismatch: {path.name}")
        for label in ("herg_label_admitted", "clinical_label_admitted", "model_label_admitted"):
            if any(table[label].to_pylist()):
                raise HergTrialUpliftError(f"label unexpectedly admitted: {path.name}:{label}")
    candidates = pq.read_table(output / CANDIDATE_OUTPUT).to_pylist()
    recomputed: dict[str, dict[str, set[str]]] = {
        name: {"records": set(), "ncts": set(), "keys": set(), "local_records": set(), "local_keys": set()}
        for _, name, _ in RULES
    }
    for row in candidates:
        resolutions = json.loads(str(row["cumulative_tier_resolution_json"]))
        if set(resolutions) != {name for _, name, _ in RULES}:
            raise HergTrialUpliftError("candidate cumulative tier membership mismatch")
        for _, name, _ in RULES:
            resolution = resolutions[name]
            keys = {str(value) for value in resolution["parent_structure_keys"]}
            local_keys = {str(value) for value in resolution["local_herg_structure_keys"]}
            if not local_keys.issubset(keys):
                raise HergTrialUpliftError("local hERG keys are not a subset of candidate structures")
            if keys:
                stats = recomputed[name]
                stats["records"].add(str(row["intervention_candidate_id"]))
                stats["ncts"].add(str(row["nct_id"]))
                stats["keys"].update(keys)
                if local_keys:
                    stats["local_records"].add(str(row["intervention_candidate_id"]))
                    stats["local_keys"].update(local_keys)
    recomputed_counts = {
        name: {
            "linked_intervention_record_count": len(stats["records"]),
            "linked_nct_count": len(stats["ncts"]),
            "unique_parent_structure_count": len(stats["keys"]),
            "local_herg_linked_record_count": len(stats["local_records"]),
            "local_herg_unique_structure_count": len(stats["local_keys"]),
        }
        for name, stats in recomputed.items()
    }
    if recomputed_counts != manifest.get("tier_statistics"):
        raise HergTrialUpliftError("tier statistics do not match physical candidate rows")
    expected_headline = _headline(manifest["tier_statistics"])
    if manifest.get("headline") != expected_headline:
        raise HergTrialUpliftError("headline does not match disclosed tier statistics")
    if any(
        manifest.get(name) != 0
        for name in ("herg_labels_admitted", "clinical_labels_admitted", "model_labels_admitted")
    ):
        raise HergTrialUpliftError("manifest unexpectedly admits labels")
    return {
        "verification_status": "pass",
        "manifest_internal_sha256": declared,
        "headline": expected_headline,
        "artifact_count": len(manifest["artifacts"]),
        "herg_labels_admitted": 0,
        "clinical_labels_admitted": 0,
        "model_labels_admitted": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--interventions", required=True)
    build.add_argument("--studies", required=True)
    build.add_argument("--chembl-sqlite", required=True)
    build.add_argument("--hierarchy-annotations", required=True)
    build.add_argument("--output-root", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_herg_trial_uplift(
            args.interventions,
            args.studies,
            args.chembl_sqlite,
            args.hierarchy_annotations,
            args.output_root,
        )
    else:
        result = verify_herg_trial_uplift(args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
