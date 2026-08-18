"""Leakage-aware, deterministic representations for the platform data model.

This module intentionally performs no learned embedding calculation.  It
prepares auditable molecular, protein-sequence, graph, and text inputs for
later training, and records enough information to quantify truncation and
invalid-input losses before a model is selected.

The central rule is that an experimental observation and a computational
prediction are different entity types.  Features derived from an outcome, or
from a prediction made without an out-of-fold/prospective contract, are not
admissible model inputs.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from .features import RDKIT_AVAILABLE, canonicalize_smiles, rdkit_descriptors

if RDKIT_AVAILABLE:  # pragma: no branch - availability is exercised in tests.
    from rdkit import Chem


FEATURE_REGISTRY_VERSION = "1.0.0"

LeakageRole = Literal[
    "safe_pre_outcome_input",
    "task_context_only",
    "split_or_audit_only",
    "identifier_only",
    "restricted_free_text",
    "conditional_cross_endpoint",
    "prediction_oof_or_prospective_only",
    "target_or_target_derived",
    "post_outcome",
    "unknown",
]
FeatureOrigin = Literal[
    "submitted",
    "experimentally_observed",
    "deterministically_derived",
    "externally_annotated",
    "computational_prediction",
]


@dataclass(frozen=True)
class FeatureSpec:
    """One feature-family admission rule.

    ``default_model_input`` is deliberately conservative.  A conditionally
    admissible feature must be explicitly enabled in a task configuration and
    supported by a temporal/out-of-fold lineage record.
    """

    name: str
    family: str
    dtype: str
    origin: FeatureOrigin
    leakage_role: LeakageRole
    default_model_input: bool
    fit_scope: str
    description: str
    required_lineage: str = "none"


DEFAULT_FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "canonical_smiles",
        "molecule_2d",
        "string",
        "deterministically_derived",
        "safe_pre_outcome_input",
        True,
        "global_deterministic",
        "Canonical molecular representation derived without outcome access.",
        "standardization_version_and_input_structure",
    ),
    FeatureSpec(
        "standardized_smiles",
        "molecule_2d",
        "string",
        "deterministically_derived",
        "safe_pre_outcome_input",
        True,
        "global_deterministic",
        "Parent/state-standardized structure; state policy must be versioned.",
        "standardization_version_and_input_structure",
    ),
    FeatureSpec(
        "molecular_graph",
        "molecule_graph",
        "graph",
        "deterministically_derived",
        "safe_pre_outcome_input",
        True,
        "global_deterministic",
        "Atom/bond graph generated deterministically from admitted SMILES.",
        "graph_schema_version_and_structure_hash",
    ),
    FeatureSpec(
        "morgan_fingerprint",
        "molecule_2d",
        "binary_vector",
        "deterministically_derived",
        "safe_pre_outcome_input",
        True,
        "global_deterministic",
        "RDKit Morgan fingerprint; radius and width are part of lineage.",
        "rdkit_version_radius_width_and_structure_hash",
    ),
    FeatureSpec(
        "rdkit_descriptors",
        "molecule_descriptors",
        "numeric_vector",
        "deterministically_derived",
        "safe_pre_outcome_input",
        True,
        "global_deterministic",
        "Interpretable 2D descriptors calculated from the admitted structure.",
        "rdkit_version_and_structure_hash",
    ),
    FeatureSpec(
        "protein_sequence",
        "protein_sequence",
        "string",
        "submitted",
        "safe_pre_outcome_input",
        True,
        "global_deterministic",
        "Exact construct or resolved canonical sequence; missing is not negative.",
        "sequence_source_accession_construct_and_hash",
    ),
    FeatureSpec(
        "protein_structure",
        "protein_structure",
        "coordinates",
        "externally_annotated",
        "safe_pre_outcome_input",
        False,
        "global_deterministic",
        "Experimental or predicted coordinates with method and confidence.",
        "coordinate_source_version_chain_and_prediction_cutoff",
    ),
    FeatureSpec(
        "assay_context_structured",
        "assay_context",
        "record",
        "experimentally_observed",
        "task_context_only",
        False,
        "train_partition_only_for_encoding",
        "Protocol fields known at intended prediction time; never the outcome.",
        "intended_use_availability_and_field_level_audit",
    ),
    FeatureSpec(
        "assay_description",
        "assay_text",
        "string",
        "submitted",
        "restricted_free_text",
        False,
        "train_partition_only",
        "Free text may quote endpoint values or labels and requires redaction audit.",
        "label_like_text_scan_and_manual_policy",
    ),
    FeatureSpec(
        "source_id",
        "provenance",
        "category",
        "externally_annotated",
        "split_or_audit_only",
        False,
        "never_fit_as_default_feature",
        "Source is retained for heterogeneity, holdout, and error audits.",
    ),
    FeatureSpec(
        "document_year",
        "provenance",
        "integer",
        "externally_annotated",
        "split_or_audit_only",
        False,
        "never_fit_as_default_feature",
        "Time axis for temporal evaluation; may proxy outcome availability.",
    ),
    FeatureSpec(
        "molecule_id",
        "identity",
        "string",
        "deterministically_derived",
        "identifier_only",
        False,
        "never_fit",
        "Stable join, grouping, prediction, and error-analysis identifier.",
    ),
    FeatureSpec(
        "protein_id",
        "identity",
        "string",
        "deterministically_derived",
        "identifier_only",
        False,
        "never_fit",
        "Stable resolved protein/construct identifier.",
    ),
    FeatureSpec(
        "task_id",
        "task_routing",
        "string",
        "deterministically_derived",
        "task_context_only",
        False,
        "fixed_task_definition",
        "Routes an example to an endpoint-specific head; not a pooled label shortcut.",
    ),
    FeatureSpec(
        "cross_endpoint_observation",
        "multitask_context",
        "numeric_or_categorical",
        "experimentally_observed",
        "conditional_cross_endpoint",
        False,
        "train_partition_and_time_filtered",
        "A distinct observed endpoint usable only if available at query time.",
        "observation_time_precedes_query_and_no_same-assay_target_derivation",
    ),
    FeatureSpec(
        "computational_prediction",
        "stacked_prediction",
        "numeric_or_categorical",
        "computational_prediction",
        "prediction_oof_or_prospective_only",
        False,
        "out_of_fold_training_or_locked_prospective",
        "Prediction features require model, training-cutoff, and overlap lineage.",
        "upstream_model_hash_dataset_hash_split_hash_and_prediction_timestamp",
    ),
    FeatureSpec(
        "label_value",
        "outcome",
        "numeric_or_categorical",
        "experimentally_observed",
        "target_or_target_derived",
        False,
        "target_only",
        "Observed or curated task label; never an input to its own task.",
    ),
    FeatureSpec(
        "p_value",
        "outcome",
        "numeric",
        "deterministically_derived",
        "target_or_target_derived",
        False,
        "target_only",
        "Molar concentration transform of the same observation.",
        "original_value_unit_relation_temperature_if_applicable_and_formula",
    ),
    FeatureSpec(
        "clinical_outcome_summary",
        "clinical_outcome",
        "record",
        "experimentally_observed",
        "post_outcome",
        False,
        "never_fit_for_preclinical_prediction",
        "Human outcome occurring after the intended preclinical prediction time.",
    ),
)


class FeatureRegistry:
    """Immutable feature registry with fail-closed admission checks."""

    def __init__(self, specs: Sequence[FeatureSpec] = DEFAULT_FEATURE_SPECS):
        by_name = {spec.name: spec for spec in specs}
        if len(by_name) != len(specs):
            raise ValueError("Feature names must be unique")
        self._specs = tuple(specs)
        self._by_name = by_name

    def get(self, name: str) -> FeatureSpec:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"Feature {name!r} is not registered; unregistered inputs fail closed") from exc

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(spec) for spec in self._specs]).sort_values("name").reset_index(drop=True)

    def digest(self) -> str:
        return stable_json_digest([asdict(spec) for spec in sorted(self._specs, key=lambda item: item.name)])

    def assert_model_inputs(
        self,
        names: Iterable[str],
        *,
        conditional_lineage: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        """Reject unsafe features unless their required conditional lineage is present."""

        conditional_lineage = conditional_lineage or {}
        forbidden: list[str] = []
        incomplete: list[str] = []
        for name in names:
            spec = self.get(name)
            if spec.default_model_input:
                continue
            if spec.leakage_role not in {
                "conditional_cross_endpoint",
                "prediction_oof_or_prospective_only",
                "task_context_only",
            }:
                forbidden.append(f"{name}:{spec.leakage_role}")
                continue
            lineage = conditional_lineage.get(name, {})
            if not bool(lineage.get("admission_approved")) or not str(lineage.get("lineage_digest", "")):
                incomplete.append(name)
        if forbidden or incomplete:
            details = []
            if forbidden:
                details.append(f"forbidden={sorted(forbidden)}")
            if incomplete:
                details.append(f"missing_conditional_lineage={sorted(incomplete)}")
            raise ValueError("Unsafe model input selection: " + "; ".join(details))


def stable_json_digest(payload: Any) -> str:
    """SHA-256 over canonical JSON for portable lineage records."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_text(value: object) -> str:
    """Unicode-normalize and collapse whitespace without semantic rewriting."""

    if value is None:
        return ""
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return ""
    normalized = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", normalized).strip()


SMILES_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+\]|Br|Cl|Si|Se|Na|Li|Ca|Mg|Al|@@?|%\d{2}|\d|[A-Z][a-z]?|[bcnops]|"
    r"\(|\)|\.|=|#|-|\+|\\|/|:|~|\?|>|\*)"
)


def tokenize_smiles(smiles: object) -> list[str]:
    """Tokenize SMILES losslessly; malformed/uncovered characters fail loudly."""

    text = normalize_text(smiles).replace(" ", "")
    tokens = SMILES_TOKEN_PATTERN.findall(text)
    if "".join(tokens) != text:
        covered = "".join(tokens)
        raise ValueError(f"SMILES tokenizer did not cover the input: input={text!r}, covered={covered!r}")
    return tokens


GRAPH_SCHEMA_VERSION = "molecular_graph_v1"


def prepare_molecular_graph(smiles: object) -> dict[str, Any]:
    """Create a deterministic, JSON-safe directed molecular graph."""

    text = normalize_text(smiles).replace(" ", "")
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for molecular graph preparation")
    mol = Chem.MolFromSmiles(text) if text else None
    if mol is None:
        return {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "valid": False,
            "canonical_smiles": "",
            "structure_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "nodes": [],
            "edges": [],
            "error": "invalid_or_empty_smiles",
        }
    canonical = str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    nodes: list[dict[str, Any]] = []
    for atom in mol.GetAtoms():
        nodes.append(
            {
                "atom_index": int(atom.GetIdx()),
                "atomic_number": int(atom.GetAtomicNum()),
                "formal_charge": int(atom.GetFormalCharge()),
                "total_degree": int(atom.GetTotalDegree()),
                "total_hydrogens": int(atom.GetTotalNumHs()),
                "is_aromatic": bool(atom.GetIsAromatic()),
                "is_in_ring": bool(atom.IsInRing()),
                "hybridization": str(atom.GetHybridization()),
                "chirality": str(atom.GetChiralTag()),
            }
        )
    edges: list[dict[str, Any]] = []
    for bond in mol.GetBonds():
        source = int(bond.GetBeginAtomIdx())
        target = int(bond.GetEndAtomIdx())
        values = {
            "bond_type": str(bond.GetBondType()),
            "is_aromatic": bool(bond.GetIsAromatic()),
            "is_conjugated": bool(bond.GetIsConjugated()),
            "is_in_ring": bool(bond.IsInRing()),
            "stereo": str(bond.GetStereo()),
        }
        edges.extend(
            [
                {"source": source, "target": target, **values},
                {"source": target, "target": source, **values},
            ]
        )
    edges.sort(key=lambda item: (item["source"], item["target"], item["bond_type"]))
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "valid": True,
        "canonical_smiles": canonical,
        "structure_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "nodes": nodes,
        "edges": edges,
        "error": "",
    }


PROTEIN_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")


@dataclass(frozen=True)
class PreparedSequence:
    normalized_sequence: str
    sequence_sha256: str
    original_length: int
    effective_length: int
    valid: bool
    invalid_characters: tuple[str, ...]
    chunks: tuple[str, ...]
    chunk_starts: tuple[int, ...]
    max_length: int | None
    overlap: int
    truncated: bool
    policy: str


def normalize_protein_sequence(sequence: object) -> tuple[str, tuple[str, ...]]:
    """Normalize FASTA-like protein text while reporting every invalid symbol."""

    text = "" if sequence is None else unicodedata.normalize("NFKC", str(sequence))
    lines = [line.strip() for line in text.splitlines() if not line.lstrip().startswith(">")]
    normalized = re.sub(r"[\s\d-]+", "", "".join(lines)).upper().rstrip("*")
    invalid = tuple(sorted(set(normalized) - PROTEIN_ALPHABET))
    return normalized, invalid


def prepare_protein_sequence(
    sequence: object,
    *,
    max_length: int | None = None,
    policy: Literal["error", "right", "left", "center", "chunk"] = "chunk",
    overlap: int = 64,
) -> PreparedSequence:
    """Validate and deterministically truncate or chunk a protein sequence."""

    normalized, invalid = normalize_protein_sequence(sequence)
    if max_length is not None and max_length < 1:
        raise ValueError("max_length must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if max_length is not None and overlap >= max_length:
        raise ValueError("overlap must be smaller than max_length")
    if invalid:
        chunks: tuple[str, ...] = ()
        starts: tuple[int, ...] = ()
    elif max_length is None or len(normalized) <= max_length:
        chunks = (normalized,) if normalized else ()
        starts = (0,) if normalized else ()
    elif policy == "error":
        raise ValueError(f"Sequence length {len(normalized)} exceeds max_length={max_length}")
    elif policy == "right":
        chunks, starts = (normalized[:max_length],), (0,)
    elif policy == "left":
        chunks, starts = (normalized[-max_length:],), (len(normalized) - max_length,)
    elif policy == "center":
        start = (len(normalized) - max_length) // 2
        chunks, starts = (normalized[start : start + max_length],), (start,)
    elif policy == "chunk":
        step = max_length - overlap
        start_values = list(range(0, max(1, len(normalized) - overlap), step))
        if start_values[-1] + max_length < len(normalized):
            start_values.append(len(normalized) - max_length)
        start_values = sorted(set(start_values))
        chunks = tuple(normalized[start : start + max_length] for start in start_values)
        starts = tuple(start_values)
    else:  # pragma: no cover - Literal prevents this for typed callers.
        raise ValueError(f"Unsupported sequence policy: {policy}")
    return PreparedSequence(
        normalized_sequence=normalized,
        sequence_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        original_length=len(normalized),
        effective_length=sum(len(chunk) for chunk in chunks),
        valid=bool(normalized) and not invalid,
        invalid_characters=invalid,
        chunks=chunks,
        chunk_starts=starts,
        max_length=max_length,
        overlap=overlap,
        truncated=bool(max_length is not None and len(normalized) > max_length and policy != "chunk"),
        policy=policy,
    )


LABEL_LIKE_TEXT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "endpoint_numeric_value",
        re.compile(
            r"(?i)\b(?:IC50|EC50|Ki|Kd|hERG|QTc?|AUC|Cmax|clearance)\b.{0,24}"
            r"[<>~=]?\s*\d+(?:\.\d+)?(?:e[+-]?\d+)?",
        ),
    ),
    (
        "numeric_bioactivity_unit",
        re.compile(r"(?i)\b\d+(?:\.\d+)?\s*(?:pM|nM|uM|µM|mM|mol/?L)\b"),
    ),
    ("explicit_binary_label", re.compile(r"(?i)\b(?:active|inactive|blocker|non[- ]?blocker)\b")),
)


def scan_text_for_label_leakage(text: object) -> list[dict[str, Any]]:
    """Return label-like spans; this is a screening flag, not proof of leakage."""

    normalized = normalize_text(text)
    findings: list[dict[str, Any]] = []
    for kind, pattern in LABEL_LIKE_TEXT_PATTERNS:
        for match in pattern.finditer(normalized):
            findings.append(
                {
                    "kind": kind,
                    "start": int(match.start()),
                    "end": int(match.end()),
                    "text_sha256": hashlib.sha256(match.group(0).encode("utf-8")).hexdigest(),
                }
            )
    return findings


@dataclass(frozen=True)
class PreparedText:
    normalized_text: str
    text_sha256: str
    tokens: tuple[str, ...]
    chunks: tuple[tuple[str, ...], ...]
    original_token_count: int
    max_tokens: int | None
    overlap: int
    truncated: bool
    label_like_findings: tuple[dict[str, Any], ...]
    label_like_scan_clean: bool
    default_model_input_admitted: bool


def prepare_text(
    text: object,
    *,
    max_tokens: int | None = None,
    overlap: int = 32,
    chunk_long_text: bool = True,
) -> PreparedText:
    """Whitespace-tokenize text for audit/length analysis, never as a final tokenizer."""

    normalized = normalize_text(text)
    tokens = tuple(normalized.split()) if normalized else ()
    chunks: tuple[tuple[str, ...], ...]
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if max_tokens is not None and overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")
    if max_tokens is None or len(tokens) <= max_tokens:
        chunks = (tokens,) if tokens else ()
        truncated = False
    elif chunk_long_text:
        step = max_tokens - overlap
        starts = list(range(0, max(1, len(tokens) - overlap), step))
        if starts[-1] + max_tokens < len(tokens):
            starts.append(len(tokens) - max_tokens)
        chunks = tuple(tokens[start : start + max_tokens] for start in sorted(set(starts)))
        truncated = False
    else:
        chunks = (tokens[:max_tokens],)
        truncated = True
    findings = tuple(scan_text_for_label_leakage(normalized))
    return PreparedText(
        normalized_text=normalized,
        text_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        tokens=tokens,
        chunks=chunks,
        original_token_count=len(tokens),
        max_tokens=max_tokens,
        overlap=overlap,
        truncated=truncated,
        label_like_findings=findings,
        label_like_scan_clean=not findings,
        default_model_input_admitted=False,
    )


def length_and_truncation_analysis(
    values: Iterable[object],
    *,
    kind: Literal["characters", "smiles_tokens", "protein_residues", "whitespace_tokens"],
    candidate_max_lengths: Sequence[int],
) -> pd.DataFrame:
    """Quantify coverage at candidate limits without selecting a limit implicitly."""

    lengths: list[int] = []
    invalid_count = 0
    for value in values:
        try:
            if kind == "characters":
                length = len(normalize_text(value))
            elif kind == "smiles_tokens":
                length = len(tokenize_smiles(value))
            elif kind == "protein_residues":
                sequence, invalid = normalize_protein_sequence(value)
                invalid_count += int(bool(invalid))
                length = len(sequence)
            else:
                length = len(normalize_text(value).split())
        except ValueError:
            invalid_count += 1
            length = 0
        lengths.append(length)
    array = np.asarray(lengths, dtype=int)
    rows: list[dict[str, Any]] = []
    for maximum in sorted(set(int(item) for item in candidate_max_lengths)):
        if maximum < 1:
            raise ValueError("candidate_max_lengths must be positive")
        affected = array > maximum
        lost = np.maximum(array - maximum, 0)
        rows.append(
            {
                "kind": kind,
                "candidate_max_length": maximum,
                "n_records": int(len(array)),
                "n_nonempty": int(np.sum(array > 0)),
                "n_invalid": int(invalid_count),
                "n_affected": int(np.sum(affected)),
                "fraction_affected": float(np.mean(affected)) if len(array) else np.nan,
                "total_units_lost_if_hard_truncated": int(np.sum(lost)),
                "fraction_units_lost_if_hard_truncated": (
                    float(np.sum(lost) / np.sum(array)) if np.sum(array) else np.nan
                ),
                "length_min": int(array.min()) if len(array) else None,
                "length_median": float(np.median(array)) if len(array) else None,
                "length_p90": float(np.quantile(array, 0.90)) if len(array) else None,
                "length_p95": float(np.quantile(array, 0.95)) if len(array) else None,
                "length_p99": float(np.quantile(array, 0.99)) if len(array) else None,
                "length_max": int(array.max()) if len(array) else None,
            }
        )
    return pd.DataFrame(rows)


def feature_failure_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize model-input availability without interpreting missing as negative."""

    candidate_columns = {
        "molecule": ("standardized_smiles", "canonical_smiles", "submitted_smiles", "smiles"),
        "protein_sequence": ("sequence", "protein_sequence"),
        "protein_id": ("protein_id", "canonical_target_id", "target_id"),
        "assay_text": ("assay_description", "description"),
    }
    rows: list[dict[str, Any]] = []
    for family, candidates in candidate_columns.items():
        available = next((column for column in candidates if column in frame.columns), None)
        if available is None:
            rows.append(
                {
                    "feature_family": family,
                    "resolved_column": None,
                    "n_records": int(len(frame)),
                    "n_present": 0,
                    "fraction_present": 0.0 if len(frame) else np.nan,
                    "status": "column_unavailable",
                }
            )
            continue
        present = frame[available].fillna("").astype(str).str.strip().ne("")
        rows.append(
            {
                "feature_family": family,
                "resolved_column": available,
                "n_records": int(len(frame)),
                "n_present": int(present.sum()),
                "fraction_present": float(present.mean()) if len(frame) else np.nan,
                "status": "available",
            }
        )
    return pd.DataFrame(rows)


def deterministic_descriptor_frame(
    records: pd.DataFrame,
    *,
    smiles_column: str = "standardized_smiles",
    id_column: str = "molecule_id",
) -> pd.DataFrame:
    """Build a static descriptor table with explicit failure and lineage fields."""

    required = {smiles_column, id_column}
    missing = sorted(required - set(records.columns))
    if missing:
        raise ValueError(f"Descriptor input is missing columns: {missing}")
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for platform descriptor artifacts")
    smiles = records[smiles_column].fillna("").astype(str)
    descriptors = rdkit_descriptors(smiles)
    output = pd.concat(
        [records[[id_column]].reset_index(drop=True), descriptors.reset_index(drop=True)], axis=1
    )
    output["feature_origin"] = "deterministically_derived"
    output["leakage_role"] = "safe_pre_outcome_input"
    output["feature_registry_version"] = FEATURE_REGISTRY_VERSION
    output["input_structure_sha256"] = [
        hashlib.sha256(canonicalize_smiles(value).encode("utf-8")).hexdigest() for value in smiles
    ]
    return output


def categorical_imbalance_summary(
    values: Iterable[object], *, missing_label: str = "__MISSING__"
) -> dict[str, Any]:
    """Compact representation summary used by feature and subgroup audits."""

    normalized = [missing_label if pd.isna(value) else str(value) for value in values]
    counts = Counter(normalized)
    n = len(normalized)
    probabilities = np.asarray(list(counts.values()), dtype=float) / n if n else np.asarray([])
    entropy = float(-np.sum(probabilities * np.log2(probabilities))) if n else np.nan
    return {
        "n": n,
        "n_categories": len(counts),
        "counts": dict(sorted(counts.items())),
        "entropy_bits": entropy,
        "majority_fraction": float(max(counts.values()) / n) if n else np.nan,
    }
