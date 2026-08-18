"""Chemical structure normalization and stable identity helpers.

The module intentionally treats RDKit as an optional import so the public-data
pipeline can still inspect and quarantine records in a minimal environment.  A
fallback structure key is provided in that case, but every fallback record is
marked as unvalidated.  Publication and production builds should require
RDKit, which callers can enforce with ``require_rdkit=True``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
from typing import Any

import pandas as pd

STANDARDIZATION_POLICY_BASE = "rdkit-cleanup-neutralize-v2"


def standardization_version(*, strip_salts: bool = True, canonicalize_tautomer: bool = False) -> str:
    """Return the identity namespace for the complete adjustable structure policy."""

    return (
        f"{STANDARDIZATION_POLICY_BASE}-"
        f"fragment-parent-{int(strip_salts)}-tautomer-{int(canonicalize_tautomer)}"
    )


STANDARDIZATION_VERSION = standardization_version()

try:  # pragma: no cover - availability depends on the execution environment.
    from rdkit import Chem, rdBase
    from rdkit.Chem.MolStandardize import rdMolStandardize

    _RDKIT_IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover - covered through fallback tests.
    Chem = None  # type: ignore[assignment]
    rdBase = None  # type: ignore[assignment]
    rdMolStandardize = None  # type: ignore[assignment]
    _RDKIT_IMPORT_ERROR = str(exc)


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def rdkit_available() -> bool:
    """Return whether the optional RDKit chemistry toolkit is importable."""

    return Chem is not None and rdMolStandardize is not None


def structure_id_from_smiles(
    smiles: object,
    *,
    namespace: str = STANDARDIZATION_VERSION,
    prefix: str = "STR",
    digest_length: int = 20,
) -> str:
    """Create a deterministic, non-semantic structure identifier.

    The identifier is stable for the supplied standardized representation and
    standardization namespace.  It is not intended to replace an InChIKey or a
    compound-registration identifier.
    """

    text = _clean_text(smiles)
    if not text:
        return ""
    digest = sha256(f"{namespace}\0{text}".encode()).hexdigest()
    return f"{prefix}-{digest[:digest_length].upper()}"


@dataclass(frozen=True)
class StandardizedStructure:
    """Traceable result of structure normalization."""

    original_smiles: str
    canonical_smiles: str
    standardized_smiles: str
    standard_inchi_key: str
    structure_id: str
    full_structure_id: str
    structure_valid: bool | None
    structure_standardization_status: str
    structure_error: str
    structure_standardization_version: str
    rdkit_version: str
    fragment_count: int | None
    formal_charge: int | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _empty_result(
    original: str,
    status: str,
    error: str = "",
    *,
    strip_salts: bool = True,
    canonicalize_tautomer: bool = False,
) -> StandardizedStructure:
    version = standardization_version(
        strip_salts=strip_salts,
        canonicalize_tautomer=canonicalize_tautomer,
    )
    return StandardizedStructure(
        original_smiles=original,
        canonical_smiles="",
        standardized_smiles="",
        standard_inchi_key="",
        structure_id="",
        full_structure_id="",
        structure_valid=False if original else None,
        structure_standardization_status=status,
        structure_error=error,
        structure_standardization_version=version,
        rdkit_version="" if rdBase is None else str(rdBase.rdkitVersion),
        fragment_count=None,
        formal_charge=None,
    )


@lru_cache(maxsize=100_000)
def _standardize_cached(
    original: str,
    strip_salts: bool,
    canonicalize_tautomer: bool,
) -> StandardizedStructure:
    version = standardization_version(
        strip_salts=strip_salts,
        canonicalize_tautomer=canonicalize_tautomer,
    )
    if not original:
        return _empty_result(
            original,
            "missing_structure",
            strip_salts=strip_salts,
            canonicalize_tautomer=canonicalize_tautomer,
        )

    if not rdkit_available():
        # The raw key allows non-chemistry stages to remain usable, while the
        # status and RAW prefix prevent it from being mistaken for a validated
        # parent-structure identifier.
        raw_id = structure_id_from_smiles(
            original,
            namespace="unvalidated-raw-smiles-v1",
            prefix="RAW",
        )
        return StandardizedStructure(
            original_smiles=original,
            canonical_smiles=original,
            standardized_smiles=original,
            standard_inchi_key="",
            structure_id=raw_id,
            full_structure_id=raw_id,
            structure_valid=None,
            structure_standardization_status="rdkit_unavailable",
            structure_error=_RDKIT_IMPORT_ERROR,
            structure_standardization_version=version,
            rdkit_version="",
            fragment_count=original.count(".") + 1,
            formal_charge=None,
        )

    try:
        molecule = Chem.MolFromSmiles(original)
        if molecule is None:
            return _empty_result(
                original,
                "invalid_smiles",
                "RDKit could not parse SMILES",
                strip_salts=strip_salts,
                canonicalize_tautomer=canonicalize_tautomer,
            )

        fragment_count = len(Chem.GetMolFrags(molecule))
        cleaned = rdMolStandardize.Cleanup(molecule)
        canonical_smiles = Chem.MolToSmiles(cleaned, canonical=True, isomericSmiles=True)

        parent = rdMolStandardize.FragmentParent(cleaned) if strip_salts else cleaned
        parent = rdMolStandardize.Uncharger().uncharge(parent)
        if canonicalize_tautomer:
            parent = rdMolStandardize.TautomerEnumerator().Canonicalize(parent)
        Chem.SanitizeMol(parent)

        standardized_smiles = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
        inchi_key = Chem.MolToInchiKey(parent)
        formal_charge = int(sum(atom.GetFormalCharge() for atom in parent.GetAtoms()))
        return StandardizedStructure(
            original_smiles=original,
            canonical_smiles=canonical_smiles,
            standardized_smiles=standardized_smiles,
            standard_inchi_key=inchi_key,
            structure_id=structure_id_from_smiles(
                standardized_smiles,
                namespace=version,
            ),
            full_structure_id=structure_id_from_smiles(
                canonical_smiles,
                namespace="rdkit-cleanup-full-structure-v2",
                prefix="FULL",
            ),
            structure_valid=True,
            structure_standardization_status="standardized",
            structure_error="",
            structure_standardization_version=version,
            rdkit_version=str(rdBase.rdkitVersion),
            fragment_count=fragment_count,
            formal_charge=formal_charge,
        )
    except Exception as exc:  # RDKit raises several exception classes.
        return _empty_result(
            original,
            "standardization_failed",
            f"{type(exc).__name__}: {exc}",
            strip_salts=strip_salts,
            canonicalize_tautomer=canonicalize_tautomer,
        )


def standardize_smiles(
    smiles: object,
    *,
    strip_salts: bool = True,
    canonicalize_tautomer: bool = False,
    require_rdkit: bool = False,
) -> StandardizedStructure:
    """Validate and standardize one SMILES while retaining its original form.

    Tautomers are not canonicalized by default because tautomer collapsing can
    erase chemically meaningful distinctions and should be a dataset-level,
    versioned decision.
    """

    if require_rdkit and not rdkit_available():
        raise RuntimeError(
            "RDKit is required for structure standardization but is unavailable. "
            "Install the project's optional 'chem' dependencies."
        )
    return _standardize_cached(_clean_text(smiles), strip_salts, canonicalize_tautomer)


def standardize_structure_table(
    table: pd.DataFrame,
    *,
    smiles_column: str = "smiles",
    strip_salts: bool = True,
    canonicalize_tautomer: bool = False,
    require_rdkit: bool = False,
    replace_smiles: bool = True,
) -> pd.DataFrame:
    """Add traceable standardized-structure columns to a dataframe.

    ``original_smiles`` and ``original_inchi_key`` are never overwritten.
    ``smiles`` remains the modeling representation for backward compatibility.
    """

    out = table.copy()
    if smiles_column not in out.columns:
        out[smiles_column] = ""
    if out.empty:
        for field in StandardizedStructure.__dataclass_fields__:
            if field not in out.columns:
                out[field] = pd.Series(index=out.index, dtype=object)
        return out

    original_values = out[smiles_column].map(_clean_text)
    if "original_smiles" not in out.columns:
        out["original_smiles"] = original_values
    else:
        stored = out["original_smiles"].map(_clean_text)
        out["original_smiles"] = stored.where(stored != "", original_values)

    if "original_inchi_key" not in out.columns:
        if "inchi_key" in out.columns:
            out["original_inchi_key"] = out["inchi_key"].map(_clean_text)
        else:
            out["original_inchi_key"] = ""

    results = [
        standardize_smiles(
            value,
            strip_salts=strip_salts,
            canonicalize_tautomer=canonicalize_tautomer,
            require_rdkit=require_rdkit,
        ).as_dict()
        for value in original_values
    ]
    result_table = pd.DataFrame(results, index=out.index)
    for name in result_table.columns:
        if name == "original_smiles":
            continue
        out[name] = result_table[name]

    # Use the generated standard InChIKey where available while preserving the
    # submitted key in ``original_inchi_key``.
    if "inchi_key" not in out.columns:
        out["inchi_key"] = ""
    generated_key = out["standard_inchi_key"].map(_clean_text)
    original_key = out["original_inchi_key"].map(_clean_text)
    out["inchi_key"] = generated_key.where(generated_key != "", original_key)

    if replace_smiles:
        standardized = out["standardized_smiles"].map(_clean_text)
        out["smiles"] = standardized.where(standardized != "", original_values)
    return out


__all__ = [
    "STANDARDIZATION_VERSION",
    "standardization_version",
    "StandardizedStructure",
    "rdkit_available",
    "standardize_smiles",
    "standardize_structure_table",
    "structure_id_from_smiles",
]
