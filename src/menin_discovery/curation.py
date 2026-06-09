"""Curation utilities for activity, hERG, and PK/ADMET data."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    BIOACTIVITY_ENDPOINTS,
    HERG_TARGET,
    MENIN_TARGET,
    PK_ADMET_KEYWORDS,
    PK_ADMET_STANDARD_TYPES,
)

ENDPOINT_CANONICAL = {
    "IC50": "IC50",
    "KI": "Ki",
    "KD": "Kd",
    "EC50": "EC50",
}

UNIT_TO_NM = {
    "NM": 1.0,
    "NANOMOLAR": 1.0,
    "UM": 1000.0,
    "ΜM": 1000.0,
    "µM": 1000.0,
    "MICROMOLAR": 1000.0,
    "MM": 1_000_000.0,
    "MILLIMOLAR": 1_000_000.0,
    "M": 1_000_000_000.0,
    "PM": 0.001,
    "PICOMOLAR": 0.001,
}


def col(df: pd.DataFrame, name: str, default: object = "") -> pd.Series:
    """Return a column or an index-aligned default series."""

    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


def extract_relation(value: object, explicit: object = "") -> str:
    if explicit is not None and not (isinstance(explicit, float) and math.isnan(explicit)):
        text = str(explicit).strip()
        if text:
            return text
    match = re.match(r"\s*([<>]=?|=|~)", str(value))
    return match.group(1) if match else "="


def parse_numeric(value: object) -> float:
    """Parse the first numeric token from assay values like '<10' or '> 1,000'."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    text = str(value).replace(",", "").strip()
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return np.nan
    try:
        return float(match.group(0))
    except ValueError:
        return np.nan


def normalize_endpoint(endpoint: object) -> str:
    text = str(endpoint).strip()
    return ENDPOINT_CANONICAL.get(text.upper(), text)


def unit_factor_to_nm(unit: object) -> float:
    if unit is None or (isinstance(unit, float) and math.isnan(unit)):
        return 1.0
    text = str(unit).strip()
    if not text:
        return 1.0
    return UNIT_TO_NM.get(text.upper(), 1.0)


def p_value_from_nm(value_nm: object) -> float:
    value = parse_numeric(value_nm)
    if not np.isfinite(value) or value <= 0:
        return np.nan
    return 9.0 - np.log10(value)


def chembl_to_long(df: pd.DataFrame, source_detail: str) -> pd.DataFrame:
    """Convert ChEMBL activity rows into the common long schema."""

    if df.empty:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "source": "ChEMBL",
            "source_record_id": col(df, "activity_id"),
            "compound_id": col(df, "molecule_chembl_id"),
            "compound_name": col(df, "molecule_pref_name"),
            "smiles": col(df, "canonical_smiles"),
            "inchi_key": "",
            "target_name": col(df, "target_pref_name"),
            "target_id": col(df, "target_chembl_id"),
            "endpoint": col(df, "standard_type"),
            "relation": col(df, "standard_relation").combine(
                col(df, "standard_value"), lambda rel, val: extract_relation(val, rel)
            ),
            "value_raw": col(df, "standard_value"),
            "standard_units": col(df, "standard_units"),
            "assay_description": col(df, "assay_description"),
            "assay_type": col(df, "assay_type"),
            "document_id": col(df, "document_chembl_id"),
            "document_year": col(df, "document_year"),
            "reference": col(df, "document_journal"),
            "source_detail": source_detail,
        }
    )


def pubchem_to_long(df: pd.DataFrame, catalog: pd.DataFrame | None = None) -> pd.DataFrame:
    """Convert PubChem BioAssay CSV rows into the common long schema."""

    if df.empty:
        return pd.DataFrame()

    if catalog is not None and not catalog.empty:
        df = df.merge(catalog[["aid", "assay_name", "assay_description"]], on="aid", how="left")
    else:
        df["assay_name"] = ""
        df["assay_description"] = ""

    endpoint = col(df, "Standard Type").fillna("")
    value = pd.Series(np.nan, index=df.index, dtype=object)
    units = pd.Series("nM", index=df.index, dtype=object)

    if "Standard Value" in df.columns:
        standard_value = col(df, "Standard Value")
        standard_has_numeric = standard_value.map(parse_numeric).notna()
        value.loc[standard_has_numeric] = standard_value.loc[standard_has_numeric]
        units.loc[standard_has_numeric] = col(df, "Standard Units").loc[standard_has_numeric].replace("", np.nan).fillna("nM")

    for assay_endpoint in BIOACTIVITY_ENDPOINTS:
        if assay_endpoint in df.columns:
            endpoint_value = col(df, assay_endpoint)
            mask = value.map(parse_numeric).isna() & endpoint_value.map(parse_numeric).notna()
            value.loc[mask] = endpoint_value.loc[mask]
            endpoint.loc[mask & (endpoint.astype(str).str.strip() == "")] = assay_endpoint
            unit_column = f"{assay_endpoint} Units"
            if unit_column in df.columns:
                units.loc[mask] = col(df, unit_column).loc[mask].replace("", np.nan).fillna("nM")
            else:
                units.loc[mask] = "nM"

    if "PubChem Standard Value" in df.columns:
        pubchem_value = col(df, "PubChem Standard Value")
        mask = value.map(parse_numeric).isna() & pubchem_value.map(parse_numeric).notna()
        value.loc[mask] = pubchem_value.loc[mask]
        units.loc[mask] = col(df, "PubChem Standard Value Units").loc[mask].replace("", np.nan).fillna("MICROMOLAR")

    return pd.DataFrame(
        {
            "source": "PubChem",
            "source_record_id": col(df, "aid").astype(str) + ":" + col(df, "PUBCHEM_RESULT_TAG").astype(str),
            "compound_id": col(df, "PUBCHEM_CID").fillna(col(df, "PUBCHEM_SID")),
            "compound_name": col(df, "Ligand"),
            "smiles": col(df, "PUBCHEM_EXT_DATASOURCE_SMILES"),
            "inchi_key": "",
            "target_name": col(df, "Target"),
            "target_id": col(df, "Target Accession(s)"),
            "endpoint": endpoint.replace("", np.nan).fillna("IC50"),
            "relation": col(df, "Standard Relation").fillna("="),
            "value_raw": value,
            "standard_units": units,
            "assay_description": col(df, "assay_description").fillna(col(df, "assay_name")),
            "assay_type": col(df, "PUBCHEM_ACTIVITY_OUTCOME"),
            "document_id": col(df, "aid"),
            "document_year": "",
            "reference": "PubChem BioAssay AID " + col(df, "aid").astype(str),
            "source_detail": "PubChem BioAssay CSV",
        }
    )


def normalize_activity_table(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize endpoints, units, relations, and p-values."""

    if df.empty:
        return df

    out = df.copy()
    out["endpoint"] = out["endpoint"].map(normalize_endpoint)
    out["relation"] = [
        extract_relation(value, relation)
        for value, relation in zip(out["value_raw"], out.get("relation", ""), strict=False)
    ]
    values = out["value_raw"].map(parse_numeric)
    factors = out["standard_units"].map(unit_factor_to_nm)
    out["value_nm"] = values * factors
    out["p_value"] = out["value_nm"].map(p_value_from_nm)
    out["is_exact"] = out["relation"].fillna("").astype(str).str.strip().isin(["", "="])
    out["is_core_endpoint"] = out["endpoint"].isin(BIOACTIVITY_ENDPOINTS)
    out["smiles"] = out["smiles"].fillna("").astype(str).str.strip()
    out = out.replace({np.inf: np.nan, -np.inf: np.nan})
    return out


def aggregate_compounds(activity: pd.DataFrame, *, exact_only: bool = True) -> pd.DataFrame:
    """Aggregate measurement-level bioactivity to a compound-level modeling table."""

    if activity.empty:
        return pd.DataFrame()

    data = activity[
        activity["is_core_endpoint"]
        & activity["smiles"].notna()
        & (activity["smiles"] != "")
        & activity["value_nm"].notna()
        & (activity["value_nm"] > 0)
        & activity["p_value"].notna()
    ].copy()
    if exact_only:
        exact = data[data["is_exact"]]
        if not exact.empty:
            data = exact

    if data.empty:
        return pd.DataFrame()

    def join_unique(values: pd.Series, limit: int = 12) -> str:
        items = [str(v) for v in values.dropna().unique() if str(v).strip()]
        return ";".join(sorted(items)[:limit])

    grouped = data.groupby("smiles", dropna=False)
    compound = grouped.agg(
        n_measurements=("p_value", "size"),
        n_sources=("source", "nunique"),
        p_activity_median=("p_value", "median"),
        p_activity_best=("p_value", "max"),
        value_nm_median=("value_nm", "median"),
        value_nm_best=("value_nm", "min"),
        endpoints=("endpoint", join_unique),
        sources=("source", join_unique),
        compound_ids=("compound_id", join_unique),
        target_names=("target_name", join_unique),
        document_years=("document_year", join_unique),
    ).reset_index()

    compound["potency_class"] = np.select(
        [
            compound["value_nm_median"] <= 100,
            compound["value_nm_median"] <= 1000,
            compound["value_nm_median"] <= 10000,
        ],
        ["high_potency_<=100nM", "moderate_100nM_to_1uM", "weak_1uM_to_10uM"],
        default="low_or_inactive_>10uM",
    )
    compound["active_100nM"] = compound["value_nm_median"] <= 100
    compound["active_1uM"] = compound["value_nm_median"] <= 1000
    return compound.sort_values(["p_activity_median", "n_measurements"], ascending=[False, False])


def curate_pk_admet(molecule_activity: pd.DataFrame) -> pd.DataFrame:
    """Filter molecule-level ChEMBL rows for likely PK/ADMET observations."""

    if molecule_activity.empty:
        return pd.DataFrame()

    df = molecule_activity.copy()
    desc = (
        col(df, "assay_description").fillna("").astype(str).str.lower()
        + " "
        + col(df, "standard_type").fillna("").astype(str).str.lower()
    )
    keyword_mask = desc.apply(lambda text: any(keyword in text for keyword in PK_ADMET_KEYWORDS))
    type_mask = col(df, "standard_type").fillna("").astype(str).isin(PK_ADMET_STANDARD_TYPES)
    data = df[keyword_mask | type_mask].copy()

    if data.empty:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "source": "ChEMBL",
            "activity_id": col(data, "activity_id"),
            "molecule_chembl_id": col(data, "molecule_chembl_id"),
            "smiles": col(data, "canonical_smiles"),
            "standard_type": col(data, "standard_type"),
            "standard_relation": col(data, "standard_relation"),
            "standard_value": col(data, "standard_value"),
            "standard_units": col(data, "standard_units"),
            "assay_description": col(data, "assay_description"),
            "assay_type": col(data, "assay_type"),
            "target_chembl_id": col(data, "target_chembl_id"),
            "target_pref_name": col(data, "target_pref_name"),
            "document_chembl_id": col(data, "document_chembl_id"),
            "document_year": col(data, "document_year"),
        }
    )
    return out.drop_duplicates()


def curate_herg_compounds(herg_activity: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hERG/KCNH2 activity and add coarse liability labels."""

    compounds = aggregate_compounds(herg_activity, exact_only=True)
    if compounds.empty:
        return compounds

    compounds = compounds.rename(
        columns={
            "p_activity_median": "p_herg_median",
            "p_activity_best": "p_herg_best",
            "value_nm_median": "herg_value_nm_median",
            "value_nm_best": "herg_value_nm_best",
        }
    )
    compounds["herg_blocker_label"] = np.nan
    compounds.loc[compounds["herg_value_nm_median"] <= 10_000, "herg_blocker_label"] = 1.0
    compounds.loc[compounds["herg_value_nm_median"] >= 30_000, "herg_blocker_label"] = 0.0
    compounds["herg_label_policy"] = "<=10 uM blocker, >=30 uM non-blocker, 10-30 uM omitted"
    compounds["target_names"] = HERG_TARGET["name"]
    return compounds


def source_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["source", "n_measurements", "n_compounds"])
    return (
        df.groupby("source")
        .agg(n_measurements=("source", "size"), n_compounds=("smiles", "nunique"))
        .reset_index()
        .sort_values("n_measurements", ascending=False)
    )


def write_processed_tables(
    *,
    chembl_menin_raw: pd.DataFrame,
    bindingdb_raw: pd.DataFrame,
    pubchem_raw: pd.DataFrame,
    pubchem_catalog: pd.DataFrame,
    herg_raw: pd.DataFrame,
    pk_raw: pd.DataFrame,
    processed_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Create all processed tables from raw dataframes."""

    processed_dir.mkdir(parents=True, exist_ok=True)

    menin_parts = [
        chembl_to_long(chembl_menin_raw, source_detail=f"ChEMBL target {MENIN_TARGET['chembl_id']}"),
    ]
    if not bindingdb_raw.empty:
        from .bindingdb import bindingdb_to_long

        menin_parts.append(bindingdb_to_long(bindingdb_raw))
    if not pubchem_raw.empty:
        menin_parts.append(pubchem_to_long(pubchem_raw, pubchem_catalog))

    menin_activity = normalize_activity_table(pd.concat(menin_parts, ignore_index=True, sort=False))
    menin_activity.to_csv(processed_dir / "menin_activity_measurements.csv", index=False)

    menin_compounds = aggregate_compounds(menin_activity, exact_only=True)
    menin_compounds.to_csv(processed_dir / "menin_compounds_curated.csv", index=False)

    herg_activity = normalize_activity_table(
        chembl_to_long(herg_raw, source_detail=f"ChEMBL target {HERG_TARGET['chembl_id']}")
    )
    herg_activity.to_csv(processed_dir / "herg_activity_measurements.csv", index=False)
    herg_compounds = curate_herg_compounds(herg_activity)
    herg_compounds.to_csv(processed_dir / "herg_compounds_curated.csv", index=False)

    pk_admet = curate_pk_admet(pk_raw)
    pk_admet.to_csv(processed_dir / "pk_admet_observations.csv", index=False)

    source_summary(menin_activity).to_csv(processed_dir / "source_summary.csv", index=False)

    return {
        "menin_activity": menin_activity,
        "menin_compounds": menin_compounds,
        "herg_activity": herg_activity,
        "herg_compounds": herg_compounds,
        "pk_admet": pk_admet,
    }
