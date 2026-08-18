"""Deterministic chemical intelligence and evidence-first prioritization.

This module deliberately separates observed Menin activity, model-derived hERG
triage, medicinal-chemistry heuristics, and evidence completeness.  Missing or
out-of-domain safety predictions are represented as unknown and never receive
favorable score credit.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .chemistry import standardize_smiles
from .features import RDKIT_AVAILABLE, rdkit_descriptors, scaffold_key
from .provenance import sha256_file

try:  # Publication analysis requires RDKit; import remains explicit for diagnostics.
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import QED, FilterCatalog, rdFingerprintGenerator, rdMMPA
    from rdkit.ML.Cluster import Butina
except ImportError:  # pragma: no cover - settings reject enabled analysis without RDKit.
    Chem = DataStructs = rdBase = None  # type: ignore[assignment]
    FilterCatalog = QED = rdFingerprintGenerator = rdMMPA = Butina = None  # type: ignore[assignment]


ANALYSIS_SCHEMA_VERSION = "chemical-intelligence-v1"
DEFAULT_ALERT_CATALOGS = ("PAINS", "BRENK", "NIH")


def _as_bool(series: pd.Series) -> pd.Series:
    """Parse booleans without treating a non-empty string such as ``False`` as true."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.fillna("").astype(str).str.strip().str.casefold()
    return normalized.isin({"true", "1", "yes"})


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def _stable_id(prefix: str, value: str, length: int = 16) -> str:
    return f"{prefix}-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:length].upper()}"


def _join_unique(values: Iterable[object]) -> str:
    cleaned = {
        str(value).strip()
        for value in values
        if value is not None and not (isinstance(value, float) and math.isnan(value)) and str(value).strip()
    }
    return ";".join(sorted(cleaned))


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _analysis_config(settings: dict[str, Any]) -> dict[str, Any]:
    return dict(settings.get("analysis", {}))


def _primary_population(
    scored: pd.DataFrame,
    *,
    endpoint: str,
    assay_family: str,
) -> pd.DataFrame:
    required = {
        "structure_id",
        "endpoint",
        "assay_family",
        "standardized_smiles",
        "standard_inchi_key",
        "p_activity_median",
        "predicted_herg_blocker_probability",
        "herg_inside_applicability_domain",
    }
    _require_columns(scored, required, label="Menin hERG-scored candidate table")
    population = scored[
        scored["endpoint"].fillna("").astype(str).str.casefold().eq(endpoint.casefold())
        & scored["assay_family"].fillna("").astype(str).str.casefold().eq(assay_family.casefold())
    ].copy()
    if population.empty:
        raise ValueError(f"No candidate rows match primary task {endpoint} × {assay_family}")

    population["structure_id"] = population["structure_id"].fillna("").astype(str).str.strip()
    if population["structure_id"].eq("").any():
        raise ValueError("Primary candidate population contains missing structure_id values")
    conflicts: list[str] = []
    comparison_columns = [
        "standardized_smiles",
        "standard_inchi_key",
        "p_activity_median",
        "predicted_herg_blocker_probability",
        "herg_inside_applicability_domain",
    ]
    for structure_id, group in population.groupby("structure_id", sort=True):
        if len(group) > 1 and any(group[column].nunique(dropna=False) > 1 for column in comparison_columns):
            conflicts.append(str(structure_id))
    if conflicts:
        raise ValueError(f"Conflicting duplicate candidate identities: {conflicts[:10]}")
    population = population.drop_duplicates("structure_id").sort_values("structure_id").reset_index(drop=True)
    population["p_activity_median"] = pd.to_numeric(population["p_activity_median"], errors="coerce")
    population["predicted_herg_blocker_probability"] = pd.to_numeric(
        population["predicted_herg_blocker_probability"], errors="coerce"
    )
    return population


def _build_alert_catalogs(names: Iterable[str]) -> dict[str, Any]:
    if not RDKIT_AVAILABLE or FilterCatalog is None:
        raise ImportError("RDKit is required for structural-alert analysis")
    catalogs: dict[str, Any] = {}
    for name in names:
        normalized = str(name).strip().upper()
        params = FilterCatalog.FilterCatalogParams()
        enum = getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, normalized, None)
        if enum is None:
            raise ValueError(f"Unsupported RDKit alert catalog: {name}")
        params.AddCatalog(enum)
        catalogs[normalized] = FilterCatalog.FilterCatalog(params)
    return catalogs


def _molecules_and_fingerprints(
    smiles: pd.Series,
    *,
    radius: int,
    n_bits: int,
    include_chirality: bool,
) -> tuple[list[Any], list[Any | None]]:
    if not RDKIT_AVAILABLE or Chem is None or rdFingerprintGenerator is None:
        raise ImportError("RDKit is required for chemical-intelligence analysis")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=include_chirality,
    )
    molecules: list[Any] = []
    fingerprints: list[Any | None] = []
    for value in smiles.fillna("").astype(str):
        molecule = Chem.MolFromSmiles(value)
        molecules.append(molecule)
        fingerprints.append(generator.GetFingerprint(molecule) if molecule is not None else None)
    return molecules, fingerprints


def _nearest_neighbors(
    fingerprints: list[Any | None],
    structure_ids: pd.Series,
) -> tuple[np.ndarray, list[str]]:
    maxima = np.zeros(len(fingerprints), dtype=float)
    neighbors = [""] * len(fingerprints)
    for index, fingerprint in enumerate(fingerprints):
        if fingerprint is None:
            continue
        references = [
            candidate for position, candidate in enumerate(fingerprints) if position != index and candidate
        ]
        reference_positions = [
            position for position, candidate in enumerate(fingerprints) if position != index and candidate
        ]
        if not references:
            continue
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, references)
        best_similarity = max(similarities)
        tied_positions = [
            reference_positions[position]
            for position, similarity in enumerate(similarities)
            if math.isclose(float(similarity), float(best_similarity), rel_tol=0.0, abs_tol=1e-12)
        ]
        best_position = min(tied_positions, key=lambda position: str(structure_ids.iloc[position]))
        maxima[index] = float(best_similarity)
        neighbors[index] = str(structure_ids.iloc[best_position])
    return maxima, neighbors


def _property_violations(row: pd.Series, windows: dict[str, Any]) -> tuple[int, str]:
    violations: list[str] = []
    for column, bounds in sorted(windows.items()):
        if column not in row or not isinstance(bounds, dict):
            continue
        value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
        if pd.isna(value):
            violations.append(f"{column}:missing")
            continue
        lower = bounds.get("min")
        upper = bounds.get("max")
        if lower is not None and float(value) < float(lower):
            violations.append(f"{column}:below_min")
        if upper is not None and float(value) > float(upper):
            violations.append(f"{column}:above_max")
    return len(violations), ";".join(violations)


def medicinal_chemistry_profiles(
    population: pd.DataFrame,
    *,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, list[Any], list[Any | None], list[Any | None]]:
    """Add interpretable RDKit properties, alerts, identity groups, and local novelty."""

    config = _analysis_config(settings)
    modeling = settings.get("modeling", {})
    radius = int(config.get("fingerprint_radius", modeling.get("fingerprint_radius", 2)))
    n_bits = int(config.get("fingerprint_bits", modeling.get("fingerprint_bits", 2048)))
    medicinal = config.get("medicinal_chemistry", {})
    alert_names = [str(value).upper() for value in medicinal.get("alert_catalogs", DEFAULT_ALERT_CATALOGS)]
    windows = dict(medicinal.get("property_windows", {}))

    profiles = population.copy()
    descriptor_frame = rdkit_descriptors(profiles["standardized_smiles"])
    profiles = pd.concat([profiles.reset_index(drop=True), descriptor_frame.reset_index(drop=True)], axis=1)
    molecules, achiral = _molecules_and_fingerprints(
        profiles["standardized_smiles"], radius=radius, n_bits=n_bits, include_chirality=False
    )
    _, chiral = _molecules_and_fingerprints(
        profiles["standardized_smiles"], radius=radius, n_bits=n_bits, include_chirality=True
    )
    catalogs = _build_alert_catalogs(alert_names)
    qed_values: list[float] = []
    alert_payload: dict[str, list[Any]] = {f"{name.casefold()}_alert_count": [] for name in alert_names}
    alert_payload.update({f"{name.casefold()}_alerts": [] for name in alert_names})
    for molecule in molecules:
        if molecule is None:
            qed_values.append(float("nan"))
            for name in alert_names:
                alert_payload[f"{name.casefold()}_alert_count"].append(0)
                alert_payload[f"{name.casefold()}_alerts"].append("")
            continue
        qed_values.append(float(QED.qed(molecule)))
        for name, catalog in catalogs.items():
            descriptions = sorted({str(entry.GetDescription()) for entry in catalog.GetMatches(molecule)})
            alert_payload[f"{name.casefold()}_alert_count"].append(len(descriptions))
            alert_payload[f"{name.casefold()}_alerts"].append(";".join(descriptions))
    profiles["qed"] = qed_values
    for column, values in alert_payload.items():
        profiles[column] = values

    scaffold_values = [scaffold_key(value) for value in profiles["standardized_smiles"]]
    profiles["scaffold_key"] = [value[0] for value in scaffold_values]
    profiles["scaffold_method"] = [value[1] for value in scaffold_values]
    profiles["series_id"] = profiles["scaffold_key"].map(lambda value: _stable_id("SER", str(value)))
    profiles["series_size"] = profiles.groupby("series_id")["structure_id"].transform("size")
    profiles["connectivity_key"] = profiles["standard_inchi_key"].fillna("").astype(str).str.split("-").str[0]
    profiles.loc[profiles["connectivity_key"].eq(""), "connectivity_key"] = profiles.loc[
        profiles["connectivity_key"].eq(""), "structure_id"
    ].map(lambda value: f"missing:{value}")
    profiles["connectivity_group_size"] = profiles.groupby("connectivity_key")["structure_id"].transform(
        "size"
    )

    achiral_similarity, achiral_neighbor = _nearest_neighbors(achiral, profiles["structure_id"])
    chiral_similarity, chiral_neighbor = _nearest_neighbors(chiral, profiles["structure_id"])
    profiles["nearest_neighbor_achiral_tanimoto"] = achiral_similarity
    profiles["nearest_neighbor_achiral_structure_id"] = achiral_neighbor
    profiles["nearest_neighbor_chiral_tanimoto"] = chiral_similarity
    profiles["nearest_neighbor_chiral_structure_id"] = chiral_neighbor
    profiles["local_novelty_achiral"] = 1.0 - profiles["nearest_neighbor_achiral_tanimoto"]

    property_results = profiles.apply(lambda row: _property_violations(row, windows), axis=1)
    profiles["property_window_violation_count"] = [value[0] for value in property_results]
    profiles["property_window_violations"] = [value[1] for value in property_results]
    n_windows = max(1, len(windows))
    profiles["property_desirability"] = (1.0 - profiles["property_window_violation_count"] / n_windows).clip(
        0.0, 1.0
    )
    profiles["ro5_violation_count"] = (
        profiles["mol_wt"].gt(500).astype(int)
        + profiles["logp"].gt(5).astype(int)
        + profiles["h_bond_donors"].gt(5).astype(int)
        + profiles["h_bond_acceptors"].gt(10).astype(int)
    )
    profiles["veber_violation_count"] = profiles["rotatable_bonds"].gt(10).astype(int) + profiles["tpsa"].gt(
        140
    ).astype(int)
    potency = pd.to_numeric(profiles["p_activity_median"], errors="coerce")
    heavy_atoms = pd.to_numeric(profiles["heavy_atom_count"], errors="coerce").replace(0, np.nan)
    profiles["apparent_ligand_efficiency_kcal_per_mol_per_heavy_atom"] = 1.364 * potency / heavy_atoms
    profiles["apparent_lipophilic_ligand_efficiency"] = potency - profiles["logp"]
    return profiles, molecules, achiral, chiral


def assign_similarity_clusters(
    profiles: pd.DataFrame,
    fingerprints: list[Any | None],
    *,
    similarity_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign deterministic Butina clusters using a user-facing similarity threshold."""

    valid_positions = [index for index, fingerprint in enumerate(fingerprints) if fingerprint is not None]
    members = profiles[["structure_id", "series_id"]].copy()
    members["similarity_cluster_id"] = ""
    members["similarity_cluster_size"] = 0
    members["similarity_cluster_representative"] = ""
    if not valid_positions:
        return members, pd.DataFrame(
            columns=["similarity_cluster_id", "cluster_size", "representative_structure_id"]
        )
    valid_fingerprints = [fingerprints[index] for index in valid_positions]
    distances: list[float] = []
    for index in range(1, len(valid_fingerprints)):
        similarities = DataStructs.BulkTanimotoSimilarity(
            valid_fingerprints[index], valid_fingerprints[:index]
        )
        distances.extend(1.0 - float(value) for value in similarities)
    raw_clusters = Butina.ClusterData(
        distances,
        len(valid_fingerprints),
        distThresh=1.0 - float(similarity_threshold),
        isDistData=True,
        reordering=True,
    )
    cluster_records: list[dict[str, Any]] = []
    assignments: dict[int, tuple[str, int, str]] = {}
    for raw_cluster in raw_clusters:
        positions = sorted(valid_positions[index] for index in raw_cluster)
        structure_ids = sorted(str(profiles.iloc[position]["structure_id"]) for position in positions)
        cluster_id = _stable_id("CLU", "\0".join(structure_ids))
        if len(positions) == 1:
            representative_position = positions[0]
        else:
            candidates: list[tuple[float, str, int]] = []
            for position in positions:
                other_fingerprints = [fingerprints[other] for other in positions if other != position]
                total_similarity = sum(
                    DataStructs.BulkTanimotoSimilarity(fingerprints[position], other_fingerprints)
                )
                candidates.append(
                    (-float(total_similarity), str(profiles.iloc[position]["structure_id"]), position)
                )
            representative_position = min(candidates)[2]
        representative = str(profiles.iloc[representative_position]["structure_id"])
        for position in positions:
            assignments[position] = (cluster_id, len(positions), representative)
        cluster_records.append(
            {
                "similarity_cluster_id": cluster_id,
                "cluster_size": len(positions),
                "representative_structure_id": representative,
                "member_structure_ids": ";".join(structure_ids),
                "n_scaffold_series": int(profiles.iloc[positions]["series_id"].nunique()),
                "median_p_activity": float(
                    pd.to_numeric(profiles.iloc[positions]["p_activity_median"], errors="coerce").median()
                ),
                "maximum_p_activity": float(
                    pd.to_numeric(profiles.iloc[positions]["p_activity_median"], errors="coerce").max()
                ),
            }
        )
    for position, assignment in assignments.items():
        members.loc[position, "similarity_cluster_id"] = assignment[0]
        members.loc[position, "similarity_cluster_size"] = assignment[1]
        members.loc[position, "similarity_cluster_representative"] = assignment[2]
    summary = pd.DataFrame(cluster_records).sort_values(
        ["cluster_size", "similarity_cluster_id"], ascending=[False, True]
    )
    return members, summary.reset_index(drop=True)


def _context_sets(measurements: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    context: dict[str, dict[str, set[str]]] = {}
    if measurements.empty or "structure_id" not in measurements:
        return context
    for structure_id, group in measurements.groupby("structure_id", sort=True):
        context[str(structure_id)] = {
            column: {
                str(value).strip()
                for value in group.get(column, pd.Series(dtype=str)).dropna()
                if str(value).strip()
            }
            for column in ("assay_id", "document_id", "source")
        }
    return context


def _pair_context(
    structure_a: str,
    structure_b: str,
    contexts: dict[str, dict[str, set[str]]],
) -> tuple[bool, bool, str]:
    left = contexts.get(structure_a, {})
    right = contexts.get(structure_b, {})
    same_assay = bool(left.get("assay_id", set()) & right.get("assay_id", set()))
    same_document = bool(left.get("document_id", set()) & right.get("document_id", set()))
    grade = "same_assay" if same_assay else "same_document" if same_document else "cross_context"
    return same_assay, same_document, grade


def identify_activity_cliffs(
    profiles: pd.DataFrame,
    achiral_fingerprints: list[Any | None],
    chiral_fingerprints: list[Any | None],
    *,
    similarity_threshold: float,
    minimum_delta_pactivity: float,
    contexts: dict[str, dict[str, set[str]]],
) -> pd.DataFrame:
    """Find high-similarity, large-potency-difference pairs within one assay task."""

    records: list[dict[str, Any]] = []
    potency = pd.to_numeric(profiles["p_activity_median"], errors="coerce").to_numpy(dtype=float)
    for right in range(1, len(profiles)):
        if achiral_fingerprints[right] is None or not np.isfinite(potency[right]):
            continue
        positions = [
            left
            for left in range(right)
            if achiral_fingerprints[left] is not None and np.isfinite(potency[left])
        ]
        if not positions:
            continue
        achiral_values = DataStructs.BulkTanimotoSimilarity(
            achiral_fingerprints[right], [achiral_fingerprints[left] for left in positions]
        )
        chiral_values = DataStructs.BulkTanimotoSimilarity(
            chiral_fingerprints[right], [chiral_fingerprints[left] for left in positions]
        )
        for left, achiral_similarity, chiral_similarity in zip(
            positions, achiral_values, chiral_values, strict=True
        ):
            delta = abs(float(potency[right] - potency[left]))
            if float(achiral_similarity) < similarity_threshold or delta < minimum_delta_pactivity:
                continue
            row_left = profiles.iloc[left]
            row_right = profiles.iloc[right]
            if str(row_left["structure_id"]) <= str(row_right["structure_id"]):
                first, second = row_left, row_right
            else:
                first, second = row_right, row_left
            higher = (
                first if float(first["p_activity_median"]) >= float(second["p_activity_median"]) else second
            )
            same_assay, same_document, evidence = _pair_context(
                str(first["structure_id"]), str(second["structure_id"]), contexts
            )
            similarity = float(achiral_similarity)
            records.append(
                {
                    "structure_id_a": first["structure_id"],
                    "structure_id_b": second["structure_id"],
                    "standardized_smiles_a": first["standardized_smiles"],
                    "standardized_smiles_b": second["standardized_smiles"],
                    "p_activity_a": first["p_activity_median"],
                    "p_activity_b": second["p_activity_median"],
                    "absolute_delta_pactivity": delta,
                    "higher_potency_structure_id": higher["structure_id"],
                    "achiral_morgan_tanimoto": similarity,
                    "chiral_morgan_tanimoto": float(chiral_similarity),
                    "sali_achiral": delta / (1.0 - similarity) if similarity < 0.999 else np.nan,
                    "achiral_fingerprint_identical": math.isclose(similarity, 1.0, abs_tol=1e-12),
                    "same_connectivity_key": first["connectivity_key"] == second["connectivity_key"],
                    "same_scaffold_series": first["series_id"] == second["series_id"],
                    "shared_assay_id": same_assay,
                    "shared_document_id": same_document,
                    "evidence_context_grade": evidence,
                }
            )
    columns = [
        "structure_id_a",
        "structure_id_b",
        "standardized_smiles_a",
        "standardized_smiles_b",
        "p_activity_a",
        "p_activity_b",
        "absolute_delta_pactivity",
        "higher_potency_structure_id",
        "achiral_morgan_tanimoto",
        "chiral_morgan_tanimoto",
        "sali_achiral",
        "achiral_fingerprint_identical",
        "same_connectivity_key",
        "same_scaffold_series",
        "shared_assay_id",
        "shared_document_id",
        "evidence_context_grade",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(records, columns=columns)
        .sort_values(
            ["absolute_delta_pactivity", "achiral_morgan_tanimoto", "structure_id_a", "structure_id_b"],
            ascending=[False, False, True, True],
        )
        .reset_index(drop=True)
    )


def identify_matched_molecular_pairs(
    profiles: pd.DataFrame,
    molecules: list[Any],
    achiral_fingerprints: list[Any | None],
    *,
    max_variable_heavy_atoms: int,
    max_variable_fraction: float,
    min_core_heavy_atoms: int,
    minimum_delta_pactivity: float,
    contexts: dict[str, dict[str, set[str]]],
) -> pd.DataFrame:
    """Generate conservative single-cut matched molecular pairs with RDKit MMPA."""

    fragment_index: dict[str, set[tuple[int, str, int, int]]] = defaultdict(set)
    for position, molecule in enumerate(molecules):
        if molecule is None:
            continue
        total_heavy = max(1, int(molecule.GetNumHeavyAtoms()))
        fragments = rdMMPA.FragmentMol(
            molecule,
            minCuts=1,
            maxCuts=1,
            maxCutBonds=20,
            resultsAsMols=False,
        )
        for _, fragment_smiles in fragments:
            parts = str(fragment_smiles).split(".")
            if len(parts) != 2:
                continue
            parsed: list[tuple[int, str]] = []
            for part in parts:
                fragment = Chem.MolFromSmiles(part)
                parsed.append((int(fragment.GetNumHeavyAtoms()) if fragment is not None else 0, part))
            parsed.sort(key=lambda value: (-value[0], value[1]))
            core_heavy, core = parsed[0]
            variable_heavy, variable = parsed[1]
            if (
                core_heavy >= min_core_heavy_atoms
                and variable_heavy <= max_variable_heavy_atoms
                and variable_heavy / total_heavy <= max_variable_fraction
            ):
                fragment_index[core].add((position, variable, core_heavy, variable_heavy))

    pair_candidates: dict[tuple[int, int], tuple[int, str, str, str, int, int]] = {}
    for core, raw_items in fragment_index.items():
        items = sorted(raw_items)
        for left_index in range(len(items)):
            for right_index in range(left_index + 1, len(items)):
                left_position, left_variable, core_heavy, left_heavy = items[left_index]
                right_position, right_variable, other_core_heavy, right_heavy = items[right_index]
                if left_position == right_position or left_variable == right_variable:
                    continue
                if left_position > right_position:
                    left_position, right_position = right_position, left_position
                    left_variable, right_variable = right_variable, left_variable
                    left_heavy, right_heavy = right_heavy, left_heavy
                key = (left_position, right_position)
                candidate = (
                    min(core_heavy, other_core_heavy),
                    core,
                    left_variable,
                    right_variable,
                    left_heavy,
                    right_heavy,
                )
                if key not in pair_candidates or candidate[0] > pair_candidates[key][0]:
                    pair_candidates[key] = candidate

    records: list[dict[str, Any]] = []
    for (left, right), candidate in sorted(pair_candidates.items()):
        core_heavy, core, variable_left, variable_right, left_heavy, right_heavy = candidate
        first = profiles.iloc[left]
        second = profiles.iloc[right]
        structure_a = str(first["structure_id"])
        structure_b = str(second["structure_id"])
        if structure_a > structure_b:
            first, second = second, first
            structure_a, structure_b = structure_b, structure_a
            variable_left, variable_right = variable_right, variable_left
            left_heavy, right_heavy = right_heavy, left_heavy
        delta = abs(float(first["p_activity_median"]) - float(second["p_activity_median"]))
        similarity = float(
            DataStructs.TanimotoSimilarity(achiral_fingerprints[left], achiral_fingerprints[right])
        )
        same_assay, same_document, evidence = _pair_context(structure_a, structure_b, contexts)
        records.append(
            {
                "structure_id_a": structure_a,
                "structure_id_b": structure_b,
                "core_smiles": core,
                "variable_fragment_a": variable_left,
                "variable_fragment_b": variable_right,
                "transformation": f"{variable_left}>>{variable_right}",
                "core_heavy_atoms": core_heavy,
                "variable_heavy_atoms_a": left_heavy,
                "variable_heavy_atoms_b": right_heavy,
                "p_activity_a": first["p_activity_median"],
                "p_activity_b": second["p_activity_median"],
                "absolute_delta_pactivity": delta,
                "achiral_morgan_tanimoto": similarity,
                "is_activity_cliff": delta >= minimum_delta_pactivity,
                "shared_assay_id": same_assay,
                "shared_document_id": same_document,
                "evidence_context_grade": evidence,
            }
        )
    columns = [
        "structure_id_a",
        "structure_id_b",
        "core_smiles",
        "variable_fragment_a",
        "variable_fragment_b",
        "transformation",
        "core_heavy_atoms",
        "variable_heavy_atoms_a",
        "variable_heavy_atoms_b",
        "p_activity_a",
        "p_activity_b",
        "absolute_delta_pactivity",
        "achiral_morgan_tanimoto",
        "is_activity_cliff",
        "shared_assay_id",
        "shared_document_id",
        "evidence_context_grade",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(records, columns=columns)
        .sort_values(
            ["is_activity_cliff", "absolute_delta_pactivity", "structure_id_a", "structure_id_b"],
            ascending=[False, False, True, True],
        )
        .reset_index(drop=True)
    )


def _desirability(value: pd.Series, lower: float, upper: float, *, maximize: bool = True) -> pd.Series:
    numeric = pd.to_numeric(value, errors="coerce")
    score = ((numeric - lower) / (upper - lower)).clip(0.0, 1.0)
    return score if maximize else 1.0 - score


def pareto_ranks(values: np.ndarray, eligible: np.ndarray | None = None) -> np.ndarray:
    """Return deterministic non-dominated sorting ranks; ineligible rows receive NaN."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Pareto values must be a two-dimensional matrix")
    valid = np.all(np.isfinite(matrix), axis=1)
    if eligible is not None:
        valid &= np.asarray(eligible, dtype=bool)
    ranks = np.full(len(matrix), np.nan, dtype=float)
    remaining = set(np.flatnonzero(valid).tolist())
    rank = 1
    while remaining:
        front: list[int] = []
        for candidate in sorted(remaining):
            dominated = False
            for other in remaining:
                if other == candidate:
                    continue
                if np.all(matrix[other] >= matrix[candidate]) and np.any(matrix[other] > matrix[candidate]):
                    dominated = True
                    break
            if not dominated:
                front.append(candidate)
        if not front:  # Defensive; strict dominance should always yield a front.
            raise RuntimeError("Could not resolve Pareto front")
        ranks[front] = rank
        remaining.difference_update(front)
        rank += 1
    return ranks


def _rank_series(
    frame: pd.DataFrame,
    eligible: pd.Series,
    columns: list[str],
    ascending: list[bool],
) -> pd.Series:
    result = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    subset = frame.loc[eligible].sort_values(columns, ascending=ascending, kind="stable")
    result.loc[subset.index] = np.arange(1, len(subset) + 1)
    return result


def prioritize_candidates(
    profiles: pd.DataFrame,
    *,
    observed_herg: pd.DataFrame,
    pk: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct transparent tiers, scores, Pareto fronts, traces, and data gaps."""

    config = _analysis_config(settings)
    policy = dict(config.get("prioritization", {}))
    herg_cfg = settings.get("herg", {})
    herg_endpoint = str(config.get("herg_endpoint", herg_cfg.get("primary_endpoint", "IC50")))
    herg_family = str(
        config.get("herg_assay_family", herg_cfg.get("primary_assay_family", "electrophysiology_functional"))
    )
    candidates = profiles.copy()
    observed_columns = [
        "structure_id",
        "herg_blocker_label",
        "p_herg_median",
        "herg_value_nm_median",
        "n_measurements",
    ]
    if not observed_herg.empty:
        observed = observed_herg[
            observed_herg["endpoint"].fillna("").astype(str).str.casefold().eq(herg_endpoint.casefold())
            & observed_herg["assay_family"].fillna("").astype(str).str.casefold().eq(herg_family.casefold())
        ].copy()
        observed = observed[[column for column in observed_columns if column in observed.columns]].rename(
            columns={
                "herg_blocker_label": "observed_herg_blocker_label",
                "p_herg_median": "observed_p_herg_median",
                "herg_value_nm_median": "observed_herg_value_nm_median",
                "n_measurements": "observed_herg_measurement_count",
            }
        )
        if observed["structure_id"].duplicated().any():
            raise ValueError("Primary observed hERG table contains duplicate structure identities")
        candidates = candidates.merge(observed, on="structure_id", how="left", validate="one_to_one")
    else:
        candidates["observed_herg_blocker_label"] = np.nan
        candidates["observed_p_herg_median"] = np.nan
        candidates["observed_herg_value_nm_median"] = np.nan
        candidates["observed_herg_measurement_count"] = 0

    if not pk.empty and "structure_id" in pk:
        pk_summary = (
            pk.groupby("structure_id", as_index=False)
            .agg(
                pk_observation_count=("structure_id", "size"),
                pk_endpoint_count=("admet_endpoint", "nunique"),
                pk_endpoints=("admet_endpoint", _join_unique),
                pk_species=("species", _join_unique),
                pk_matrices=("matrix", _join_unique),
            )
            .sort_values("structure_id")
        )
        candidates = candidates.merge(pk_summary, on="structure_id", how="left", validate="one_to_one")
    for column in ("pk_observation_count", "pk_endpoint_count"):
        candidates[column] = pd.to_numeric(candidates.get(column, 0), errors="coerce").fillna(0).astype(int)
    for column in ("pk_endpoints", "pk_species", "pk_matrices"):
        candidates[column] = candidates.get(column, pd.Series("", index=candidates.index)).fillna("")

    inside = _as_bool(candidates["herg_inside_applicability_domain"])
    probability = pd.to_numeric(candidates["predicted_herg_blocker_probability"], errors="coerce")
    observed_label = pd.to_numeric(candidates["observed_herg_blocker_label"], errors="coerce")
    lower_herg = float(policy.get("lower_herg_probability", 0.30))
    high_herg = float(policy.get("high_herg_probability", 0.70))
    status = pd.Series("unknown_missing_prediction", index=candidates.index, dtype=object)
    status.loc[probability.notna() & ~inside] = "unknown_outside_applicability_domain"
    status.loc[probability.notna() & inside & probability.lt(lower_herg)] = "predicted_lower_concern"
    status.loc[probability.notna() & inside & probability.ge(lower_herg) & probability.lt(high_herg)] = (
        "predicted_indeterminate"
    )
    status.loc[probability.notna() & inside & probability.ge(high_herg)] = "predicted_high_concern"
    status.loc[observed_label.eq(0)] = "observed_non_blocker"
    status.loc[observed_label.eq(1)] = "observed_blocker"
    candidates["herg_evidence_status"] = status
    candidates["herg_probability_used_for_decision"] = probability.where(inside & observed_label.isna())
    candidates["herg_prediction_used"] = inside & probability.notna() & observed_label.isna()
    candidates["observed_herg_used"] = observed_label.notna()

    potency_lower = float(policy.get("potency_desirability_lower", 6.0))
    potency_upper = float(policy.get("potency_desirability_upper", 9.0))
    candidates["potency_desirability"] = _desirability(
        candidates.get("p_activity_min", candidates["p_activity_median"]), potency_lower, potency_upper
    )
    candidates["qed_desirability"] = pd.to_numeric(candidates["qed"], errors="coerce").clip(0.0, 1.0)
    measurement_score = (
        pd.to_numeric(candidates.get("n_exact_measurements", 0), errors="coerce").fillna(0) / 3.0
    ).clip(0.0, 1.0)
    source_score = (pd.to_numeric(candidates.get("n_sources", 0), errors="coerce").fillna(0) / 2.0).clip(
        0.0, 1.0
    )
    spread = pd.to_numeric(candidates.get("activity_range_log10", np.nan), errors="coerce")
    consistency_score = (
        1.0 - spread.fillna(1.0) / max(1e-9, float(policy.get("maximum_activity_range_log10", 1.0)))
    ).clip(0.0, 1.0)
    candidates["evidence_desirability"] = (measurement_score + source_score + consistency_score) / 3.0
    candidates["herg_desirability"] = np.nan
    candidates.loc[observed_label.eq(0), "herg_desirability"] = 1.0
    candidates.loc[observed_label.eq(1), "herg_desirability"] = 0.0
    candidates.loc[inside & observed_label.isna() & probability.notna(), "herg_desirability"] = (
        1.0 - probability.loc[inside & observed_label.isna() & probability.notna()]
    )

    weights = {
        "potency": 0.35,
        "qed": 0.15,
        "property": 0.15,
        "evidence": 0.10,
        "herg": 0.25,
        **dict(policy.get("weights", {})),
    }
    component_columns = {
        "potency": "potency_desirability",
        "qed": "qed_desirability",
        "property": "property_desirability",
        "evidence": "evidence_desirability",
        "herg": "herg_desirability",
    }
    no_safety_names = [name for name in component_columns if name != "herg"]
    no_safety_weight = sum(float(weights[name]) for name in no_safety_names)
    candidates["discovery_score_without_safety"] = (
        sum(candidates[component_columns[name]] * float(weights[name]) for name in no_safety_names)
        / no_safety_weight
    )
    complete_weight = sum(float(weights[name]) for name in component_columns)
    candidates["complete_evidence_score"] = (
        sum(candidates[component_columns[name]] * float(weights[name]) for name in component_columns)
        / complete_weight
    )
    candidates.loc[candidates["herg_desirability"].isna(), "complete_evidence_score"] = np.nan

    invalid = pd.to_numeric(candidates["invalid_structure"], errors="coerce").fillna(1).gt(0)
    heterogeneous = _as_bool(
        candidates.get("is_activity_heterogeneous", pd.Series(False, index=candidates.index))
    )
    strong = candidates["p_activity_median"].ge(float(policy.get("strong_pactivity", 7.0)))
    spread_ok = spread.fillna(np.inf).le(float(policy.get("maximum_activity_range_log10", 1.0)))
    property_ok = candidates["property_window_violation_count"].le(
        int(policy.get("maximum_property_violations", 1))
    )
    pains_count = pd.to_numeric(candidates.get("pains_alert_count", 0), errors="coerce").fillna(0)
    pains_ok = (
        pains_count.eq(0)
        if bool(policy.get("require_no_pains", True))
        else pd.Series(True, index=candidates.index)
    )
    chemistry_gate = ~invalid & ~heterogeneous & strong & spread_ok & property_ok & pains_ok
    candidates["chemistry_evidence_gate"] = chemistry_gate

    balanced = status.isin({"observed_non_blocker", "predicted_lower_concern"})
    unknown_safety = status.str.startswith("unknown_")
    liability = status.isin({"observed_blocker", "predicted_high_concern", "predicted_indeterminate"})
    tier = pd.Series("priority_5_context_only", index=candidates.index, dtype=object)
    tier.loc[strong & ~chemistry_gate] = "priority_4_chemistry_review"
    tier.loc[chemistry_gate & liability] = "priority_3_potent_liability_flag"
    tier.loc[chemistry_gate & unknown_safety] = "priority_2_potent_safety_data_gap"
    tier.loc[chemistry_gate & balanced] = "priority_1_balanced_public_evidence"
    candidates["experimental_followup_tier"] = tier

    discovery_values = np.column_stack(
        [
            candidates["potency_desirability"],
            candidates["qed_desirability"],
            candidates["property_desirability"],
            candidates["evidence_desirability"],
        ]
    )
    candidates["potency_property_pareto_rank"] = pd.array(
        pareto_ranks(discovery_values, (~invalid).to_numpy()), dtype="Int64"
    )
    safety_values = np.column_stack(
        [
            candidates["potency_desirability"],
            candidates["qed_desirability"],
            candidates["property_desirability"],
            candidates["evidence_desirability"],
            candidates["herg_desirability"],
        ]
    )
    safety_eligible = (~invalid & candidates["herg_desirability"].notna()).to_numpy()
    candidates["complete_evidence_pareto_rank"] = pd.array(
        pareto_ranks(safety_values, safety_eligible), dtype="Int64"
    )
    candidates["discovery_rank_without_safety"] = _rank_series(
        candidates,
        ~invalid & candidates["discovery_score_without_safety"].notna(),
        [
            "potency_property_pareto_rank",
            "discovery_score_without_safety",
            "p_activity_median",
            "structure_id",
        ],
        [True, False, False, True],
    )
    candidates["complete_evidence_rank"] = _rank_series(
        candidates,
        pd.Series(safety_eligible, index=candidates.index),
        ["complete_evidence_pareto_rank", "complete_evidence_score", "p_activity_median", "structure_id"],
        [True, False, False, True],
    )

    trace_records: list[dict[str, Any]] = []
    for _index, row in candidates.iterrows():
        for objective, column in component_columns.items():
            available = pd.notna(row[column])
            trace_records.append(
                {
                    "structure_id": row["structure_id"],
                    "objective": objective,
                    "desirability_column": column,
                    "desirability": row[column],
                    "weight": float(weights[objective]),
                    "included_in_complete_score": bool(available),
                    "weighted_contribution": (
                        float(row[column]) * float(weights[objective]) / complete_weight
                        if available
                        else np.nan
                    ),
                    "missing_policy": "unknown_no_credit" if not available else "observed_or_in_domain",
                }
            )
    trace = pd.DataFrame(trace_records)

    data_gap_records: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        gaps: list[str] = []
        if str(row["herg_evidence_status"]).startswith("unknown_"):
            gaps.append(str(row["herg_evidence_status"]))
        if int(row["pk_observation_count"]) == 0:
            gaps.append("no_context_compatible_pk_observation")
        if int(row.get("n_exact_measurements", 0)) < 2:
            gaps.append("no_replicated_primary_menin_measurement")
        if int(row.get("n_sources", 0)) < 2:
            gaps.append("single_public_source")
        if int(row.get("property_window_violation_count", 0)) > 0:
            gaps.append("medicinal_chemistry_property_review")
        if int(row.get("pains_alert_count", 0)) > 0:
            gaps.append("pains_review_alert")
        data_gap_records.append(
            {
                "structure_id": row["structure_id"],
                "n_data_gaps": len(gaps),
                "data_gaps": ";".join(gaps),
                "requires_herg_measurement": str(row["herg_evidence_status"]).startswith("unknown_"),
                "requires_pk_context": int(row["pk_observation_count"]) == 0,
            }
        )
    gaps = pd.DataFrame(data_gap_records)
    candidates = candidates.sort_values(
        ["experimental_followup_tier", "discovery_rank_without_safety", "structure_id"],
        kind="stable",
    ).reset_index(drop=True)
    return candidates, trace, gaps


def prioritization_sensitivity(
    candidates: pd.DataFrame,
    *,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit rank stability under deterministic leave-one-out and emphasis scenarios."""

    policy = dict(_analysis_config(settings).get("prioritization", {}))
    base_weights = {
        "potency": 0.35,
        "qed": 0.15,
        "property": 0.15,
        "evidence": 0.10,
        "herg": 0.25,
        **dict(policy.get("weights", {})),
    }
    component_columns = {
        "potency": "potency_desirability",
        "qed": "qed_desirability",
        "property": "property_desirability",
        "evidence": "evidence_desirability",
        "herg": "herg_desirability",
    }
    sensitivity_cfg = dict(policy.get("sensitivity", {}))
    emphasis_factor = float(sensitivity_cfg.get("emphasis_factor", 2.0))
    scenarios: dict[str, dict[str, float]] = {"base_complete": dict(base_weights)}
    for objective in component_columns:
        leave_out = dict(base_weights)
        leave_out[objective] = 0.0
        scenarios[f"leave_out_{objective}"] = leave_out
        emphasized = dict(base_weights)
        emphasized[objective] *= emphasis_factor
        scenarios[f"emphasize_{objective}"] = emphasized

    records: list[dict[str, Any]] = []
    for scenario, weights in sorted(scenarios.items()):
        active = [name for name, weight in weights.items() if float(weight) > 0]
        total_weight = sum(float(weights[name]) for name in active)
        available = pd.Series(True, index=candidates.index)
        score = pd.Series(0.0, index=candidates.index, dtype=float)
        for objective in active:
            component = pd.to_numeric(candidates[component_columns[objective]], errors="coerce")
            available &= component.notna() & np.isfinite(component)
            score += component.fillna(0.0) * float(weights[objective]) / total_weight
        valid_structure = pd.to_numeric(candidates["invalid_structure"], errors="coerce").fillna(1).eq(0)
        available &= valid_structure
        ranked = candidates.loc[available, ["structure_id"]].assign(_score=score.loc[available])
        ranked = ranked.sort_values(["_score", "structure_id"], ascending=[False, True], kind="stable")
        ranks = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index)
        for index, row in candidates.iterrows():
            records.append(
                {
                    "structure_id": row["structure_id"],
                    "scenario": scenario,
                    "active_objectives": ";".join(active),
                    "requires_herg_evidence": "herg" in active,
                    "eligible": bool(available.loc[index]),
                    "scenario_score": float(score.loc[index]) if available.loc[index] else np.nan,
                    "scenario_rank": int(ranks.loc[index]) if index in ranks.index else pd.NA,
                }
            )
    sensitivity = pd.DataFrame(records)
    eligible = sensitivity[sensitivity["eligible"]].copy()
    if eligible.empty:
        stability = pd.DataFrame(
            columns=[
                "structure_id",
                "sensitivity_scenarios_eligible",
                "sensitivity_rank_min",
                "sensitivity_rank_median",
                "sensitivity_rank_max",
                "sensitivity_rank_span",
            ]
        )
    else:
        eligible["scenario_rank"] = pd.to_numeric(eligible["scenario_rank"], errors="coerce")
        stability = (
            eligible.groupby("structure_id", as_index=False)
            .agg(
                sensitivity_scenarios_eligible=("scenario", "size"),
                sensitivity_rank_min=("scenario_rank", "min"),
                sensitivity_rank_median=("scenario_rank", "median"),
                sensitivity_rank_max=("scenario_rank", "max"),
            )
            .sort_values("structure_id")
        )
        stability["sensitivity_rank_span"] = (
            stability["sensitivity_rank_max"] - stability["sensitivity_rank_min"]
        )
    return sensitivity, stability


def build_prospective_selection_plan(
    candidates: pd.DataFrame,
    activity_cliffs: pd.DataFrame,
    *,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a diversity-constrained public-data experiment design, not a lead claim."""

    selection_cfg = dict(_analysis_config(settings).get("prospective_selection", {}))
    quotas = {
        "potent_safety_gap": 12,
        "liability_characterization": 8,
        "novel_scaffold_exploration": 8,
        "activity_cliff_confirmation": 8,
        "negative_control": 6,
        "pk_bridge": 6,
        **dict(selection_cfg.get("quotas", {})),
    }
    output_columns = [
        "selection_order",
        "selection_category",
        "selection_rationale",
        "paired_cliff_id",
        "structure_id",
        "standardized_smiles",
        "p_activity_median",
        "series_id",
        "series_size",
        "qed",
        "property_window_violation_count",
        "herg_evidence_status",
        "pk_observation_count",
        "local_novelty_achiral",
        "experimental_followup_tier",
    ]
    summary_columns = ["selection_category", "requested_quota", "selected_structures", "shortfall"]
    if not bool(selection_cfg.get("enabled", True)) or candidates.empty:
        return pd.DataFrame(columns=output_columns), pd.DataFrame(columns=summary_columns)

    maximum_per_series = int(selection_cfg.get("maximum_per_scaffold_series", 2))
    selected: set[str] = set()
    series_counts: dict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []

    def add_row(row: pd.Series, category: str, rationale: str, pair_id: str = "") -> bool:
        structure_id = str(row["structure_id"])
        series_id = str(row.get("series_id", ""))
        if structure_id in selected or series_counts[series_id] >= maximum_per_series:
            return False
        selected.add(structure_id)
        series_counts[series_id] += 1
        payload = {
            "selection_order": len(rows) + 1,
            "selection_category": category,
            "selection_rationale": rationale,
            "paired_cliff_id": pair_id,
        }
        payload.update({column: row.get(column, np.nan) for column in output_columns[4:]})
        rows.append(payload)
        return True

    def add_from_frame(
        frame: pd.DataFrame,
        category: str,
        rationale: str,
        sort_columns: list[str],
        ascending: list[bool],
    ) -> None:
        quota = int(quotas[category])
        count_before = sum(record["selection_category"] == category for record in rows)
        ordered = frame.sort_values(sort_columns, ascending=ascending, kind="stable")
        for _, row in ordered.iterrows():
            if sum(record["selection_category"] == category for record in rows) - count_before >= quota:
                break
            add_row(row, category, rationale)

    add_from_frame(
        candidates[
            candidates["experimental_followup_tier"].astype(str).eq("priority_2_potent_safety_data_gap")
        ],
        "potent_safety_gap",
        "potent/property-qualified public evidence; obtain direct hERG and compatible PK before progression",
        ["discovery_rank_without_safety", "structure_id"],
        [True, True],
    )
    add_from_frame(
        candidates[
            candidates["experimental_followup_tier"].astype(str).eq("priority_3_potent_liability_flag")
        ],
        "liability_characterization",
        "potent chemistry with modeled/observed hERG concern; characterize liability and exposure margin",
        ["complete_evidence_rank", "p_activity_median", "structure_id"],
        [True, False, True],
    )
    novel = candidates[
        pd.to_numeric(candidates["invalid_structure"], errors="coerce").fillna(1).eq(0)
        & candidates["p_activity_median"].ge(6.0)
    ]
    add_from_frame(
        novel,
        "novel_scaffold_exploration",
        "exploration arm selected for local structural novelty and scaffold diversity",
        ["series_size", "local_novelty_achiral", "p_activity_median", "structure_id"],
        [True, False, False, True],
    )

    cliff_quota = int(quotas["activity_cliff_confirmation"])
    candidate_index = candidates.set_index("structure_id", drop=False)
    if not activity_cliffs.empty:
        ordered_cliffs = activity_cliffs.sort_values(
            ["shared_assay_id", "absolute_delta_pactivity", "achiral_morgan_tanimoto", "structure_id_a"],
            ascending=[False, False, False, True],
            kind="stable",
        )
        cliff_selected = 0
        for _, cliff in ordered_cliffs.iterrows():
            if cliff_selected >= cliff_quota:
                break
            structure_a = str(cliff["structure_id_a"])
            structure_b = str(cliff["structure_id_b"])
            if structure_a not in candidate_index.index or structure_b not in candidate_index.index:
                continue
            pair_id = _stable_id("CLF", f"{structure_a}\0{structure_b}")
            added_a = add_row(
                candidate_index.loc[structure_a],
                "activity_cliff_confirmation",
                "confirm a high-similarity ≥100-fold potency cliff in a harmonized repeat assay",
                pair_id,
            )
            added_b = False
            if added_a and cliff_selected + 1 < cliff_quota:
                added_b = add_row(
                    candidate_index.loc[structure_b],
                    "activity_cliff_confirmation",
                    "confirm a high-similarity ≥100-fold potency cliff in a harmonized repeat assay",
                    pair_id,
                )
            if added_a and added_b:
                cliff_selected += 2
            elif added_a:
                # A cliff requires both partners; remove an unpaired provisional selection.
                record = rows.pop()
                selected.remove(str(record["structure_id"]))
                series_counts[str(record["series_id"])] -= 1

    negative = candidates[
        candidates["p_activity_median"].le(5.0) & candidates["property_window_violation_count"].le(1)
    ]
    add_from_frame(
        negative,
        "negative_control",
        "lower-potency, property-compatible negative control for prospective decision utility",
        ["qed", "p_activity_median", "structure_id"],
        [False, True, True],
    )
    pk_bridge = candidates[candidates["pk_observation_count"].gt(0) & candidates["p_activity_median"].ge(6.0)]
    add_from_frame(
        pk_bridge,
        "pk_bridge",
        "public PK/ADMET context exists; repeat under a single compatible protocol before modeling",
        ["pk_endpoint_count", "p_activity_median", "structure_id"],
        [False, False, True],
    )
    plan = pd.DataFrame(rows, columns=output_columns)
    summary = pd.DataFrame(
        [
            {
                "selection_category": category,
                "requested_quota": int(quota),
                "selected_structures": int(
                    plan["selection_category"].eq(category).sum() if not plan.empty else 0
                ),
                "shortfall": int(quota)
                - int(plan["selection_category"].eq(category).sum() if not plan.empty else 0),
            }
            for category, quota in quotas.items()
        ],
        columns=summary_columns,
    )
    return plan, summary


def _series_tables(profiles: pd.DataFrame, *, minimum_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    members = profiles[
        [
            "series_id",
            "scaffold_key",
            "scaffold_method",
            "series_size",
            "structure_id",
            "standardized_smiles",
            "p_activity_median",
            "sources",
            "compound_ids",
            "qed",
            "mol_wt",
            "logp",
            "tpsa",
        ]
    ].sort_values(
        ["series_size", "series_id", "p_activity_median", "structure_id"],
        ascending=[False, True, False, True],
    )
    summary = profiles.groupby(["series_id", "scaffold_key", "scaffold_method"], as_index=False).agg(
        series_size=("structure_id", "size"),
        median_p_activity=("p_activity_median", "median"),
        best_p_activity=("p_activity_median", "max"),
        minimum_p_activity=("p_activity_median", "min"),
        median_qed=("qed", "median"),
        median_mol_wt=("mol_wt", "median"),
        median_logp=("logp", "median"),
        source_count=(
            "sources",
            lambda values: len(set(";".join(values.fillna("").astype(str)).split(";")) - {""}),
        ),
        representative_structure_id=("structure_id", "min"),
    )
    summary["activity_span_log10"] = summary["best_p_activity"] - summary["minimum_p_activity"]
    summary["meets_series_minimum"] = summary["series_size"].ge(minimum_size)
    return members.reset_index(drop=True), summary.sort_values(
        ["series_size", "best_p_activity", "series_id"], ascending=[False, False, True]
    ).reset_index(drop=True)


def _connectivity_tables(
    profiles: pd.DataFrame, *, minimum_delta: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shared = profiles[profiles["connectivity_group_size"].ge(2)].copy()
    member_columns = [
        "connectivity_key",
        "connectivity_group_size",
        "structure_id",
        "standard_inchi_key",
        "standardized_smiles",
        "p_activity_median",
        "series_id",
        "sources",
        "compound_ids",
    ]
    members = shared[member_columns].sort_values(
        ["connectivity_group_size", "connectivity_key", "p_activity_median", "structure_id"],
        ascending=[False, True, False, True],
    )
    if shared.empty:
        summary = pd.DataFrame(
            columns=[
                "connectivity_key",
                "structures",
                "maximum_p_activity",
                "minimum_p_activity",
                "activity_span_log10",
            ]
        )
        cliffs = pd.DataFrame(
            columns=["connectivity_key", "structure_id_a", "structure_id_b", "absolute_delta_pactivity"]
        )
        return members, summary, cliffs
    summary = (
        shared.groupby("connectivity_key", as_index=False)
        .agg(
            structures=("structure_id", "size"),
            maximum_p_activity=("p_activity_median", "max"),
            minimum_p_activity=("p_activity_median", "min"),
            structure_ids=("structure_id", _join_unique),
        )
        .sort_values(["structures", "connectivity_key"], ascending=[False, True])
    )
    summary["activity_span_log10"] = summary["maximum_p_activity"] - summary["minimum_p_activity"]
    records: list[dict[str, Any]] = []
    for key, group in shared.groupby("connectivity_key", sort=True):
        ordered = group.sort_values("structure_id").reset_index(drop=True)
        for left in range(len(ordered)):
            for right in range(left + 1, len(ordered)):
                delta = abs(
                    float(ordered.iloc[left]["p_activity_median"])
                    - float(ordered.iloc[right]["p_activity_median"])
                )
                if delta >= minimum_delta:
                    records.append(
                        {
                            "connectivity_key": key,
                            "structure_id_a": ordered.iloc[left]["structure_id"],
                            "structure_id_b": ordered.iloc[right]["structure_id"],
                            "standard_inchi_key_a": ordered.iloc[left]["standard_inchi_key"],
                            "standard_inchi_key_b": ordered.iloc[right]["standard_inchi_key"],
                            "p_activity_a": ordered.iloc[left]["p_activity_median"],
                            "p_activity_b": ordered.iloc[right]["p_activity_median"],
                            "absolute_delta_pactivity": delta,
                            "interpretation": "connectivity-equivalent variant; stereochemistry/protonation/source identity requires review",
                        }
                    )
    cliffs = pd.DataFrame(records)
    if not cliffs.empty:
        cliffs = cliffs.sort_values(
            ["absolute_delta_pactivity", "connectivity_key", "structure_id_a"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
    return members.reset_index(drop=True), summary.reset_index(drop=True), cliffs


def reference_compound_coverage(
    profiles: pd.DataFrame,
    all_measurements: pd.DataFrame,
    primary_achiral_fingerprints: list[Any | None],
    primary_chiral_fingerprints: list[Any | None],
    *,
    settings: dict[str, Any],
) -> pd.DataFrame:
    """Benchmark configured approved reference structures against public Menin coverage."""

    config = _analysis_config(settings)
    records = list(config.get("reference_compounds", []))
    columns = [
        "name",
        "pubchem_cid",
        "regulatory_status",
        "source_checked_at",
        "approval_context",
        "reference_structure_id",
        "reference_standard_inchi_key",
        "pubchem_inchi_matches_standardized",
        "has_exact_primary_task_structure",
        "has_any_public_menin_measurement",
        "public_menin_endpoints",
        "public_menin_assay_families",
        "nearest_primary_structure_id",
        "nearest_primary_p_activity",
        "maximum_primary_achiral_tanimoto",
        "maximum_primary_chiral_tanimoto",
        "reference_scaffold_series_id",
        "reference_scaffold_present_in_primary",
        "mol_wt",
        "logp",
        "tpsa",
        "qed",
        "pains_alert_count",
        "brenk_alert_count",
        "nih_alert_count",
        "pubchem_url",
        "regulatory_url",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    curation = settings.get("curation", {})
    modeling = settings.get("modeling", {})
    radius = int(config.get("fingerprint_radius", modeling.get("fingerprint_radius", 2)))
    n_bits = int(config.get("fingerprint_bits", modeling.get("fingerprint_bits", 2048)))
    standardized = [
        standardize_smiles(
            record["pubchem_isomeric_smiles"],
            strip_salts=bool(curation.get("strip_salts", True)),
            canonicalize_tautomer=bool(curation.get("canonicalize_tautomers", False)),
            require_rdkit=True,
        )
        for record in records
    ]
    reference_smiles = pd.Series([value.standardized_smiles for value in standardized], dtype=str)
    descriptor_frame = rdkit_descriptors(reference_smiles)
    molecules, achiral = _molecules_and_fingerprints(
        reference_smiles, radius=radius, n_bits=n_bits, include_chirality=False
    )
    _, chiral = _molecules_and_fingerprints(
        reference_smiles, radius=radius, n_bits=n_bits, include_chirality=True
    )
    medicinal = config.get("medicinal_chemistry", {})
    alert_names = [str(value).upper() for value in medicinal.get("alert_catalogs", DEFAULT_ALERT_CATALOGS)]
    catalogs = _build_alert_catalogs(alert_names)

    output: list[dict[str, Any]] = []
    for index, (record, structure) in enumerate(zip(records, standardized, strict=True)):
        similarities_achiral = [
            float(DataStructs.TanimotoSimilarity(achiral[index], fingerprint))
            if fingerprint is not None
            else 0.0
            for fingerprint in primary_achiral_fingerprints
        ]
        similarities_chiral = [
            float(DataStructs.TanimotoSimilarity(chiral[index], fingerprint))
            if fingerprint is not None
            else 0.0
            for fingerprint in primary_chiral_fingerprints
        ]
        maximum_achiral = max(similarities_achiral, default=0.0)
        tied = [
            position
            for position, similarity in enumerate(similarities_achiral)
            if math.isclose(similarity, maximum_achiral, rel_tol=0.0, abs_tol=1e-12)
        ]
        nearest_position = (
            min(tied, key=lambda position: str(profiles.iloc[position]["structure_id"])) if tied else None
        )
        scaffold, _method = scaffold_key(structure.standardized_smiles)
        series_id = _stable_id("SER", scaffold)
        public_rows = (
            all_measurements[
                all_measurements.get("standard_inchi_key", pd.Series("", index=all_measurements.index))
                .fillna("")
                .astype(str)
                .eq(structure.standard_inchi_key)
            ]
            if not all_measurements.empty
            else pd.DataFrame()
        )
        alert_counts: dict[str, int] = {}
        for name, catalog in catalogs.items():
            alert_counts[name.casefold()] = len(
                {str(entry.GetDescription()) for entry in catalog.GetMatches(molecules[index])}
            )
        output.append(
            {
                "name": str(record["name"]),
                "pubchem_cid": int(record["pubchem_cid"]),
                "regulatory_status": str(record["regulatory_status"]),
                "source_checked_at": str(record["source_checked_at"]),
                "approval_context": str(record["approval_context"]),
                "reference_structure_id": structure.structure_id,
                "reference_standard_inchi_key": structure.standard_inchi_key,
                "pubchem_inchi_matches_standardized": structure.standard_inchi_key
                == str(record["pubchem_inchi_key"]),
                "has_exact_primary_task_structure": bool(
                    profiles["standard_inchi_key"].astype(str).eq(structure.standard_inchi_key).any()
                ),
                "has_any_public_menin_measurement": not public_rows.empty,
                "public_menin_endpoints": _join_unique(public_rows.get("endpoint", pd.Series(dtype=str))),
                "public_menin_assay_families": _join_unique(
                    public_rows.get("assay_family", pd.Series(dtype=str))
                ),
                "nearest_primary_structure_id": (
                    profiles.iloc[nearest_position]["structure_id"] if nearest_position is not None else ""
                ),
                "nearest_primary_p_activity": (
                    profiles.iloc[nearest_position]["p_activity_median"]
                    if nearest_position is not None
                    else np.nan
                ),
                "maximum_primary_achiral_tanimoto": maximum_achiral,
                "maximum_primary_chiral_tanimoto": max(similarities_chiral, default=0.0),
                "reference_scaffold_series_id": series_id,
                "reference_scaffold_present_in_primary": bool(profiles["series_id"].eq(series_id).any()),
                "mol_wt": descriptor_frame.iloc[index]["mol_wt"],
                "logp": descriptor_frame.iloc[index]["logp"],
                "tpsa": descriptor_frame.iloc[index]["tpsa"],
                "qed": float(QED.qed(molecules[index])),
                "pains_alert_count": alert_counts.get("pains", 0),
                "brenk_alert_count": alert_counts.get("brenk", 0),
                "nih_alert_count": alert_counts.get("nih", 0),
                "pubchem_url": str(record["pubchem_url"]),
                "regulatory_url": str(record["regulatory_url"]),
            }
        )
    return pd.DataFrame(output, columns=columns).sort_values("name").reset_index(drop=True)


def write_chemical_intelligence(
    processed_dir: Path,
    reports_dir: Path,
    analysis_dir: Path,
    *,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Run and write the complete public-data chemical-intelligence layer."""

    if not RDKIT_AVAILABLE:
        raise ImportError("Enabled chemical-intelligence analysis requires RDKit")
    config = _analysis_config(settings)
    modeling = settings.get("modeling", {})
    endpoint = str(config.get("menin_endpoint", modeling.get("primary_menin_endpoint", "IC50")))
    assay_family = str(
        config.get("menin_assay_family", modeling.get("primary_menin_assay_family", "biochemical_binding"))
    )
    input_paths = {
        "scored_menin_herg": reports_dir / "menin_with_predicted_herg_risk.csv",
        "menin_measurements": processed_dir / "menin_activity_measurements.csv",
        "observed_herg": processed_dir / "herg_compounds_curated.csv",
        "pk_admet": processed_dir / "pk_admet_observations.csv",
    }
    missing_inputs = [name for name, path in input_paths.items() if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(f"Chemical-intelligence inputs are missing: {missing_inputs}")
    scored = _read_csv(input_paths["scored_menin_herg"])
    population = _primary_population(scored, endpoint=endpoint, assay_family=assay_family)
    profiles, molecules, achiral, chiral = medicinal_chemistry_profiles(population, settings=settings)

    measurements = _read_csv(input_paths["menin_measurements"])
    all_measurements = measurements.copy()
    if not measurements.empty:
        measurements = measurements[
            measurements["endpoint"].fillna("").astype(str).str.casefold().eq(endpoint.casefold())
            & measurements["assay_family"].fillna("").astype(str).str.casefold().eq(assay_family.casefold())
        ].copy()
        if "is_modeling_eligible" in measurements:
            measurements = measurements[_as_bool(measurements["is_modeling_eligible"])]
        if "is_exact" in measurements:
            measurements = measurements[_as_bool(measurements["is_exact"])]
    contexts = _context_sets(measurements)
    reference_coverage = reference_compound_coverage(
        profiles,
        all_measurements,
        achiral,
        chiral,
        settings=settings,
    )

    cluster_cfg = dict(config.get("clustering", {}))
    cluster_members, cluster_summary = assign_similarity_clusters(
        profiles,
        achiral,
        similarity_threshold=float(cluster_cfg.get("similarity_threshold", 0.65)),
    )
    profiles = profiles.merge(
        cluster_members.drop(columns=["series_id"]), on="structure_id", how="left", validate="one_to_one"
    )
    cliff_cfg = dict(config.get("activity_cliffs", {}))
    cliff_threshold = float(cliff_cfg.get("similarity_threshold", 0.80))
    minimum_delta = float(cliff_cfg.get("minimum_delta_pactivity", 2.0))
    cliffs = identify_activity_cliffs(
        profiles,
        achiral,
        chiral,
        similarity_threshold=cliff_threshold,
        minimum_delta_pactivity=minimum_delta,
        contexts=contexts,
    )
    mmp_cfg = dict(config.get("matched_molecular_pairs", {}))
    mmps = identify_matched_molecular_pairs(
        profiles,
        molecules,
        achiral,
        max_variable_heavy_atoms=int(mmp_cfg.get("max_variable_heavy_atoms", 12)),
        max_variable_fraction=float(mmp_cfg.get("max_variable_fraction", 0.35)),
        min_core_heavy_atoms=int(mmp_cfg.get("min_core_heavy_atoms", 10)),
        minimum_delta_pactivity=minimum_delta,
        contexts=contexts,
    )
    series_members, series_summary = _series_tables(
        profiles, minimum_size=int(config.get("series_minimum_size", 3))
    )
    connectivity_members, connectivity_summary, connectivity_cliffs = _connectivity_tables(
        profiles, minimum_delta=float(cliff_cfg.get("connectivity_minimum_delta_pactivity", 1.0))
    )
    observed_herg = _read_csv(input_paths["observed_herg"])
    pk = _read_csv(input_paths["pk_admet"])
    priorities, trace, gaps = prioritize_candidates(
        profiles, observed_herg=observed_herg, pk=pk, settings=settings
    )
    sensitivity, stability = prioritization_sensitivity(priorities, settings=settings)
    priorities = priorities.merge(stability, on="structure_id", how="left", validate="one_to_one")
    selection_plan, selection_summary = build_prospective_selection_plan(
        priorities,
        cliffs,
        settings=settings,
    )
    frontier = priorities[
        priorities["complete_evidence_pareto_rank"].eq(1) & priorities["chemistry_evidence_gate"]
    ].copy()

    analysis_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "medicinal_chemistry_profiles.csv": profiles,
        "candidate_priorities.csv": priorities,
        "priority_decision_trace.csv": trace,
        "priority_data_gaps.csv": gaps,
        "priority_sensitivity.csv": sensitivity,
        "priority_frontier.csv": frontier,
        "chemical_series_members.csv": series_members,
        "chemical_series_summary.csv": series_summary,
        "similarity_cluster_members.csv": cluster_members,
        "similarity_cluster_summary.csv": cluster_summary,
        "activity_cliffs.csv": cliffs,
        "matched_molecular_pairs.csv": mmps,
        "matched_molecular_pair_cliffs.csv": mmps[mmps.get("is_activity_cliff", False)].copy(),
        "connectivity_variant_members.csv": connectivity_members,
        "connectivity_variant_summary.csv": connectivity_summary,
        "connectivity_variant_cliffs.csv": connectivity_cliffs,
        "approved_reference_coverage.csv": reference_coverage,
        "prospective_selection_plan.csv": selection_plan,
        "prospective_selection_summary.csv": selection_summary,
    }
    for filename, frame in outputs.items():
        frame.to_csv(analysis_dir / filename, index=False)

    summary = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "complete",
        "task": {"endpoint": endpoint, "assay_family": assay_family},
        "candidate_count": int(len(priorities)),
        "unique_scaffold_series": int(profiles["series_id"].nunique()),
        "series_meeting_minimum": int(series_summary["meets_series_minimum"].sum()),
        "similarity_clusters": int(len(cluster_summary)),
        "activity_cliffs": int(len(cliffs)),
        "matched_molecular_pairs": int(len(mmps)),
        "matched_molecular_pair_cliffs": int(mmps.get("is_activity_cliff", pd.Series(dtype=bool)).sum()),
        "connectivity_variant_groups": int(len(connectivity_summary)),
        "connectivity_variant_cliffs": int(len(connectivity_cliffs)),
        "approved_reference_compounds": int(len(reference_coverage)),
        "approved_references_with_exact_primary_coverage": int(
            reference_coverage.get("has_exact_primary_task_structure", pd.Series(dtype=bool)).sum()
        ),
        "prospective_selection_count": int(len(selection_plan)),
        "prospective_selection_shortfall": int(
            selection_summary.get("shortfall", pd.Series(dtype=int)).sum()
        ),
        "priority_tier_counts": {
            str(key): int(value)
            for key, value in priorities["experimental_followup_tier"].value_counts().sort_index().items()
        },
        "herg_evidence_status_counts": {
            str(key): int(value)
            for key, value in priorities["herg_evidence_status"].value_counts().sort_index().items()
        },
        "algorithm_contract": {
            "rdkit_version": str(rdBase.rdkitVersion),
            "fingerprint": {
                "type": "Morgan",
                "radius": int(config.get("fingerprint_radius", modeling.get("fingerprint_radius", 2))),
                "bits": int(config.get("fingerprint_bits", modeling.get("fingerprint_bits", 2048))),
                "achiral_primary_with_chiral_sensitivity": True,
            },
            "scaffold": "RDKit Bemis-Murcko; exact canonical identity for acyclic structures",
            "clustering": "Butina with configured similarity converted to distance",
            "activity_cliff_thresholds": {
                "minimum_achiral_tanimoto": cliff_threshold,
                "minimum_delta_pactivity": minimum_delta,
            },
            "alerts_are_review_flags_not_exclusions": True,
            "outside_domain_herg_policy": "unknown_no_safety_credit",
        },
        "input_sha256": {name: sha256_file(path) for name, path in sorted(input_paths.items())},
    }
    (analysis_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
