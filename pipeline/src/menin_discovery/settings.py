"""Versioned, adjustable pipeline settings.

The default configuration lives in ``pipeline/config/pipeline.yaml``.  A user-supplied
YAML file is merged recursively over those defaults so experiments can change
policies without editing source code.
"""

from __future__ import annotations

import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_SOURCE_ROOT_CANDIDATE = Path(__file__).resolve().parents[3]
_DEFAULT_ROOT = (
    _SOURCE_ROOT_CANDIDATE
    if (_SOURCE_ROOT_CANDIDATE / "pipeline" / "config" / "pipeline.yaml").exists()
    else Path.cwd()
)
ROOT = Path(os.environ.get("MENIN_PROJECT_ROOT", _DEFAULT_ROOT)).expanduser().resolve()
REPOSITORY_SETTINGS_PATH = ROOT / "pipeline" / "config" / "pipeline.yaml"
PACKAGE_SETTINGS_PATH = Path(__file__).with_name("default_config.yaml")
DEFAULT_SETTINGS_PATH = (
    REPOSITORY_SETTINGS_PATH if REPOSITORY_SETTINGS_PATH.exists() else PACKAGE_SETTINGS_PATH
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Pipeline configuration not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Pipeline configuration must be a mapping: {path}")
    return payload


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    """Load defaults and recursively merge an optional YAML override."""

    defaults = _read_yaml(DEFAULT_SETTINGS_PATH)
    if path is None:
        return validate_settings(defaults)
    override_path = Path(path).expanduser().resolve()
    if override_path == DEFAULT_SETTINGS_PATH.resolve():
        return validate_settings(defaults)
    return validate_settings(_deep_merge(defaults, _read_yaml(override_path)))


def validate_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Fail early on contradictory or unsafe analytical configuration."""

    project = settings.get("project", {})
    paths = settings.get("paths", {})
    curation = settings.get("curation", {})
    herg = settings.get("herg", {})
    modeling = settings.get("modeling", {})
    analysis = settings.get("analysis", {})
    analysis_enabled = analysis.get("enabled", False)
    if not isinstance(analysis_enabled, bool):
        raise ValueError("analysis.enabled must be a boolean")
    required_paths = (
        "raw",
        "processed",
        "models",
        *(("analysis",) if analysis_enabled else ()),
        "reports",
    )
    missing_paths = [name for name in required_paths if not str(paths.get(name, "")).strip()]
    if missing_paths:
        raise ValueError(f"Missing configured output paths: {', '.join(missing_paths)}")
    normalized_paths = [str(Path(str(paths[name])).expanduser()) for name in required_paths]
    if len(set(normalized_paths)) != len(normalized_paths):
        raise ValueError("raw, processed, models, and reports paths must be distinct")
    try:
        int(project.get("random_state", 13))
    except (TypeError, ValueError) as exc:
        raise ValueError("project.random_state must be an integer") from exc
    blocker = float(herg.get("blocker_max_nm", 10_000.0))
    nonblocker = float(herg.get("nonblocker_min_nm", 30_000.0))
    if blocker <= 0 or nonblocker <= 0 or blocker >= nonblocker:
        raise ValueError("herg thresholds must be positive and blocker_max_nm < nonblocker_min_nm")
    if float(curation.get("max_within_compound_log_spread", 1.0)) <= 0:
        raise ValueError("curation.max_within_compound_log_spread must be positive")
    if int(modeling.get("fingerprint_bits", 2048)) <= 0:
        raise ValueError("modeling.fingerprint_bits must be positive")
    if int(modeling.get("fingerprint_radius", 2)) < 0:
        raise ValueError("modeling.fingerprint_radius must be non-negative")
    domain_quantile = float(modeling.get("applicability_domain_quantile", 0.05))
    if not 0 < domain_quantile < 1:
        raise ValueError("modeling.applicability_domain_quantile must lie between 0 and 1")
    uncertainty = float(modeling.get("uncertainty_coverage", 0.90))
    if not 0 < uncertainty < 1:
        raise ValueError("modeling.uncertainty_coverage must lie between 0 and 1")
    test_size = float(modeling.get("test_size", 0.2))
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("modeling.test_size must lie between 0.05 and 0.5")
    if int(modeling.get("cv_folds", 3)) < 2:
        raise ValueError("modeling.cv_folds must be at least 2")
    for name in (
        "min_regression_compounds",
        "min_classification_compounds",
        "tree_estimators",
    ):
        if int(modeling.get(name, 1)) <= 0:
            raise ValueError(f"modeling.{name} must be positive")
    if int(modeling.get("bootstrap_iterations", 500)) < 0:
        raise ValueError("modeling.bootstrap_iterations must be non-negative")
    allowed_splits = {"scaffold", "chemical", "temporal", "random"}
    configured_splits = [
        str(modeling.get("primary_split", "scaffold")),
        *[str(value) for value in modeling.get("evaluation_splits", [])],
    ]
    unsupported = sorted(set(configured_splits) - allowed_splits)
    if unsupported:
        raise ValueError(f"Unsupported modeling split strategies: {unsupported}")
    if analysis_enabled:
        _validate_analysis_settings(analysis)
    return settings


def _finite_number(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_analysis_settings(analysis: dict[str, Any]) -> None:
    """Validate the complete, adjustable chemical-intelligence policy."""

    for name in ("menin_endpoint", "menin_assay_family", "herg_endpoint", "herg_assay_family"):
        if not str(analysis.get(name, "")).strip():
            raise ValueError(f"analysis.{name} must be non-empty")
    if int(analysis.get("fingerprint_bits", 2048)) <= 0:
        raise ValueError("analysis.fingerprint_bits must be positive")
    if int(analysis.get("fingerprint_radius", 2)) < 0:
        raise ValueError("analysis.fingerprint_radius must be non-negative")
    if int(analysis.get("series_minimum_size", 3)) <= 0:
        raise ValueError("analysis.series_minimum_size must be positive")
    references = analysis.get("reference_compounds", [])
    if not isinstance(references, list):
        raise ValueError("analysis.reference_compounds must be a list")
    reference_names: list[str] = []
    reference_cids: list[int] = []
    for index, record in enumerate(references):
        if not isinstance(record, dict):
            raise ValueError(f"analysis.reference_compounds[{index}] must be a mapping")
        for field in (
            "name",
            "pubchem_cid",
            "pubchem_inchi_key",
            "pubchem_isomeric_smiles",
            "regulatory_status",
            "source_checked_at",
            "approval_context",
            "pubchem_url",
            "regulatory_url",
        ):
            if not str(record.get(field, "")).strip():
                raise ValueError(f"analysis.reference_compounds[{index}].{field} must be non-empty")
        reference_names.append(str(record["name"]).strip().casefold())
        try:
            reference_cids.append(int(record["pubchem_cid"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"analysis.reference_compounds[{index}].pubchem_cid must be an integer") from exc
    if len(reference_names) != len(set(reference_names)) or len(reference_cids) != len(set(reference_cids)):
        raise ValueError("analysis.reference_compounds names and PubChem CIDs must be unique")

    clustering_similarity = _finite_number(
        analysis.get("clustering", {}).get("similarity_threshold", 0.65),
        name="analysis.clustering.similarity_threshold",
    )
    if not 0 < clustering_similarity <= 1:
        raise ValueError("analysis.clustering.similarity_threshold must lie in (0, 1]")
    cliffs = analysis.get("activity_cliffs", {})
    cliff_similarity = _finite_number(
        cliffs.get("similarity_threshold", 0.80),
        name="analysis.activity_cliffs.similarity_threshold",
    )
    if not 0 < cliff_similarity <= 1:
        raise ValueError("analysis.activity_cliffs.similarity_threshold must lie in (0, 1]")
    for name in ("minimum_delta_pactivity", "connectivity_minimum_delta_pactivity"):
        if _finite_number(cliffs.get(name, 1.0), name=f"analysis.activity_cliffs.{name}") <= 0:
            raise ValueError(f"analysis.activity_cliffs.{name} must be positive")

    matched = analysis.get("matched_molecular_pairs", {})
    for name in ("max_variable_heavy_atoms", "min_core_heavy_atoms"):
        if int(matched.get(name, 1)) <= 0:
            raise ValueError(f"analysis.matched_molecular_pairs.{name} must be positive")
    max_fraction = _finite_number(
        matched.get("max_variable_fraction", 0.35),
        name="analysis.matched_molecular_pairs.max_variable_fraction",
    )
    if not 0 < max_fraction < 1:
        raise ValueError("analysis.matched_molecular_pairs.max_variable_fraction must lie in (0, 1)")

    medicinal = analysis.get("medicinal_chemistry", {})
    catalogs = [str(value).strip().upper() for value in medicinal.get("alert_catalogs", [])]
    supported_catalogs = {"PAINS", "PAINS_A", "PAINS_B", "PAINS_C", "BRENK", "NIH", "ZINC", "CHEMBL"}
    if not catalogs or len(catalogs) != len(set(catalogs)):
        raise ValueError("analysis.medicinal_chemistry.alert_catalogs must be non-empty and unique")
    unsupported_catalogs = sorted(set(catalogs) - supported_catalogs)
    if unsupported_catalogs:
        raise ValueError(f"Unsupported analysis alert catalogs: {unsupported_catalogs}")
    windows = medicinal.get("property_windows", {})
    if not isinstance(windows, dict) or not windows:
        raise ValueError("analysis.medicinal_chemistry.property_windows must be non-empty")
    for descriptor, bounds in windows.items():
        if not isinstance(bounds, dict) or not {"min", "max"}.intersection(bounds):
            raise ValueError(f"Property window {descriptor} must define min and/or max")
        lower = (
            _finite_number(bounds["min"], name=f"property window {descriptor}.min")
            if bounds.get("min") is not None
            else None
        )
        upper = (
            _finite_number(bounds["max"], name=f"property window {descriptor}.max")
            if bounds.get("max") is not None
            else None
        )
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError(f"Property window {descriptor} requires min < max")

    policy = analysis.get("prioritization", {})
    potency_lower = _finite_number(
        policy.get("potency_desirability_lower", 6.0),
        name="analysis.prioritization.potency_desirability_lower",
    )
    potency_upper = _finite_number(
        policy.get("potency_desirability_upper", 9.0),
        name="analysis.prioritization.potency_desirability_upper",
    )
    if potency_lower >= potency_upper:
        raise ValueError("analysis prioritization potency bounds require lower < upper")
    if (
        _finite_number(
            policy.get("maximum_activity_range_log10", 1.0),
            name="analysis.prioritization.maximum_activity_range_log10",
        )
        <= 0
    ):
        raise ValueError("analysis.prioritization.maximum_activity_range_log10 must be positive")
    if int(policy.get("maximum_property_violations", 1)) < 0:
        raise ValueError("analysis.prioritization.maximum_property_violations must be non-negative")
    if not isinstance(policy.get("require_no_pains", True), bool):
        raise ValueError("analysis.prioritization.require_no_pains must be boolean")
    lower_herg = _finite_number(
        policy.get("lower_herg_probability", 0.30),
        name="analysis.prioritization.lower_herg_probability",
    )
    high_herg = _finite_number(
        policy.get("high_herg_probability", 0.70),
        name="analysis.prioritization.high_herg_probability",
    )
    if not 0 <= lower_herg < high_herg <= 1:
        raise ValueError("analysis hERG probability thresholds require 0 <= lower < high <= 1")
    expected_weights = {"potency", "qed", "property", "evidence", "herg"}
    weights = policy.get("weights", {})
    unknown_weights = sorted(set(weights) - expected_weights)
    if unknown_weights:
        raise ValueError(f"Unknown analysis prioritization weights: {unknown_weights}")
    resolved_weights = {
        "potency": 0.35,
        "qed": 0.15,
        "property": 0.15,
        "evidence": 0.10,
        "herg": 0.25,
        **weights,
    }
    numbers = [
        _finite_number(value, name=f"analysis.prioritization.weights.{name}")
        for name, value in resolved_weights.items()
    ]
    if any(value < 0 for value in numbers) or sum(numbers) <= 0:
        raise ValueError("analysis prioritization weights must be non-negative with positive total")
    emphasis_factor = _finite_number(
        policy.get("sensitivity", {}).get("emphasis_factor", 2.0),
        name="analysis.prioritization.sensitivity.emphasis_factor",
    )
    if emphasis_factor <= 1:
        raise ValueError("analysis prioritization sensitivity emphasis_factor must be greater than 1")
    selection = analysis.get("prospective_selection", {})
    if not isinstance(selection.get("enabled", True), bool):
        raise ValueError("analysis.prospective_selection.enabled must be boolean")
    if int(selection.get("maximum_per_scaffold_series", 2)) <= 0:
        raise ValueError("analysis.prospective_selection.maximum_per_scaffold_series must be positive")
    expected_quotas = {
        "potent_safety_gap",
        "liability_characterization",
        "novel_scaffold_exploration",
        "activity_cliff_confirmation",
        "negative_control",
        "pk_bridge",
    }
    quotas = selection.get("quotas", {})
    unknown_quotas = sorted(set(quotas) - expected_quotas)
    if unknown_quotas:
        raise ValueError(f"Unknown prospective selection quotas: {unknown_quotas}")
    for name in expected_quotas:
        if int(quotas.get(name, 0)) < 0:
            raise ValueError(f"analysis.prospective_selection.quotas.{name} must be non-negative")


def resolve_project_path(value: str | Path, *, root: Path = ROOT) -> Path:
    """Resolve a configured path relative to the repository root."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def settings_snapshot(settings: dict[str, Any]) -> str:
    """Return a stable YAML representation suitable for build metadata."""

    return yaml.safe_dump(settings, sort_keys=True, allow_unicode=True)
