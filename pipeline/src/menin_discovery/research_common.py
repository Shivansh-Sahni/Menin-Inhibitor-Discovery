"""Shared, deliberately non-git utilities for the PK/hERG research program.

The research command is separated from :mod:`menin_discovery.cli`: it neither
reads repository state nor records commit identifiers.  Scientific outputs are
written through sibling temporary files so a failed stage cannot truncate the
last validated artifact.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def load_research_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load the PK/hERG configuration and resolve every declared project path."""

    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("Research configuration must contain a YAML mapping")
    project_value = payload.get("project", {}).get("root", str(PACKAGE_ROOT))
    project_root = Path(str(project_value)).expanduser()
    if not project_root.is_absolute():
        project_root = (config_path.parent / project_root).resolve()
    payload["project_root"] = project_root
    for section_name in ("inputs", "paths"):
        section = payload.get(section_name, {})
        if not isinstance(section, dict):
            raise TypeError(f"{section_name} must be a mapping")
        payload[section_name] = {
            key: (
                Path(str(value)).expanduser().resolve()
                if Path(str(value)).expanduser().is_absolute()
                else (project_root / str(value)).resolve()
            )
            for key, value in section.items()
        }
    return payload


def _atomic_path(path: Path) -> tuple[Path, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    return Path(name), descriptor


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> Path:
    """Atomically write readable JSON without run hashes or repository metadata."""

    def json_safe(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [json_safe(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if hasattr(value, "item") and callable(value.item):
            try:
                return json_safe(value.item())
            except (TypeError, ValueError):
                pass
        return value

    target = Path(path)
    temporary, descriptor = _atomic_path(target)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, indent=2, sort_keys=True, default=str, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_write_text(path: str | os.PathLike[str], text: str) -> Path:
    target = Path(path)
    temporary, descriptor = _atomic_path(target)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_write_csv(path: str | os.PathLike[str], frame: pd.DataFrame) -> Path:
    target = Path(path)
    temporary, descriptor = _atomic_path(target)
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_write_parquet(path: str | os.PathLike[str], frame: pd.DataFrame) -> Path:
    target = Path(path)
    temporary, descriptor = _atomic_path(target)
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        # Read-back is the promotion gate.  This catches missing engines,
        # truncated files, and schema serialization failures before replace.
        validated = pd.read_parquet(temporary, engine="pyarrow")
        if len(validated) != len(frame) or list(validated.columns) != list(frame.columns):
            raise ValueError(f"Parquet round-trip mismatch for {target}")
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def require_columns(frame: pd.DataFrame, columns: set[str], *, label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def jsonable_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    """Convert numpy/pandas scalars in a shallow mapping for model cards."""

    converted: dict[str, Any] = {}
    for key, value in values.items():
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                pass
        if isinstance(value, float) and (pd.isna(value) or value in (float("inf"), float("-inf"))):
            value = None
        converted[str(key)] = value
    return converted
