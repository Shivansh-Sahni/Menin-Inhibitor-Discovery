from __future__ import annotations

import json
from pathlib import Path

import pytest
from menin_discovery.platform_adjacent_training_surfaces import (
    AdjacentTrainingSurfaceError,
    _broad_base_id,
    _split,
    _structure,
    validate_adjacent_training_surfaces,
)


def test_structure_identity_and_scaffold_split_are_deterministic() -> None:
    first = _structure("CCO")
    second = _structure("OCC")
    assert first == second
    assert first[2] is not None and first[3] is not None
    assert _split(first[3]) == _split(second[3])


def test_invalid_structures_are_not_eligible() -> None:
    assert _structure(None) == (None, None, None, None)
    assert _structure("-666") == (None, None, None, None)


def test_broad_batch_ids_normalize_to_cross_source_compound_identity() -> None:
    assert _broad_base_id("BRD-A00055058-001-01-0") == "BRD-A00055058"
    assert _broad_base_id("BRD-A00055058") == "BRD-A00055058"
    assert _broad_base_id("not-a-broad-id") is None


def test_production_adjacent_training_surfaces_validate() -> None:
    root = Path("research/data/platform/processed/multimodal/v1_0_adjacent_training_surfaces")
    if root.exists():
        manifest = validate_adjacent_training_surfaces(root)
        assert manifest["counts"]["prism_structure_linked_training_values"] >= 8_000_000
        assert manifest["counts"]["lincs_compound_instance_rows"] >= 900_000


def test_manifest_tampering_is_rejected(tmp_path: Path) -> None:
    root = Path("research/data/platform/processed/multimodal/v1_0_adjacent_training_surfaces")
    if not root.exists():
        pytest.skip("production release is not built")
    target = tmp_path / "release"
    target.mkdir()
    for source in root.iterdir():
        (target / source.name).write_bytes(source.read_bytes())
    manifest_path = target / "adjacent_training_manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["counts"]["prism_structure_linked_training_values"] += 1
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(AdjacentTrainingSurfaceError, match="self-hash"):
        validate_adjacent_training_surfaces(target)
