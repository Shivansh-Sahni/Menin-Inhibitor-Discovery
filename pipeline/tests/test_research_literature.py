from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from menin_discovery.research_cli import (
    CATALOG_COLUMNS,
    LOCKED_CORE_ASSETS,
    MISSING_ASSET_COLUMNS,
    REQUIRED_MEETING_FILES,
    REQUIRED_SYNTHESIS_FILES,
    _literature_stage,
)

PARENTS = {
    "sun_2026_jcim_6c00163_dataset": "sun_2026_jcim_6c00163_article",
    "sun_2026_jcim_6c00163_methods": "sun_2026_jcim_6c00163_article",
    "miyashita_2024_si": "miyashita_2024_article",
    "mavroudis_2023_si": "mavroudis_2023_article",
    "poongavanam_2022_methods": "poongavanam_2022_article",
    "poongavanam_2022_assay_data": "poongavanam_2022_article",
    "qi_2025_si": "qi_2025_article",
}


def _write_fixture(project_root: Path) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    literature_root = project_root / "research" / "literature"
    rows: list[dict[str, str]] = []
    for literature_id, (role, relative_path) in LOCKED_CORE_ASSETS.items():
        parent_id = PARENTS.get(literature_id, "")
        family_id = parent_id or literature_id
        asset = project_root / relative_path
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(b"test asset")
        rows.append(
            {
                "literature_id": literature_id,
                "parent_literature_id": parent_id,
                "doi": f"10.test/{family_id}",
                "title": literature_id,
                "year": "2025",
                "topic": f"topic_{family_id}",
                "role": role,
                "local_path": relative_path,
                "data_availability": "local test fixture",
                "applicability_to_internal_series": "test only",
                "review_status": "reviewed",
            }
        )
    catalog = pd.DataFrame(rows, columns=sorted(CATALOG_COLUMNS))
    catalog.to_csv(literature_root / "catalog.csv", index=False)

    article_rows = catalog.loc[catalog["role"].eq("article")]
    for local_path in article_rows["local_path"]:
        package = (project_root / local_path).parent
        (package / "evidence_review.md").write_text("# Evidence review\n", encoding="utf-8")

    missing_assets = pd.DataFrame(
        [
            {
                "record_id": "fixture_missing_asset",
                "domain": "PK",
                "asset": "raw profiles",
                "why_needed": "mechanistic validation",
                "availability": "not attached",
                "status": "missing",
                "priority": "critical",
                "next_action": "request from authors",
            }
        ],
        columns=sorted(MISSING_ASSET_COLUMNS),
    )
    missing_assets.to_csv(literature_root / "missing_assets.csv", index=False)

    for relative_path in REQUIRED_SYNTHESIS_FILES:
        synthesis_file = literature_root / relative_path
        synthesis_file.parent.mkdir(parents=True, exist_ok=True)
        synthesis_file.write_text("test\n", encoding="utf-8")
    for meeting_date, filenames in REQUIRED_MEETING_FILES.items():
        meeting_root = project_root / "research" / "notes" / "meetings" / meeting_date
        meeting_root.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            (meeting_root / filename).write_bytes(b"test meeting record")

    config: dict[str, object] = {
        "project_root": project_root,
        "paths": {"literature": literature_root},
    }
    return config, catalog, missing_assets


def _rewrite_catalog(config: dict[str, object], catalog: pd.DataFrame) -> None:
    literature_root = config["paths"]["literature"]
    assert isinstance(literature_root, Path)
    catalog.to_csv(literature_root / "catalog.csv", index=False)


def _rewrite_missing_assets(config: dict[str, object], missing_assets: pd.DataFrame) -> None:
    literature_root = config["paths"]["literature"]
    assert isinstance(literature_root, Path)
    missing_assets.to_csv(literature_root / "missing_assets.csv", index=False)


def test_literature_stage_validates_locked_hierarchy(tmp_path: Path) -> None:
    config, _, _ = _write_fixture(tmp_path)
    literature_root = tmp_path / "research" / "literature"
    (literature_root / "herg" / "rendered" / "page.png").parent.mkdir(parents=True)
    (literature_root / "herg" / "rendered" / "page.png").write_bytes(b"render output")
    (literature_root / ".temp" / "copy.pdf").parent.mkdir(parents=True)
    (literature_root / ".temp" / "copy.pdf").write_bytes(b"temporary output")

    summary = _literature_stage(config)

    assert summary == {
        "catalog_rows": len(LOCKED_CORE_ASSETS),
        "canonical_binary_assets": len(LOCKED_CORE_ASSETS),
        "evidence_reviews": 6,
        "article_packages": 6,
        "missing_asset_rows": 1,
        "meeting_folders": 2,
    }


def test_literature_stage_rejects_duplicate_literature_id(tmp_path: Path) -> None:
    config, catalog, _ = _write_fixture(tmp_path)
    catalog = pd.concat([catalog, catalog.iloc[[0]]], ignore_index=True)
    _rewrite_catalog(config, catalog)

    with pytest.raises(ValueError, match="duplicate literature_id"):
        _literature_stage(config)


def test_literature_stage_rejects_unknown_parent_article(tmp_path: Path) -> None:
    config, catalog, _ = _write_fixture(tmp_path)
    catalog.loc[catalog["literature_id"].eq("miyashita_2024_si"), "parent_literature_id"] = "absent_article"
    _rewrite_catalog(config, catalog)

    with pytest.raises(ValueError, match="unknown parent article"):
        _literature_stage(config)


def test_literature_stage_rejects_role_extension_mismatch(tmp_path: Path) -> None:
    config, catalog, _ = _write_fixture(tmp_path)
    catalog.loc[catalog["literature_id"].eq("sun_2026_jcim_6c00163_dataset"), "role"] = "article"
    _rewrite_catalog(config, catalog)

    with pytest.raises(ValueError, match="role/extension mismatch"):
        _literature_stage(config)


def test_literature_stage_rejects_missing_cataloged_local_asset(tmp_path: Path) -> None:
    config, catalog, _ = _write_fixture(tmp_path)
    local_path = catalog.loc[catalog["literature_id"].eq("lau_2024_article"), "local_path"].item()
    (tmp_path / local_path).unlink()

    with pytest.raises(FileNotFoundError, match="local asset is missing"):
        _literature_stage(config)


def test_literature_stage_rejects_uncataloged_package_asset(tmp_path: Path) -> None:
    config, catalog, _ = _write_fixture(tmp_path)
    package = (tmp_path / catalog.loc[catalog["role"].eq("article"), "local_path"].iloc[0]).parent
    (package / "orphan_supplement.pdf").write_bytes(b"unrepresented")

    with pytest.raises(ValueError, match="not represented in catalog.csv"):
        _literature_stage(config)


def test_literature_stage_catalogs_and_validates_mmcif_coordinates(tmp_path: Path) -> None:
    config, catalog, _ = _write_fixture(tmp_path)
    parent_id = "miyashita_2024_article"
    parent = catalog.loc[catalog["literature_id"].eq(parent_id)].iloc[0]
    package = (tmp_path / parent["local_path"]).parent
    relative_path = (package / "9CHP.cif").relative_to(tmp_path).as_posix()
    (tmp_path / relative_path).write_text(
        """data_9CHP
_entry.id 9CHP
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.auth_asym_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.pdbx_PDB_model_num
ATOM 1 A 1.0 2.0 3.0 1
#
""",
        encoding="utf-8",
    )
    coordinate = {
        **parent.to_dict(),
        "literature_id": "miyashita_2024_9chp_coordinate",
        "parent_literature_id": parent_id,
        "title": "test coordinate",
        "role": "structure coordinate",
        "local_path": relative_path,
    }
    catalog = pd.concat([catalog, pd.DataFrame([coordinate])], ignore_index=True)
    _rewrite_catalog(config, catalog)

    summary = _literature_stage(config)

    assert summary["canonical_binary_assets"] == len(LOCKED_CORE_ASSETS) + 1


def test_literature_stage_rejects_coordinate_filename_entry_mismatch(tmp_path: Path) -> None:
    config, catalog, _ = _write_fixture(tmp_path)
    parent_id = "miyashita_2024_article"
    parent = catalog.loc[catalog["literature_id"].eq(parent_id)].iloc[0]
    package = (tmp_path / parent["local_path"]).parent
    relative_path = (package / "9CHP.cif").relative_to(tmp_path).as_posix()
    (tmp_path / relative_path).write_text(
        """data_8ZYN
_entry.id 8ZYN
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.auth_asym_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.pdbx_PDB_model_num
ATOM 1 A 1.0 2.0 3.0 1
#
""",
        encoding="utf-8",
    )
    coordinate = {
        **parent.to_dict(),
        "literature_id": "miyashita_2024_9chp_coordinate",
        "parent_literature_id": parent_id,
        "title": "test coordinate",
        "role": "structure coordinate",
        "local_path": relative_path,
    }
    catalog = pd.concat([catalog, pd.DataFrame([coordinate])], ignore_index=True)
    _rewrite_catalog(config, catalog)

    with pytest.raises(ValueError, match="expected identity"):
        _literature_stage(config)


def test_literature_stage_rejects_missing_evidence_review(tmp_path: Path) -> None:
    config, catalog, _ = _write_fixture(tmp_path)
    package = (
        tmp_path / catalog.loc[catalog["literature_id"].eq("qi_2025_article"), "local_path"].item()
    ).parent
    (package / "evidence_review.md").unlink()

    with pytest.raises(FileNotFoundError, match="lacks evidence_review.md"):
        _literature_stage(config)


def test_literature_stage_rejects_changed_locked_mapping(tmp_path: Path) -> None:
    config, catalog, _ = _write_fixture(tmp_path)
    selected = catalog["literature_id"].eq("lau_2024_article")
    original = tmp_path / catalog.loc[selected, "local_path"].item()
    renamed = original.with_name("renamed_article.pdf")
    original.rename(renamed)
    catalog.loc[selected, "local_path"] = renamed.relative_to(tmp_path).as_posix()
    _rewrite_catalog(config, catalog)

    with pytest.raises(ValueError, match="changed locked role/path mappings"):
        _literature_stage(config)


@pytest.mark.parametrize("status", ["available", "", "MISSING"])
def test_literature_stage_rejects_invalid_missing_asset_status(tmp_path: Path, status: str) -> None:
    config, _, missing_assets = _write_fixture(tmp_path)
    missing_assets.loc[0, "status"] = status
    _rewrite_missing_assets(config, missing_assets)

    with pytest.raises(ValueError, match="blank required values|invalid statuses"):
        _literature_stage(config)


def test_literature_stage_rejects_incomplete_missing_asset_schema(tmp_path: Path) -> None:
    config, _, missing_assets = _write_fixture(tmp_path)
    _rewrite_missing_assets(config, missing_assets.drop(columns="next_action"))

    with pytest.raises(ValueError, match="missing required columns.*next_action"):
        _literature_stage(config)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("domain", "ADME", "invalid domains"),
        ("priority", "urgent", "invalid priorities"),
    ],
)
def test_literature_stage_rejects_invalid_missing_asset_enums(
    tmp_path: Path,
    column: str,
    value: str,
    message: str,
) -> None:
    config, _, missing_assets = _write_fixture(tmp_path)
    missing_assets.loc[0, column] = value
    _rewrite_missing_assets(config, missing_assets)

    with pytest.raises(ValueError, match=message):
        _literature_stage(config)


def test_literature_stage_rejects_malformed_meeting_folder(tmp_path: Path) -> None:
    config, _, _ = _write_fixture(tmp_path)
    malformed = tmp_path / "research" / "notes" / "meetings" / "July-21"
    malformed.mkdir()
    (malformed / "context.md").write_text("test\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not an ISO date"):
        _literature_stage(config)


def test_literature_stage_requires_exact_meeting_record(tmp_path: Path) -> None:
    config, _, _ = _write_fixture(tmp_path)
    required = (
        tmp_path
        / "research"
        / "notes"
        / "meetings"
        / "2026-07-21"
        / "data_availability_2026-07-21_11-39-44.png"
    )
    required.unlink()

    with pytest.raises(FileNotFoundError, match="Required meeting records are missing"):
        _literature_stage(config)
