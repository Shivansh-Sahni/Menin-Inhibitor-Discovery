from __future__ import annotations

from pathlib import Path

import pytest
from menin_discovery.research_structures import validate_mmcif_coordinate


def _minimal_mmcif(entry_id: str = "9CHP") -> str:
    return f"""data_{entry_id}
#
_entry.id {entry_id}
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.auth_asym_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.pdbx_PDB_model_num
ATOM 1 C A 1.0 2.0 3.0 1
HETATM 2 K B 4.0 5.0 6.0 1
#
"""


def test_mmcif_coordinate_validation_recovers_identity_and_coordinate_counts(tmp_path: Path) -> None:
    path = tmp_path / "9CHP.cif"
    path.write_text(_minimal_mmcif(), encoding="utf-8")

    qc = validate_mmcif_coordinate(path, expected_entry_id="9CHP")

    assert qc.entry_id == "9CHP"
    assert qc.atom_count == 2
    assert qc.polymer_atom_count == 1
    assert qc.hetero_atom_count == 1
    assert qc.auth_chains == ("A", "B")
    assert qc.model_numbers == ("1",)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (_minimal_mmcif("8ZYN"), "expected identity"),
        (_minimal_mmcif().replace("1.0 2.0 3.0", "nan 2.0 3.0"), "Non-finite"),
        (_minimal_mmcif().replace("HETATM 2", "ATOM 1"), "Duplicate atom_site.id"),
    ],
)
def test_mmcif_coordinate_validation_rejects_identity_and_atom_table_failures(
    tmp_path: Path,
    text: str,
    message: str,
) -> None:
    path = tmp_path / "9CHP.cif"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_mmcif_coordinate(path, expected_entry_id="9CHP")
