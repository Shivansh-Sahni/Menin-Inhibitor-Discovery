"""Validation utilities for deposited receptor coordinate records.

The research program keeps deposited PDBx/mmCIF records as raw evidence.  A
successful validation here means that the entry identity and coordinate table
are internally parseable; it does not imply biological-assembly selection,
missing-residue repair, protonation, membrane placement, or simulation
readiness.
"""

from __future__ import annotations

import math
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class MMCIFCoordinateQC:
    """Minimal, auditable facts recovered from one deposited coordinate file."""

    entry_id: str
    atom_count: int
    polymer_atom_count: int
    hetero_atom_count: int
    auth_chain_count: int
    auth_chains: tuple[str, ...]
    model_count: int
    model_numbers: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a serialization-friendly representation."""

        return asdict(self)


def _scalar_value(lines: list[str], key: str) -> str:
    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped.startswith(key):
            continue
        tokens = shlex.split(stripped, posix=True)
        if len(tokens) >= 2:
            return tokens[1]
        for continuation in lines[index + 1 :]:
            if continuation.strip():
                values = shlex.split(continuation.strip(), posix=True)
                if values:
                    return values[0]
                break
    raise ValueError(f"PDBx/mmCIF scalar is missing: {key}")


def _atom_site_table(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    for loop_index, raw_line in enumerate(lines):
        if raw_line.strip() != "loop_":
            continue
        headers: list[str] = []
        row_index = loop_index + 1
        while row_index < len(lines) and lines[row_index].lstrip().startswith("_"):
            headers.append(lines[row_index].split(maxsplit=1)[0])
            row_index += 1
        if not headers or not headers[0].startswith("_atom_site."):
            continue

        rows: list[list[str]] = []
        buffered: list[str] = []
        while row_index < len(lines):
            stripped = lines[row_index].strip()
            row_index += 1
            if not stripped:
                continue
            if stripped == "#":
                if buffered:
                    raise ValueError("PDBx/mmCIF atom_site loop ends with a partial row")
                break
            buffered.extend(shlex.split(stripped, posix=True))
            while len(buffered) >= len(headers):
                rows.append(buffered[: len(headers)])
                buffered = buffered[len(headers) :]
        if buffered:
            raise ValueError("PDBx/mmCIF atom_site loop contains a partial row")
        if not rows:
            raise ValueError("PDBx/mmCIF atom_site loop contains no coordinates")
        return headers, rows
    raise ValueError("PDBx/mmCIF file has no atom_site coordinate loop")


def validate_mmcif_coordinate(
    path: str | Path,
    *,
    expected_entry_id: str | None = None,
) -> MMCIFCoordinateQC:
    """Parse and validate the identity and full ``atom_site`` table.

    This deliberately checks the deposited coordinate evidence, not whether the
    file is a prepared molecular-dynamics receptor system.
    """

    coordinate_path = Path(path)
    if coordinate_path.suffix.casefold() not in {".cif", ".mmcif"}:
        raise ValueError(f"Expected a PDBx/mmCIF coordinate file: {coordinate_path}")
    try:
        lines = coordinate_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"Coordinate file is not UTF-8 PDBx/mmCIF text: {coordinate_path}") from exc
    if not lines:
        raise ValueError(f"Coordinate file is empty: {coordinate_path}")

    data_ids = [line.strip()[5:] for line in lines if line.strip().casefold().startswith("data_")]
    if len(data_ids) != 1 or not data_ids[0]:
        raise ValueError(f"Expected exactly one named PDBx/mmCIF data block: {coordinate_path}")
    entry_id = _scalar_value(lines, "_entry.id").upper()
    if data_ids[0].upper() != entry_id:
        raise ValueError(f"PDBx/mmCIF data-block and _entry.id disagree: {data_ids[0]!r} != {entry_id!r}")
    if expected_entry_id and entry_id != expected_entry_id.upper():
        raise ValueError(
            f"PDBx/mmCIF entry does not match expected identity: {entry_id!r} != "
            f"{expected_entry_id.upper()!r}"
        )

    headers, rows = _atom_site_table(lines)
    header_index = {header: index for index, header in enumerate(headers)}
    required = {
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.auth_asym_id",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.pdbx_PDB_model_num",
    }
    if missing := sorted(required - set(header_index)):
        raise ValueError(f"PDBx/mmCIF atom_site loop is missing required fields: {missing}")

    atom_ids: set[str] = set()
    chains: set[str] = set()
    models: set[str] = set()
    polymer_atoms = 0
    hetero_atoms = 0
    for row_number, row in enumerate(rows, start=1):
        atom_id = row[header_index["_atom_site.id"]]
        if atom_id in atom_ids:
            raise ValueError(f"Duplicate atom_site.id {atom_id!r} at coordinate row {row_number}")
        atom_ids.add(atom_id)
        try:
            coordinates = (
                float(row[header_index["_atom_site.Cartn_x"]]),
                float(row[header_index["_atom_site.Cartn_y"]]),
                float(row[header_index["_atom_site.Cartn_z"]]),
            )
        except ValueError as exc:
            raise ValueError(f"Non-numeric coordinate at atom_site row {row_number}") from exc
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"Non-finite coordinate at atom_site row {row_number}")
        group = row[header_index["_atom_site.group_PDB"]].upper()
        if group == "ATOM":
            polymer_atoms += 1
        elif group == "HETATM":
            hetero_atoms += 1
        else:
            raise ValueError(f"Unsupported atom_site.group_PDB {group!r} at row {row_number}")
        chains.add(row[header_index["_atom_site.auth_asym_id"]])
        models.add(row[header_index["_atom_site.pdbx_PDB_model_num"]])

    if polymer_atoms + hetero_atoms != len(rows):
        raise ValueError("PDBx/mmCIF coordinate-row classification is inconsistent")
    return MMCIFCoordinateQC(
        entry_id=entry_id,
        atom_count=len(rows),
        polymer_atom_count=polymer_atoms,
        hetero_atom_count=hetero_atoms,
        auth_chain_count=len(chains),
        auth_chains=tuple(sorted(chains)),
        model_count=len(models),
        model_numbers=tuple(sorted(models)),
    )


__all__ = ["MMCIFCoordinateQC", "validate_mmcif_coordinate"]
