#!/usr/bin/env python3
"""Extract the 2026-07-28 Ascentage hERG DOCX into traceable chemical records.

The source contains editable ChemDraw OLE objects rather than machine-readable
SMILES.  Each table row is linked to its exact OLE relationship, the embedded
CDX is preserved, and Open Babel is used only as a format converter.  RDKit
then performs the repository-standard structure normalization and validation.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import olefile
import pandas as pd
from menin_discovery.chemistry import standardize_smiles
from menin_discovery.features import rdkit_descriptors, scaffold_key
from menin_discovery.research_common import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
)
from rdkit import Chem
from rdkit.Chem import Descriptors, rdDepictor

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "research/data/internal/ascentage_herg_2026-07-28/source_document.docx"
DEFAULT_OUTPUT = ROOT / "research/data/pk_herg/canonical/ascentage_herg_2026_07_28"
OPEN_BABEL = Path("/opt/homebrew/bin/obabel")

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
O_NS = "urn:schemas-microsoft-com:office:office"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"w": W_NS, "r": R_NS, "o": O_NS}

EXPECTED_ROWS = 76
PIC50_NONBLOCKER_BOUND = 6.0 - math.log10(30.0)
SOURCE_CLARIFICATION_DATE = "2026-07-29"
SOURCE_CLARIFICATION_PROVIDER = "Dr. Angelo Aguilar"


def _cell_text(cell: ET.Element) -> str:
    parts = [node.text or "" for node in cell.findall(".//w:t", NS)]
    return " ".join(" ".join(parts).split())


def _relationship_map(payload: bytes) -> dict[str, str]:
    root = ET.fromstring(payload)
    mapping: dict[str, str] = {}
    for relationship in root.findall(f"{{{PKG_REL_NS}}}Relationship"):
        identifier = relationship.attrib.get("Id", "")
        target = relationship.attrib.get("Target", "")
        if identifier and target:
            mapping[identifier] = target
    return mapping


def _word_target(target: str) -> str:
    normalized = PurePosixPath("word") / PurePosixPath(target)
    parts: list[str] = []
    for part in normalized.parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part != ".":
            parts.append(part)
    return str(PurePosixPath(*parts))


def _extract_rows(source: Path) -> tuple[list[dict[str, str | bytes]], dict[str, int]]:
    with ZipFile(source) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
        relationships = _relationship_map(archive.read("word/_rels/document.xml.rels"))
        tables = document.findall(".//w:tbl", NS)
        if len(tables) != 1:
            raise ValueError(f"Expected exactly one table; found {len(tables)}")
        table_rows = tables[0].findall("./w:tr", NS)
        if len(table_rows) != EXPECTED_ROWS + 1:
            raise ValueError(f"Expected header plus {EXPECTED_ROWS} rows; found {len(table_rows)} rows")

        records: list[dict[str, str | bytes]] = []
        for source_row, row in enumerate(table_rows[1:], start=1):
            cells = row.findall("./w:tc", NS)
            if len(cells) != 4:
                raise ValueError(f"Row {source_row} has {len(cells)} cells instead of four")

            internal_id = _cell_text(cells[1]).replace(" ", "")
            ascentage_id = _cell_text(cells[2]).replace(" ", "")
            submitted_result = _cell_text(cells[3]).replace(" ", "")
            if not re.fullmatch(r"M-\d{4}", internal_id):
                raise ValueError(f"Invalid internal ID at row {source_row}: {internal_id!r}")
            if ascentage_id and not re.fullmatch(r"AS\d{5}", ascentage_id):
                raise ValueError(f"Invalid Ascentage ID at row {source_row}: {ascentage_id!r}")

            ole_objects = cells[0].findall(".//o:OLEObject", NS)
            relationship_ids = [node.attrib.get(f"{{{R_NS}}}id", "") for node in ole_objects]
            relationship_ids = [value for value in relationship_ids if value]
            if len(relationship_ids) != 1:
                raise ValueError(
                    f"Expected one ChemDraw OLE object for {internal_id}; found {len(relationship_ids)}"
                )
            relationship_id = relationship_ids[0]
            if relationship_id not in relationships:
                raise ValueError(f"Missing relationship {relationship_id} for {internal_id}")
            member = _word_target(relationships[relationship_id])
            embedded = archive.read(member)
            with olefile.OleFileIO(io.BytesIO(embedded)) as container:
                if not container.exists("CONTENTS"):
                    raise ValueError(f"ChemDraw CONTENTS stream missing for {internal_id}")
                cdx = container.openstream("CONTENTS").read()
            if not cdx.startswith(b"VjCD0100"):
                raise ValueError(f"Embedded CONTENTS is not recognized CDX for {internal_id}")
            records.append(
                {
                    "source_row": source_row,
                    "internal_id": internal_id,
                    "ascentage_id": ascentage_id,
                    "submitted_herg_ic50_um": submitted_result,
                    "ole_relationship_id": relationship_id,
                    "ole_member": member,
                    "cdx": cdx,
                }
            )

    return records, {
        "table_rows_including_header": len(table_rows),
        "compound_rows": len(records),
        "embedded_cdx_count": len(records),
    }


def _cdx_to_smiles(cdx: bytes, identifier: str) -> str:
    if not OPEN_BABEL.exists():
        raise FileNotFoundError(f"Open Babel is required at {OPEN_BABEL}")
    with tempfile.TemporaryDirectory(prefix="ascentage-cdx-") as temporary:
        input_path = Path(temporary) / f"{identifier}.cdx"
        input_path.write_bytes(cdx)
        result = subprocess.run(
            [str(OPEN_BABEL), "-icdx", str(input_path), "-osmi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    if result.returncode != 0:
        raise ValueError(f"Open Babel failed for {identifier}: {result.stderr.strip()}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"Expected one converted molecule for {identifier}; found {len(lines)}")
    smiles = lines[0].split()[0]
    if Chem.MolFromSmiles(smiles) is None:
        raise ValueError(f"Open Babel returned invalid SMILES for {identifier}")
    return smiles


def _parse_measurement(submitted: str) -> dict[str, object]:
    if submitted == "":
        return {
            "herg_ic50_value_um": float("nan"),
            "herg_ic50_relation": "",
            "herg_ic50_censoring": "missing",
            "herg_ic50_lower_bound_um": float("nan"),
            "herg_ic50_upper_bound_um": float("nan"),
            "herg_pic50_value": float("nan"),
            "herg_pic50_relation": "",
            "herg_pic50_lower_bound": float("nan"),
            "herg_pic50_upper_bound": float("nan"),
            "observed_decisive_class": float("nan"),
        }
    if submitted.startswith(">"):
        bound = float(submitted[1:])
        pic50_bound = 6.0 - math.log10(bound)
        return {
            "herg_ic50_value_um": bound,
            "herg_ic50_relation": ">",
            "herg_ic50_censoring": "right",
            "herg_ic50_lower_bound_um": bound,
            "herg_ic50_upper_bound_um": float("nan"),
            "herg_pic50_value": pic50_bound,
            "herg_pic50_relation": "<",
            "herg_pic50_lower_bound": float("nan"),
            "herg_pic50_upper_bound": pic50_bound,
            "observed_decisive_class": 0.0,
        }
    value = float(submitted)
    pic50 = 6.0 - math.log10(value)
    decisive = 1.0 if pic50 >= 5.0 else (0.0 if pic50 <= PIC50_NONBLOCKER_BOUND else float("nan"))
    return {
        "herg_ic50_value_um": value,
        "herg_ic50_relation": "=",
        "herg_ic50_censoring": "none",
        "herg_ic50_lower_bound_um": value,
        "herg_ic50_upper_bound_um": value,
        "herg_pic50_value": pic50,
        "herg_pic50_relation": "=",
        "herg_pic50_lower_bound": pic50,
        "herg_pic50_upper_bound": pic50,
        "observed_decisive_class": decisive,
    }


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        if temporary.stat().st_size != source.stat().st_size:
            raise ValueError("Source-document copy failed size validation")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_sdf(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    writer = Chem.SDWriter(str(temporary))
    try:
        for row in frame.itertuples(index=False):
            molecule = Chem.MolFromSmiles(row.standardized_smiles)
            if molecule is None:
                raise ValueError(f"Cannot write invalid structure for {row.internal_id}")
            rdDepictor.Compute2DCoords(molecule)
            molecule.SetProp("_Name", row.internal_id)
            molecule.SetProp("internal_id", row.internal_id)
            molecule.SetProp("ascentage_id", row.ascentage_id)
            molecule.SetProp("structure_id", row.structure_id)
            molecule.SetProp("submitted_herg_ic50_um", row.submitted_herg_ic50_um)
            molecule.SetProp("synthesis_status", row.synthesis_status)
            molecule.SetProp("series_status", row.series_status)
            writer.write(molecule)
    finally:
        writer.close()
    supplier = Chem.SDMolSupplier(str(temporary), removeHs=False)
    if sum(molecule is not None for molecule in supplier) != len(frame):
        temporary.unlink(missing_ok=True)
        raise ValueError("SDF read-back count mismatch")
    os.replace(temporary, path)


def ingest(source: Path, output: Path) -> pd.DataFrame:
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    extracted, package_qc = _extract_rows(source)

    rows: list[dict[str, object]] = []
    cdx_dir = output / "source_cdx"
    cdx_dir.mkdir(parents=True, exist_ok=True)
    for record in extracted:
        internal_id = str(record["internal_id"])
        cdx = bytes(record.pop("cdx"))
        smiles = _cdx_to_smiles(cdx, internal_id)
        standardized = standardize_smiles(smiles, require_rdkit=True)
        if standardized.structure_valid is not True:
            raise ValueError(f"Standardization failed for {internal_id}: {standardized.structure_error}")
        molecule = Chem.MolFromSmiles(standardized.standardized_smiles)
        assert molecule is not None
        source_cdx = cdx_dir / f"{internal_id}.cdx"
        source_cdx.write_bytes(cdx)
        row: dict[str, object] = {
            **record,
            "source_document": "source_document.docx",
            "source_cdx": str(source_cdx.relative_to(output)),
            "submitted_smiles": smiles,
            **standardized.as_dict(),
            "computed_mw_g_mol": float(Descriptors.MolWt(molecule)),
            "scaffold": scaffold_key(standardized.standardized_smiles)[0],
            **_parse_measurement(str(record["submitted_herg_ic50_um"])),
            "synthesis_status": (
                "not_synthesized" if str(record["submitted_herg_ic50_um"]) == "" else "synthesized_by_cro"
            ),
            "herg_measurement_status": (
                "not_applicable_not_synthesized"
                if str(record["submitted_herg_ic50_um"]) == ""
                else (
                    "right_censored_no_50pct_inhibition_at_30um"
                    if str(record["submitted_herg_ic50_um"]).startswith(">")
                    else "exact_summary_ic50_reported"
                )
            ),
            "herg_missing_reason": (
                "compound_not_synthesized" if str(record["submitted_herg_ic50_um"]) == "" else ""
            ),
            "menin_potency_status": "not_available_from_source_provider",
            "series_status": "same_internal_medicinal_chemistry_series_confirmed",
            "synthesis_provider": (
                "" if str(record["submitted_herg_ic50_um"]) == "" else "contract_research_organization"
            ),
            "source_clarification_date": SOURCE_CLARIFICATION_DATE,
            "source_clarification_provider": SOURCE_CLARIFICATION_PROVIDER,
            "assay_protocol_status": "not_reported_in_source",
            "evaluation_role": "retrospective_same_series_extension_evaluation",
            "decision_track_eligible": False,
            "decision_track_exclusion": (
                "assay platform, cell system, voltage protocol, temperature, pH, "
                "incubation, and free-concentration basis are absent; chemistry is from "
                "the same medicinal chemistry series"
            ),
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame["internal_id"].duplicated().any():
        raise ValueError("Duplicate internal IDs in source")
    if frame["structure_id"].duplicated().any():
        duplicates = frame.loc[frame["structure_id"].duplicated(False), "internal_id"].tolist()
        raise ValueError(f"Duplicate standardized structures in source: {duplicates}")

    descriptors = rdkit_descriptors(frame["standardized_smiles"])
    descriptor_keep = [
        "mw",
        "logp",
        "tpsa",
        "hbd",
        "hba",
        "rotatable_bonds",
        "fraction_csp3",
        "aromatic_rings",
        "formal_charge",
    ]
    descriptor_keep = [column for column in descriptor_keep if column in descriptors]
    frame = pd.concat(
        [frame.reset_index(drop=True), descriptors[descriptor_keep].add_prefix("audit_")],
        axis=1,
    )

    exact = frame["herg_ic50_relation"].eq("=")
    censored = frame["herg_ic50_relation"].eq(">")
    missing = frame["herg_ic50_censoring"].eq("missing")
    recalculated_delta = (
        frame["computed_mw_g_mol"] - pd.to_numeric(frame["audit_mw"], errors="coerce")
        if "audit_mw" in frame
        else pd.Series(0.0, index=frame.index)
    )
    qc = {
        **package_qc,
        "source_file": str(source),
        "visual_page_review": {
            "status": "passed",
            "pages_reviewed": 31,
            "blank_pages": [31],
        },
        "records": len(frame),
        "unique_structures": int(frame["structure_id"].nunique()),
        "exact_measurements": int(exact.sum()),
        "right_censored_measurements": int(censored.sum()),
        "missing_measurements": int(missing.sum()),
        "not_synthesized_structures": int(frame["synthesis_status"].eq("not_synthesized").sum()),
        "synthesized_by_cro_structures": int(frame["synthesis_status"].eq("synthesized_by_cro").sum()),
        "minimum_exact_ic50_um": float(frame.loc[exact, "herg_ic50_value_um"].min()),
        "maximum_exact_ic50_um": float(frame.loc[exact, "herg_ic50_value_um"].max()),
        "minimum_computed_mw_g_mol": float(frame["computed_mw_g_mol"].min()),
        "maximum_computed_mw_g_mol": float(frame["computed_mw_g_mol"].max()),
        "mw_descriptor_recalculation_max_abs_delta": float(recalculated_delta.abs().max()),
        "all_structures_standardized": bool(frame["structure_valid"].eq(True).all()),
        "all_censored_pic50_bounds_correct": bool(
            (frame.loc[censored, "herg_pic50_upper_bound"] - PIC50_NONBLOCKER_BOUND).abs().lt(1e-10).all()
        ),
        "missing_internal_ids": frame.loc[missing, "internal_id"].tolist(),
        "right_censored_internal_ids": frame.loc[censored, "internal_id"].tolist(),
        "source_clarifications": {
            "date_received": SOURCE_CLARIFICATION_DATE,
            "provider": SOURCE_CLARIFICATION_PROVIDER,
            "right_censoring_meaning": (
                "For >30 uM records, 50% hERG inhibition was not reached at the highest "
                "tested concentration of 30 uM."
            ),
            "blank_entry_meaning": "The eight blank-entry structures were not synthesized.",
            "menin_potency_availability": "Not available from Dr. Aguilar.",
            "synthesis_provenance": "Measured compounds were synthesized by a CRO.",
            "series_relation": "All records belong to the same medicinal chemistry series.",
        },
        "source_limitations": [
            "No hERG assay protocol metadata are supplied.",
            "The source provides summary IC50 values rather than replicate concentration-response data.",
            "Eight drawn structures were not synthesized and therefore have no hERG measurement.",
            "Menin potency is unavailable for these molecules.",
            "The chemistry is from the same series and is not independent-series evidence.",
        ],
    }

    output.mkdir(parents=True, exist_ok=True)
    _atomic_copy(source, output / "source_document.docx")
    review_columns = [
        "source_row",
        "internal_id",
        "ascentage_id",
        "submitted_herg_ic50_um",
        "herg_ic50_value_um",
        "herg_ic50_relation",
        "herg_pic50_value",
        "herg_pic50_relation",
        "synthesis_status",
        "herg_measurement_status",
        "herg_missing_reason",
        "menin_potency_status",
        "series_status",
        "synthesis_provider",
        "standardized_smiles",
        "standard_inchi_key",
        "structure_id",
        "computed_mw_g_mol",
        "scaffold",
        "assay_protocol_status",
        "decision_track_eligible",
    ]
    atomic_write_csv(output / "normalized_records.csv", frame[review_columns])
    atomic_write_parquet(output / "normalized_records.parquet", frame)
    atomic_write_json(output / "ingestion_qc.json", qc)
    _write_sdf(frame, output / "structures.sdf")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frame = ingest(args.source, args.output)
    summary = {
        "records": len(frame),
        "exact": int(frame["herg_ic50_relation"].eq("=").sum()),
        "right_censored": int(frame["herg_ic50_relation"].eq(">").sum()),
        "missing": int(frame["herg_ic50_censoring"].eq("missing").sum()),
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
