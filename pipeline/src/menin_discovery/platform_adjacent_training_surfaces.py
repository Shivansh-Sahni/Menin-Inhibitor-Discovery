"""Freeze structure-linked PRISM and LINCS training surfaces without expanding matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

SCHEMA_VERSION = "platform-adjacent-training-surfaces/1.0"
SPLIT_SEED = "20260809"
EXPECTED_OUTPUTS = {
    "ADJACENT_TRAINING_SURFACES.md",
    "adjacent_training_manifest.json",
    "lincs_compound_instances.parquet",
    "lincs_gene_registry.parquet",
    "prism_cell_registry.parquet",
    "prism_treatment_registry.parquet",
    "training_surface_contract.json",
}


class AdjacentTrainingSurfaceError(RuntimeError):
    """Raised when a large adjacent-modality surface cannot be proven."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_hash(payload: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    raw = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _schema_hash(path: Path) -> str:
    return hashlib.sha256(pq.read_schema(path).serialize().to_pybytes()).hexdigest()


def _structure(raw_smiles: object) -> tuple[str | None, str | None, str | None, str | None]:
    text = str(raw_smiles).strip() if pd.notna(raw_smiles) else ""
    if not text or text.lower() in {"nan", "na", "-666", "restricted"}:
        return None, None, None, None
    # PRISM reports some stereoisomer/component sets as comma-separated SMILES.
    # Preserve every reported component as a disconnected graph; never select one.
    parse_text = ".".join(part.strip() for part in text.split(",") if part.strip())
    molecule = Chem.MolFromSmiles(parse_text)
    if molecule is None:
        return None, None, None, None
    smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    inchikey = Chem.MolToInchiKey(molecule)
    structure_id = "STR_" + hashlib.sha256(inchikey.encode()).hexdigest()[:24]
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    scaffold_key = scaffold or smiles
    scaffold_group_id = "SCF_" + hashlib.sha256(scaffold_key.encode()).hexdigest()[:24]
    return smiles, inchikey, structure_id, scaffold_group_id


def _split(scaffold_group_id: str) -> str:
    value = int(hashlib.sha256(f"{SPLIT_SEED}:{scaffold_group_id}".encode()).hexdigest()[:16], 16)
    bucket = value % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _broad_base_id(value: object) -> str | None:
    text = str(value).strip().upper() if pd.notna(value) else ""
    if len(text) >= 13 and text.startswith("BRD-") and text[4].isalpha() and text[5:13].isdigit():
        return text[:13]
    return None


def _bind_input(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _bind_parquet(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "rows": pq.read_metadata(path).num_rows,
        "arrow_schema_sha256": _schema_hash(path),
    }


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, path, compression="zstd", use_dictionary=True)


def _apply_global_leakage_groups(
    prism_treatments: pd.DataFrame, lincs_instances: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unify exact structures and scaffold variants into one cross-modality split group."""

    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            low, high = sorted((left_root, right_root))
            parent[high] = low

    pairs = (
        pd.concat(
            [
                prism_treatments[["structure_id", "scaffold_group_id"]],
                lincs_instances[["structure_id", "scaffold_group_id"]],
            ],
            ignore_index=True,
        )
        .dropna()
        .drop_duplicates()
    )
    for structure_id, scaffold_group_id in pairs.itertuples(index=False, name=None):
        union(f"structure:{structure_id}", f"scaffold:{scaffold_group_id}")
    compound_links = (
        pd.concat(
            [
                prism_treatments[["structure_id", "cross_source_compound_id"]],
                lincs_instances[["structure_id", "cross_source_compound_id"]],
            ],
            ignore_index=True,
        )
        .dropna()
        .drop_duplicates()
    )
    for structure_id, compound_id in compound_links.itertuples(index=False, name=None):
        union(f"structure:{structure_id}", f"compound:{compound_id}")
    roots = {node: find(node) for node in parent}
    members: dict[str, list[str]] = {}
    for node, root in roots.items():
        members.setdefault(root, []).append(node)
    group_by_root = {
        root: "LKG_" + hashlib.sha256("|".join(sorted(nodes)).encode()).hexdigest()[:24]
        for root, nodes in members.items()
    }
    structure_to_group = {
        node.removeprefix("structure:"): group_by_root[root]
        for node, root in roots.items()
        if node.startswith("structure:")
    }
    outputs: list[pd.DataFrame] = []
    for frame in (prism_treatments, lincs_instances):
        frame = frame.copy()
        frame["leakage_group_id"] = frame["structure_id"].map(structure_to_group)
        frame["model_split"] = frame["leakage_group_id"].map(
            lambda value: _split(value) if pd.notna(value) else None
        )
        outputs.append(frame)
    return outputs[0], outputs[1]


def _prism_registry(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], int]:
    treatment_frames: list[pd.DataFrame] = []
    cell_frames: list[pd.DataFrame] = []
    matrix_contracts: list[dict[str, Any]] = []
    eligible_values = 0
    for phase in ("primary", "secondary"):
        treatment_path = root / f"{phase}-screen-replicate-collapsed-treatment-info.csv"
        cell_path = root / f"{phase}-screen-cell-line-info.csv"
        matrix_path = root / f"{phase}-screen-replicate-collapsed-logfold-change.csv"
        treatment = pd.read_csv(treatment_path, low_memory=False)
        cell = pd.read_csv(cell_path, low_memory=False)
        required_treatment = {"column_name", "broad_id", "name", "dose", "screen_id", "smiles"}
        if not required_treatment.issubset(treatment.columns) or "row_name" not in cell:
            raise AdjacentTrainingSurfaceError(f"unexpected PRISM {phase} metadata schema")
        structures = treatment["smiles"].map(_structure)
        treatment[["standardized_smiles", "standard_inchi_key", "structure_id", "scaffold_group_id"]] = (
            pd.DataFrame(structures.tolist(), index=treatment.index)
        )
        treatment["model_split"] = treatment["scaffold_group_id"].map(
            lambda value: _split(value) if pd.notna(value) else None
        )
        treatment["structure_representation_scope"] = (
            treatment["smiles"]
            .astype(str)
            .map(lambda value: "reported_component_set" if "," in str(value) else "single_reported_structure")
        )
        treatment["cross_source_compound_id"] = treatment["broad_id"].map(_broad_base_id)
        treatment["screen_phase"] = phase
        treatment["target_semantics"] = "replicate_collapsed_logfold_change_viability"
        treatment["training_eligible"] = treatment["structure_id"].notna()
        keep = [
            "screen_phase",
            "column_name",
            "broad_id",
            "name",
            "dose",
            "screen_id",
            "smiles",
            "standardized_smiles",
            "standard_inchi_key",
            "structure_id",
            "scaffold_group_id",
            "model_split",
            "structure_representation_scope",
            "cross_source_compound_id",
            "target_semantics",
            "training_eligible",
        ]
        treatment_frames.append(treatment[keep].copy())
        cell = cell.copy()
        cell["screen_phase"] = phase
        cell["cell_context_id"] = cell["row_name"].astype(str)

        header = pd.read_csv(matrix_path, nrows=0).columns.tolist()
        matrix_columns = header[1:]
        if set(matrix_columns) != set(treatment["column_name"].astype(str)):
            raise AdjacentTrainingSurfaceError(f"PRISM {phase} matrix/treatment order mismatch")
        observed = 0
        eligible = 0
        row_count = 0
        matrix_cell_ids: list[str] = []
        known_cell_ids = set(cell["row_name"].astype(str))
        if "passed_str_profiling" in cell:
            eligible_cell_ids = set(
                cell.loc[cell["passed_str_profiling"].astype(str).str.upper() == "TRUE", "row_name"].astype(
                    str
                )
            )
        else:
            eligible_cell_ids = known_cell_ids
        eligible_mask = (
            treatment.set_index("column_name")["training_eligible"]
            .reindex(matrix_columns)
            .fillna(False)
            .to_numpy()
        )
        for chunk in pd.read_csv(matrix_path, chunksize=64):
            ids = chunk.iloc[:, 0].astype(str)
            matrix_cell_ids.extend(ids.tolist())
            values = chunk.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
            finite = pd.DataFrame(
                np.isfinite(values.to_numpy(dtype=float)),
                index=values.index,
                columns=values.columns,
            )
            observed += int(finite.to_numpy().sum())
            eligible_rows = ids.isin(eligible_cell_ids).to_numpy()
            eligible += int(finite.iloc[eligible_rows, eligible_mask].to_numpy().sum())
            row_count += len(chunk)
        if len(matrix_cell_ids) != len(set(matrix_cell_ids)) or row_count > len(cell):
            raise AdjacentTrainingSurfaceError(f"PRISM {phase} matrix cell identity mismatch")
        cell["has_matrix_measurements"] = cell["row_name"].astype(str).isin(matrix_cell_ids)
        cell["cell_training_eligible"] = (
            cell["row_name"].astype(str).isin(eligible_cell_ids & set(matrix_cell_ids))
        )
        missing_cell_ids = sorted(set(matrix_cell_ids) - known_cell_ids)
        if missing_cell_ids:
            placeholder = pd.DataFrame(
                {
                    "row_name": missing_cell_ids,
                    "screen_phase": phase,
                    "cell_context_id": missing_cell_ids,
                    "has_matrix_measurements": True,
                    "cell_training_eligible": False,
                }
            )
            cell = pd.concat([cell, placeholder], ignore_index=True)
        cell_frames.append(cell)
        eligible_values += eligible
        matrix_contracts.append(
            {
                "phase": phase,
                "matrix_path": str(matrix_path.resolve()),
                "matrix_bytes": matrix_path.stat().st_size,
                "matrix_sha256": _sha256(matrix_path),
                "cell_rows": row_count,
                "cell_metadata_rows": len(cell),
                "matrix_only_unresolved_cell_rows": len(missing_cell_ids),
                "treatment_columns": len(matrix_columns),
                "possible_values": row_count * len(matrix_columns),
                "observed_finite_values": observed,
                "structure_linked_training_values": eligible,
                "target": "replicate_collapsed_logfold_change",
                "matrix_expanded_to_long_format": False,
                "streaming_loader_required": True,
            }
        )
    treatments = pd.concat(treatment_frames, ignore_index=True)
    cells = pd.concat(cell_frames, ignore_index=True)
    return treatments, cells, matrix_contracts, eligible_values


def _lincs_registry(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], int]:
    instances_out: list[pd.DataFrame] = []
    genes_out: list[pd.DataFrame] = []
    matrix_contracts: list[dict[str, Any]] = []
    eligible_positions = 0
    releases = (
        ("GSE92742", root / "lincs_gse92742"),
        ("GSE70138", root / "lincs_gse70138"),
    )
    for accession, release_root in releases:
        pert_path = next(release_root.glob("*pert_info*"))
        inst_path = next(release_root.glob("*inst_info*"))
        gene_path = next(release_root.glob("*gene_info*"))
        matrix_paths = sorted(release_root.glob("*Level2_GEX*.gctx.gz"))
        pert = pd.read_csv(pert_path, sep="\t", compression="gzip", dtype=str, low_memory=False)
        inst = pd.read_csv(inst_path, sep="\t", compression="gzip", dtype=str, low_memory=False)
        gene = pd.read_csv(gene_path, sep="\t", compression="gzip", dtype=str, low_memory=False)
        if not {"pert_id", "canonical_smiles"}.issubset(pert) or not {
            "inst_id",
            "pert_id",
            "pert_type",
            "cell_id",
        }.issubset(inst):
            raise AdjacentTrainingSurfaceError(f"unexpected LINCS {accession} metadata schema")
        pert = pert.copy()
        structures = pert["canonical_smiles"].map(_structure)
        pert[["standardized_smiles", "standard_inchi_key", "structure_id", "scaffold_group_id"]] = (
            pd.DataFrame(structures.tolist(), index=pert.index)
        )
        compound = pert.loc[
            pert["standardized_smiles"].notna(),
            [
                "pert_id",
                "pert_iname",
                "canonical_smiles",
                "standardized_smiles",
                "standard_inchi_key",
                "structure_id",
                "scaffold_group_id",
            ],
        ].drop_duplicates("pert_id")
        joined = inst.merge(compound, on="pert_id", how="inner", validate="many_to_one")
        joined = joined.loc[joined["pert_type"].astype(str).str.startswith("trt_cp")].copy()
        joined["release_accession"] = accession
        joined["model_split"] = joined["scaffold_group_id"].map(_split)
        joined["target_semantics"] = "level2_landmark_gene_expression"
        joined["training_eligible"] = True
        joined["cross_source_compound_id"] = joined["pert_id"].map(_broad_base_id)
        keep = [
            "release_accession",
            "inst_id",
            "pert_id",
            "pert_iname_x",
            "pert_type",
            "pert_dose",
            "pert_dose_unit",
            "pert_time",
            "pert_time_unit",
            "cell_id",
            "canonical_smiles",
            "standardized_smiles",
            "standard_inchi_key",
            "structure_id",
            "scaffold_group_id",
            "model_split",
            "cross_source_compound_id",
            "target_semantics",
            "training_eligible",
        ]
        joined = joined.rename(columns={"pert_iname_x": "perturbagen_name"})
        if "perturbagen_name" not in joined and "pert_iname" in joined:
            joined = joined.rename(columns={"pert_iname": "perturbagen_name"})
        keep = ["perturbagen_name" if name == "pert_iname_x" else name for name in keep]
        instances_out.append(joined[keep])

        landmarks = gene.loc[gene["pr_is_lm"].astype(str) == "1"].copy()
        if len(landmarks) != 978:
            raise AdjacentTrainingSurfaceError(f"LINCS {accession} does not have 978 landmarks")
        landmarks["release_accession"] = accession
        genes_out.append(landmarks[["release_accession", "pr_gene_id", "pr_gene_symbol", "pr_gene_title"]])
        positions = len(joined) * len(landmarks)
        eligible_positions += positions
        matrix_contracts.append(
            {
                "release_accession": accession,
                "matrix_paths": [str(path.resolve()) for path in matrix_paths],
                "matrix_bindings": [_bind_input(path) for path in matrix_paths],
                "all_instances": len(inst),
                "structure_linked_compound_instances": len(joined),
                "landmark_genes": len(landmarks),
                "metadata_derived_structure_linked_positions": positions,
                "target": "level2_expression_value_by_landmark_gene",
                "compressed_gctx_requires_staging_or_stream_capable_loader": True,
                "matrix_values_or_axis_membership_scanned": False,
                "matrix_expanded_to_profile_gene_long_format": False,
            }
        )
    instances = pd.concat(instances_out, ignore_index=True)
    genes = pd.concat(genes_out, ignore_index=True)
    return instances, genes, matrix_contracts, eligible_positions


def build_adjacent_training_surfaces(*, source_root: Path, output_root: Path) -> dict[str, Any]:
    """Build compact training indices and immutable contracts around large source matrices."""

    prism_root = source_root / "prism_repurposing"
    prism_treatments, prism_cells, prism_matrices, prism_values = _prism_registry(prism_root)
    lincs_instances, lincs_genes, lincs_matrices, lincs_positions = _lincs_registry(source_root)
    prism_treatments, lincs_instances = _apply_global_leakage_groups(prism_treatments, lincs_instances)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        _write_parquet(prism_treatments, staging / "prism_treatment_registry.parquet")
        _write_parquet(prism_cells, staging / "prism_cell_registry.parquet")
        _write_parquet(lincs_instances, staging / "lincs_compound_instances.parquet")
        _write_parquet(lincs_genes, staging / "lincs_gene_registry.parquet")
        contract = {
            "schema_version": SCHEMA_VERSION,
            "purpose": "structure_linked_training_surfaces_for_future_multimodal_platform",
            "prism": {
                "prediction_unit": "compound_treatment_by_cell_line",
                "matrices": prism_matrices,
                "default_generalization": "compound_scaffold_cold",
                "cell_context_is_input_not_target": True,
                "structure_linked_training_values": prism_values,
            },
            "lincs": {
                "prediction_unit": "compound_dose_time_cell_profile_by_landmark_gene",
                "matrices": lincs_matrices,
                "default_generalization": "compound_scaffold_cold",
                "dose_time_cell_are_inputs_not_targets": True,
                "metadata_derived_structure_linked_positions": lincs_positions,
            },
            "shared_contract": {
                "split_group": (
                    "cross_modality_exact_structure_Bemis_Murcko_and_normalized_Broad_compound_"
                    "identity_connected_component"
                ),
                "exact_structure_or_scaffold_cross_split_allowed": False,
                "targets_expanded_or_duplicated": False,
                "production_features_generated": False,
                "substantive_training_started": False,
                "hpc_executed": False,
                "rights_boundary": (
                    "PRISM publisher metadata records CC BY 4.0; LINCS/GEO source provenance and "
                    "checksum records are bound, but downstream release must retain attribution."
                ),
            },
        }
        contract_path = staging / "training_surface_contract.json"
        contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")

        report_path = staging / "ADJACENT_TRAINING_SURFACES.md"
        report_path.write_text(
            "# Structure-linked adjacent training surfaces — 2026-08-09\n\n"
            "PRISM is now indexed as a structure-linked drug-by-cell response surface with "
            f"{prism_values:,} eligible finite log-fold-change targets. Reported component sets "
            "remain explicit disconnected structures rather than being silently reduced to one component. "
            "Cell contexts that failed or lack STR metadata are retained but excluded from eligible counts.\n\n"
            "LINCS is now indexed at the compound-instance grain with "
            f"{len(lincs_instances):,} structure-resolved compound profiles and "
            f"{lincs_positions:,} metadata-derived, addressable landmark-gene positions. These positions are "
            "not claimed as scanned finite values. The compressed GCTX matrices "
            "remain source-bound rather than being wastefully expanded to long format; training requires a "
            "GCTX-capable loader or deterministic staging.\n\n"
            "Exact structures, Bemis–Murcko scaffolds, and normalized Broad compound IDs are joined into "
            "cross-modality connected leakage groups before deterministic train, validation, and test "
            "assignment. PRISM viability and LINCS "
            "expression remain separate context-dependent objectives. No molecular features, model fitting, "
            "HPC work, figure, or presentation table was generated.\n",
            encoding="utf-8",
        )

        inputs: list[dict[str, Any]] = []
        for path in sorted(prism_root.glob("*.csv")):
            if "replicate-collapsed" in path.name or "cell-line-info" in path.name:
                inputs.append(_bind_input(path))
        for path in sorted(prism_root.glob("*.txt")) + sorted(prism_root.glob("*.json")):
            inputs.append(_bind_input(path))
        for release in (source_root / "lincs_gse92742", source_root / "lincs_gse70138"):
            for pattern in (
                "*pert_info*",
                "*inst_info*",
                "*gene_info*",
                "*Level2_GEX*.gctx.gz",
                "*SHA512SUMS*",
                "*accession_brief*",
            ):
                inputs.extend(_bind_input(path) for path in sorted(release.glob(pattern)))
        parquet_names = [
            "prism_treatment_registry.parquet",
            "prism_cell_registry.parquet",
            "lincs_compound_instances.parquet",
            "lincs_gene_registry.parquet",
        ]
        artifacts = {name: _bind_parquet(staging / name) for name in parquet_names}
        artifacts[contract_path.name] = {
            "path": contract_path.name,
            "bytes": contract_path.stat().st_size,
            "sha256": _sha256(contract_path),
        }
        artifacts[report_path.name] = {
            "path": report_path.name,
            "bytes": report_path.stat().st_size,
            "sha256": _sha256(report_path),
        }
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "implementation": _bind_input(Path(__file__)),
            "inputs": inputs,
            "artifacts": artifacts,
            "counts": {
                "prism_treatment_rows": len(prism_treatments),
                "prism_unique_structures": int(prism_treatments["structure_id"].nunique()),
                "prism_cell_context_rows": len(prism_cells),
                "prism_structure_linked_training_values": prism_values,
                "lincs_compound_instance_rows": len(lincs_instances),
                "lincs_unique_structures": int(lincs_instances["structure_id"].nunique()),
                "lincs_landmark_gene_rows_across_releases": len(lincs_genes),
                "lincs_metadata_derived_profile_gene_positions": lincs_positions,
                "normalized_broad_ids_shared_across_prism_lincs": len(
                    set(prism_treatments["cross_source_compound_id"].dropna())
                    & set(lincs_instances["cross_source_compound_id"].dropna())
                ),
            },
            "status": {
                "prism_matrix_training_surface": "trainable_with_streaming_matrix_loader",
                "lincs_compound_profile_surface": "trainable_after_compressed_gctx_staging_or_stream_loader",
                "not_a_single_pooled_endpoint": True,
                "no_figures_or_presentation_tables_generated": True,
            },
        }
        manifest["manifest_sha256"] = _manifest_hash(manifest)
        (staging / "adjacent_training_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if output_root.exists():
            raise AdjacentTrainingSurfaceError(f"refusing to overwrite existing release: {output_root}")
        staging.rename(output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_adjacent_training_surfaces(output_root)


def validate_adjacent_training_surfaces(output_root: Path) -> dict[str, Any]:
    """Validate bindings, counts, structure/scaffold splits, and claim boundaries."""

    actual = {path.name for path in output_root.iterdir() if path.is_file()}
    if actual != EXPECTED_OUTPUTS:
        raise AdjacentTrainingSurfaceError(
            f"unexpected output membership: {sorted(actual ^ EXPECTED_OUTPUTS)}"
        )
    manifest = json.loads((output_root / "adjacent_training_manifest.json").read_text())
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise AdjacentTrainingSurfaceError("manifest self-hash mismatch")
    implementation = manifest["implementation"]
    implementation_path = Path(implementation["path"])
    if (
        not implementation_path.is_file()
        or implementation_path.stat().st_size != implementation["bytes"]
        or _sha256(implementation_path) != implementation["sha256"]
    ):
        raise AdjacentTrainingSurfaceError("implementation binding failed")
    for binding in manifest["inputs"]:
        path = Path(binding["path"])
        if (
            not path.is_file()
            or path.stat().st_size != binding["bytes"]
            or _sha256(path) != binding["sha256"]
        ):
            raise AdjacentTrainingSurfaceError(f"input binding failed: {path}")
    for name, binding in manifest["artifacts"].items():
        path = output_root / name
        if (
            not path.is_file()
            or path.stat().st_size != binding["bytes"]
            or _sha256(path) != binding["sha256"]
        ):
            raise AdjacentTrainingSurfaceError(f"artifact binding failed: {name}")
        if path.suffix == ".parquet":
            if (
                pq.read_metadata(path).num_rows != binding["rows"]
                or _schema_hash(path) != binding["arrow_schema_sha256"]
            ):
                raise AdjacentTrainingSurfaceError(f"Parquet binding failed: {name}")
    treatments = pd.read_parquet(output_root / "prism_treatment_registry.parquet")
    cells = pd.read_parquet(output_root / "prism_cell_registry.parquet")
    instances = pd.read_parquet(output_root / "lincs_compound_instances.parquet")
    if (cells["cell_training_eligible"] & ~cells["has_matrix_measurements"]).any():
        raise AdjacentTrainingSurfaceError("PRISM cell is eligible without matrix measurements")
    for label, frame in (("PRISM", treatments), ("LINCS", instances)):
        eligible = frame.loc[frame["training_eligible"]]
        if eligible[["structure_id", "scaffold_group_id", "model_split"]].isna().any().any():
            raise AdjacentTrainingSurfaceError(f"{label} eligible identities are incomplete")
        if eligible.groupby("structure_id")["model_split"].nunique().max() != 1:
            raise AdjacentTrainingSurfaceError(f"{label} structure crosses splits")
        if eligible.groupby("scaffold_group_id")["model_split"].nunique().max() != 1:
            raise AdjacentTrainingSurfaceError(f"{label} scaffold crosses splits")
        if eligible.groupby("leakage_group_id")["model_split"].nunique().max() != 1:
            raise AdjacentTrainingSurfaceError(f"{label} leakage group crosses splits")
    combined = pd.concat(
        [
            treatments.loc[treatments["training_eligible"]],
            instances.loc[instances["training_eligible"]],
        ],
        ignore_index=True,
    )
    if combined.groupby("structure_id")["model_split"].nunique().max() != 1:
        raise AdjacentTrainingSurfaceError("structure crosses PRISM/LINCS splits")
    if combined.groupby("scaffold_group_id")["model_split"].nunique().max() != 1:
        raise AdjacentTrainingSurfaceError("scaffold crosses PRISM/LINCS splits")
    compound = combined.dropna(subset=["cross_source_compound_id"])
    if compound.groupby("cross_source_compound_id")["model_split"].nunique().max() != 1:
        raise AdjacentTrainingSurfaceError("normalized Broad compound crosses PRISM/LINCS splits")
    contract = json.loads((output_root / "training_surface_contract.json").read_text())
    counts = manifest["counts"]
    if counts["prism_structure_linked_training_values"] < 8_000_000:
        raise AdjacentTrainingSurfaceError("PRISM structure-linked scale unexpectedly below eight million")
    if (
        counts["lincs_compound_instance_rows"] < 900_000
        or counts["lincs_metadata_derived_profile_gene_positions"] < 900_000_000
    ):
        raise AdjacentTrainingSurfaceError("LINCS structure-linked scale unexpectedly low")
    if not manifest["status"]["not_a_single_pooled_endpoint"]:
        raise AdjacentTrainingSurfaceError("adjacent tasks cannot be pooled")
    if any(matrix["matrix_values_or_axis_membership_scanned"] for matrix in contract["lincs"]["matrices"]):
        raise AdjacentTrainingSurfaceError("LINCS matrix scanning claim is inconsistent")
    if any(
        contract["shared_contract"][key]
        for key in ("production_features_generated", "substantive_training_started", "hpc_executed")
    ):
        raise AdjacentTrainingSurfaceError("execution boundary violated")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_adjacent_training_surfaces(args.output_root)
    else:
        if args.source_root is None:
            parser.error("--source-root is required when building")
        build_adjacent_training_surfaces(source_root=args.source_root, output_root=args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
