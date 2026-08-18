"""Create split-contained Q1 matched-molecular-pair evidence before HPC."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import rdMMPA
from scipy.stats import spearmanr

from .platform_herg_master_dataset import validate_herg_master_dataset

SCHEMA_VERSION = "platform-herg-mmp-analysis/1.0"
TASK_ID = "Q1_QUANTITATIVE_PIC50"
MAX_CORE_MEMBERS = 64
MAX_PAIRS_PER_CORE = 2_016


class HergMmpAnalysisError(RuntimeError):
    """Raised when MMP analysis violates its split or evidence contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_hash(path: Path) -> str:
    return hashlib.sha256(pq.read_schema(path).serialize().to_pybytes()).hexdigest()


def _manifest_hash(payload: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _fragment_members(smiles_by_id: dict[str, str]) -> tuple[dict[str, list[tuple[str, str, int]]], int]:
    cores: dict[str, set[tuple[str, str, int]]] = defaultdict(set)
    failures = 0
    for structure_id, smiles in sorted(smiles_by_id.items()):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            failures += 1
            continue
        total_heavy = max(1, molecule.GetNumHeavyAtoms())
        try:
            fragments = rdMMPA.FragmentMol(
                molecule, minCuts=1, maxCuts=1, maxCutBonds=20, resultsAsMols=False
            )
        except (RuntimeError, ValueError):
            failures += 1
            continue
        for _, fragment_smiles in fragments:
            parts = str(fragment_smiles).split(".")
            if len(parts) != 2:
                continue
            parsed = []
            for part in parts:
                fragment = Chem.MolFromSmiles(part)
                parsed.append((fragment.GetNumHeavyAtoms() if fragment is not None else 0, part))
            parsed.sort(key=lambda item: (-item[0], item[1]))
            core_heavy, core = parsed[0]
            variable_heavy, variable = parsed[1]
            if core_heavy >= 10 and variable_heavy <= 12 and variable_heavy / total_heavy <= 0.35:
                cores[core].add((structure_id, variable, variable_heavy))
    return {core: sorted(rows) for core, rows in cores.items()}, failures


def _pair_registry(
    membership: pd.DataFrame, structures: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    smiles = structures.set_index("structure_id")["standardized_smiles"].astype(str).to_dict()
    best: dict[tuple[str, str], dict[str, Any]] = {}
    skipped_oversized: list[dict[str, Any]] = []
    fragmentation_failures = 0
    for split, group in membership.groupby("model_split", sort=True):
        split_smiles = {sid: smiles[sid] for sid in sorted(set(group["structure_id"]))}
        cores, failures = _fragment_members(split_smiles)
        fragmentation_failures += failures
        for core, raw_members in sorted(cores.items()):
            # One deterministic representative per variable fragment avoids duplicate chemistry.
            by_variable: dict[str, tuple[str, str, int]] = {}
            for member in raw_members:
                by_variable.setdefault(member[1], member)
            members = sorted(by_variable.values())
            if len(members) > MAX_CORE_MEMBERS:
                skipped_oversized.append(
                    {
                        "model_split": split,
                        "core_smiles": core,
                        "unique_variable_fragments": len(members),
                        "reason": "core_exceeds_preregistered_pair_enumeration_cap",
                    }
                )
                continue
            for left_index, left in enumerate(members):
                for right in members[left_index + 1 :]:
                    structure_a, variable_a, variable_heavy_a = left
                    structure_b, variable_b, variable_heavy_b = right
                    if structure_a > structure_b:
                        structure_a, structure_b = structure_b, structure_a
                        variable_a, variable_b = variable_b, variable_a
                        variable_heavy_a, variable_heavy_b = variable_heavy_b, variable_heavy_a
                    key = (structure_a, structure_b)
                    core_molecule = Chem.MolFromSmiles(core)
                    core_heavy = core_molecule.GetNumHeavyAtoms() if core_molecule is not None else 0
                    candidate = {
                        "pair_id": "MMP-"
                        + hashlib.sha256("\x1f".join(key).encode()).hexdigest()[:24].upper(),
                        "model_split": split,
                        "structure_id_a": structure_a,
                        "structure_id_b": structure_b,
                        "core_smiles": core,
                        "variable_fragment_a": variable_a,
                        "variable_fragment_b": variable_b,
                        "transformation": f"{variable_a}>>{variable_b}",
                        "core_heavy_atoms": core_heavy,
                        "variable_heavy_atoms_a": variable_heavy_a,
                        "variable_heavy_atoms_b": variable_heavy_b,
                        "pair_definition_uses_labels": False,
                    }
                    if key not in best or (core_heavy, core) > (
                        best[key]["core_heavy_atoms"],
                        best[key]["core_smiles"],
                    ):
                        best[key] = candidate
    rows = sorted(best.values(), key=lambda row: (row["model_split"], row["pair_id"]))
    return rows, skipped_oversized, fragmentation_failures


def _training_effects(
    registry: pd.DataFrame, membership: pd.DataFrame, structures: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = membership[membership["model_split"].eq("train")].copy()
    targets = train.groupby("structure_id", sort=True)["target_pic50_point"].agg(["median", "size"])
    descriptors = structures.set_index("structure_id")
    descriptor_columns = [
        "molecular_weight",
        "mol_logp",
        "topological_polar_surface_area",
        "hydrogen_bond_donors",
        "hydrogen_bond_acceptors",
        "rotatable_bonds",
        "fraction_csp3",
    ]
    means = descriptors.loc[targets.index, descriptor_columns].mean()
    scales = descriptors.loc[targets.index, descriptor_columns].std(ddof=1).replace(0, np.nan)
    z = (descriptors[descriptor_columns] - means) / scales
    rows: list[dict[str, Any]] = []
    for pair in registry[registry["model_split"].eq("train")].itertuples(index=False):
        if pair.structure_id_a not in targets.index or pair.structure_id_b not in targets.index:
            continue
        a = pair.structure_id_a
        b = pair.structure_id_b
        row: dict[str, Any] = {
            "pair_id": pair.pair_id,
            "structure_id_a": a,
            "structure_id_b": b,
            "transformation": pair.transformation,
            "pic50_median_a": float(targets.loc[a, "median"]),
            "pic50_median_b": float(targets.loc[b, "median"]),
            "pic50_observations_a": int(targets.loc[a, "size"]),
            "pic50_observations_b": int(targets.loc[b, "size"]),
            "delta_pic50_b_minus_a": float(targets.loc[b, "median"] - targets.loc[a, "median"]),
            "absolute_delta_pic50": float(abs(targets.loc[b, "median"] - targets.loc[a, "median"])),
            "activity_cliff_ge_1_pic50": bool(abs(targets.loc[b, "median"] - targets.loc[a, "median"]) >= 1),
            "exploratory_training_only": True,
        }
        for descriptor in descriptor_columns:
            row[f"delta_{descriptor}_b_minus_a"] = float(
                descriptors.loc[b, descriptor] - descriptors.loc[a, descriptor]
            )
        interaction_a = z.loc[a, "mol_logp"] * z.loc[a, "topological_polar_surface_area"]
        interaction_b = z.loc[b, "mol_logp"] * z.loc[b, "topological_polar_surface_area"]
        row["delta_standardized_logp_x_tpsa_b_minus_a"] = float(interaction_b - interaction_a)
        rows.append(row)
    effects = pd.DataFrame(rows).sort_values(["absolute_delta_pic50", "pair_id"], ascending=[False, True])
    association_rows = []
    effect_columns = [f"delta_{name}_b_minus_a" for name in descriptor_columns] + [
        "delta_standardized_logp_x_tpsa_b_minus_a"
    ]
    for column in effect_columns:
        finite = effects[[column, "delta_pic50_b_minus_a"]].replace([np.inf, -np.inf], np.nan).dropna()
        rho, p_value = spearmanr(finite[column], finite["delta_pic50_b_minus_a"])
        association_rows.append(
            {
                "contrast": column.removeprefix("delta_").removesuffix("_b_minus_a"),
                "training_pairs": len(finite),
                "spearman_rho": None if math.isnan(float(rho)) else float(rho),
                "nominal_p_value_not_multiplicity_adjusted": (
                    None if math.isnan(float(p_value)) else float(p_value)
                ),
                "exploratory_training_only": True,
                "causal_or_mechanistic_claim_allowed": False,
            }
        )
    return effects, pd.DataFrame(association_rows)


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "rows": pq.read_metadata(path).num_rows,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "arrow_schema_sha256": _schema_hash(path),
    }


def _input_record(path: Path, role: str) -> dict[str, Any]:
    path = path.resolve()
    record: dict[str, Any] = {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix == ".parquet":
        record["rows"] = pq.read_metadata(path).num_rows
        record["arrow_schema_sha256"] = _schema_hash(path)
    return record


def _assert_mmp_split_exclusivity(registry: pd.DataFrame, structures: pd.DataFrame) -> None:
    split_by_structure: dict[str, set[str]] = defaultdict(set)
    for row in registry.itertuples(index=False):
        split_by_structure[row.structure_id_a].add(row.model_split)
        split_by_structure[row.structure_id_b].add(row.model_split)
    if any(len(splits) != 1 for splits in split_by_structure.values()):
        raise HergMmpAnalysisError("structure crosses MMP splits")
    scaffold_by_structure = structures.set_index("structure_id")["scaffold_group_id"].to_dict()
    split_by_scaffold: dict[str, set[str]] = defaultdict(set)
    for structure_id, splits in split_by_structure.items():
        if structure_id not in scaffold_by_structure:
            raise HergMmpAnalysisError("MMP structure is absent from bound structure master")
        split_by_scaffold[str(scaffold_by_structure[structure_id])].update(splits)
    if any(len(splits) != 1 for splits in split_by_scaffold.values()):
        raise HergMmpAnalysisError("scaffold crosses MMP splits")


def build_mmp_analysis(*, master_root: Path, output_root: Path) -> dict[str, Any]:
    master_manifest = validate_herg_master_dataset(master_root)
    task_path = master_root / "task_membership.parquet"
    structure_path = master_root / "structure_master.parquet"
    routing_columns = ["task_id", "structure_id", "model_split", "eligible", "use_as_training_label"]
    routing = pq.read_table(task_path, columns=routing_columns).to_pandas()
    routing = routing[routing["task_id"].eq(TASK_ID) & routing["eligible"] & routing["use_as_training_label"]]
    # Outcome projection is predicate-pushed to the training partition only.
    training_labels = pq.read_table(
        task_path,
        columns=["task_id", "structure_id", "model_split", "target_pic50_point"],
        filters=[("task_id", "=", TASK_ID), ("model_split", "=", "train")],
    ).to_pandas()
    if training_labels.empty or training_labels["target_pic50_point"].isna().any():
        raise HergMmpAnalysisError("training-only Q1 target projection is empty or non-exact")
    membership = routing
    structures = pq.read_table(
        structure_path,
        columns=[
            "structure_id",
            "standardized_smiles",
            "scaffold_group_id",
            "molecular_weight",
            "mol_logp",
            "topological_polar_surface_area",
            "hydrogen_bond_donors",
            "hydrogen_bond_acceptors",
            "rotatable_bonds",
            "fraction_csp3",
        ],
    ).to_pandas()
    structures = structures[structures["structure_id"].isin(set(membership["structure_id"]))]
    registry_rows, skipped_rows, failures = _pair_registry(membership, structures)
    registry = pd.DataFrame(registry_rows)
    effects, associations = _training_effects(registry, training_labels, structures)
    skipped = pd.DataFrame(
        skipped_rows,
        columns=["model_split", "core_smiles", "unique_variable_fragments", "reason"],
    )
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "mmp_pair_registry.parquet": registry,
        "training_mmp_effects.parquet": effects,
        "training_mmp_descriptor_associations.parquet": associations,
        "skipped_oversized_mmp_cores.parquet": skipped,
    }
    artifacts = {}
    for name, frame in paths.items():
        path = output_root / name
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd")
        artifacts[name] = _artifact(path)
    counts_by_split = registry.groupby("model_split").size().to_dict()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "inputs": [
            _input_record(master_root / "herg_master_manifest.json", "master_manifest"),
            _input_record(task_path, "task_membership"),
            _input_record(structure_path, "structure_master"),
            _input_record(Path(__file__), "builder_implementation"),
        ],
        "master_manifest_sha256": master_manifest["manifest_sha256"],
        "artifacts": artifacts,
        "counts": {
            "q1_structures": int(membership["structure_id"].nunique()),
            "mmp_pairs": len(registry),
            "train_pairs_with_effects": len(effects),
            "validation_pair_definitions": int(counts_by_split.get("validation", 0)),
            "test_pair_definitions": int(counts_by_split.get("test", 0)),
            "activity_cliffs_ge_1_pic50_train": int(effects["activity_cliff_ge_1_pic50"].sum()),
            "skipped_oversized_cores": len(skipped),
            "fragmentation_failures": failures,
        },
        "pair_definition": {
            "cuts": 1,
            "minimum_core_heavy_atoms": 10,
            "maximum_variable_heavy_atoms": 12,
            "maximum_variable_fraction": 0.35,
            "maximum_unique_variable_fragments_per_core": MAX_CORE_MEMBERS,
            "maximum_pairs_per_core": MAX_PAIRS_PER_CORE,
            "deduplication": "one_pair_then_largest_core_lexical_tiebreak",
        },
        "scientific_contract": {
            "models_trained": False,
            "production_model_input_feature_store_generated": False,
            "analysis_descriptors_only": True,
            "pair_registry_uses_labels": False,
            "label_columns_opened_by_builder": True,
            "label_projection_filter": "task_Q1_and_model_split_train",
            "nontraining_label_values_returned_to_analysis_frame": False,
            "physical_page_level_nonaccess_proven": False,
            "nontraining_label_values_retained_in_outputs": False,
            "effect_estimation_partition": "train_only",
            "validation_test_pairs_are_definition_only": True,
            "mechanistic_or_causal_claim": False,
        },
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    (output_root / "mmp_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def validate_mmp_analysis(output_root: Path) -> dict[str, Any]:
    manifest = json.loads((output_root / "mmp_analysis_manifest.json").read_text())
    if manifest["manifest_sha256"] != _manifest_hash(manifest):
        raise HergMmpAnalysisError("manifest self-hash mismatch")
    for binding in manifest["inputs"]:
        path = Path(binding["path"])
        if (
            not path.is_file()
            or path.stat().st_size != binding["bytes"]
            or _sha256(path) != binding["sha256"]
        ):
            raise HergMmpAnalysisError(f"input binding failed: {path}")
        if path.suffix == ".parquet" and (
            pq.read_metadata(path).num_rows != binding["rows"]
            or _schema_hash(path) != binding["arrow_schema_sha256"]
        ):
            raise HergMmpAnalysisError(f"input Parquet binding failed: {path}")
    for name, binding in manifest["artifacts"].items():
        path = output_root / name
        if (
            not path.is_file()
            or path.stat().st_size != binding["bytes"]
            or _sha256(path) != binding["sha256"]
            or pq.read_metadata(path).num_rows != binding["rows"]
            or _schema_hash(path) != binding["arrow_schema_sha256"]
        ):
            raise HergMmpAnalysisError(f"artifact binding failed: {name}")
    registry = pq.read_table(output_root / "mmp_pair_registry.parquet").to_pandas()
    effects = pq.read_table(output_root / "training_mmp_effects.parquet").to_pandas()
    if (
        registry["pair_id"].duplicated().any()
        or (registry["structure_id_a"] == registry["structure_id_b"]).any()
    ):
        raise HergMmpAnalysisError("invalid or duplicate pair registry")
    counts = manifest["counts"]
    skipped = pq.read_table(output_root / "skipped_oversized_mmp_cores.parquet").to_pandas()
    if (
        len(registry) != counts["mmp_pairs"]
        or len(effects) != counts["train_pairs_with_effects"]
        or int(effects["activity_cliff_ge_1_pic50"].sum()) != counts["activity_cliffs_ge_1_pic50_train"]
        or len(skipped) != counts["skipped_oversized_cores"]
    ):
        raise HergMmpAnalysisError("manifest counts do not replay from artifacts")
    structure_binding = next(row for row in manifest["inputs"] if row["role"] == "structure_master")
    structures = pq.read_table(
        Path(structure_binding["path"]), columns=["structure_id", "scaffold_group_id"]
    ).to_pandas()
    _assert_mmp_split_exclusivity(registry, structures)
    if set(effects["pair_id"]) - set(registry.loc[registry["model_split"].eq("train"), "pair_id"]):
        raise HergMmpAnalysisError("nontraining effect leaked into output")
    forbidden = {"pic50", "target", "class"}
    if any(any(token in name.lower() for token in forbidden) for name in registry.columns) or any(
        "label" in name.lower() and name != "pair_definition_uses_labels" for name in registry.columns
    ):
        raise HergMmpAnalysisError("pair registry contains outcome-like columns")
    if registry["pair_definition_uses_labels"].any():
        raise HergMmpAnalysisError("pair registry label-independence flag failed")
    expected_scientific = {
        "analysis_descriptors_only": True,
        "effect_estimation_partition": "train_only",
        "label_columns_opened_by_builder": True,
        "label_projection_filter": "task_Q1_and_model_split_train",
        "mechanistic_or_causal_claim": False,
        "models_trained": False,
        "nontraining_label_values_retained_in_outputs": False,
        "nontraining_label_values_returned_to_analysis_frame": False,
        "pair_registry_uses_labels": False,
        "physical_page_level_nonaccess_proven": False,
        "production_model_input_feature_store_generated": False,
        "validation_test_pairs_are_definition_only": True,
    }
    if manifest["scientific_contract"] != expected_scientific:
        raise HergMmpAnalysisError("scientific contract weakened")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-root", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_mmp_analysis(args.output_root)
    else:
        if args.master_root is None:
            parser.error("--master-root is required when building")
        build_mmp_analysis(master_root=args.master_root, output_root=args.output_root)
        validate_mmp_analysis(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
