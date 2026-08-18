#!/usr/bin/env python3
"""Create publication-oriented figures for the bounded local M3 physics pilot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "research/simulations/pk_herg/local_m3_pilot"
FIGURE_DIR = ROOT / "research/reports/pk_herg/local_m3/figures"
SITE_EXTENSION = PILOT / "site_specific_extension"
COMPOUND_ORDER = (
    "CMP-D6B900FDA91C13513900",
    "CMP-642D79F70A93767590D0",
    "CMP-47DADB26C12A7C3D5CB5",
    "CMP-593B478B10007352C89B",
)
LABELS = {
    "CMP-D6B900FDA91C13513900": "MP1 blocker",
    "CMP-642D79F70A93767590D0": "MP1 nonblocker",
    "CMP-47DADB26C12A7C3D5CB5": "MP2 blocker",
    "CMP-593B478B10007352C89B": "MP2 nonblocker",
}
COLORS = {
    "CMP-D6B900FDA91C13513900": "#D55E00",
    "CMP-642D79F70A93767590D0": "#0072B2",
    "CMP-47DADB26C12A7C3D5CB5": "#E69F00",
    "CMP-593B478B10007352C89B": "#56B4E9",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _paired_panel(
    axis: plt.Axes,
    paired: pd.DataFrame,
    *,
    observable: str,
    ylabel: str,
    panel_label: str,
) -> None:
    subset = paired[paired["observable"].eq(observable)]
    jitter = np.asarray([-0.18, -0.06, 0.06, 0.18])
    for x, compound_id in enumerate(COMPOUND_ORDER):
        values = (
            subset[subset["compound_id"].eq(compound_id)]
            .sort_values(["seed", "basin_index_within_seed"])["water_minus_chloroform"]
            .to_numpy(dtype=float)
        )
        axis.scatter(
            x + jitter[: len(values)],
            values,
            s=18,
            color=COLORS[compound_id],
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        median = float(np.median(values))
        axis.plot([x - 0.24, x + 0.24], [median, median], color="black", linewidth=1.2)
    axis.axhline(0.0, color="#777777", linewidth=0.7, linestyle="--", zorder=1)
    axis.set_xticks(range(len(COMPOUND_ORDER)))
    axis.set_xticklabels(
        [LABELS[compound_id] for compound_id in COMPOUND_ORDER],
        rotation=35,
        ha="right",
    )
    axis.set_ylabel(ylabel)
    axis.text(-0.18, 1.05, panel_label, transform=axis.transAxes, fontweight="bold")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def make_local_physics_summary() -> None:
    thresholds = pd.read_csv(PILOT / "rare_state_flux_thresholds.csv")
    paired = pd.read_csv(PILOT / "xtb_basin_analysis/paired_environment_responses.csv")
    neutral = thresholds[thresholds["ph"].eq(7.4) & thresholds["target_neutral_flux_fraction"].eq(0.5)]

    figure, axes = plt.subplots(1, 4, figsize=(7.2, 2.25), constrained_layout=True)
    for compound_id in COMPOUND_ORDER:
        values = neutral[neutral["compound_id"].eq(compound_id)].sort_values("common_basic_pka_offset")
        pair_number = 1 if compound_id in COMPOUND_ORDER[:2] else 2
        axes[0].plot(
            values["common_basic_pka_offset"],
            values["neutral_macrostate_fraction"],
            color=COLORS[compound_id],
            marker="o",
            markersize=3.5,
            linewidth=1.1,
            linestyle="-" if pair_number == 1 else "--",
            label=LABELS[compound_id],
        )
    axes[0].set_yscale("log")
    axes[0].set_xticks([-1, 0, 1])
    axes[0].set_xlabel("Common pKa offset")
    axes[0].set_ylabel("Neutral macrostate fraction")
    axes[0].text(-0.18, 1.05, "A", transform=axes[0].transAxes, fontweight="bold")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].legend(frameon=False, loc="lower left", handlelength=1.8)

    _paired_panel(
        axes[1],
        paired,
        observable="radius_of_gyration_angstrom",
        ylabel=r"$\Delta R_g$ water − CHCl$_3$ (Å)",
        panel_label="B",
    )
    _paired_panel(
        axes[2],
        paired,
        observable="polar_heavy_atom_sasa_angstrom2",
        ylabel=r"$\Delta$ polar SASA water − CHCl$_3$ (Å$^2$)",
        panel_label="C",
    )
    _paired_panel(
        axes[3],
        paired,
        observable="dipole_magnitude_debye",
        ylabel=r"$\Delta$ dipole water − CHCl$_3$ (D)",
        panel_label="D",
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(
            FIGURE_DIR / f"local_physics_pilot_summary.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def make_receptor_state_summary() -> None:
    selection = pd.read_csv(PILOT / "receptor_state_analysis/receptor_state_selection.csv")
    selection = selection.set_index("pdb_id").loc[["8ZYN", "8ZYP", "8ZYO", "8ZYQ", "9CHP", "9CHQ"]]
    x = np.arange(len(selection))
    width = 0.36
    figure, axis = plt.subplots(figsize=(4.4, 2.35), constrained_layout=True)
    axis.bar(
        x - width / 2,
        selection["filter_rmsd_vs_8ZYN_angstrom"],
        width,
        label="Selectivity filter",
        color="#0072B2",
    )
    axis.bar(
        x + width / 2,
        selection["cavity_rmsd_vs_8ZYN_angstrom"],
        width,
        label="Cavity/S6",
        color="#E69F00",
    )
    for position, tier in zip(x, selection["production_tier"], strict=True):
        if tier == "sensitivity":
            axis.text(position, -0.10, "S", ha="center", va="top", fontweight="bold")
    axis.set_xticks(x)
    axis.set_xticklabels(selection.index)
    axis.set_ylabel("Post-scaffold alignment RMSD (Å)")
    axis.legend(frameon=False, ncol=2)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.text(
        0.99,
        0.98,
        "S = sensitivity-only coordinate",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(
            FIGURE_DIR / f"herg_receptor_coordinate_contrasts.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def make_site_specific_microstate_summary() -> None:
    ranking = pd.read_csv(SITE_EXTENSION / "site_ranking_hamiltonian_sensitivity.csv")
    context = pd.read_csv(SITE_EXTENSION / "near_degenerate_site_uncertainty_vs_pair_signal.csv")
    compounds = (
        "CMP-47DADB26C12A7C3D5CB5",
        "CMP-593B478B10007352C89B",
    )
    titles = ("MP2 blocker", "MP2 nonblocker")
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.4, 2.35),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.0, 1.0, 1.25)},
    )
    method_styles = {
        "GFN2": ("#D55E00", "relative_energy_within_compound_seed_kcal_mol"),
        "GFN1": ("#0072B2", "gfn1_relative_energy_within_compound_seed_kcal_mol"),
    }
    for panel, (axis, compound_id, title) in enumerate(zip(axes[:2], compounds, titles, strict=True)):
        subset = ranking[ranking["compound_id"].eq(compound_id)]
        sites = sorted(subset["protonated_atom_index_zero_based"].astype(int).unique())
        site_positions = np.arange(len(sites))
        for method, (color, value_column) in method_styles.items():
            for seed_index, (_seed, group) in enumerate(subset.groupby("seed", sort=True)):
                values = (
                    group.set_index("protonated_atom_index_zero_based")
                    .loc[sites, value_column]
                    .to_numpy(dtype=float)
                )
                axis.plot(
                    site_positions,
                    values,
                    color=color,
                    marker="o",
                    markersize=3,
                    linewidth=1.0,
                    linestyle="-" if seed_index == 0 else "--",
                    label=f"{method}, seed {seed_index + 1}",
                )
        axis.axhline(
            2.8307469378972656,
            color="#777777",
            linewidth=0.7,
            linestyle=":",
        )
        axis.set_xticks(site_positions)
        axis.set_xticklabels(sites)
        axis.set_xlabel("Protonated N atom index")
        axis.set_ylabel("Relative local energy (kcal/mol)")
        axis.set_title(title)
        axis.set_ylim(bottom=-0.5)
        axis.text(
            -0.18,
            1.05,
            chr(ord("A") + panel),
            transform=axis.transAxes,
            fontweight="bold",
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=5.6, handlelength=1.7, loc="center left")

    label_map = {
        "cation_fragment_sasa_angstrom2": "Cation SASA",
        "cation_fragment_xtb_charge": "Cation charge",
        "edited_ring_xtb_charge": "Edited-ring charge",
        "edited_ring_nitrogen_xtb_charge": "Edited-ring N charge",
        "cation_to_edited_ring_centroid_distance_angstrom": "Cation–ring distance",
        "positive_negative_charge_centroid_separation_angstrom": "Charge separation",
        "radius_of_gyration_angstrom": r"$R_g$",
        "polar_heavy_atom_sasa_angstrom2": "Polar SASA",
    }
    context = context.copy()
    context["label"] = context["observable"].map(label_map)
    context = context.sort_values("site_range_to_pair_difference_ratio")
    y = np.arange(len(context))
    ratios = context["site_range_to_pair_difference_ratio"].to_numpy(dtype=float)
    axes[2].barh(
        y,
        ratios,
        color=np.where(ratios > 1.0, "#D55E00", "#0072B2"),
    )
    axes[2].axvline(1.0, color="#333333", linewidth=0.8, linestyle="--")
    axes[2].set_xscale("log")
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(context["label"])
    axes[2].set_xlabel("Near-site range / N28 pair difference")
    axes[2].text(-0.18, 1.05, "C", transform=axes[2].transAxes, fontweight="bold")
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(
            FIGURE_DIR / f"site_specific_microstate_extension.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    _style()
    make_local_physics_summary()
    make_receptor_state_summary()
    make_site_specific_microstate_summary()


if __name__ == "__main__":
    main()
