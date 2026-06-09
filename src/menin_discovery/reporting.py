"""Generate project reports and simple figures."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import HERG_TARGET, MENIN_TARGET


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown_table(df: pd.DataFrame, *, floatfmt: str = ".3g") -> str:
    """Small dependency-free Markdown table writer."""

    if df.empty:
        return "_No rows available._"
    frame = df.copy()
    for column in frame.columns:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].map(lambda value: format(value, floatfmt))
    headers = [str(c) for c in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        values = [str(row[column]).replace("\n", " ") for column in frame.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _plot_potency(compounds: pd.DataFrame, figures_dir: Path) -> str | None:
    if compounds.empty or "p_activity_median" not in compounds:
        return None
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "menin_potency_distribution.png"
    plt.figure(figsize=(7.5, 4.5))
    plt.hist(compounds["p_activity_median"].dropna(), bins=24, color="#2f6f6d", edgecolor="white")
    plt.xlabel("Median pActivity")
    plt.ylabel("Compound count")
    plt.title("Curated Menin Potency Distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return str(path)


def _plot_endpoint_counts(activity: pd.DataFrame, figures_dir: Path) -> str | None:
    if activity.empty or "endpoint" not in activity:
        return None
    counts = activity[activity["is_core_endpoint"]]["endpoint"].value_counts()
    if counts.empty:
        return None
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "menin_endpoint_counts.png"
    plt.figure(figsize=(7.5, 4.5))
    counts.plot(kind="bar", color="#6f7f2f")
    plt.xlabel("Endpoint")
    plt.ylabel("Measurement count")
    plt.title("Core Menin Bioactivity Endpoint Counts")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return str(path)


def _plot_herg_risk(risk: pd.DataFrame, figures_dir: Path) -> str | None:
    if risk.empty or "predicted_herg_risk" not in risk:
        return None
    counts = risk["predicted_herg_risk"].value_counts().reindex(["low", "medium", "high"]).dropna()
    if counts.empty:
        return None
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "predicted_herg_risk_counts.png"
    plt.figure(figsize=(7.5, 4.5))
    counts.plot(kind="bar", color=["#3e7c59", "#b88a2b", "#a04747"])
    plt.xlabel("Predicted hERG risk")
    plt.ylabel("Menin compound count")
    plt.title("Predicted hERG Liability Triage")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return str(path)


def write_summary_report(processed_dir: Path, reports_dir: Path, models_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"

    menin_activity = _read_csv(processed_dir / "menin_activity_measurements.csv")
    menin_compounds = _read_csv(processed_dir / "menin_compounds_curated.csv")
    herg_activity = _read_csv(processed_dir / "herg_activity_measurements.csv")
    herg_compounds = _read_csv(processed_dir / "herg_compounds_curated.csv")
    pk_admet = _read_csv(processed_dir / "pk_admet_observations.csv")
    source_summary = _read_csv(processed_dir / "source_summary.csv")
    herg_risk = _read_csv(reports_dir / "menin_with_predicted_herg_risk.csv")
    menin_metrics = _read_json(reports_dir / "menin_activity_model_metrics.json")
    herg_metrics = _read_json(reports_dir / "herg_classifier_metrics.json")

    _plot_potency(menin_compounds, figures_dir)
    _plot_endpoint_counts(menin_activity, figures_dir)
    _plot_herg_risk(herg_risk, figures_dir)

    top = menin_compounds.head(15).copy()
    top_cols = [
        "compound_ids",
        "value_nm_median",
        "p_activity_median",
        "n_measurements",
        "endpoints",
        "sources",
    ]
    top_md = _markdown_table(top[top_cols], floatfmt=".3g") if not top.empty and set(top_cols).issubset(top.columns) else "_No curated compound table available._"

    source_md = (
        _markdown_table(source_summary)
        if not source_summary.empty
        else "_No source summary available._"
    )

    potency_counts = (
        _markdown_table(
            menin_compounds["potency_class"]
            .value_counts()
            .rename_axis("potency_class")
            .reset_index(name="count")
        )
        if not menin_compounds.empty and "potency_class" in menin_compounds
        else "_No potency labels available._"
    )

    lines = [
        "# Menin Inhibitor Discovery Preliminary Work",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Executive summary",
        "",
        (
            "This repo builds a reproducible public-data foundation for menin inhibitor "
            "modeling. It anchors target activity on human Menin/MEN1 "
            f"({MENIN_TARGET['chembl_id']}, UniProt {MENIN_TARGET['uniprot']}), "
            "adds BindingDB Menin and Menin/KMT2A TSV exports, searches PubChem "
            "BioAssay for MEN1/menin assays, and builds baseline ML-ready tables for "
            "activity, PK/ADMET observations, and hERG liability."
        ),
        "",
        "## Current dataset snapshot",
        "",
        f"- Menin measurement rows: {len(menin_activity):,}",
        f"- Curated unique Menin SMILES strings: {len(menin_compounds):,}",
        f"- hERG/KCNH2 measurement rows: {len(herg_activity):,}",
        f"- hERG/KCNH2 curated unique SMILES strings: {len(herg_compounds):,}",
        f"- Menin-molecule PK/ADMET observation rows: {len(pk_admet):,}",
        "",
        "### Source coverage",
        "",
        source_md,
        "",
        "### Potency label counts",
        "",
        potency_counts,
        "",
        "## Highest-potency public compounds in the curated table",
        "",
        top_md,
        "",
        "## Baseline model status",
        "",
        "### Menin activity regression",
        "",
        "```json",
        json.dumps(menin_metrics, indent=2),
        "```",
        "",
        "### hERG liability classifier",
        "",
        "```json",
        json.dumps(herg_metrics, indent=2),
        "```",
        "",
        "## How to interpret the models",
        "",
        (
            "The current models are intentionally dependency-light baselines using hashed "
            "SMILES character n-grams plus simple string descriptors. They are useful for "
            "pipeline validation, triage, and demonstrating the full workflow. They should "
            "not be treated as final medicinal chemistry decision models until the next "
            "steps are complete: RDKit descriptors/fingerprints, scaffold/time splits, "
            "assay-family stratification, uncertainty estimates, and prospective validation."
        ),
        "",
        "## Immediate next steps for the Wang lab",
        "",
        "1. Add internal compound IDs, SD files, and assay metadata under the same schema.",
        "2. Confirm which endpoint should be the primary model target: biochemical FP/HTRF IC50, Kd/Ki, cell EC50, or a weighted multitask formulation.",
        "3. Replace string-features with RDKit Morgan fingerprints, physicochemical descriptors, and optional graph neural network embeddings.",
        "4. Build scaffold-split and time-split evaluations to estimate prospective performance.",
        "5. Expand hERG labels with lab-preferred thresholds and assay-specific annotations.",
        "6. Turn PK/ADMET observations into endpoint-specific models only when each endpoint has enough clean, comparable rows.",
        "",
        "## Questions to clarify with the lab",
        "",
        "- Are internal menin compounds available as SDF/SMILES with exact batch IDs?",
        "- Which assay should define the official activity label when multiple public measurements disagree?",
        "- Should censored values such as `>10000 nM` be retained for classification, regression with censoring, or excluded from the first model?",
        "- Does the lab want prospective design suggestions, virtual-screening rank lists, or only data/model infrastructure first?",
        "",
        "## Key public-source anchors",
        "",
        f"- ChEMBL Menin target: {MENIN_TARGET['chembl_id']} / UniProt {MENIN_TARGET['uniprot']}",
        f"- ChEMBL hERG/KCNH2 target: {HERG_TARGET['chembl_id']} / UniProt {HERG_TARGET['uniprot']}",
        "- BindingDB target-index TSVs for Menin and Menin/KMT2A complex",
        "- PubChem BioAssay search terms in `src/menin_discovery/config.py`",
    ]

    path = reports_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
