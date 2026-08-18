#!/usr/bin/env python3
"""Validate the hERG landscape v2 machine-readable artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(name: str) -> dict:
    with (ROOT / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    model_csv = load_csv("model_comparison_matrix.csv")
    model_json = load_json("model_comparison_matrix.json")
    priority_csv = load_csv("benchmark_adapter_priority_matrix.csv")
    priority_json = load_json("benchmark_adapter_priority_matrix.json")

    csv_ids = [row["model_id"] for row in model_csv]
    json_ids = [row["model_id"] for row in model_json["models"]]
    assert len(csv_ids) == len(set(csv_ids)), "duplicate model_id in CSV"
    assert len(json_ids) == len(set(json_ids)), "duplicate model_id in JSON"
    assert set(csv_ids) == set(json_ids), "CSV/JSON model IDs differ"
    assert {row["work_item"] for row in priority_csv} == {
        row["work_item"] for row in priority_json["items"]
    }, "CSV/JSON priority work items differ"

    for row in model_csv:
        words = row["five_word_issue"].split()
        assert len(words) == 5, f"{row['model_id']} issue has {len(words)} words: {words}"
        assert row["primary_source"].startswith("http"), f"missing source: {row['model_id']}"
        assert row["comparison_priority"].split(maxsplit=1)[0] in {"P0", "P1", "P2"}
    for row in model_json["models"]:
        assert len(row["five_word_issue"].split()) == 5
        assert str(row["source"]).startswith("http")
        assert row["priority"] in {"P0", "P1", "P2"}
    for row in priority_csv:
        assert row["priority"] in {"P0", "P1", "P2"}
        assert row["success_gate"].strip()

    required = {
        "MaxQSARing", "Transformer_Morgan", "XGB_ISE", "HERGAI",
        "hERGBoost", "hERGAT", "hERG_MFFGNN", "TDMFLSGAT",
        "MultiCTox", "CardioSafe_v1_1", "UQ4DD_censored",
    }
    assert required <= set(csv_ids), f"required comparators missing: {required - set(csv_ids)}"
    report = (ROOT / "CURRENT_HERG_MODEL_LANDSCAPE_2025_2026.md").read_text(encoding="utf-8")
    assert "Predictive superiority remains a hypothesis" in report
    assert "343,909 confirmed-WT observations" in report
    assert "369,546 unique structures" in report
    assert "343,909 confirmed-WT structures" not in report

    print(f"PASS: {len(model_csv)} model/method rows")
    print(f"PASS: {len(priority_csv)} benchmark/adapter priorities")
    print("PASS: CSV/JSON IDs, five-word issues, sources, priorities, and claim boundaries")


if __name__ == "__main__":
    main()
