from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from menin_discovery.platform_non_hpc_completion import (
    NonHPCCompletionError,
    _safe_project_file,
    _validate_literature,
    _write_immutable,
    document_with_sha256,
    verify_document_sha256,
)


def test_document_identity_detects_tamper() -> None:
    document = document_with_sha256({"status": "bounded", "substantive_training_started": False})
    assert verify_document_sha256(document)
    document["status"] = "changed"
    assert not verify_document_sha256(document)


def test_safe_project_file_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("evidence", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(NonHPCCompletionError, match="symlinked"):
        _safe_project_file(tmp_path, Path("link"))


def test_literature_counts_and_probable_reference_are_exact(tmp_path: Path) -> None:
    root = tmp_path
    reports = root / "research/reports/platform/literature_decisions"
    reports.mkdir(parents=True)
    for name in ("comprehensive_literature_review.md", "decision_recommendations.md"):
        (reports / name).write_text("x" * 1001, encoding="utf-8")
    sources = [{"source_id": f"S{index:02d}"} for index in range(1, 50)]
    (reports / "source_bibliography.json").write_text(json.dumps({"sources": sources}), encoding="utf-8")
    fields = ["candidate_id", "candidate_name", "decision"]
    with (reports / "model_candidate_decision_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(1, 18):
            writer.writerow(
                {
                    "candidate_id": f"M{index:02d}",
                    "candidate_name": "RoseTTAFold All-Atom" if index == 1 else f"Model {index}",
                    "decision": (
                        "probable_meeting_reference_and_later_comparator" if index == 1 else "candidate"
                    ),
                }
            )
    result = _validate_literature(root)
    assert result["source_count"] == 49
    assert result["model_candidate_count"] == 17
    assert result["checkpoint_downloaded"] is False


def test_immutable_writer_is_idempotent_but_refuses_change(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    document = document_with_sha256({"substantive_training_started": False})
    _write_immutable(path, document)
    _write_immutable(path, document)
    with pytest.raises(NonHPCCompletionError, match="refusing to overwrite"):
        _write_immutable(path, document_with_sha256({"substantive_training_started": True}))
