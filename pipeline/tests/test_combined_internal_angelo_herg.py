from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_combined_internal_angelo_herg import _prepare_data  # noqa: E402


def test_combined_training_uses_unique_nonconflicting_structures() -> None:
    (
        _compounds,
        internal,
        extension,
        combined,
        overlap,
        _audit,
        ambiguous,
        controls,
    ) = _prepare_data()

    assert len(controls) == 9
    assert len(internal) == internal["compound_id"].nunique() == 70
    assert len(extension) == extension["compound_id"].nunique() == 46
    assert len(combined) == combined["compound_id"].nunique() == 116
    assert len(overlap) == 22
    assert len(ambiguous) == 5
    assert not set(internal["standardized_smiles"]) & set(extension["standardized_smiles"])
    exact = np.isfinite(combined["pic50_lower"]) & np.isclose(
        combined["pic50_lower"], combined["pic50_upper"]
    )
    assert int(exact.sum()) == 100
    assert int((~exact).sum()) == 16
