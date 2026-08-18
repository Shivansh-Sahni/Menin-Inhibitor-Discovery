from __future__ import annotations

from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "run_local_herg_fundamental_optimization_v6.sh"


def test_launcher_is_fixed_six_core_resumable_entrypoint() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "OMP_NUM_THREADS=6" in text
    assert "--workers 6" in text
    assert "herg_fundamental_optimization_v6" in text
    assert "caffeinate -dimsu" in text
    assert "accepts no arguments" in text
