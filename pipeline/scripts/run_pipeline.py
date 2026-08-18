#!/usr/bin/env python3
"""Compatibility wrapper for the installed ``menin-pipeline`` command."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "pipeline" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from menin_discovery.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
