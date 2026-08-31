"""Test-only path fix: ensure ``import ultralytics`` hits the inner package (see ``pyproject.toml`` ``pythonpath``)."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
_vend = _root / "ultralytics"
if _vend.is_dir():
    # Parent of ``ultralytics/ultralytics/``; must precede repo root on ``sys.path``.
    sys.path.insert(0, str(_vend))
