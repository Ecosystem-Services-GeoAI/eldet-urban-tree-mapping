"""Tests for examples/resume_utils.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from resume_utils import (
    resolve_rfdetr_resume,
    resolve_yolo_resume,
    try_resolve_rfdetr_resume,
    try_resolve_yolo_resume,
)


def test_resolve_rfdetr_resume_last(tmp_path: Path) -> None:
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"x")
    assert resolve_rfdetr_resume("", tmp_path) == str(ckpt.resolve())


def test_resolve_rfdetr_resume_explicit_file(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint_9.ckpt"
    ckpt.write_bytes(b"x")
    assert resolve_rfdetr_resume(str(ckpt), tmp_path) == str(ckpt.resolve())


def test_resolve_yolo_resume_run_dir(tmp_path: Path) -> None:
    last = tmp_path / "train-3" / "weights" / "last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"x")
    assert resolve_yolo_resume(str(tmp_path / "train-3"), tmp_path) == str(last.resolve())


def test_resolve_yolo_resume_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_rfdetr_resume("", tmp_path)


def test_try_resolve_rfdetr_resume_missing(tmp_path: Path) -> None:
    assert try_resolve_rfdetr_resume(tmp_path) is None


def test_try_resolve_rfdetr_resume_found(tmp_path: Path) -> None:
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"x")
    assert try_resolve_rfdetr_resume(tmp_path) == str(ckpt.resolve())


def test_try_resolve_yolo_resume_found(tmp_path: Path) -> None:
    last = tmp_path / "train-1" / "weights" / "last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"x")
    assert try_resolve_yolo_resume(tmp_path) == str(last.resolve())


def test_try_resolve_yolo_resume_missing(tmp_path: Path) -> None:
    assert try_resolve_yolo_resume(tmp_path) is None
