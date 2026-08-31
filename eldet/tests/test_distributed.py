"""Distributed helpers (single-process no-op paths)."""

from __future__ import annotations

from eldet.distributed import broadcast_object_list, broadcast_phase_decisions, is_distributed


def test_single_process_broadcast_phase() -> None:
    assert is_distributed() is False
    assert broadcast_phase_decisions(2, 5) == (2, 5)
    lst = [1, "a"]
    broadcast_object_list(lst)
    assert lst == [1, "a"]
