"""Minimal DDP helpers for sharing ELDET phase decisions across ranks (torch only, no framework imports)."""

from __future__ import annotations

from typing import Any


def is_distributed() -> bool:
    """True when ``torch.distributed`` is initialized with world size > 1."""
    import torch.distributed as dist

    return bool(dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1)


def broadcast_object_list(objects: list[Any], src: int = 0) -> None:
    """In-place broadcast of picklable objects from rank ``src``; no-op when not in distributed mode."""
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return
    dist.broadcast_object_list(objects, src=src)


def broadcast_phase_decisions(
    t_loc: int | None,
    t_cls: int | None,
    *,
    src: int = 0,
) -> tuple[int | None, int | None]:
    """Broadcast ``(t_loc, t_cls)`` from rank ``src``; returns the same pair on every rank.

    Non-src ranks should pass their local guesses (typically ignored); after the call, all ranks
    share the source values.
    """
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return t_loc, t_cls
    if int(dist.get_rank()) == int(src):
        payload: list[int | None] = [t_loc, t_cls]
    else:
        payload = [None, None]
    dist.broadcast_object_list(payload, src=src)
    return payload[0], payload[1]
