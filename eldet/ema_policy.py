"""EMA update for ELDET teacher (global decay + optional faster decay on classification params)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import torch
from torch import nn


def update_teacher_from_student(
    teacher: nn.Module,
    student: nn.Module,
    *,
    global_decay: float,
    cls_decay: Optional[float] = None,
    is_cls_param: Optional[Callable[[str], bool]] = None,
    ema_buffers: bool = True,
) -> None:
    """In-place EMA: ``θ_t ← d * θ_t + (1-d) * θ_s`` with ``d`` in ``(0, 1)`` (high ``d`` = slow teacher).

    Args:
        teacher: Frozen (no grad) teacher module; tensors updated in place.
        student: Training student.
        global_decay: Decay ``d`` for non-classification parameters (paper ~0.999).
        cls_decay: If set and ``is_cls_param`` is set, use this ``d`` for matching parameter **names**.
        is_cls_param: Predicate on full parameter name (e.g. ``lambda n: "class_embed" in n``).
        ema_buffers: If True, apply ``global_decay`` to floating **buffers**; if False, skip buffers.
    """
    if not (0.0 < global_decay < 1.0):
        raise ValueError("global_decay must be in (0, 1)")
    if cls_decay is not None:
        if not (0.0 < cls_decay < 1.0):
            raise ValueError("cls_decay must be in (0, 1)")
        if is_cls_param is None:
            raise ValueError("is_cls_param is required when cls_decay is set")

    student_params = dict(student.named_parameters())
    with torch.no_grad():
        for name, t_p in teacher.named_parameters():
            if name not in student_params:
                continue
            s_p = student_params[name]
            if not t_p.dtype.is_floating_point:
                continue
            d = global_decay
            if cls_decay is not None and is_cls_param is not None and is_cls_param(name):
                d = cls_decay
            t_p.data.mul_(d).add_(s_p.data, alpha=(1.0 - d))

        if ema_buffers:
            student_bufs = dict(student.named_buffers())
            for name, t_b in teacher.named_buffers():
                if name not in student_bufs:
                    continue
                s_b = student_bufs[name]
                if not t_b.dtype.is_floating_point:
                    continue
                t_b.data.mul_(global_decay).add_(s_b.data, alpha=(1.0 - global_decay))


def yolo_detect_cls_param(name: str) -> bool:
    """Heuristic: Ultralytics ``Detect`` head classification convs often contain ``cv3``."""
    return "cv3" in name


def rfdetr_cls_param(name: str) -> bool:
    """Heuristic: RF-DETR / DETR-style classification weights."""
    lowered = name.lower()
    return any(k in lowered for k in ("class_embed", "cls_score", "class_head", "logits"))
