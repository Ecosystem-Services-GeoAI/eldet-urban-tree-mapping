"""KD loss between two RF-DETR-style output dicts (``pred_logits``, ``pred_boxes``).

**Phase 2.9 behaviour**

- **Matcher-aligned (default):** ``compute_rf_detr_kd_at_indices`` / ``compute_rf_detr_kd_auto`` with
  ``use_matcher_indices=True`` distill only queries selected by the **same** Hungarian ``indices`` as the detection
  loss. Requires the vendored ``SetCriterion`` patch (``_eldet_last_indices``).
- **Dense (legacy):** ``compute_rf_detr_kd`` or ``compute_rf_detr_kd_auto`` when indices are missing flattens all
  queries for cls + box terms.
"""

from __future__ import annotations

from typing import Any

import torch

from eldet.kd_losses import kl_div_logits, l1_box_loss


def compute_rf_detr_kd_at_indices(
    criterion: Any,
    outputs_student: dict[str, Any],
    outputs_teacher: dict[str, Any],
    *,
    lambda_cls: float = 1.0,
    lambda_box: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KD on **Hungarian-matched** queries only (same ``indices`` as the last detection loss).

    Expects ``criterion._eldet_last_indices`` to have been set during the preceding
    ``SetCriterion.forward(outputs_student, targets)`` call (see vendored ``criterion.py`` patch).
    """
    indices = getattr(criterion, "_eldet_last_indices", None)
    if indices is None:
        return compute_rf_detr_kd(
            outputs_student,
            outputs_teacher,
            lambda_cls=lambda_cls,
            lambda_box=lambda_box,
            temperature=temperature,
        )
    batch_idx, src_idx = criterion._get_src_permutation_idx(indices)
    if batch_idx.numel() == 0:
        z = outputs_student["pred_logits"]
        return z.sum() * 0.0
    ls = outputs_student["pred_logits"][batch_idx, src_idx]
    lt = outputs_teacher["pred_logits"][batch_idx, src_idx]
    bs = outputs_student["pred_boxes"][batch_idx, src_idx]
    bt = outputs_teacher["pred_boxes"][batch_idx, src_idx]
    return lambda_cls * kl_div_logits(ls, lt, temperature=temperature) + lambda_box * l1_box_loss(bs, bt)


def compute_rf_detr_kd(
    outputs_student: dict[str, Any],
    outputs_teacher: dict[str, Any],
    *,
    lambda_cls: float = 1.0,
    lambda_box: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Flatten ``(B, Q, *)`` predictions and combine cls + box terms (dense KD over all queries)."""
    ls = outputs_student["pred_logits"].reshape(-1, outputs_student["pred_logits"].shape[-1])
    lt = outputs_teacher["pred_logits"].reshape(-1, outputs_teacher["pred_logits"].shape[-1])
    bs = outputs_student["pred_boxes"].reshape(-1, 4)
    bt = outputs_teacher["pred_boxes"].reshape(-1, 4)
    return lambda_cls * kl_div_logits(ls, lt, temperature=temperature) + lambda_box * l1_box_loss(bs, bt)


def compute_rf_detr_kd_auto(
    criterion: Any,
    outputs_student: dict[str, Any],
    outputs_teacher: dict[str, Any],
    *,
    lambda_cls: float = 1.0,
    lambda_box: float = 1.0,
    temperature: float = 1.0,
    use_matcher_indices: bool = True,
) -> torch.Tensor:
    """Matcher-aligned KD when ``use_matcher_indices`` and indices are available; else dense KD."""
    if use_matcher_indices and getattr(criterion, "_eldet_last_indices", None) is not None:
        return compute_rf_detr_kd_at_indices(
            criterion,
            outputs_student,
            outputs_teacher,
            lambda_cls=lambda_cls,
            lambda_box=lambda_box,
            temperature=temperature,
        )
    return compute_rf_detr_kd(
        outputs_student,
        outputs_teacher,
        lambda_cls=lambda_cls,
        lambda_box=lambda_box,
        temperature=temperature,
    )
