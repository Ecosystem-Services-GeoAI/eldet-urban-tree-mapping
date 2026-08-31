"""KD on YOLO v8 ``Detect`` predictions (``boxes``, ``scores``, ``feats``) vs a frozen teacher.

**Phase 3.9 (v1):** Uses ``ultralytics.utils.tal.make_anchors`` on the **student** ``feats`` grid, then decodes both
student and teacher box distributions with the **student** ``criterion.bbox_decode``, then ``kd_detection_v1`` on
flattened cls logits and decoded boxes. Same anchor geometry for both branches (no separate teacher ``Detect`` forward
with different strides).
"""

from __future__ import annotations

from typing import Any

import torch

from eldet.kd_losses import kd_detection_v1


def _yolo_detect_preds_for_kd(preds: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Return a v8-style ``{boxes, scores, feats}`` dict (unwrap ``one2many`` for end2end heads)."""
    if isinstance(preds, dict) and "one2many" in preds:
        return preds["one2many"]
    return preds


def _yolo_kd_criterion(criterion: Any) -> Any:
    """``v8DetectionLoss`` used for anchor decode (``E2ELoss.one2many`` for YOLO26)."""
    sub = getattr(criterion, "one2many", None)
    return sub if sub is not None else criterion


def compute_yolo_kd(
    preds_student: dict[str, torch.Tensor],
    preds_teacher: dict[str, torch.Tensor],
    criterion: Any,
    *,
    lambda_cls: float = 1.0,
    lambda_box: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Distillation term using the **student** criterion's decode path (anchors + DFL).

    Expects ``preds_*`` from :meth:`ultralytics.nn.tasks.DetectionModel.forward` on ``batch[\"img\"]`` only.
    For end2end (YOLO26) models, pass the full forward dict; ``one2many`` branches are used for KD.
    """
    from ultralytics.utils.tal import make_anchors

    crit = _yolo_kd_criterion(criterion)
    preds_student = _yolo_detect_preds_for_kd(preds_student)
    preds_teacher = _yolo_detect_preds_for_kd(preds_teacher)

    pred_distri_s = preds_student["boxes"].permute(0, 2, 1).contiguous()
    pred_scores_s = preds_student["scores"].permute(0, 2, 1).contiguous()
    anchor_points, stride_tensor = make_anchors(preds_student["feats"], crit.stride, 0.5)
    pred_bboxes_s = crit.bbox_decode(anchor_points, pred_distri_s)

    pred_distri_t = preds_teacher["boxes"].permute(0, 2, 1).contiguous()
    pred_scores_t = preds_teacher["scores"].permute(0, 2, 1).contiguous()
    pred_bboxes_t = crit.bbox_decode(anchor_points, pred_distri_t)

    box_s = (pred_bboxes_s * stride_tensor).reshape(-1, 4)
    box_t = (pred_bboxes_t * stride_tensor).reshape(-1, 4)
    nc = pred_scores_s.shape[-1]
    ls = pred_scores_s.reshape(-1, nc)
    lt = pred_scores_t.reshape(-1, nc)
    return kd_detection_v1(
        ls,
        lt,
        box_s,
        box_t,
        lambda_cls=lambda_cls,
        lambda_box=lambda_box,
        temperature=temperature,
        use_giou=False,
    )
