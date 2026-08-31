"""Knowledge-distillation helpers (torch): classification + box terms (ELDET Sec. 4.3, v1 non-CrossKD)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_div_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL( σ(z_s/T) || σ(z_t/T) ) with binary per-element treatment (mean reduction).

    Shapes must broadcast; typical use: same shape ``(..., C)`` for per-anchor class logits.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    p = torch.sigmoid(student_logits / temperature)
    q = torch.sigmoid(teacher_logits / temperature).detach()
    eps = 1e-7
    p = p.clamp(eps, 1.0 - eps)
    q = q.clamp(eps, 1.0 - eps)
    kl = p * (p.log() - q.log()) + (1.0 - p) * ((1.0 - p).log() - (1.0 - q).log())
    return kl.mean()


def l1_box_loss(student_boxes: torch.Tensor, teacher_boxes: torch.Tensor) -> torch.Tensor:
    """Mean L1 between boxes (same shape ``(*, 4)`` xyxy or any consistent 4-vector)."""
    return F.l1_loss(student_boxes, teacher_boxes.detach(), reduction="mean")


def pairwise_giou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Generalized IoU between two sets of xyxy boxes: ``boxes1`` ``(N,4)``, ``boxes2`` ``(M,4)`` → ``(N,M)``."""
    # Based on standard giou formulation for axis-aligned boxes
    b1 = boxes1.unsqueeze(1)
    b2 = boxes2.unsqueeze(0)
    x1 = torch.max(b1[..., 0], b2[..., 0])
    y1 = torch.max(b1[..., 1], b2[..., 1])
    x2 = torch.min(b1[..., 2], b2[..., 2])
    y2 = torch.min(b1[..., 3], b2[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a1 = (b1[..., 2] - b1[..., 0]).clamp(min=0) * (b1[..., 3] - b1[..., 1]).clamp(min=0)
    a2 = (b2[..., 2] - b2[..., 0]).clamp(min=0) * (b2[..., 3] - b2[..., 1]).clamp(min=0)
    union = a1 + a2 - inter + 1e-7
    iou = inter / union
    c_x1 = torch.min(b1[..., 0], b2[..., 0])
    c_y1 = torch.min(b1[..., 1], b2[..., 1])
    c_x2 = torch.max(b1[..., 2], b2[..., 2])
    c_y2 = torch.max(b1[..., 3], b2[..., 3])
    c_area = (c_x2 - c_x1).clamp(min=0) * (c_y2 - c_y1).clamp(min=0) + 1e-7
    return iou - (c_area - union) / c_area


def kd_detection_v1(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_boxes: torch.Tensor,
    teacher_boxes: torch.Tensor,
    *,
    lambda_cls: float = 1.0,
    lambda_box: float = 1.0,
    temperature: float = 1.0,
    use_giou: bool = False,
) -> torch.Tensor:
    """Scalar ``λ_cls * L_cls + λ_box * L_box`` for aligned tensors (same leading dims / flattenable)."""
    if use_giou:
        if student_boxes.ndim != 2 or student_boxes.shape[-1] != 4:
            raise ValueError("use_giou expects student_boxes (N,4)")
        giou = torch.diag(pairwise_giou_xyxy(student_boxes, teacher_boxes.detach()))
        box_term = (1.0 - giou).mean()
    else:
        box_term = l1_box_loss(student_boxes, teacher_boxes)
    cls_term = kl_div_logits(student_logits, teacher_logits, temperature=temperature)
    return lambda_cls * cls_term + lambda_box * box_term
