"""Train-set CA / GTBA for YOLO v8 detection (pixel xyxy, greedy top-``k`` anchors per image)."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
import torch

from eldet.metrics import (
    ImageDetections,
    ImageGroundTruth,
    ca_localization_recall_at_iou,
    gtba_classification_accuracy,
)


def _batch_targets_to_per_image_gt(
    gt_labels: torch.Tensor,
    gt_bboxes: torch.Tensor,
    mask_gt: torch.Tensor,
    batch_size: int,
) -> list[ImageGroundTruth]:
    """``gt_*`` from :meth:`ultralytics.utils.loss.v8DetectionLoss.preprocess` layout."""
    out: list[ImageGroundTruth] = []
    for b in range(batch_size):
        m = mask_gt[b, :, 0].bool()
        if not m.any():
            out.append(
                ImageGroundTruth(
                    boxes_xyxy=np.zeros((0, 4), dtype=np.float64),
                    class_ids=np.zeros((0,), dtype=np.int64),
                )
            )
            continue
        bx = gt_bboxes[b, m].detach().cpu().numpy().astype(np.float64)
        cid = gt_labels[b, m, 0].detach().cpu().numpy().astype(np.int64)
        out.append(ImageGroundTruth(boxes_xyxy=bx, class_ids=cid))
    return out


def _preds_to_topk_detections(
    pred_bboxes_px: torch.Tensor,
    pred_scores: torch.Tensor,
    *,
    topk: int,
) -> list[ImageDetections]:
    """``pred_bboxes_px`` ``(B, A, 4)`` xyxy pixels; ``pred_scores`` ``(B, A, nc)`` logits."""
    b, a, nc = pred_scores.shape
    k = min(topk, a)
    out: list[ImageDetections] = []
    probs = pred_scores.sigmoid()
    for bi in range(b):
        scores, cls = probs[bi].max(dim=-1)
        idx = torch.topk(scores, k=k, largest=True).indices
        pb = pred_bboxes_px[bi, idx].detach().cpu().numpy()
        pr_scores = scores[idx].detach().cpu().numpy()
        pl = cls[idx].detach().cpu().numpy().astype(np.int64)
        out.append(ImageDetections(boxes_xyxy=pb, scores=pr_scores, class_ids=pl))
    return out


@torch.no_grad()
def probe_yolo_train_ca_gtba(
    batch: dict[str, Any],
    preds: dict[str, torch.Tensor],
    criterion: Any,
    *,
    topk: int = 200,
    gtba_tau: float = 0.1,
) -> tuple[float, float]:
    """One batch's train-set CA + GTBA (``preds`` vs ``batch`` GT)."""
    from ultralytics.utils.tal import make_anchors

    from eldet.yolo.kd_yolo import _yolo_detect_preds_for_kd, _yolo_kd_criterion

    criterion = _yolo_kd_criterion(criterion)
    preds = _yolo_detect_preds_for_kd(preds)

    pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
    pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
    anchor_points, stride_tensor = make_anchors(preds["feats"], criterion.stride, 0.5)
    pred_bboxes = criterion.bbox_decode(anchor_points, pred_distri)
    pred_bboxes_px = pred_bboxes * stride_tensor

    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    imgsz = torch.tensor(preds["feats"][0].shape[2:], device=criterion.device, dtype=dtype) * criterion.stride[0]
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    targets = criterion.preprocess(targets.to(criterion.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 4), 2)
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

    gts = _batch_targets_to_per_image_gt(gt_labels, gt_bboxes, mask_gt, batch_size)
    dets = _preds_to_topk_detections(pred_bboxes_px, pred_scores, topk=topk)
    ca = ca_localization_recall_at_iou(gts, dets, iou_threshold=0.5)
    gtba = gtba_classification_accuracy(gts, dets, tau=gtba_tau)
    return float(ca), float(gtba)


def iter_yolo_train_batches(trainer: Any, *, max_batches: int) -> Iterator[dict[str, Any]]:
    """Yield preprocessed batches from the train loader (rank 0 / single-GPU only)."""
    from ultralytics.utils import RANK

    if RANK not in {-1, 0}:
        return
    loader = getattr(trainer, "train_loader", None)
    if loader is None:
        return
    it = iter(loader)
    for _ in range(max_batches):
        try:
            raw = next(it)
        except StopIteration:
            break
        yield trainer.preprocess_batch(raw)
