"""Train-set CA / GTBA probes for RF-DETR (normalized cxcywh → xyxy in unit square for IoU)."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
import torch
from torch import nn

from eldet.metrics import (
    ImageDetections,
    ImageGroundTruth,
    ca_localization_recall_at_iou,
    gtba_classification_accuracy,
)


def _cxcywh01_to_xyxy01(boxes: torch.Tensor) -> torch.Tensor:
    """``boxes`` (*, 4) cxcywh in ``[0,1]`` → xyxy in same normalized frame."""
    from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy

    return box_cxcywh_to_xyxy(boxes)


def _batch_pred_to_image_detections(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    *,
    topk: int = 50,
) -> list[ImageDetections]:
    """One ``ImageDetections`` per batch image using top-``topk`` queries by max class score."""
    b, q, c = pred_logits.shape
    probs = pred_logits.sigmoid()
    scores, _ = probs.max(dim=-1)
    boxes_cxcywh = pred_boxes
    out: list[ImageDetections] = []
    for i in range(b):
        s = scores[i]
        k = min(topk, q)
        idx = torch.topk(s, k=k, largest=True).indices
        pb = _cxcywh01_to_xyxy01(boxes_cxcywh[i, idx]).detach().cpu().numpy()
        pr_scores = s[idx].detach().cpu().numpy()
        pl = probs[i, idx].argmax(dim=-1).detach().cpu().numpy().astype(np.int64)
        out.append(ImageDetections(boxes_xyxy=pb, scores=pr_scores, class_ids=pl))
    return out


def _targets_to_image_gts(targets: list[dict[str, Any]]) -> list[ImageGroundTruth]:
    gts: list[ImageGroundTruth] = []
    for t in targets:
        bx = t["boxes"]
        if bx.numel() == 0:
            empty = ImageGroundTruth(
                boxes_xyxy=np.zeros((0, 4), dtype=np.float64),
                class_ids=np.zeros((0,), dtype=np.int64),
            )
            gts.append(empty)
            continue
        xyxy = _cxcywh01_to_xyxy01(bx).detach().cpu().numpy()
        lab = t["labels"].detach().cpu().numpy().astype(np.int64)
        gts.append(ImageGroundTruth(boxes_xyxy=xyxy, class_ids=lab))
    return gts


@torch.no_grad()
def probe_train_metrics_one_batch(
    model: nn.Module,
    samples: Any,
    targets: list[dict[str, Any]],
    *,
    topk: int = 50,
    gtba_tau: float = 0.1,
) -> tuple[float, float]:
    """Return ``(ca_recall@0.5, gtba_acc)`` for one batch (model in caller-controlled mode)."""
    outputs = model(samples)
    logits = outputs["pred_logits"]
    boxes = outputs["pred_boxes"]
    preds = _batch_pred_to_image_detections(logits, boxes, topk=topk)
    gts = _targets_to_image_gts(targets)
    ca = ca_localization_recall_at_iou(gts, preds, iou_threshold=0.5)
    gtba = gtba_classification_accuracy(gts, preds, tau=gtba_tau)
    return float(ca), float(gtba)


def iter_train_metric_batches(
    trainer: Any,
    pl_module: Any,
    *,
    max_batches: int,
) -> Iterator[tuple[Any, list[dict[str, Any]]]]:
    """Yield ``(samples, targets)`` from the train dataloader (rank 0)."""
    if getattr(trainer, "global_rank", 0) != 0:
        return
    dm = getattr(trainer, "datamodule", None)
    if dm is None:
        return
    loader = dm.train_dataloader()
    it = iter(loader)
    for _ in range(max_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        yield batch
