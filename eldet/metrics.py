"""Class-agnostic (CA) and ground-truth box allocation (GTBA) train-set metrics (numpy only).

Conventions: boxes are ``xyxy`` in the same coordinate system (pixels or normalized).

Use :class:`MetricConfig` plus :func:`metrics_should_run` / :func:`cap_probe_count` to align probe cadence and
cost across framework adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of axis-aligned boxes.

    Args:
        a: ``(N, 4)`` xyxy
        b: ``(M, 4)`` xyxy

    Returns:
        ``(N, M)`` IoU matrix.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 4 or b.ndim != 2 or b.shape[1] != 4:
        raise ValueError("boxes must be (N,4) and (M,4) xyxy")

    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    iw = np.clip(inter_x2 - inter_x1, a_min=0.0, a_max=None)
    ih = np.clip(inter_y2 - inter_y1, a_min=0.0, a_max=None)
    inter = iw * ih

    area_a = np.clip(ax2 - ax1, 1e-6, None) * np.clip(ay2 - ay1, 1e-6, None)
    area_b = np.clip(bx2 - bx1, 1e-6, None) * np.clip(by2 - by1, 1e-6, None)
    union = area_a + area_b - inter
    return inter / np.clip(union, 1e-6, None)


@dataclass
class ImageDetections:
    """Per-image predictions (lists of arrays for one image)."""

    boxes_xyxy: np.ndarray  # (P, 4)
    scores: np.ndarray  # (P,)
    class_ids: np.ndarray  # (P,) int


@dataclass
class ImageGroundTruth:
    """Per-image ground truth."""

    boxes_xyxy: np.ndarray  # (G, 4)
    class_ids: np.ndarray  # (G,) int


def ca_localization_recall_at_iou(
    gts: Sequence[ImageGroundTruth],
    preds: Sequence[ImageDetections],
    *,
    iou_threshold: float = 0.5,
) -> float:
    """Class-agnostic localization proxy: mean recall @ ``iou_threshold`` with greedy score match.

    All classes are ignored for matching; predictions are pooled by box+score only.
    Greedy: sort predictions by descending score, each pred matches at most one GT, TP if IoU >= thr.
    Per-image recall = TP / max(G, 1); return mean over images with at least one GT.
    """
    if len(gts) != len(preds):
        raise ValueError("gts and preds must have same length")
    recalls: list[float] = []
    for gt, pr in zip(gts, preds):
        gb = np.asarray(gt.boxes_xyxy, dtype=np.float64)
        g = gb.shape[0]
        if g == 0:
            continue
        pb = np.asarray(pr.boxes_xyxy, dtype=np.float64)
        ps = np.asarray(pr.scores, dtype=np.float64).ravel()
        if pb.shape[0] == 0:
            recalls.append(0.0)
            continue
        order = np.argsort(-ps)
        used_gt = np.zeros(g, dtype=bool)
        tp = 0
        ious = box_iou_xyxy(pb, gb)
        for j in order:
            best_i, best_iou = -1, 0.0
            for i in range(g):
                if used_gt[i]:
                    continue
                iou = float(ious[j, i])
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i >= 0 and best_iou >= iou_threshold:
                used_gt[best_i] = True
                tp += 1
        recalls.append(tp / g)
    if not recalls:
        return 0.0
    return float(np.mean(recalls))


def gtba_classification_accuracy(
    gts: Sequence[ImageGroundTruth],
    preds: Sequence[ImageDetections],
    *,
    tau: float = 0.1,
) -> float:
    """GTBA: for each GT, take highest-IoU pred; if IoU >= ``tau``, count class equality (paper default τ=0.1).

    Denominator: number of GT objects with at least one prediction (any); if none, returns 0.0.
    """
    if len(gts) != len(preds):
        raise ValueError("gts and preds must have same length")
    correct = 0
    total = 0
    for gt, pr in zip(gts, preds):
        gb = np.asarray(gt.boxes_xyxy, dtype=np.float64)
        gc = np.asarray(gt.class_ids, dtype=np.int64).ravel()
        pb = np.asarray(pr.boxes_xyxy, dtype=np.float64)
        pc = np.asarray(pr.class_ids, dtype=np.int64).ravel()
        g = gb.shape[0]
        if g == 0:
            continue
        if pb.shape[0] == 0:
            continue
        ious = box_iou_xyxy(gb, pb)
        for i in range(g):
            j = int(np.argmax(ious[i]))
            if float(ious[i, j]) >= tau:
                total += 1
                if int(pc[j]) == int(gc[i]):
                    correct += 1
    if total == 0:
        return 0.0
    return correct / total


@dataclass(frozen=True)
class MetricConfig:
    """Shared knobs for train-set CA/GTBA probes (framework adapters may map their fields onto this)."""

    tau: float = 0.1
    """GTBA: IoU threshold for allocating a prediction box to a GT (paper default 0.1)."""

    ca_iou_threshold: float = 0.5
    """CA proxy: greedy match threshold for class-agnostic localization recall."""

    max_images_per_epoch: int | None = None
    """If set, adapters should cap probe images/batches to this count per epoch (cost control)."""

    metrics_every_n_epochs: int = 1
    """Run CA/GTBA probes every N epochs (1 = each epoch)."""


def metrics_should_run(epoch_0based: int, cfg: MetricConfig) -> bool:
    """True when ``epoch_0based`` (0-based) should run a scheduled probe for ``cfg.metrics_every_n_epochs``."""
    every = int(cfg.metrics_every_n_epochs)
    if every <= 1:
        return True
    return (int(epoch_0based) + 1) % every == 0


def cap_probe_count(requested: int, cfg: MetricConfig) -> int:
    """Upper bound on probe images/batches: ``min(requested, max_images_per_epoch)`` when set."""
    r = max(0, int(requested))
    cap = cfg.max_images_per_epoch
    if cap is None:
        return r
    return min(r, max(0, int(cap)))


def ca_localization_recall_with_config(
    gts: Sequence[ImageGroundTruth],
    preds: Sequence[ImageDetections],
    cfg: MetricConfig,
) -> float:
    """Same as :func:`ca_localization_recall_at_iou` using ``cfg.ca_iou_threshold``."""
    return ca_localization_recall_at_iou(gts, preds, iou_threshold=cfg.ca_iou_threshold)


def gtba_classification_accuracy_with_config(
    gts: Sequence[ImageGroundTruth],
    preds: Sequence[ImageDetections],
    cfg: MetricConfig,
) -> float:
    """Same as :func:`gtba_classification_accuracy` using ``cfg.tau``."""
    return gtba_classification_accuracy(gts, preds, tau=cfg.tau)
