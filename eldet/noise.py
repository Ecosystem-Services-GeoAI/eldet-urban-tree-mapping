"""Optional synthetic localization and categorization label noise (paper Sec. 5.1).

Use only when you need **controlled ablations** (e.g. reproducing paper corruption rates on an otherwise
clean benchmark). This repo’s default workflow assumes **labels already reflect real noise** in the
dataset; frameworks should not inject synthetic noise unless you opt in.

All operations are **numpy** arrays, xyxy pixel (or same-unit) boxes. No framework imports.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_numpy(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError(f"boxes must be (N,4) xyxy, got shape {a.shape}")
    return a


def _as_int_labels(labels: Any) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64).ravel()
    return y


def apply_localization_noise_xyxy(
    boxes_xyxy: np.ndarray | Any,
    *,
    fraction: float,
    epsilon: float = 0.5,
    image_size_xy: tuple[float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Jitter box corners; perturbation scale is ``epsilon`` times local width/height (paper default 0.5).

    Args:
        boxes_xyxy: ``(N, 4)`` in ``x1,y1,x2,y2`` order.
        fraction: Fraction of boxes to corrupt in ``[0, 1]``.
        epsilon: Max relative shift per coordinate vs that box's width/height.
        image_size_xy: Optional ``(W, H)`` to clip boxes after noise.
        rng: ``numpy.random.Generator``; default ``numpy.random.default_rng()``.

    Returns:
        A **copy** of the boxes array with a random subset jittered.
    """
    rng = rng or np.random.default_rng()
    b = _as_numpy(boxes_xyxy).copy()
    n = b.shape[0]
    if n == 0 or fraction <= 0:
        return b
    k = min(n, max(1, int(round(fraction * n))))
    idx = rng.choice(n, size=k, replace=False)
    for i in idx:
        x1, y1, x2, y2 = b[i]
        w = max(float(x2 - x1), 1e-6)
        h = max(float(y2 - y1), 1e-6)
        dx = rng.uniform(-epsilon, epsilon) * w
        dy = rng.uniform(-epsilon, epsilon) * h
        b[i, 0] += dx
        b[i, 1] += dy
        b[i, 2] += dx
        b[i, 3] += dy
    if image_size_xy is not None:
        w_img, h_img = float(image_size_xy[0]), float(image_size_xy[1])
        b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0.0, w_img)
        b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0.0, h_img)
        # ensure x2>=x1, y2>=y1
        b[:, 2] = np.maximum(b[:, 2], b[:, 0] + 1e-3)
        b[:, 3] = np.maximum(b[:, 3], b[:, 1] + 1e-3)
    return b


def apply_categorization_noise(
    labels: np.ndarray | Any,
    num_classes: int,
    *,
    fraction: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Replace a random subset of labels with a **uniform wrong** class (paper Sec. 5.1).

    Args:
        labels: ``(N,)`` integer class ids in ``[0, num_classes-1]``.
        num_classes: Number of classes ``C``; wrong label drawn from ``C \\ {c_true}``.
        fraction: Fraction of instances to relabel in ``[0, 1]``.

    Returns:
        A **copy** of ``labels``.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2 for categorization noise")
    rng = rng or np.random.default_rng()
    y = _as_int_labels(labels).copy()
    n = y.shape[0]
    if n == 0 or fraction <= 0:
        return y
    k = min(n, max(1, int(round(fraction * n))))
    idx = rng.choice(n, size=k, replace=False)
    for i in idx:
        c = int(y[i])
        choices = [j for j in range(num_classes) if j != c]
        y[i] = int(rng.choice(choices))
    return y
