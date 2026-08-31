"""KD helper for YOLO (requires ``ultralytics`` for ``make_anchors``)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("ultralytics")

from eldet.yolo.kd_yolo import compute_yolo_kd


class _FakeCriterion:
    device = torch.device("cpu")
    stride = torch.tensor([8.0, 16.0, 32.0])

    def bbox_decode(self, anchor_points: torch.Tensor, pred_distri: torch.Tensor) -> torch.Tensor:
        del anchor_points
        b, a, _ = pred_distri.shape
        return torch.zeros(b, a, 4, device=pred_distri.device, dtype=pred_distri.dtype)


def test_compute_yolo_kd_scalar() -> None:
    b, nc = 2, 3
    feats = [
        torch.zeros(b, 16, 20, 20),
        torch.zeros(b, 16, 10, 10),
        torch.zeros(b, 16, 5, 5),
    ]
    na = 20 * 20 + 10 * 10 + 5 * 5
    preds_s = {
        "feats": feats,
        "boxes": torch.randn(b, 4 * 16, na),
        "scores": torch.randn(b, nc, na),
    }
    preds_t = {
        "feats": [f.clone() for f in feats],
        "boxes": torch.randn(b, 4 * 16, na),
        "scores": torch.randn(b, nc, na),
    }
    crit = _FakeCriterion()
    loss = compute_yolo_kd(preds_s, preds_t, crit)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
