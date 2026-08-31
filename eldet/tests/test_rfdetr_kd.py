"""RF-DETR KD helpers (torch only; no full ``rfdetr`` model)."""

from __future__ import annotations

import torch

from eldet.rfdetr.kd_rf import compute_rf_detr_kd_at_indices, compute_rf_detr_kd_auto


class _CriterionStub:
    _eldet_last_indices = [(torch.tensor([0, 0], dtype=torch.long), torch.tensor([1, 2], dtype=torch.long))]

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i, dtype=torch.long) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


def test_compute_rf_detr_kd_at_indices_finite_grad_student() -> None:
    b, q, c = 1, 4, 3
    os_ = {
        "pred_logits": torch.randn(b, q, c, requires_grad=True),
        "pred_boxes": torch.randn(b, q, 4, requires_grad=True),
    }
    ot = {"pred_logits": torch.randn(b, q, c), "pred_boxes": torch.randn(b, q, 4)}
    loss = compute_rf_detr_kd_at_indices(_CriterionStub(), os_, ot)
    loss.backward()
    assert os_["pred_logits"].grad is not None


def test_compute_rf_detr_kd_auto_falls_back_when_no_indices() -> None:
    b, q, c = 1, 2, 3
    os_ = {
        "pred_logits": torch.randn(b, q, c, requires_grad=True),
        "pred_boxes": torch.randn(b, q, 4, requires_grad=True),
    }
    ot = {"pred_logits": torch.randn(b, q, c), "pred_boxes": torch.randn(b, q, 4)}
    class _NoIdx:
        pass

    loss = compute_rf_detr_kd_auto(_NoIdx(), os_, ot, use_matcher_indices=True)
    loss.backward()
    assert os_["pred_logits"].grad is not None


def test_compute_rf_detr_kd_auto_empty_matches_zero() -> None:
    class _Empty:
        _eldet_last_indices = [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))]

        def _get_src_permutation_idx(self, indices):
            batch_idx = torch.cat([torch.full_like(src, i, dtype=torch.long) for i, (src, _) in enumerate(indices)])
            src_idx = torch.cat([src for (src, _) in indices])
            return batch_idx, src_idx

    b, q, c = 1, 2, 3
    os_ = {
        "pred_logits": torch.randn(b, q, c, requires_grad=True),
        "pred_boxes": torch.randn(b, q, 4, requires_grad=True),
    }
    ot = {"pred_logits": torch.randn(b, q, c), "pred_boxes": torch.randn(b, q, 4)}
    loss = compute_rf_detr_kd_at_indices(_Empty(), os_, ot)
    assert float(loss.item()) == 0.0
