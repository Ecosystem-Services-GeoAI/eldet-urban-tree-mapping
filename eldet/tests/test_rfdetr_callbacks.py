"""RF-DETR ELDET callbacks (PyTorch Lightning only; no vendored ``rfdetr`` required)."""

from __future__ import annotations

import pytest

pytest.importorskip("pytorch_lightning")

from unittest.mock import MagicMock

import torch
from torch import nn

from eldet.rfdetr.callbacks import ELDETTeacherEMACallback, _eldet_epoch_kd_mean, eldet_rf_detr_callbacks
from eldet.rfdetr.config import ELDETRFDETRConfig


def test_eldet_epoch_kd_mean_helper() -> None:
    pl = MagicMock()
    pl._eldet_kd_epoch_n = 0
    assert _eldet_epoch_kd_mean(pl) is None
    pl._eldet_kd_epoch_sum = 1.5
    pl._eldet_kd_epoch_n = 3
    assert _eldet_epoch_kd_mean(pl) == pytest.approx(0.5)


def test_eldet_rf_detr_callbacks_disabled() -> None:
    assert eldet_rf_detr_callbacks(ELDETRFDETRConfig(enabled=False)) == []


def test_eldet_rf_detr_callbacks_enabled() -> None:
    cbs = eldet_rf_detr_callbacks(ELDETRFDETRConfig(enabled=True))
    assert len(cbs) == 2
    assert type(cbs[0]).__name__ == "ELDETPhaseCallback"
    assert type(cbs[1]).__name__ == "ELDETTeacherEMACallback"


def test_teacher_ema_callback_updates_linear_pair() -> None:
    """Duck-typed pl_module: teacher EMA blends toward student."""

    class _Pl(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.eldet = ELDETRFDETRConfig(enabled=True, teacher_decay=0.5, teacher_cls_decay=0.1)
            self.model = nn.Linear(4, 2, bias=True)
            self.eldet_teacher = nn.Linear(4, 2, bias=True)
            self.eldet_teacher.load_state_dict(self.model.state_dict())
            self.model.weight.data.fill_(1.0)
            self.model.bias.data.fill_(0.0)
            self.eldet_teacher.weight.data.zero_()
            self.eldet_teacher.bias.data.zero_()

        def init_eldet_teacher(self) -> None:
            pass

        def eldet_should_use_cls_teacher_decay(self) -> bool:
            return False

    pl = _Pl()
    cb = ELDETTeacherEMACallback(pl.eldet)
    trainer = MagicMock()
    cb.on_train_batch_end(trainer, pl, None, None, 0)
    assert torch.allclose(pl.eldet_teacher.weight, torch.full_like(pl.eldet_teacher.weight, 0.5))
    assert torch.allclose(pl.eldet_teacher.bias, torch.zeros_like(pl.eldet_teacher.bias))
