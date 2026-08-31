"""Attach ELDET callbacks to a Lightning Trainer (no RF-DETR train loop)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("pytorch_lightning")

from eldet.rfdetr.callbacks import attach_eldet_callbacks_to_trainer, eldet_rf_detr_callbacks
from eldet.rfdetr.config import ELDETRFDETRConfig


def test_attach_eldet_callbacks_extends_list() -> None:
    trainer = MagicMock()
    trainer.callbacks = [object()]
    cfg = ELDETRFDETRConfig(enabled=True)
    tc = MagicMock(use_ema=False)
    attach_eldet_callbacks_to_trainer(trainer, cfg, tc)
    assert len(trainer.callbacks) == 1 + len(eldet_rf_detr_callbacks(cfg))


def test_attach_strict_raises_if_use_ema_with_eldet() -> None:
    trainer = MagicMock()
    trainer.callbacks = []
    cfg = ELDETRFDETRConfig(enabled=True)
    tc = MagicMock(use_ema=True)
    with pytest.raises(ValueError, match="use_ema"):
        attach_eldet_callbacks_to_trainer(trainer, cfg, tc, strict_stock_ema=True)


def test_attach_allow_stock_ema_relaxed() -> None:
    trainer = MagicMock()
    trainer.callbacks = []
    cfg = ELDETRFDETRConfig(enabled=True)
    tc = MagicMock(use_ema=True)
    attach_eldet_callbacks_to_trainer(trainer, cfg, tc, strict_stock_ema=False)
    assert len(trainer.callbacks) == len(eldet_rf_detr_callbacks(cfg))


def test_eldet_rf_detr_callbacks_disabled_empty() -> None:
    assert eldet_rf_detr_callbacks(ELDETRFDETRConfig(enabled=False)) == []


def test_eldet_rf_detr_callbacks_pth_export_optional() -> None:
    cfg = ELDETRFDETRConfig(enabled=True, export_teacher_pth=False, export_student_last_pth=False)
    assert len(eldet_rf_detr_callbacks(cfg)) == 2
    cfg_all = ELDETRFDETRConfig(enabled=True, export_teacher_pth=True, export_student_last_pth=True)
    assert len(eldet_rf_detr_callbacks(cfg_all)) == 3
