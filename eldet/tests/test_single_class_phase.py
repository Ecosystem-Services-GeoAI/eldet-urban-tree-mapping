"""Single-class ``t_cls`` collapse helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from eldet.rfdetr.config import ELDETRFDETRConfig
from eldet.single_class_phase import (
    apply_single_class_t_cls_collapse_rfdetr,
    apply_single_class_t_cls_collapse_yolo,
    should_collapse_t_cls,
)


def test_should_collapse_explicit() -> None:
    assert should_collapse_t_cls(single_class=True, num_classes=80) is True
    assert should_collapse_t_cls(single_class=False, num_classes=1) is False
    assert should_collapse_t_cls(single_class=None, num_classes=1) is True
    assert should_collapse_t_cls(single_class=None, num_classes=2) is False
    assert should_collapse_t_cls(single_class=None, num_classes=None) is False


def test_rfdetr_collapse_sets_t_cls() -> None:
    mc = MagicMock(num_classes=1)
    pl = MagicMock(model_config=mc, eldet_t_loc_0based=5, eldet_t_cls_0based=None)
    eldet = ELDETRFDETRConfig(enabled=True, single_class=None)
    apply_single_class_t_cls_collapse_rfdetr(pl, eldet)
    assert pl.eldet_t_cls_0based == 5


def test_rfdetr_no_collapse_multi_class() -> None:
    mc = MagicMock(num_classes=3)
    pl = MagicMock(model_config=mc, eldet_t_loc_0based=5, eldet_t_cls_0based=9)
    eldet = ELDETRFDETRConfig(enabled=True, single_class=None)
    apply_single_class_t_cls_collapse_rfdetr(pl, eldet)
    assert pl.eldet_t_cls_0based == 9


def test_yolo_collapse_uses_detect_nc() -> None:
    pytest.importorskip("ultralytics")
    head = MagicMock(nc=1)
    seq = MagicMock()
    seq.__getitem__ = MagicMock(return_value=head)
    um = MagicMock(model=seq)
    trainer = MagicMock()
    trainer.eldet = MagicMock(single_class=None)
    trainer.model = object()
    trainer.eldet_t_loc_0based = 2
    trainer.eldet_t_cls_0based = None

    with patch("ultralytics.utils.torch_utils.unwrap_model", return_value=um):
        apply_single_class_t_cls_collapse_yolo(trainer)
    assert trainer.eldet_t_cls_0based == 2
