"""RF-DETR + ELDET training config guards."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from eldet.rfdetr.config import ELDETRFDETRConfig
from eldet.rfdetr.train_checks import eldet_rfdetr_training_issues, ensure_eldet_rfdetr_training_config


def test_eldet_rfdetr_training_issues_empty_when_disabled() -> None:
    tc = MagicMock(use_ema=True)
    assert eldet_rfdetr_training_issues(tc, ELDETRFDETRConfig(enabled=False)) == []


def test_eldet_rfdetr_training_issues_when_stock_ema_on() -> None:
    tc = MagicMock(use_ema=True)
    issues = eldet_rfdetr_training_issues(tc, ELDETRFDETRConfig(enabled=True))
    assert len(issues) == 1
    assert "use_ema" in issues[0].lower()


def test_ensure_strict_raises() -> None:
    tc = MagicMock(use_ema=True)
    with pytest.raises(ValueError, match="use_ema"):
        ensure_eldet_rfdetr_training_config(tc, ELDETRFDETRConfig(enabled=True), strict=True)


def test_ensure_not_strict_no_raise() -> None:
    tc = MagicMock(use_ema=True)
    ensure_eldet_rfdetr_training_config(tc, ELDETRFDETRConfig(enabled=True), strict=False)
