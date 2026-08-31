"""YOLO + ELDET strict config validation (train_checks)."""

from __future__ import annotations

import pytest

from eldet.yolo.config import ELDETYoloConfig
from eldet.yolo.train_checks import eldet_yolo_config_issues, ensure_eldet_yolo_config


def test_issues_empty_when_disabled_even_if_noise() -> None:
    c = ELDETYoloConfig(enabled=False, noise_cat_fraction=0.5, num_classes=None)
    assert eldet_yolo_config_issues(c) == []


def test_issues_when_noise_without_num_classes() -> None:
    c = ELDETYoloConfig(enabled=True, noise_cat_fraction=0.1, num_classes=None)
    assert len(eldet_yolo_config_issues(c)) == 1


def test_issues_ok_when_num_classes_set() -> None:
    c = ELDETYoloConfig(enabled=True, noise_cat_fraction=0.1, num_classes=80)
    assert eldet_yolo_config_issues(c) == []


def test_ensure_strict_raises() -> None:
    c = ELDETYoloConfig(enabled=True, noise_cat_fraction=0.1, num_classes=None)
    with pytest.raises(ValueError, match="num_classes"):
        ensure_eldet_yolo_config(c, strict=True)


def test_ensure_non_strict_no_raise() -> None:
    c = ELDETYoloConfig(enabled=True, noise_cat_fraction=0.1, num_classes=None)
    ensure_eldet_yolo_config(c, strict=False)
