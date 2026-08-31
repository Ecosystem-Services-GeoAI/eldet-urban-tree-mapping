"""ELDET YOLO config coercion (no ``ultralytics`` import)."""

from __future__ import annotations

import pytest

from eldet.yolo.config import coerce_eldet_yolo_config


def test_coerce_none() -> None:
    assert coerce_eldet_yolo_config(None).enabled is False


def test_coerce_dict() -> None:
    c = coerce_eldet_yolo_config({"enabled": True, "lambda_kd": 2.0, "unknown": 99})
    assert c.enabled is True
    assert c.lambda_kd == 2.0


def test_coerce_dict_noise_requires_num_classes() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        coerce_eldet_yolo_config({"enabled": True, "noise_cat_fraction": 0.1})


def test_trainer_subclasses_detection() -> None:
    pytest.importorskip("ultralytics")
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    from eldet.yolo.trainer import ELDETDetectionTrainer

    assert issubclass(ELDETDetectionTrainer, DetectionTrainer)
