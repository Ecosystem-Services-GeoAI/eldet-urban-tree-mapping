"""Ultralytics YOLO ELDET integration (imports ``ultralytics`` from ``eldet.yolo.trainer``)."""

from __future__ import annotations

from eldet.yolo.config import ELDETYoloConfig, coerce_eldet_yolo_config
from eldet.yolo.train_checks import eldet_yolo_config_issues, ensure_eldet_yolo_config

__all__ = [
    "ELDETDetectionTrainer",
    "ELDETYoloConfig",
    "coerce_eldet_yolo_config",
    "compute_yolo_kd",
    "eldet_yolo_config_issues",
    "ensure_eldet_yolo_config",
]


def __getattr__(name: str):
    if name == "ELDETDetectionTrainer":
        from eldet.yolo.trainer import ELDETDetectionTrainer

        return ELDETDetectionTrainer
    if name == "compute_yolo_kd":
        from eldet.yolo.kd_yolo import compute_yolo_kd

        return compute_yolo_kd
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
