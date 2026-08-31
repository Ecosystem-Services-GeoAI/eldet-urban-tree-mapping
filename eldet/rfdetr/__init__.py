"""RF-DETR + PyTorch Lightning integration (optional ``rfdetr`` / vendor install for training modules)."""

from __future__ import annotations

from eldet.rfdetr.callbacks import (
    ELDETPhaseCallback,
    ELDETTeacherEMACallback,
    attach_eldet_callbacks_to_trainer,
    eldet_rf_detr_callbacks,
)
from eldet.rfdetr.config import ELDETRFDETRConfig
from eldet.rfdetr.train_checks import eldet_rfdetr_training_issues, ensure_eldet_rfdetr_training_config

__all__ = [
    "ELDETPhaseCallback",
    "ELDETRFDETRConfig",
    "ELDETRFDETRModelModule",
    "ELDETTeacherEMACallback",
    "attach_eldet_callbacks_to_trainer",
    "compute_rf_detr_kd",
    "compute_rf_detr_kd_at_indices",
    "compute_rf_detr_kd_auto",
    "eldet_rf_detr_callbacks",
    "eldet_rfdetr_training_issues",
    "ensure_eldet_rfdetr_training_config",
]


def __getattr__(name: str):
    if name == "ELDETRFDETRModelModule":
        from eldet.rfdetr.module import ELDETRFDETRModelModule

        return ELDETRFDETRModelModule
    if name == "compute_rf_detr_kd":
        from eldet.rfdetr.kd_rf import compute_rf_detr_kd

        return compute_rf_detr_kd
    if name == "compute_rf_detr_kd_at_indices":
        from eldet.rfdetr.kd_rf import compute_rf_detr_kd_at_indices

        return compute_rf_detr_kd_at_indices
    if name == "compute_rf_detr_kd_auto":
        from eldet.rfdetr.kd_rf import compute_rf_detr_kd_auto

        return compute_rf_detr_kd_auto
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
