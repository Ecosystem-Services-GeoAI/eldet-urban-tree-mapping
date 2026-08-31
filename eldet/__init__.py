"""ELDET: Early-Learning Distillation helpers (Phase 1 core — no Ultralytics / RF-DETR imports)."""

from eldet._version import __version__
from eldet.distributed import broadcast_object_list, broadcast_phase_decisions, is_distributed
from eldet.ema_policy import (
    rfdetr_cls_param,
    update_teacher_from_student,
    yolo_detect_cls_param,
)
from eldet.kd_losses import kd_detection_v1, kl_div_logits, l1_box_loss
from eldet.metrics import (
    ImageDetections,
    ImageGroundTruth,
    MetricConfig,
    box_iou_xyxy,
    ca_localization_recall_at_iou,
    ca_localization_recall_with_config,
    cap_probe_count,
    gtba_classification_accuracy,
    gtba_classification_accuracy_with_config,
    metrics_should_run,
)
from eldet.noise import apply_categorization_noise, apply_localization_noise_xyxy
from eldet.transition import (
    MemorizationEpoch,
    find_memorization_epoch,
    find_memorization_epoch_finite_diff,
    fit_exponential_saturation,
    moving_average_smooth,
)

__all__ = [
    "__version__",
    "MemorizationEpoch",
    "MetricConfig",
    "apply_categorization_noise",
    "apply_localization_noise_xyxy",
    "box_iou_xyxy",
    "broadcast_object_list",
    "broadcast_phase_decisions",
    "ca_localization_recall_at_iou",
    "cap_probe_count",
    "ca_localization_recall_with_config",
    "find_memorization_epoch",
    "find_memorization_epoch_finite_diff",
    "fit_exponential_saturation",
    "gtba_classification_accuracy",
    "gtba_classification_accuracy_with_config",
    "ImageDetections",
    "ImageGroundTruth",
    "is_distributed",
    "kd_detection_v1",
    "kl_div_logits",
    "l1_box_loss",
    "metrics_should_run",
    "moving_average_smooth",
    "rfdetr_cls_param",
    "update_teacher_from_student",
    "yolo_detect_cls_param",
]
