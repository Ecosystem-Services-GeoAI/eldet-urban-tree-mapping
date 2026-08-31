"""MetricConfig and probe scheduling helpers."""

from __future__ import annotations

import numpy as np

from eldet.metrics import (
    ImageDetections,
    ImageGroundTruth,
    MetricConfig,
    ca_localization_recall_with_config,
    cap_probe_count,
    gtba_classification_accuracy_with_config,
    metrics_should_run,
)


def test_metrics_should_run_every_n_epochs() -> None:
    cfg = MetricConfig(metrics_every_n_epochs=3)
    assert not metrics_should_run(0, cfg)
    assert not metrics_should_run(1, cfg)
    assert metrics_should_run(2, cfg)


def test_cap_probe_count() -> None:
    assert cap_probe_count(100, MetricConfig(max_images_per_epoch=None)) == 100
    assert cap_probe_count(100, MetricConfig(max_images_per_epoch=10)) == 10


def test_with_config_wrappers_match_explicit() -> None:
    gt = ImageGroundTruth(
        boxes_xyxy=np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float64),
        class_ids=np.array([1], dtype=np.int64),
    )
    pr = ImageDetections(
        boxes_xyxy=np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float64),
        scores=np.array([0.9], dtype=np.float64),
        class_ids=np.array([1], dtype=np.int64),
    )
    cfg = MetricConfig(tau=0.05, ca_iou_threshold=0.5)
    assert ca_localization_recall_with_config([gt], [pr], cfg) >= 0.0
    assert gtba_classification_accuracy_with_config([gt], [pr], cfg) == 1.0
