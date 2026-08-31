"""YOLO phase hooks: DDP sync and fixed-schedule paths."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("ultralytics")

from eldet.yolo.config import ELDETYoloConfig
from eldet.yolo.phase import _eldet_yolo_epoch_kd_mean, eldet_yolo_on_train_epoch_end


def test_eldet_yolo_epoch_kd_mean() -> None:
    trainer = MagicMock()
    trainer._eldet_kd_epoch_count = 0
    assert _eldet_yolo_epoch_kd_mean(trainer) is None
    trainer._eldet_kd_epoch_sum = 2.0
    trainer._eldet_kd_epoch_count = 4
    assert _eldet_yolo_epoch_kd_mean(trainer) == pytest.approx(0.5)


def test_fixed_schedule_epoch_end_calls_sync() -> None:
    trainer = MagicMock()
    trainer.eldet = ELDETYoloConfig(enabled=True, use_metric_phase_detection=False, teacher_start_epoch=1)
    trainer.epoch = 0
    trainer.eldet_teacher = object()
    with patch("eldet.yolo.phase._eldet_yolo_sync_phase_and_teacher") as sync:
        eldet_yolo_on_train_epoch_end(trainer)
        sync.assert_called_once()


def test_metric_path_calls_sync_when_no_probe_batches() -> None:
    trainer = MagicMock()
    trainer.eldet = ELDETYoloConfig(enabled=True, use_metric_phase_detection=True, metric_every_n_epochs=1)
    trainer.epoch = 0
    trainer.model = MagicMock(training=True)
    trainer.train_loader = object()
    trainer._eldet_ca_hist = []
    trainer._eldet_gtba_hist = []
    trainer.eldet_t_loc_0based = None
    trainer.eldet_t_cls_0based = None
    with patch("eldet.yolo.phase.iter_yolo_train_batches", return_value=[]):
        with patch("eldet.yolo.phase._eldet_yolo_sync_phase_and_teacher") as sync:
            with patch("ultralytics.utils.RANK", 0):
                eldet_yolo_on_train_epoch_end(trainer)
                sync.assert_called_once()
