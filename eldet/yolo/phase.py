"""Ultralytics callback hooks for ELDET phase detection and fixed teacher schedules."""

from __future__ import annotations

from typing import Any

from eldet.distributed import broadcast_phase_decisions
from eldet.series_csv import maybe_append_eldet_series_row
from eldet.transition import find_memorization_epoch
from eldet.yolo.config import ELDETYoloConfig
from eldet.yolo.metrics_probe import iter_yolo_train_batches, probe_yolo_train_ca_gtba


def _eldet_yolo_epoch_kd_mean(trainer: Any) -> float | None:
    n = int(getattr(trainer, "_eldet_kd_epoch_count", 0))
    if n <= 0:
        return None
    return float(getattr(trainer, "_eldet_kd_epoch_sum", 0.0)) / n


def _trainer_ok(t: Any) -> bool:
    return getattr(t, "eldet", None) is not None and isinstance(t.eldet, ELDETYoloConfig)


def _eldet_yolo_sync_phase_and_teacher(trainer: Any) -> None:
    """Broadcast ``t_loc`` / ``t_cls`` from rank 0; single-class collapse; create teacher when needed (DDP)."""
    from eldet.single_class_phase import apply_single_class_t_cls_collapse_yolo

    t_loc, t_cls = broadcast_phase_decisions(
        getattr(trainer, "eldet_t_loc_0based", None),
        getattr(trainer, "eldet_t_cls_0based", None),
        src=0,
    )
    trainer.eldet_t_loc_0based, trainer.eldet_t_cls_0based = t_loc, t_cls
    apply_single_class_t_cls_collapse_yolo(trainer)
    if trainer.eldet_t_loc_0based is not None and getattr(trainer, "eldet_teacher", None) is None:
        eldet_yolo_init_teacher(trainer)


def eldet_yolo_on_train_epoch_start(trainer: Any) -> None:
    if not _trainer_ok(trainer) or not trainer.eldet.enabled:
        return
    trainer._eldet_kd_epoch_sum = 0.0
    trainer._eldet_kd_epoch_count = 0
    if trainer.eldet.use_metric_phase_detection:
        return
    te = trainer.eldet.teacher_start_epoch
    if te == 0 and trainer.epoch == 0 and getattr(trainer, "eldet_teacher", None) is None:
        eldet_yolo_init_teacher(trainer)
        return
    if te is None or te <= 0:
        return
    if trainer.epoch == te and getattr(trainer, "eldet_teacher", None) is None:
        eldet_yolo_init_teacher(trainer)


def eldet_yolo_on_train_epoch_end(trainer: Any) -> None:
    if not _trainer_ok(trainer) or not trainer.eldet.enabled:
        return

    from ultralytics.utils import RANK

    if not trainer.eldet.use_metric_phase_detection:
        te = trainer.eldet.teacher_start_epoch
        if te is not None and te > 0 and trainer.epoch == te - 1 and getattr(trainer, "eldet_teacher", None) is None:
            eldet_yolo_init_teacher(trainer)
        _eldet_yolo_sync_phase_and_teacher(trainer)
        if trainer.eldet.series_csv_path and RANK in {-1, 0}:
            maybe_append_eldet_series_row(
                trainer.eldet.series_csv_path,
                epoch=int(trainer.epoch),
                ca=None,
                gtba=None,
                t_loc_0based=trainer.eldet_t_loc_0based,
                t_cls_0based=trainer.eldet_t_cls_0based,
                loss_kd=_eldet_yolo_epoch_kd_mean(trainer),
            )
        return

    every = trainer.eldet.metric_every_n_epochs
    if every > 1 and (trainer.epoch + 1) % every != 0:
        _eldet_yolo_sync_phase_and_teacher(trainer)
        if trainer.eldet.series_csv_path and RANK in {-1, 0}:
            maybe_append_eldet_series_row(
                trainer.eldet.series_csv_path,
                epoch=int(trainer.epoch),
                ca=None,
                gtba=None,
                t_loc_0based=trainer.eldet_t_loc_0based,
                t_cls_0based=trainer.eldet_t_cls_0based,
                loss_kd=_eldet_yolo_epoch_kd_mean(trainer),
            )
        return

    if RANK in {-1, 0}:
        was_training = trainer.model.training
        ca_vals: list[float] = []
        gtba_vals: list[float] = []
        for batch in iter_yolo_train_batches(trainer, max_batches=trainer.eldet.metric_max_train_batches):
            trainer.model.eval()
            try:
                preds = trainer.model(batch["img"])
                from ultralytics.utils.torch_utils import unwrap_model

                um = unwrap_model(trainer.model)
                ca, gtba = probe_yolo_train_ca_gtba(
                    batch,
                    preds,
                    um.criterion,
                    topk=trainer.eldet.metric_topk_anchors,
                    gtba_tau=trainer.eldet.gtba_tau,
                )
            finally:
                trainer.model.train(was_training)
            ca_vals.append(ca)
            gtba_vals.append(gtba)

        if ca_vals:
            ca_mean = float(sum(ca_vals) / len(ca_vals))
            gtba_mean = float(sum(gtba_vals) / len(gtba_vals))

            freeze_after_cls = bool(trainer.eldet.freeze_transitions_after_cls)
            transitions_frozen = freeze_after_cls and trainer.eldet_t_cls_0based is not None

            if not transitions_frozen:
                trainer._eldet_ca_hist.append(ca_mean)
                trainer._eldet_gtba_hist.append(gtba_mean)

            maybe_append_eldet_series_row(
                trainer.eldet.series_csv_path,
                epoch=int(trainer.epoch),
                ca=ca_mean,
                gtba=gtba_mean,
                t_loc_0based=trainer.eldet_t_loc_0based,
                t_cls_0based=trainer.eldet_t_cls_0based,
                loss_kd=_eldet_yolo_epoch_kd_mean(trainer),
            )

            if not transitions_frozen:
                if getattr(trainer, "eldet_teacher", None) is None and len(trainer._eldet_ca_hist) >= 3:
                    m = find_memorization_epoch(trainer._eldet_ca_hist, gamma=trainer.eldet.transition_gamma)
                    if m is not None and m.epoch_0based == len(trainer._eldet_ca_hist) - 1:
                        trainer.eldet_t_loc_0based = m.epoch_0based
                        eldet_yolo_init_teacher(trainer)

                if (
                    getattr(trainer, "eldet_teacher", None) is not None
                    and trainer.eldet_t_cls_0based is None
                    and len(trainer._eldet_gtba_hist) >= 3
                ):
                    m2 = find_memorization_epoch(trainer._eldet_gtba_hist, gamma=trainer.eldet.transition_gamma)
                    if m2 is not None and m2.epoch_0based == len(trainer._eldet_gtba_hist) - 1:
                        trainer.eldet_t_cls_0based = m2.epoch_0based

    _eldet_yolo_sync_phase_and_teacher(trainer)


def eldet_yolo_init_teacher(trainer: Any) -> None:
    """Deep-copy student into ``trainer.eldet_teacher`` (frozen, same device)."""
    if getattr(trainer, "eldet_teacher", None) is not None:
        return
    from copy import deepcopy

    from ultralytics.utils.torch_utils import unwrap_model

    core = unwrap_model(trainer.model)
    teacher = deepcopy(core).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    trainer.eldet_teacher = teacher.to(trainer.device)


def eldet_yolo_should_use_cls_teacher_decay(trainer: Any) -> bool:
    if trainer.eldet_t_loc_0based is None or getattr(trainer, "eldet_teacher", None) is None:
        return False
    e = int(trainer.epoch)
    if e < trainer.eldet_t_loc_0based:
        return False
    if trainer.eldet_t_cls_0based is not None and e >= trainer.eldet_t_cls_0based:
        return False
    return True
