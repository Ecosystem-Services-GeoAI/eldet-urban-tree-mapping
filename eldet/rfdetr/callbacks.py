"""PyTorch Lightning callbacks: ELDET teacher EMA and optional CA/GTBA phase detection."""

from __future__ import annotations

from typing import Any

from pytorch_lightning import Callback, Trainer

from eldet.distributed import broadcast_phase_decisions
from eldet.ema_policy import rfdetr_cls_param, update_teacher_from_student
from eldet.rfdetr.config import ELDETRFDETRConfig
from eldet.rfdetr.metrics_probe import iter_train_metric_batches, probe_train_metrics_one_batch
from eldet.rfdetr.train_checks import ensure_eldet_rfdetr_training_config
from eldet.series_csv import maybe_append_eldet_series_row
from eldet.transition import find_memorization_epoch


def _eldet_rfdetr_pl_module_ok(pl_module: Any) -> bool:
    """Avoid importing :mod:`eldet.rfdetr.module` (pulls ``rfdetr``) so tests can run without vendor installs."""
    return bool(
        getattr(pl_module, "eldet", None) is not None
        and callable(getattr(pl_module, "init_eldet_teacher", None))
        and getattr(pl_module, "model", None) is not None
    )


def _eldet_epoch_kd_mean(pl_module: Any) -> float | None:
    n = int(getattr(pl_module, "_eldet_kd_epoch_n", 0))
    if n <= 0:
        return None
    return float(getattr(pl_module, "_eldet_kd_epoch_sum", 0.0)) / n


def _eldet_sync_phase_and_teacher(_trainer: Trainer, pl_module: Any, eldet: ELDETRFDETRConfig) -> None:
    """Broadcast ``t_loc``/``t_cls``; single-class collapse; create ``eldet_teacher`` when needed (DDP)."""
    from eldet.single_class_phase import apply_single_class_t_cls_collapse_rfdetr

    t_loc, t_cls = broadcast_phase_decisions(pl_module.eldet_t_loc_0based, pl_module.eldet_t_cls_0based, src=0)
    pl_module.eldet_t_loc_0based, pl_module.eldet_t_cls_0based = t_loc, t_cls
    apply_single_class_t_cls_collapse_rfdetr(pl_module, eldet)
    if pl_module.eldet_t_loc_0based is not None and getattr(pl_module, "eldet_teacher", None) is None:
        pl_module.init_eldet_teacher()


class ELDETTeacherEMACallback(Callback):
    """Update ``eldet_teacher`` from student after each training batch (post-optimizer in default PTL order)."""

    def __init__(self, eldet: ELDETRFDETRConfig) -> None:
        super().__init__()
        self._eldet = eldet

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx
        if not self._eldet.enabled or not _eldet_rfdetr_pl_module_ok(pl_module):
            return
        teacher = getattr(pl_module, "eldet_teacher", None)
        if teacher is None:
            return
        cls_decay = self._eldet.teacher_cls_decay if pl_module.eldet_should_use_cls_teacher_decay() else None
        update_teacher_from_student(
            teacher,
            pl_module.model,
            global_decay=self._eldet.teacher_decay,
            cls_decay=cls_decay,
            is_cls_param=rfdetr_cls_param if cls_decay is not None else None,
            ema_buffers=True,
        )


class ELDETPhaseCallback(Callback):
    """Optional CA/GTBA probe + transition detection, or fixed ``teacher_start_epoch`` (all ranks for teacher init)."""

    def __init__(self, eldet: ELDETRFDETRConfig) -> None:
        super().__init__()
        self._eldet = eldet

    def on_train_epoch_start(self, trainer: Trainer, pl_module: Any) -> None:
        if not self._eldet.enabled or not _eldet_rfdetr_pl_module_ok(pl_module):
            return
        if self._eldet.use_metric_phase_detection:
            return
        te = self._eldet.teacher_start_epoch
        if te == 0 and trainer.current_epoch == 0 and getattr(pl_module, "eldet_teacher", None) is None:
            pl_module.init_eldet_teacher()
            return
        if te is None or te <= 0:
            return
        if trainer.current_epoch == te and getattr(pl_module, "eldet_teacher", None) is None:
            pl_module.init_eldet_teacher()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: Any) -> None:
        if not self._eldet.enabled or not _eldet_rfdetr_pl_module_ok(pl_module):
            return

        if not self._eldet.use_metric_phase_detection:
            te = self._eldet.teacher_start_epoch
            teacher = getattr(pl_module, "eldet_teacher", None)
            if te is not None and te > 0 and trainer.current_epoch == te - 1 and teacher is None:
                pl_module.init_eldet_teacher()
            _eldet_sync_phase_and_teacher(trainer, pl_module, self._eldet)
            if int(getattr(trainer, "global_rank", 0)) == 0 and self._eldet.series_csv_path:
                maybe_append_eldet_series_row(
                    self._eldet.series_csv_path,
                    epoch=int(trainer.current_epoch),
                    ca=None,
                    gtba=None,
                    t_loc_0based=pl_module.eldet_t_loc_0based,
                    t_cls_0based=pl_module.eldet_t_cls_0based,
                    loss_kd=_eldet_epoch_kd_mean(pl_module),
                )
            return

        every = self._eldet.metric_every_n_epochs
        if every > 1 and (trainer.current_epoch + 1) % every != 0:
            _eldet_sync_phase_and_teacher(trainer, pl_module, self._eldet)
            rank_skip = int(getattr(trainer, "global_rank", 0))
            if rank_skip == 0 and self._eldet.series_csv_path:
                maybe_append_eldet_series_row(
                    self._eldet.series_csv_path,
                    epoch=int(trainer.current_epoch),
                    ca=None,
                    gtba=None,
                    t_loc_0based=pl_module.eldet_t_loc_0based,
                    t_cls_0based=pl_module.eldet_t_cls_0based,
                    loss_kd=_eldet_epoch_kd_mean(pl_module),
                )
            return

        rank = int(getattr(trainer, "global_rank", 0))
        if rank == 0:
            was_training = pl_module.model.training
            ca_vals: list[float] = []
            gtba_vals: list[float] = []
            for batch in iter_train_metric_batches(
                trainer, pl_module, max_batches=self._eldet.metric_max_train_batches
            ):
                samples, targets = batch
                pl_module.model.eval()
                try:
                    ca, gtba = probe_train_metrics_one_batch(
                        pl_module.model,
                        samples,
                        targets,
                        topk=50,
                        gtba_tau=self._eldet.gtba_tau,
                    )
                finally:
                    pl_module.model.train(was_training)
                ca_vals.append(ca)
                gtba_vals.append(gtba)

            if ca_vals:
                ca_mean = float(sum(ca_vals) / len(ca_vals))
                gtba_mean = float(sum(gtba_vals) / len(gtba_vals))

                freeze_after_cls = bool(self._eldet.freeze_transitions_after_cls)
                transitions_frozen = freeze_after_cls and pl_module.eldet_t_cls_0based is not None

                if not transitions_frozen:
                    pl_module._eldet_ca_hist.append(ca_mean)
                    pl_module._eldet_gtba_hist.append(gtba_mean)

                logger = getattr(trainer, "logger", None)
                if logger is not None and callable(getattr(logger, "log_metrics", None)):
                    logger.log_metrics(
                        {"eldet/train_ca": ca_mean, "eldet/train_gtba": gtba_mean},
                        step=trainer.global_step,
                    )

                maybe_append_eldet_series_row(
                    self._eldet.series_csv_path,
                    epoch=int(trainer.current_epoch),
                    ca=ca_mean,
                    gtba=gtba_mean,
                    t_loc_0based=pl_module.eldet_t_loc_0based,
                    t_cls_0based=pl_module.eldet_t_cls_0based,
                    loss_kd=_eldet_epoch_kd_mean(pl_module),
                )

                if not transitions_frozen:
                    teacher = getattr(pl_module, "eldet_teacher", None)
                    if teacher is None and len(pl_module._eldet_ca_hist) >= 3:
                        m = find_memorization_epoch(pl_module._eldet_ca_hist, gamma=self._eldet.transition_gamma)
                        if m is not None and m.epoch_0based == len(pl_module._eldet_ca_hist) - 1:
                            pl_module.eldet_t_loc_0based = m.epoch_0based
                            pl_module.init_eldet_teacher()

                    if (
                        getattr(pl_module, "eldet_teacher", None) is not None
                        and pl_module.eldet_t_cls_0based is None
                        and len(pl_module._eldet_gtba_hist) >= 3
                    ):
                        m2 = find_memorization_epoch(
                            pl_module._eldet_gtba_hist, gamma=self._eldet.transition_gamma
                        )
                        if m2 is not None and m2.epoch_0based == len(pl_module._eldet_gtba_hist) - 1:
                            pl_module.eldet_t_cls_0based = m2.epoch_0based

        _eldet_sync_phase_and_teacher(trainer, pl_module, self._eldet)


def eldet_rf_detr_callbacks(eldet: ELDETRFDETRConfig) -> list[Callback]:
    """Callbacks to append after ``rfdetr.training.trainer.build_trainer`` defaults when ELDET is enabled."""
    if not eldet.enabled:
        return []
    out: list[Callback] = [ELDETPhaseCallback(eldet), ELDETTeacherEMACallback(eldet)]
    if eldet.export_teacher_pth or eldet.export_student_last_pth:
        from eldet.rfdetr.pth_export_callback import ELDETStrippedPthExportCallback

        out.append(ELDETStrippedPthExportCallback(eldet))
    return out


def attach_eldet_callbacks_to_trainer(
    trainer: Any,
    eldet: ELDETRFDETRConfig,
    train_config: Any | None = None,
    *,
    strict_stock_ema: bool = True,
) -> None:
    """Append ELDET callbacks to a Lightning ``Trainer`` (e.g. from ``build_trainer``).

    Args:
        trainer: PyTorch Lightning ``Trainer``.
        eldet: ELDET options (callbacks are no-ops when ``enabled`` is False).
        train_config: Optional :class:`rfdetr.config.TrainConfig`. When provided with
            ``strict_stock_ema=True`` and ELDET enabled, raises if ``use_ema`` is still True
            (Phase 2.5 — avoid stock ``RFDETREMACallback`` alongside the ELDET teacher).
        strict_stock_ema: When ``True`` and ``train_config`` is given, enforce ``use_ema=False``.
    """
    if train_config is not None:
        ensure_eldet_rfdetr_training_config(train_config, eldet, strict=strict_stock_ema)
    for cb in eldet_rf_detr_callbacks(eldet):
        trainer.callbacks.append(cb)
