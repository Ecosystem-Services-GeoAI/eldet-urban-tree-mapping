"""ELDET options for RF-DETR / PyTorch Lightning training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ELDETRFDETRConfig:
    """Optional ELDET behaviour on top of :class:`rfdetr.training.module_model.RFDETRModelModule`.

    When ``enabled`` is False, the Lightning module behaves like the stock RF-DETR module.

    **Teacher schedule**

    - If ``use_metric_phase_detection`` is True (default), ``ELDETPhaseCallback`` appends CA/GTBA scores each epoch
      and sets ``t_loc`` / ``t_cls`` from :mod:`eldet.transition` on the running series (rank 0).
    - If False, set ``teacher_start_epoch`` to a **0-based** epoch index; the teacher is created at the **end** of
      epoch ``teacher_start_epoch - 1`` (i.e. before the first batch of epoch ``teacher_start_epoch``), or at end of
      epoch 0 when ``teacher_start_epoch == 0``.

    **EMA conflict:** set ``TrainConfig.use_ema=False`` when ELDET is enabled (stock RF-DETR EMA is incompatible with
    asymmetric teacher decay).

    **CSV / resume (Phase 4):** ``series_csv_path`` appends epoch-level CA/GTBA and transition columns (rank 0).
    ``freeze_transitions_after_cls`` (default True) stops extending memorization histories after ``t_cls`` is set.
    Lightning checkpoints also store ``eldet_series_state`` (histories + ``t_loc`` / ``t_cls``) via the ELDET module
    hooks.

    **Single-class:** ``single_class=None`` (default) auto-collapses ``t_cls`` to ``t_loc`` when
    ``model_config.num_classes == 1`` so GTBA memorization is not used for a non-existent categorization phase.
    Set ``single_class=False`` on multi-class runs. ``single_class=True`` forces that collapse whenever ``t_loc`` is set.

    **Synthetic noise fields** (``noise_*``): reserved for optional ``eldet/noise.py`` integration; keep at ``0`` when
    training on **already-noisy** dataset labels (default for this workspace).

    **KD:** ``kd_use_matcher_indices`` (default True) distills on the same matched queries as the detection loss,
    using indices stashed on ``SetCriterion`` (see vendored ``rfdetr/models/criterion.py`` ELDET patch). Set False for
    legacy dense KD over all queries.

    **Stripped ``.pth`` exports:** Lightning still writes ``*.ckpt`` for resume. Stock ``BestModelCallback`` only
    snapshots **student** weights into ``checkpoint_best_regular.pth``. When ``export_teacher_pth`` is True (default),
    :class:`~eldet.rfdetr.pth_export_callback.ELDETStrippedPthExportCallback` also writes ``last_teacher.pth`` each
    epoch (once a teacher exists) and ``checkpoint_best_teacher.pth`` when ``val/mAP_50_95`` improves — same payload
    layout as upstream RF-DETR (``model``, ``args``, ``epoch``, …). There is no ``checkpoint_best_ema.pth`` while
    ``use_ema=False``; the ELDET teacher is the slow-weight track. Set ``export_student_last_pth`` to also mirror the
    student into ``last_student.pth`` each epoch without opening ``last.ckpt``.
    """

    single_class: bool | None = None  # None: auto when num_classes==1; True/False: force on/off
    enabled: bool = False
    lambda_kd: float = 1.0
    lambda_kd_cls: float = 1.0
    lambda_kd_box: float = 1.0
    kd_temperature: float = 1.0
    kd_use_matcher_indices: bool = True  # True: KD on Hungarian pairs via criterion._eldet_last_indices
    teacher_decay: float = 0.999
    teacher_cls_decay: float = 0.1
    use_metric_phase_detection: bool = True
    teacher_start_epoch: int | None = None
    transition_gamma: float = 0.9
    gtba_tau: float = 0.1
    metric_max_train_batches: int = 32
    metric_every_n_epochs: int = 1
    noise_loc_fraction: float = 0.0
    noise_cat_fraction: float = 0.0
    noise_epsilon: float = 0.5
    num_classes: int | None = None  # required if noise_cat_fraction > 0
    series_csv_path: str | None = None  # append epoch CA/GTBA / transitions (rank 0)
    freeze_transitions_after_cls: bool = True  # stop updating t_loc/t_cls from metrics once t_cls is set
    export_teacher_pth: bool = True  # last_teacher.pth + checkpoint_best_teacher.pth (RF-DETR stripped format)
    export_student_last_pth: bool = False  # last_student.pth each epoch (student-only; best student stays upstream .pth)
