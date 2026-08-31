"""ELDET options for Ultralytics YOLO :class:`DetectionTrainer` subclasses."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass
class ELDETYoloConfig:
    """Optional ELDET behaviour on top of stock detection training.

    Pass via ``overrides`` as ``eldet=ELDETYoloConfig(...)`` to :class:`eldet.yolo.trainer.ELDETDetectionTrainer`.

    **Stock Ultralytics EMA:** the trainer replaces the default :class:`~ultralytics.utils.torch_utils.ModelEMA` shadow
    model with a lightweight shim (validation uses the **student**). Distillation uses ``eldet_teacher`` only.

    **Teacher schedule:** same semantics as ``ELDETRFDETRConfig`` (see :mod:`eldet.rfdetr.config`): metric-driven
    transitions vs fixed ``teacher_start_epoch`` (0-based).

    **YOLO26 / end2end:** KD and CA/GTBA probes use the ``one2many`` branch (same anchor grid as v8 ``Detect``).

    **Phase 4 logging / resume:** ``series_csv_path`` appends per-epoch probe rows (rank 0).
    ``freeze_transitions_after_cls`` stops extending CA/GTBA memorization histories once ``t_cls`` is detected
    (defaults to True).

    **Single-class:** ``single_class=None`` (default) auto-collapses ``t_cls`` to ``t_loc`` when the YOLO head
    ``nc == 1``. Set ``single_class=False`` for multi-class. ``single_class=True`` forces collapse whenever ``t_loc`` is set.

    **Synthetic noise fields** (``noise_*``): reserved for optional ``eldet/noise.py`` wiring; keep at ``0`` for
    naturally noisy datasets.
    """

    single_class: bool | None = None  # None: auto when Detect nc==1; True/False: force on/off
    enabled: bool = False
    lambda_kd: float = 1.0
    lambda_kd_cls: float = 1.0
    lambda_kd_box: float = 1.0
    kd_temperature: float = 1.0
    teacher_decay: float = 0.999
    teacher_cls_decay: float = 0.1
    use_metric_phase_detection: bool = True
    teacher_start_epoch: int | None = None
    transition_gamma: float = 0.9
    gtba_tau: float = 0.1
    metric_max_train_batches: int = 32
    metric_every_n_epochs: int = 1
    metric_topk_anchors: int = 200
    noise_loc_fraction: float = 0.0
    noise_cat_fraction: float = 0.0
    noise_epsilon: float = 0.5
    num_classes: int | None = None  # required if noise_cat_fraction > 0
    series_csv_path: str | None = None
    freeze_transitions_after_cls: bool = True


def coerce_eldet_yolo_config(raw: Any) -> ELDETYoloConfig:
    """Accept ``None``, :class:`ELDETYoloConfig`, or a dict of field overrides (for YAML/CLI bridges)."""
    if raw is None:
        return ELDETYoloConfig(enabled=False)
    if isinstance(raw, ELDETYoloConfig):
        return raw
    if isinstance(raw, dict):
        valid = {f.name for f in fields(ELDETYoloConfig)}
        cfg = ELDETYoloConfig(**{k: v for k, v in raw.items() if k in valid})
        if cfg.noise_cat_fraction > 0 and cfg.num_classes is None:
            raise ValueError("ELDETYoloConfig.num_classes is required when noise_cat_fraction > 0")
        return cfg
    raise TypeError(f"eldet override must be ELDETYoloConfig or dict, got {type(raw)!r}")
