"""Pin ``t_cls = t_loc`` for single-class ELDET so GTBA memorization is not relied on.

With ``num_classes == 1``, GTBA is ~degenerate (pred/GT class indices almost always match when IoU ≥ τ), so
``find_memorization_epoch`` on the GTBA series rarely yields a meaningful ``t_cls``. Without ``t_cls``, the code
would keep using fast classification EMA on the teacher for all epochs after ``t_loc``. Collapsing
``t_cls`` to ``t_loc`` ends that window immediately (no asymmetric cls-only teacher decay), matching the idea
that there is no separate categorization memorization phase.
"""

from __future__ import annotations

from typing import Any


def should_collapse_t_cls(*, single_class: bool | None, num_classes: int | None) -> bool:
    """Whether to force ``t_cls = t_loc`` once ``t_loc`` is known.

    ``single_class``:
        ``True`` — always collapse when ``t_loc`` is set (ignore ``num_classes``).
        ``False`` — never auto-collapse.
        ``None`` — collapse when ``num_classes == 1``.
    """
    if single_class is True:
        return True
    if single_class is False:
        return False
    return num_classes == 1


def apply_single_class_t_cls_collapse_rfdetr(pl_module: Any, eldet: Any) -> None:
    """If single-class mode, set ``pl_module.eldet_t_cls_0based = eldet_t_loc_0based`` when ``t_loc`` is set."""
    mc = getattr(pl_module, "model_config", None)
    nc = getattr(mc, "num_classes", None) if mc is not None else None
    if isinstance(nc, (float, str)):
        try:
            nc = int(nc)
        except (TypeError, ValueError):
            nc = None
    sc = getattr(eldet, "single_class", None)
    if not should_collapse_t_cls(single_class=sc, num_classes=nc):
        return
    t_loc = getattr(pl_module, "eldet_t_loc_0based", None)
    if t_loc is None:
        return
    pl_module.eldet_t_cls_0based = int(t_loc)


def apply_single_class_t_cls_collapse_yolo(trainer: Any) -> None:
    """If single-class mode, set ``trainer.eldet_t_cls_0based = eldet_t_loc_0based`` when ``t_loc`` is set."""
    eldet = getattr(trainer, "eldet", None)
    if eldet is None:
        return
    model = getattr(trainer, "model", None)
    if model is None:
        return
    try:
        from ultralytics.utils.torch_utils import unwrap_model
    except ImportError:
        return
    um = unwrap_model(model)
    nc: int | None = None
    if hasattr(um, "model"):
        head = um.model[-1]
        raw = getattr(head, "nc", None)
        if raw is not None:
            try:
                nc = int(raw)
            except (TypeError, ValueError):
                nc = None
    sc = getattr(eldet, "single_class", None)
    if not should_collapse_t_cls(single_class=sc, num_classes=nc):
        return
    t_loc = getattr(trainer, "eldet_t_loc_0based", None)
    if t_loc is None:
        return
    trainer.eldet_t_cls_0based = int(t_loc)
