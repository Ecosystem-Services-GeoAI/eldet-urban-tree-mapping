"""RF-DETR + ELDET training configuration guards (Phase 2.3–2.5)."""

from __future__ import annotations

from typing import Any

from eldet.rfdetr.config import ELDETRFDETRConfig


def eldet_rfdetr_training_issues(train_config: Any, eldet: ELDETRFDETRConfig) -> list[str]:
    """Return human-readable configuration problems (empty if OK).

    Args:
        train_config: :class:`rfdetr.config.TrainConfig` (duck-typed: ``use_ema`` attribute).
        eldet: ELDET RF-DETR options.
    """
    issues: list[str] = []
    if not eldet.enabled:
        return issues
    if bool(getattr(train_config, "use_ema", False)):
        issues.append(
            "TrainConfig.use_ema is True while ELDET is enabled. "
            "Set use_ema=False so RF-DETR's RFDETREMACallback is not registered alongside the ELDET teacher "
            "(two incompatible shadow-model mechanisms)."
        )
    return issues


def ensure_eldet_rfdetr_training_config(train_config: Any, eldet: ELDETRFDETRConfig, *, strict: bool = True) -> None:
    """Raise ``ValueError`` if ``strict`` and :func:`eldet_rfdetr_training_issues` is non-empty."""
    issues = eldet_rfdetr_training_issues(train_config, eldet)
    if strict and issues:
        raise ValueError(" ".join(issues))
