"""YOLO + ELDET configuration guards (Phase 3)."""

from __future__ import annotations

from eldet.yolo.config import ELDETYoloConfig


def eldet_yolo_config_issues(eldet: ELDETYoloConfig) -> list[str]:
    """Return human-readable problems with ``ELDETYoloConfig`` (empty if OK)."""
    issues: list[str] = []
    if not eldet.enabled:
        return issues
    if eldet.noise_cat_fraction > 0 and eldet.num_classes is None:
        issues.append(
            "ELDETYoloConfig.num_classes is required when noise_cat_fraction > 0 (optional synthetic-noise path)."
        )
    return issues


def ensure_eldet_yolo_config(eldet: ELDETYoloConfig, *, strict: bool = True) -> None:
    """Raise ``ValueError`` if ``strict`` and :func:`eldet_yolo_config_issues` is non-empty."""
    issues = eldet_yolo_config_issues(eldet)
    if strict and issues:
        raise ValueError(" ".join(issues))
