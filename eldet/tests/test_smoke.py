"""Smoke tests for Phase 0 packaging."""

from eldet import (
    __version__,
    apply_localization_noise_xyxy,
    ca_localization_recall_at_iou,
    find_memorization_epoch,
    update_teacher_from_student,
)


def test_version_is_string():
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_phase1_exports_importable():
    assert callable(apply_localization_noise_xyxy)
    assert callable(ca_localization_recall_at_iou)
    assert callable(find_memorization_epoch)
    assert callable(update_teacher_from_student)
