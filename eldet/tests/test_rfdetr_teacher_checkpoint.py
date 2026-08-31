"""ELDET teacher registered for Lightning checkpoints (Phase 2.13)."""

from __future__ import annotations

import pytest

pytest.importorskip("rfdetr")

from eldet.rfdetr.config import ELDETRFDETRConfig
from eldet.rfdetr.module import ELDETRFDETRModelModule
from rfdetr.config import RFDETRNanoConfig, TrainConfig


def test_init_eldet_teacher_state_dict_contains_teacher_weights() -> None:
    mc = RFDETRNanoConfig(num_classes=2, pretrain_weights=None)
    tc = TrainConfig(
        dataset_dir=".",
        output_dir=".",
        use_ema=False,
        tensorboard=False,
        wandb=False,
        mlflow=False,
        clearml=False,
        epochs=1,
        batch_size=1,
        num_workers=0,
        devices=1,
    )
    m = ELDETRFDETRModelModule(mc, tc, eldet=ELDETRFDETRConfig(enabled=True))
    assert m.eldet_teacher is None
    m.init_eldet_teacher()
    assert m.eldet_teacher is not None
    sd = m.state_dict()
    assert any(k.startswith("eldet_teacher.") for k in sd)
