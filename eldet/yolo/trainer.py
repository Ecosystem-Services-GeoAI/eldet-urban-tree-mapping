"""Ultralytics YOLO detection trainer with optional ELDET (KD + teacher EMA + phase hooks)."""

from __future__ import annotations

import io
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from ultralytics import __version__
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, GIT, LOGGER
from ultralytics.utils.torch_utils import (
    convert_optimizer_state_dict_to_fp16,
    copy_attr,
    unwrap_model,
)

from eldet.ema_policy import update_teacher_from_student, yolo_detect_cls_param
from eldet.yolo.config import ELDETYoloConfig, coerce_eldet_yolo_config
from eldet.yolo.phase import (
    eldet_yolo_init_teacher,
    eldet_yolo_on_train_epoch_end,
    eldet_yolo_on_train_epoch_start,
    eldet_yolo_should_use_cls_teacher_decay,
)
from eldet.yolo.train_checks import ensure_eldet_yolo_config
from eldet.yolo.vendor_train_loop import eldet_yolo_do_train


class _StudentAsEMAShim:
    """Validator and checkpoint code expect ``trainer.ema.ema``; point that at the live student (no shadow ModelEMA)."""

    __slots__ = ("ema", "enabled", "updates")

    def __init__(self, model: nn.Module) -> None:
        self.ema = unwrap_model(model)
        self.updates = 0
        self.enabled = False

    def update(self, model: nn.Module) -> None:
        del model

    def update_attr(self, model: nn.Module, include: tuple = (), exclude: tuple = ("process_group", "reducer")) -> None:
        copy_attr(self.ema, unwrap_model(model), include, exclude)


class ELDETDetectionTrainer(DetectionTrainer):
    """:class:`DetectionTrainer` with ELDET distillation (v8 / end2end ``Detect`` heads).

    Usage::

        from eldet.yolo import ELDETDetectionTrainer, ELDETYoloConfig
        trainer = ELDETDetectionTrainer(overrides=dict(
            model=\"yolo26n.pt\", data=\"coco8.yaml\", epochs=3,
            eldet=ELDETYoloConfig(enabled=True),
        ))
        trainer.train()
    """

    eldet: ELDETYoloConfig
    eldet_teacher: nn.Module | None

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        overrides = dict(overrides or {})
        eldet_raw = overrides.pop("eldet", None)
        super().__init__(cfg, overrides, _callbacks)
        self.eldet = coerce_eldet_yolo_config(eldet_raw)
        ensure_eldet_yolo_config(self.eldet, strict=True)
        self.eldet_teacher = None
        self._eldet_ca_hist: list[float] = []
        self._eldet_gtba_hist: list[float] = []
        self.eldet_t_loc_0based: int | None = None
        self.eldet_t_cls_0based: int | None = None
        self._eldet_kd_epoch_sum = 0.0
        self._eldet_kd_epoch_count = 0
        if self.eldet.enabled:
            self.callbacks["on_train_epoch_start"].append(eldet_yolo_on_train_epoch_start)
            self.callbacks["on_train_epoch_end"].append(eldet_yolo_on_train_epoch_end)

    def _setup_train(self):
        super()._setup_train()
        if not self.eldet.enabled:
            return
        if self.ema is not None:
            del self.ema
        self.ema = _StudentAsEMAShim(self.model)
        LOGGER.info("ELDET: replaced Ultralytics ModelEMA with student shim (no duplicate shadow weights).")

    def _do_train(self):
        if self.eldet.enabled:
            return eldet_yolo_do_train(self)
        return super()._do_train()

    def optimizer_step(self):
        super().optimizer_step()
        if not self.eldet.enabled or self.eldet_teacher is None:
            return
        cls_decay = self.eldet.teacher_cls_decay if eldet_yolo_should_use_cls_teacher_decay(self) else None
        update_teacher_from_student(
            self.eldet_teacher,
            unwrap_model(self.model),
            global_decay=self.eldet.teacher_decay,
            cls_decay=cls_decay,
            is_cls_param=yolo_detect_cls_param if cls_decay is not None else None,
            ema_buffers=True,
        )

    def _eldet_write_student_checkpoint(self) -> bool:
        """Save checkpoints from the live student (``_StudentAsEMAShim``).

        Ultralytics ``save_model`` skips the write when any EMA tensor is non-finite. With the ELDET shim the EMA *is*
        the training student, so that gate can spuriously block every save; clamp instead of skipping.
        """
        self.ema.ema = unwrap_model(self.model)
        ema = deepcopy(self.ema.ema).half()
        for v in ema.state_dict().values():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                torch.nan_to_num_(v, nan=0.0, posinf=0.0, neginf=0.0)

        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,
                "ema": ema,
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "scaler": self.scaler.state_dict(),
                "train_args": vars(self.args),
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "git": {
                    "root": str(GIT.root),
                    "branch": GIT.branch,
                    "commit": GIT.commit,
                    "message": GIT.message,
                    "origin": GIT.origin,
                },
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()
        self.wdir.mkdir(parents=True, exist_ok=True)
        self.last.write_bytes(serialized_ckpt)
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)
        return True

    def save_model(self):
        """Persist ELDET state into checkpoint files after Ultralytics writes them.

        Augments ``last.pt``, ``best.pt``, and periodic ``epoch*.pt`` with teacher weights and phase metadata.
        """
        if self.eldet.enabled and isinstance(self.ema, _StudentAsEMAShim):
            ok = self._eldet_write_student_checkpoint()
        else:
            ok = super().save_model()
        if ok is False or not self.eldet.enabled or self.eldet_teacher is None:
            return ok
        paths: list[Path] = [Path(self.last)]
        if self.best is not None:
            paths.append(Path(self.best))
        if self.save_period > 0 and (self.epoch % self.save_period == 0):
            paths.append(self.wdir / f"epoch{self.epoch}.pt")
        seen: set[str] = set()
        for path in paths:
            key = str(path.resolve())
            if key in seen or not path.is_file():
                continue
            seen.add(key)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(ckpt, dict):
                continue
            ckpt["eldet_teacher"] = self.eldet_teacher.state_dict()
            ckpt["eldet_ca_hist"] = list(self._eldet_ca_hist)
            ckpt["eldet_gtba_hist"] = list(self._eldet_gtba_hist)
            ckpt["eldet_t_loc_0based"] = self.eldet_t_loc_0based
            ckpt["eldet_t_cls_0based"] = self.eldet_t_cls_0based
            torch.save(ckpt, path)
        return ok

    def _load_checkpoint_state(self, ckpt):
        """Restore ELDET state after Ultralytics loads weights (and may recreate ``ModelEMA``)."""
        super()._load_checkpoint_state(ckpt)
        if not self.eldet.enabled:
            return
        if not isinstance(ckpt, dict):
            return
        if self.ema is not None and not isinstance(self.ema, _StudentAsEMAShim):
            del self.ema
            self.ema = _StudentAsEMAShim(self.model)
            LOGGER.info("ELDET: restored student EMA shim after resume (replaced Ultralytics ModelEMA).")
        if "eldet_teacher" in ckpt:
            if self.eldet_teacher is None:
                eldet_yolo_init_teacher(self)
            self.eldet_teacher.load_state_dict(ckpt["eldet_teacher"], strict=False)
        self._eldet_ca_hist = list(ckpt.get("eldet_ca_hist", []))
        self._eldet_gtba_hist = list(ckpt.get("eldet_gtba_hist", []))
        self.eldet_t_loc_0based = ckpt.get("eldet_t_loc_0based")
        self.eldet_t_cls_0based = ckpt.get("eldet_t_cls_0based")
