"""Strip RF-DETR-compatible ``.pth`` exports for ELDET (teacher + optional student).

PyTorch Lightning's stock ``ModelCheckpoint`` callbacks in ``rfdetr.training.trainer`` still write
``*.ckpt`` for full-module resume (optimizer, AMP scaler, callback state, etc.). Upstream
:class:`~rfdetr.training.callbacks.best_model.BestModelCallback` writes stripped ``checkpoint_best_regular.pth``
from **student** weights only; it does not include ``eldet_teacher``. With ``TrainConfig.use_ema=False`` (required
for ELDET), ``checkpoint_best_ema.pth`` is never produced — the ELDET teacher is the slow-weight track.

This callback writes teacher (and optionally student) weights in the same payload shape as ``BestModelCallback``
(``model``, ``args``, ``epoch``, …) so ``rfdetr`` validation / eval loaders that expect ``.pth`` keep working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

from eldet.rfdetr.config import ELDETRFDETRConfig


def _output_dir(pl_module: LightningModule) -> Path:
    tc = getattr(pl_module, "train_config", None)
    out = getattr(tc, "output_dir", None) if tc is not None else None
    if isinstance(out, str) and out:
        return Path(out)
    return Path(".")


def _unwrap_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    core = getattr(module, "_orig_mod", None)
    raw: torch.nn.Module = core if isinstance(core, torch.nn.Module) else module
    return raw.state_dict()


def _build_pth_payload(
    *,
    model_state_dict: dict[str, torch.Tensor],
    pl_module: LightningModule,
    trainer: Trainer,
    model_name: str | None,
) -> dict[str, Any]:
    from rfdetr.training.callbacks.best_model import BestModelCallback

    train_config = pl_module.train_config
    dataset_class_names = getattr(trainer.datamodule, "class_names", None)
    if (
        dataset_class_names is not None
        and hasattr(train_config, "model_copy")
        and getattr(train_config, "class_names", None) is None
    ):
        train_config = train_config.model_copy(update={"class_names": dataset_class_names})
    args_dict = train_config.model_dump() if hasattr(train_config, "model_dump") else train_config
    return BestModelCallback._build_checkpoint_payload(
        model_state_dict, args_dict, trainer, model_name=model_name
    )


class ELDETStrippedPthExportCallback(Callback):
    """Write ``last_teacher.pth``, ``checkpoint_best_teacher.pth``, and optional ``last_student.pth``.

    Rank 0 only. Skips sanity-check and non-global-zero ranks.
    """

    def __init__(self, eldet: ELDETRFDETRConfig) -> None:
        super().__init__()
        self._eldet = eldet
        self._best_teacher_map: float | None = None

    def _maybe_save_student_last(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self._eldet.export_student_last_pth:
            return
        from rfdetr.training.callbacks.best_model import BestModelCallback

        out = _output_dir(pl_module)
        out.mkdir(parents=True, exist_ok=True)
        sd = _unwrap_state_dict(pl_module.model)
        model_name = BestModelCallback._resolve_model_name(pl_module)
        payload = _build_pth_payload(
            model_state_dict=sd,
            pl_module=pl_module,
            trainer=trainer,
            model_name=model_name,
        )
        torch.save(payload, out / "last_student.pth")

    def _maybe_save_teacher_last_and_best_on_val(
        self, trainer: Trainer, pl_module: LightningModule, *, is_val_end: bool
    ) -> None:
        if not self._eldet.export_teacher_pth:
            return
        teacher = getattr(pl_module, "eldet_teacher", None)
        if teacher is None:
            return
        from rfdetr.training.callbacks.best_model import BestModelCallback

        out = _output_dir(pl_module)
        out.mkdir(parents=True, exist_ok=True)
        sd = _unwrap_state_dict(teacher)
        model_name = BestModelCallback._resolve_model_name(pl_module)
        payload = _build_pth_payload(
            model_state_dict=sd,
            pl_module=pl_module,
            trainer=trainer,
            model_name=model_name,
        )
        torch.save(payload, out / "last_teacher.pth")

        if not is_val_end:
            return
        if getattr(trainer, "sanity_checking", False):
            return
        key = "val/mAP_50_95"
        metrics = getattr(trainer, "callback_metrics", {}) or {}
        if key not in metrics:
            return
        score = float(metrics[key].detach().cpu().item())
        if self._best_teacher_map is None or score > self._best_teacher_map:
            self._best_teacher_map = score
            torch.save(payload, out / "checkpoint_best_teacher.pth")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self._eldet.enabled or not getattr(trainer, "is_global_zero", True):
            return
        if getattr(trainer, "sanity_checking", False):
            return
        self._maybe_save_student_last(trainer, pl_module)
        self._maybe_save_teacher_last_and_best_on_val(trainer, pl_module, is_val_end=False)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self._eldet.enabled or not getattr(trainer, "is_global_zero", True):
            return
        if getattr(trainer, "sanity_checking", False):
            return
        self._maybe_save_teacher_last_and_best_on_val(trainer, pl_module, is_val_end=True)
