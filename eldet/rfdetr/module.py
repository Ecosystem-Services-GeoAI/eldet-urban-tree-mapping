"""Lightning module: RF-DETR + optional ELDET teacher / KD."""

from __future__ import annotations

import copy
import warnings
from typing import Any, Tuple

import torch
from rfdetr.training.module_model import RFDETRModelModule
from torch import nn

from eldet.rfdetr.config import ELDETRFDETRConfig
from eldet.rfdetr.kd_rf import compute_rf_detr_kd_auto
from rfdetr.config import ModelConfig, TrainConfig


class ELDETRFDETRModelModule(RFDETRModelModule):
    """Same as :class:`rfdetr.training.module_model.RFDETRModelModule` with optional ELDET distillation."""

    eldet: ELDETRFDETRConfig
    eldet_teacher: nn.Module | None
    _eldet_ca_hist: list[float]
    _eldet_gtba_hist: list[float]
    eldet_t_loc_0based: int | None
    eldet_t_cls_0based: int | None

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainConfig,
        *,
        eldet: ELDETRFDETRConfig | None = None,
    ) -> None:
        super().__init__(model_config, train_config)
        self.eldet = eldet or ELDETRFDETRConfig(enabled=False)
        if self.eldet.enabled and getattr(train_config, "use_ema", False):
            warnings.warn(
                "ELDET is enabled but TrainConfig.use_ema is True. "
                "Set use_ema=False to avoid two shadow models (see ELDETRFDETRConfig docstring).",
                stacklevel=2,
            )
        self.eldet_teacher: nn.Module | None = None  # replaced by add_module("eldet_teacher", ...) when created
        self._eldet_ca_hist = []
        self._eldet_gtba_hist = []
        self.eldet_t_loc_0based = None
        self.eldet_t_cls_0based = None
        self._eldet_kd_epoch_sum = 0.0
        self._eldet_kd_epoch_n = 0

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        if self.eldet.enabled:
            self._eldet_kd_epoch_sum = 0.0
            self._eldet_kd_epoch_n = 0

    @property
    def eldet_training_phase(self) -> str:
        """Coarse ELDET phase label (``pre_loc`` / ``post_loc_pre_cls`` / ``post_cls`` / ``disabled``)."""
        if not self.eldet.enabled:
            return "disabled"
        if self.eldet_teacher is None:
            return "pre_loc"
        if self.eldet_t_cls_0based is None:
            return "post_loc_pre_cls"
        e = int(self.current_epoch)
        if e < self.eldet_t_cls_0based:
            return "post_loc_pre_cls"
        return "post_cls"

    @property
    def eldet_state(self) -> dict[str, Any]:
        """Serializable-ish view of metric histories and transition indices (for logging / debugging)."""
        return {
            "ca_hist": list(self._eldet_ca_hist),
            "gtba_hist": list(self._eldet_gtba_hist),
            "t_loc_0based": self.eldet_t_loc_0based,
            "t_cls_0based": self.eldet_t_cls_0based,
            "phase": self.eldet_training_phase,
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        if not self.eldet.enabled:
            return
        checkpoint["eldet_series_state"] = {
            "_eldet_ca_hist": list(self._eldet_ca_hist),
            "_eldet_gtba_hist": list(self._eldet_gtba_hist),
            "eldet_t_loc_0based": self.eldet_t_loc_0based,
            "eldet_t_cls_0based": self.eldet_t_cls_0based,
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        if not self.eldet.enabled:
            return
        st = checkpoint.get("eldet_series_state")
        if not isinstance(st, dict):
            return
        self._eldet_ca_hist = list(st.get("_eldet_ca_hist", []))
        self._eldet_gtba_hist = list(st.get("_eldet_gtba_hist", []))
        self.eldet_t_loc_0based = st.get("eldet_t_loc_0based")
        self.eldet_t_cls_0based = st.get("eldet_t_cls_0based")

    def init_eldet_teacher(self) -> None:
        """Deep-copy the trainable model into a frozen teacher and register it for checkpointing.

        The copy is taken from ``self.model._orig_mod`` when present (``torch.compile`` wrapper) so the teacher is an
        **eager** module tree; avoid compiling the teacher separately. If you require compile on the student, keep
        teacher updates and KD on the same compiled graph semantics as upstream RF-DETR (``model_config.compile`` is
        opt-in and disabled under multi-scale training in the vendor module).
        """
        if self.eldet_teacher is not None:
            return
        core = getattr(self.model, "_orig_mod", self.model)
        teacher = copy.deepcopy(core).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher = teacher.to(self.device)
        # nn.Module.add_module refuses if ``eldet_teacher`` is already a plain attribute (we used None in __init__).
        if hasattr(self, "eldet_teacher"):
            delattr(self, "eldet_teacher")
        self.add_module("eldet_teacher", teacher)

    def eldet_should_use_cls_teacher_decay(self) -> bool:
        """True between ``t_loc`` (inclusive) and ``t_cls`` (exclusive) once both are known."""
        if self.eldet_t_loc_0based is None or self.eldet_teacher is None:
            return False
        e = int(self.current_epoch)
        if e < self.eldet_t_loc_0based:
            return False
        if self.eldet_t_cls_0based is not None and e >= self.eldet_t_cls_0based:
            return False
        return True

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self.eldet.enabled and not self.eldet.use_metric_phase_detection and self.eldet.teacher_start_epoch == 0:
            self.init_eldet_teacher()

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        if not self.eldet.enabled:
            return super().training_step(batch, batch_idx)

        samples, targets = batch
        batch_size = len(targets)
        outputs = self.model(samples, targets)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)

        if self.eldet_teacher is not None:
            with torch.no_grad():
                # LWDETR.forward uses only the first ``num_queries`` embedding rows when
                # ``self.training`` is False (inference path). The student runs in train
                # mode with ``num_queries * group_detr`` decoder slots, so Hungarian
                # ``src_idx`` from the criterion can exceed the teacher's sequence length.
                # Temporarily set train mode for this forward so teacher tensors align.
                prev_training = self.eldet_teacher.training
                try:
                    self.eldet_teacher.train()
                    out_t = self.eldet_teacher(samples, targets)
                finally:
                    self.eldet_teacher.train(prev_training)
            kd = compute_rf_detr_kd_auto(
                self.criterion,
                outputs,
                out_t,
                lambda_cls=self.eldet.lambda_kd_cls,
                lambda_box=self.eldet.lambda_kd_box,
                temperature=self.eldet.kd_temperature,
                use_matcher_indices=self.eldet.kd_use_matcher_indices,
            )
            self._eldet_kd_epoch_sum += float(kd.detach())
            self._eldet_kd_epoch_n += 1
            loss = loss + self.eldet.lambda_kd * kd
            self.log(
                "train/eldet_kd",
                kd.detach(),
                on_step=bool(self.train_config.train_log_on_step),
                on_epoch=True,
                sync_dist=bool(self.train_config.train_log_sync_dist),
                batch_size=batch_size,
            )

        loss_scaled = loss / self.trainer.accumulate_grad_batches
        train_log_sync_dist = bool(self.train_config.train_log_sync_dist)
        train_log_on_step = bool(self.train_config.train_log_on_step)
        self.log_dict(
            {f"train/{k}": v for k, v in loss_dict.items()},
            on_step=train_log_on_step,
            on_epoch=True,
            sync_dist=train_log_sync_dist,
            batch_size=batch_size,
        )
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=train_log_on_step,
            on_epoch=True,
            sync_dist=train_log_sync_dist,
            batch_size=batch_size,
        )
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        group_lrs = [pg["lr"] for pg in optimizer.param_groups if "lr" in pg]
        if group_lrs:
            base_lr = group_lrs[0]
            min_lr = min(group_lrs)
            max_lr = max(group_lrs)
            self.log("train/lr", base_lr, prog_bar=True, on_step=True, on_epoch=False)
            self.log("train/lr_min", min_lr, prog_bar=True, on_step=True, on_epoch=False)
            self.log("train/lr_max", max_lr, prog_bar=True, on_step=True, on_epoch=False)
        return loss_scaled
