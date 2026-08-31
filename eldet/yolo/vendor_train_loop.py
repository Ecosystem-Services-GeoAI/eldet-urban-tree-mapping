# Vendored from ultralytics/engine/trainer.py BaseTrainer._do_train (≈ lines 376–589), with ELDET forward path.
# When updating Ultralytics, re-sync this loop with BaseTrainer._do_train.

from __future__ import annotations

import math
import time
import warnings

import numpy as np
import torch
from torch import distributed as dist
from ultralytics.utils import LOGGER, RANK, TQDM, colorstr
from ultralytics.utils.torch_utils import autocast, unset_deterministic, unwrap_model


def eldet_yolo_do_train(trainer) -> None:
    """Training loop (same control flow as Ultralytics ``BaseTrainer._do_train``) with ELDET KD forward path."""
    if trainer.world_size > 1:
        trainer._setup_ddp()
    trainer._setup_train()

    nb = len(trainer.train_loader)
    nw = max(round(trainer.args.warmup_epochs * nb), 100) if trainer.args.warmup_epochs > 0 else -1
    last_opt_step = -1
    trainer.epoch_time = None
    trainer.epoch_time_start = time.time()
    trainer.train_time_start = time.time()
    trainer.run_callbacks("on_train_start")
    LOGGER.info(
        f"Image sizes {trainer.args.imgsz} train, {trainer.args.imgsz} val\n"
        f"Using {trainer.train_loader.num_workers * (trainer.world_size or 1)} dataloader workers\n"
        f"Logging results to {colorstr('bold', trainer.save_dir)}\n"
        f"Starting training for "
        + (f"{trainer.args.time} hours..." if trainer.args.time else f"{trainer.epochs} epochs...")
    )
    if trainer.args.close_mosaic:
        base_idx = (trainer.epochs - trainer.args.close_mosaic) * nb
        trainer.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
    epoch = trainer.start_epoch
    trainer.optimizer.zero_grad()
    trainer._oom_retries = 0
    while True:
        trainer.epoch = epoch
        trainer.run_callbacks("on_train_epoch_start")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.scheduler.step()

        trainer._model_train()
        if RANK != -1:
            trainer.train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(trainer.train_loader)
        if epoch == (trainer.epochs - trainer.args.close_mosaic):
            trainer._close_dataloader_mosaic()
            trainer.train_loader.reset()

        if RANK in {-1, 0}:
            LOGGER.info(trainer.progress_string())
            pbar = TQDM(enumerate(trainer.train_loader), total=nb)
        trainer.tloss = None
        for i, batch in pbar:
            trainer.run_callbacks("on_train_batch_start")
            ni = i + nb * epoch
            if ni <= nw:
                xi = [0, nw]
                trainer.accumulate = max(1, int(np.interp(ni, xi, [1, trainer.args.nbs / trainer.batch_size]).round()))
                for x in trainer.optimizer.param_groups:
                    x["lr"] = np.interp(
                        ni,
                        xi,
                        [
                            trainer.args.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                            x["initial_lr"] * trainer.lf(epoch),
                        ],
                    )
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [trainer.args.warmup_momentum, trainer.args.momentum])

            try:
                with autocast(trainer.amp):
                    batch = trainer.preprocess_batch(batch)
                    use_split = bool(trainer.args.compile) or (
                        getattr(trainer, "eldet", None) is not None and trainer.eldet.enabled
                    )
                    if use_split:
                        preds = trainer.model(batch["img"])
                        um = unwrap_model(trainer.model)
                        loss, trainer.loss_items = um.loss(batch, preds)
                        if trainer.eldet.enabled and getattr(trainer, "eldet_teacher", None) is not None:
                            from eldet.yolo.kd_yolo import compute_yolo_kd

                            with torch.no_grad():
                                preds_t = trainer.eldet_teacher(batch["img"])
                            kd = compute_yolo_kd(
                                preds,
                                preds_t,
                                um.criterion,
                                lambda_cls=trainer.eldet.lambda_kd_cls,
                                lambda_box=trainer.eldet.lambda_kd_box,
                                temperature=trainer.eldet.kd_temperature,
                            )
                            trainer._eldet_kd_epoch_sum += float(kd.detach())
                            trainer._eldet_kd_epoch_count += 1
                            loss = loss + trainer.eldet.lambda_kd * kd
                    else:
                        loss, trainer.loss_items = trainer.model(batch)
                    trainer.loss = loss.sum()
                    if RANK != -1:
                        trainer.loss *= trainer.world_size
                    trainer.tloss = (
                        trainer.loss_items
                        if trainer.tloss is None
                        else (trainer.tloss * i + trainer.loss_items) / (i + 1)
                    )

                trainer.scaler.scale(trainer.loss).backward()
            except RuntimeError as e:
                is_oom = isinstance(e, torch.cuda.OutOfMemoryError)
                bad_engine = ("CUDNN_STATUS_INTERNAL_ERROR", "unable to find an engine")
                if not is_oom and not any(s in str(e) for s in bad_engine):
                    raise
                if epoch > trainer.start_epoch or trainer._oom_retries >= 3 or RANK != -1:
                    raise
                trainer._oom_retries += 1
                old_batch = trainer.batch_size
                trainer.args.batch = trainer.batch_size = max(trainer.batch_size // 2, 1)
                LOGGER.warning(
                    f"{'CUDA out of memory' if is_oom else 'CUDA backend memory error'} with batch={old_batch}. "
                    f"Reducing to batch={trainer.batch_size} and retrying ({trainer._oom_retries}/3)."
                )
                batch = loss = preds = None
                trainer.loss = trainer.loss_items = trainer.tloss = None
                trainer._clear_memory()
                trainer._build_train_pipeline()
                trainer.scheduler.last_epoch = trainer.start_epoch - 1
                nb = len(trainer.train_loader)
                nw = max(round(trainer.args.warmup_epochs * nb), 100) if trainer.args.warmup_epochs > 0 else -1
                last_opt_step = -1
                trainer.optimizer.zero_grad()
                break
            if ni - last_opt_step >= trainer.accumulate:
                trainer.optimizer_step()
                last_opt_step = ni

                if trainer.args.time:
                    trainer.stop = (time.time() - trainer.train_time_start) > (trainer.args.time * 3600)
                    if RANK != -1:
                        broadcast_list = [trainer.stop if RANK == 0 else None]
                        dist.broadcast_object_list(broadcast_list, 0)
                        trainer.stop = broadcast_list[0]
                    if trainer.stop:
                        break

            if RANK in {-1, 0}:
                loss_length = trainer.tloss.shape[0] if len(trainer.tloss.shape) else 1
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                    % (
                        f"{epoch + 1}/{trainer.epochs}",
                        f"{trainer._get_memory():.3g}G",
                        *(trainer.tloss if loss_length > 1 else torch.unsqueeze(trainer.tloss, 0)),
                        batch["img"].shape[0],
                        batch["img"].shape[-1],
                    )
                )
                trainer.run_callbacks("on_batch_end")
                if trainer.args.plots and ni in trainer.plot_idx:
                    trainer.plot_training_samples(batch, ni)

            trainer.run_callbacks("on_train_batch_end")
            if trainer.stop:
                break
        else:
            trainer._oom_retries = 0

        if trainer._oom_retries and not trainer.stop:
            continue

        if hasattr(unwrap_model(trainer.model).criterion, "update"):
            unwrap_model(trainer.model).criterion.update()

        trainer.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(trainer.optimizer.param_groups)}

        trainer.run_callbacks("on_train_epoch_end")
        if RANK in {-1, 0} and trainer.ema is not None:
            trainer.ema.update_attr(trainer.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

        final_epoch = epoch + 1 >= trainer.epochs
        if trainer.args.val or final_epoch or trainer.stopper.possible_stop or trainer.stop:
            trainer._clear_memory(None if trainer.device.type == "mps" else 0.5)
            trainer.metrics, trainer.fitness = trainer.validate()

        if trainer._handle_nan_recovery(epoch):
            continue

        trainer.nan_recovery_attempts = 0
        if RANK in {-1, 0}:
            metrics = {**trainer.label_loss_items(trainer.tloss), **trainer.metrics, **trainer.lr}
            n_kd = int(getattr(trainer, "_eldet_kd_epoch_count", 0))
            if getattr(trainer, "eldet", None) is not None and trainer.eldet.enabled and n_kd > 0:
                metrics["eldet_kd"] = trainer._eldet_kd_epoch_sum / n_kd
            trainer.save_metrics(metrics=metrics)
            trainer.stop |= trainer.stopper(epoch + 1, trainer.fitness) or final_epoch
            if trainer.args.time:
                trainer.stop |= (time.time() - trainer.train_time_start) > (trainer.args.time * 3600)

            if (trainer.args.save or final_epoch) and trainer.save_model():
                trainer.run_callbacks("on_model_save")

        t = time.time()
        trainer.epoch_time = t - trainer.epoch_time_start
        trainer.epoch_time_start = t
        if trainer.args.time:
            mean_epoch_time = (t - trainer.train_time_start) / (epoch - trainer.start_epoch + 1)
            trainer.epochs = trainer.args.epochs = math.ceil(trainer.args.time * 3600 / mean_epoch_time)
            trainer._setup_scheduler()
            trainer.scheduler.last_epoch = trainer.epoch
            trainer.stop |= epoch >= trainer.epochs
        trainer.run_callbacks("on_fit_epoch_end")
        trainer._clear_memory(None if trainer.device.type == "mps" else 0.5)

        if RANK != -1:
            broadcast_list = [trainer.stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)
            trainer.stop = broadcast_list[0]
        if trainer.stop:
            break
        epoch += 1

    seconds = time.time() - trainer.train_time_start
    LOGGER.info(f"\n{epoch - trainer.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
    trainer.final_eval()
    if RANK in {-1, 0}:
        if trainer.args.plots:
            trainer.plot_metrics()
        trainer.run_callbacks("on_train_end")
    trainer._clear_memory()
    unset_deterministic()
    trainer.run_callbacks("teardown")
