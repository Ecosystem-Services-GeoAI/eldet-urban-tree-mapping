#!/usr/bin/env python3
"""RF-DETR + ELDET training entrypoint.

Requires ``pip install -e "./rf-detr[train]"`` from the repo root and a COCO-style ``dataset_dir``.

Example::

    python examples/train_rfdetr_eldet.py --dataset-dir /path/to/coco --epochs 10

Validate wiring without training (no dataset reads)::

    python examples/train_rfdetr_eldet.py --dataset-dir /path/to/coco --dry-run

By default the script resumes from ``{output_dir}/last.ckpt`` when present; use ``--fresh`` for a new run.
Set ``TrainConfig.use_ema=False`` when ELDET is enabled (the default here).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from resume_utils import resolve_rfdetr_resume, try_resolve_rfdetr_resume


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="COCO dataset root (train/valid image dirs plus annotations).",
    )
    p.add_argument("--output-dir", type=Path, default=Path("runs/eldet_rfdetr"))
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument(
        "--num-classes",
        type=int,
        default=80,
        help="Number of foreground detection classes (default 80 for COCO).",
    )
    p.add_argument(
        "--num-channels",
        type=int,
        default=3,
        help="Input image channels (default 3 for RGB).",
    )
    p.add_argument(
        "--resolution",
        type=int,
        default=704,
        help="Square input size (model resolution and dataset resize target).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device batch size (per GPU in multi-GPU runs).",
    )
    p.add_argument(
        "--devices",
        default="auto",
        help='Lightning device count / selector: default "auto" uses all visible CUDA GPUs (or CPU if none).',
    )
    p.add_argument(
        "--allow-random-resize-crops",
        action="store_true",
        help="Use stock RF-DETR train resize (OneOf direct resize vs random crop). Disables direct_train_resize and "
        "enables multi_scale.",
    )
    p.add_argument("--fast-dev-run", action="store_true", help="PyTorch Lightning smoke (few steps)")
    p.add_argument("--no-pretrain", action="store_true", help="Random init (no weight download)")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build trainer + ELDET callbacks only (no LightningModule, no dataset, no fit).",
    )
    p.add_argument(
        "--allow-stock-ema",
        action="store_true",
        help="Allow TrainConfig.use_ema=True with ELDET (sets use_ema True; pass strict_stock_ema=False on attach).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: min(8, CPU count) on Linux/Mac, 0 on Windows; use 0 only for debugging).",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Start a new run; do not resume from an existing checkpoint.",
    )
    p.add_argument(
        "--resume",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="Resume from an explicit checkpoint (default: auto-resume {output_dir}/last.ckpt). "
        "PATH may be a .ckpt file or a run directory containing last.ckpt.",
    )
    args = p.parse_args()

    output_dir = args.output_dir.resolve()
    if args.fresh:
        resume_path = None
    elif args.resume is not None:
        try:
            resume_path = resolve_rfdetr_resume(args.resume, output_dir)
        except FileNotFoundError as exc:
            p.error(str(exc))
    else:
        resume_path = try_resolve_rfdetr_resume(output_dir)

    dataset_root = args.dataset_dir

    from eldet.rfdetr import ELDETRFDETRConfig, ELDETRFDETRModelModule, attach_eldet_callbacks_to_trainer
    from rfdetr.config import RFDETRLargeConfig, TrainConfig
    from rfdetr.training import RFDETRDataModule, build_trainer

    use_stock_resize = bool(args.allow_random_resize_crops)
    mc_kwargs: dict = {
        "num_classes": args.num_classes,
        "num_channels": int(args.num_channels),
        "resolution": int(args.resolution),
    }
    if args.no_pretrain:
        mc_kwargs["pretrain_weights"] = None
    model_config = RFDETRLargeConfig(**mc_kwargs)

    devices_arg: int | str
    if args.devices == "auto":
        devices_arg = "auto"
    else:
        try:
            devices_arg = int(args.devices)
        except ValueError:
            p.error(f"Invalid --devices {args.devices!r}; use 'auto' or a positive integer.")
        if devices_arg < 1:
            p.error("--devices must be >= 1 when an integer is passed.")

    if args.num_workers is not None:
        num_workers = int(args.num_workers)
    elif os.name == "nt":
        num_workers = 0
    else:
        num_workers = min(8, (os.cpu_count() or 4))

    train_config = TrainConfig(
        dataset_dir=str(dataset_root.resolve()),
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=int(args.batch_size),
        grad_accum_steps=1,
        num_workers=num_workers,
        use_ema=bool(args.allow_stock_ema),
        tensorboard=False,
        wandb=False,
        mlflow=False,
        clearml=False,
        devices=devices_arg,
        accelerator="auto",
        multi_scale=use_stock_resize,
        expanded_scales=use_stock_resize,
        direct_train_resize=not use_stock_resize,
        progress_bar="tqdm",
        resume=resume_path,
    )
    eldet_cfg = ELDETRFDETRConfig(
        enabled=True,
        use_metric_phase_detection=False,
        teacher_start_epoch=0,
    )

    strict_ema = not bool(args.allow_stock_ema)
    trainer_kw: dict = {}
    if args.fast_dev_run:
        trainer_kw["fast_dev_run"] = True
    # Subprocess DDP + ELDET (frozen teacher, matcher KD) needs find_unused_parameters=True.
    # Pass strategy here so it still applies if the training node has an older vendored
    # ``build_trainer`` that ignores ``TrainConfig.ddp_find_unused_parameters`` (merged last).
    if train_config.strategy in ("auto", "ddp"):
        from pytorch_lightning.strategies import DDPStrategy

        trainer_kw["strategy"] = DDPStrategy(find_unused_parameters=True)

    if args.dry_run:
        trainer = build_trainer(train_config, model_config, **trainer_kw)
        attach_eldet_callbacks_to_trainer(
            trainer, eldet_cfg, train_config, strict_stock_ema=strict_ema
        )
        print("ELDET RF-DETR dry-run OK (trainer + callbacks).")
        return

    module = ELDETRFDETRModelModule(model_config, train_config, eldet=eldet_cfg)
    datamodule = RFDETRDataModule(model_config, train_config)
    trainer = build_trainer(train_config, model_config, **trainer_kw)
    attach_eldet_callbacks_to_trainer(trainer, eldet_cfg, train_config, strict_stock_ema=strict_ema)
    trainer.fit(module, datamodule, ckpt_path=resume_path)


if __name__ == "__main__":
    main()
