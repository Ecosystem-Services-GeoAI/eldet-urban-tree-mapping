#!/usr/bin/env python3
"""Ultralytics YOLO + ELDET training entrypoint.

Requires ``pip install -e "./ultralytics"`` from the repo root.

Example::

    python examples/train_yolo_eldet.py --data coco8.yaml --epochs 3

By default the script resumes from the newest ``last.pt`` under ``runs/detect`` when one exists;
use ``--fresh`` for a new run.

Use ``--metric-phases`` for CA/GTBA-driven teacher transitions. Without it, the teacher starts at epoch 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from resume_utils import resolve_yolo_resume, try_resolve_yolo_resume


def _default_ultralytics_device() -> str:
    import torch

    if torch.cuda.is_available():
        return ",".join(str(i) for i in range(torch.cuda.device_count()))
    return "cpu"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Ultralytics weights or hub name (default: YOLO11 nano).",
    )
    p.add_argument(
        "--data",
        type=str,
        default="coco8.yaml",
        help="Ultralytics dataset YAML (default: coco8.yaml).",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8, help="Batch size (per device in multi-GPU training).")
    p.add_argument("--imgsz", type=int, default=640, help="Square train/val image size.")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help='CUDA device(s) for Ultralytics, e.g. "0", "0,1,2", or "cpu". Default: all visible CUDA GPUs.',
    )
    p.add_argument(
        "--metric-phases",
        action="store_true",
        help="CA/GTBA metric phases (default: fixed teacher at epoch 0)",
    )
    p.add_argument(
        "--project",
        type=Path,
        default=Path("runs/detect"),
        help="Ultralytics project directory for run folders (default: runs/detect).",
    )
    p.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name under --project (default: Ultralytics auto-increment train, train-2, …).",
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
        help="Resume from an explicit last.pt (default: auto-resume newest under --project). "
        "PATH may be a .pt file, a run dir, or a weights/ dir.",
    )
    args = p.parse_args()

    project = args.project.resolve()
    if args.fresh:
        resume_arg: bool | str = False
    elif args.resume is not None:
        try:
            resume_arg = resolve_yolo_resume(args.resume, project)
        except FileNotFoundError as exc:
            p.error(str(exc))
    else:
        resume_arg = try_resolve_yolo_resume(project) or False

    from eldet.yolo import ELDETDetectionTrainer, ELDETYoloConfig

    device = args.device if args.device is not None else _default_ultralytics_device()

    eldet = ELDETYoloConfig(
        enabled=True,
        use_metric_phase_detection=bool(args.metric_phases),
        teacher_start_epoch=None if args.metric_phases else 0,
    )
    overrides: dict = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch_size,
        "imgsz": args.imgsz,
        "device": device,
        "project": str(args.project),
        "eldet": eldet,
    }
    if args.name is not None:
        overrides["name"] = args.name
    if resume_arg:
        overrides["resume"] = resume_arg
    trainer = ELDETDetectionTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
