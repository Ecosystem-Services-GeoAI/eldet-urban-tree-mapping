"""Checkpoint path helpers for ELDET example training scripts."""

from __future__ import annotations

from pathlib import Path


def try_resolve_rfdetr_resume(output_dir: Path) -> str | None:
    """Return ``{output_dir}/last.ckpt`` when present, else ``None`` (fresh training)."""
    ckpt = output_dir / "last.ckpt"
    if ckpt.is_file():
        return str(ckpt.resolve())
    return None


def try_resolve_yolo_resume(project: Path) -> str | None:
    """Return the newest ``last.pt`` under ``project`` when present, else ``None``."""
    from ultralytics.utils.files import get_latest_run

    latest = get_latest_run(str(project))
    return latest or None


def resolve_rfdetr_resume(resume: str | None, output_dir: Path) -> str | None:
    """Resolve ``--resume`` to a Lightning ``ckpt_path`` string.

    ``resume`` may be:
    - ``None``: fresh training
    - ``""`` (``--resume`` with no path): ``{output_dir}/last.ckpt``
    - a run directory (contains ``last.ckpt``)
    - a direct ``.ckpt`` / ``.pth`` file
    """
    if resume is None:
        return None

    if resume in {"", "last"}:
        ckpt = output_dir / "last.ckpt"
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"No RF-DETR resume checkpoint at {ckpt}. Train first or pass an explicit checkpoint path."
            )
        return str(ckpt.resolve())

    path = Path(resume).expanduser()
    if not path.is_absolute():
        path = path.resolve()

    if path.is_file():
        return str(path)

    if path.is_dir():
        for candidate in (path / "last.ckpt", path / "checkpoint.pth"):
            if candidate.is_file():
                return str(candidate.resolve())
        raise FileNotFoundError(
            f"No resume checkpoint in {path} (expected last.ckpt or checkpoint.pth)."
        )

    raise FileNotFoundError(f"Resume checkpoint not found: {path}")


def resolve_yolo_resume(resume: str | None, project: Path) -> bool | str:
    """Resolve ``--resume`` to an Ultralytics ``resume`` override.

    ``resume`` may be:
    - ``None``: fresh training
    - ``""`` (``--resume`` with no path): newest ``last.pt`` under ``project``
    - a run directory (``…/train-N`` or ``…/train-N/weights``)
    - a direct ``last.pt`` (or other ``.pt``) file
    """
    if resume is None:
        return False  # type: ignore[return-value]

    if resume in {"", "last"}:
        from ultralytics.utils.files import get_latest_run

        latest = get_latest_run(str(project))
        if not latest:
            raise FileNotFoundError(
                f"No YOLO resume checkpoint under {project} (expected **/weights/last.pt). "
                "Train first or pass an explicit checkpoint path."
            )
        return latest

    path = Path(resume).expanduser()
    if not path.is_absolute():
        path = path.resolve()

    if path.is_file():
        return str(path)

    if path.is_dir():
        if path.name == "weights":
            last = path / "last.pt"
            if last.is_file():
                return str(last.resolve())
            path = path.parent
        last = path / "weights" / "last.pt"
        if last.is_file():
            return str(last.resolve())
        raise FileNotFoundError(f"No YOLO resume checkpoint at {last}.")

    raise FileNotFoundError(f"Resume checkpoint not found: {path}")
