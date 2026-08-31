"""Append-only CSV logging for ELDET train metrics (framework-agnostic)."""

from __future__ import annotations

import csv
from pathlib import Path


def append_eldet_series_row(
    path: str | Path,
    *,
    epoch: int,
    ca: float | None,
    gtba: float | None,
    t_loc_0based: int | None,
    t_cls_0based: int | None,
    loss_kd: float | None = None,
) -> None:
    """Append one row; create file with header if ``path`` does not exist.

    Empty / unknown numeric fields are written as empty cells.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = not p.exists()
    row = [
        epoch,
        "" if ca is None else ca,
        "" if gtba is None else gtba,
        "" if t_loc_0based is None else t_loc_0based,
        "" if t_cls_0based is None else t_cls_0based,
        "" if loss_kd is None else loss_kd,
    ]
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header:
            w.writerow(["epoch", "train_ca", "train_gtba", "t_loc_0based", "t_cls_0based", "loss_kd"])
        w.writerow(row)


def maybe_append_eldet_series_row(
    path: str | None,
    *,
    epoch: int,
    ca: float | None,
    gtba: float | None,
    t_loc_0based: int | None,
    t_cls_0based: int | None,
    loss_kd: float | None = None,
) -> None:
    """No-op when ``path`` is ``None`` or empty."""
    if not path:
        return
    append_eldet_series_row(
        path,
        epoch=epoch,
        ca=ca,
        gtba=gtba,
        t_loc_0based=t_loc_0based,
        t_cls_0based=t_cls_0based,
        loss_kd=loss_kd,
    )
