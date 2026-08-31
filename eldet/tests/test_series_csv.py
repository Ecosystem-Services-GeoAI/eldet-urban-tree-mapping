"""CSV series logging (no heavy deps)."""

from __future__ import annotations

import csv
from pathlib import Path

from eldet.series_csv import append_eldet_series_row, maybe_append_eldet_series_row


def test_append_creates_header_and_rows(tmp_path: Path) -> None:
    p = tmp_path / "s.csv"
    append_eldet_series_row(
        p,
        epoch=0,
        ca=0.1,
        gtba=0.2,
        t_loc_0based=None,
        t_cls_0based=None,
        loss_kd=None,
    )
    append_eldet_series_row(
        p,
        epoch=1,
        ca=None,
        gtba=None,
        t_loc_0based=3,
        t_cls_0based=7,
        loss_kd=0.05,
    )
    rows = list(csv.reader(p.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == ["epoch", "train_ca", "train_gtba", "t_loc_0based", "t_cls_0based", "loss_kd"]
    assert rows[1] == ["0", "0.1", "0.2", "", "", ""]
    assert rows[2] == ["1", "", "", "3", "7", "0.05"]


def test_maybe_append_skips_empty_path(tmp_path) -> None:
    maybe_append_eldet_series_row(
        None,
        epoch=0,
        ca=1.0,
        gtba=1.0,
        t_loc_0based=None,
        t_cls_0based=None,
    )
    maybe_append_eldet_series_row(
        "",
        epoch=0,
        ca=1.0,
        gtba=1.0,
        t_loc_0based=None,
        t_cls_0based=None,
    )
    assert list(tmp_path.iterdir()) == []
