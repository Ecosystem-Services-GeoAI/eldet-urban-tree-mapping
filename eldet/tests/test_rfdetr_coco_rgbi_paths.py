"""RF-DETR Roboflow COCO path resolution for the alternate annotations/ layout."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest


def _alt_coco_imports() -> tuple[Any, Callable[..., Any], Callable[..., Any]]:
    pytest.importorskip("rfdetr")
    coco_mod = importlib.import_module("rfdetr.datasets.coco")
    if not hasattr(coco_mod, "resolve_roboflow_coco_paths"):
        pytest.skip("Install patched rf-detr (patches/rfdetr-datasets-coco-rgbi-layout.md)")
    from rfdetr.datasets import detect_roboflow_format
    from rfdetr.datasets.coco import is_valid_coco_dataset, resolve_roboflow_coco_paths

    return detect_roboflow_format, is_valid_coco_dataset, resolve_roboflow_coco_paths


def _write_min_coco_ann(path: Path) -> None:
    ann = {"categories": [{"id": 1, "name": "obj"}], "images": [], "annotations": []}
    path.write_text(json.dumps(ann), encoding="utf-8")


def test_detect_alternate_coco_layout(tmp_path: Path) -> None:
    detect_roboflow_format, _, _ = _alt_coco_imports()
    root = tmp_path / "alt_coco"
    (root / "train").mkdir(parents=True)
    (root / "valid").mkdir(parents=True)
    (root / "annotations").mkdir()
    _write_min_coco_ann(root / "annotations" / "instances_train.json")
    _write_min_coco_ann(root / "annotations" / "instances_valid.json")
    (root / "train" / "scene" / "chip1").mkdir(parents=True)
    (root / "train" / "scene" / "chip1" / "r0_c0.tif").write_bytes(b"")
    assert detect_roboflow_format(root) == "coco"


def test_resolve_paths_prefers_roboflow_default(tmp_path: Path) -> None:
    _, _, resolve_roboflow_coco_paths = _alt_coco_imports()
    root = tmp_path / "roboflow"
    (root / "train").mkdir(parents=True)
    ann_path = root / "train" / "_annotations.coco.json"
    _write_min_coco_ann(ann_path)
    img, ann = resolve_roboflow_coco_paths(root, "train")
    assert ann == ann_path
    assert img == root / "train"


def test_resolve_paths_alternate_coco(tmp_path: Path) -> None:
    _, is_valid_coco_dataset, resolve_roboflow_coco_paths = _alt_coco_imports()
    root = tmp_path / "alt"
    (root / "train").mkdir(parents=True)
    (root / "valid").mkdir(parents=True)
    (root / "annotations").mkdir()
    train_json = root / "annotations" / "instances_train.json"
    val_json = root / "annotations" / "instances_valid.json"
    _write_min_coco_ann(train_json)
    _write_min_coco_ann(val_json)
    assert is_valid_coco_dataset(str(root))
    assert resolve_roboflow_coco_paths(root, "train") == (root / "train", train_json)
    assert resolve_roboflow_coco_paths(root, "val") == (root / "valid", val_json)
