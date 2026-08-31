# RF-DETR datasets — alternate COCO layout

**Files:**

- `rf-detr/src/rfdetr/datasets/coco.py` — `resolve_roboflow_coco_paths`, `_semi_tree_coco_ann_and_image_dir`, extended `is_valid_coco_dataset`
- `rf-detr/src/rfdetr/datasets/__init__.py` — `detect_roboflow_format` accepts `annotations/instances_train.json` + `train/`
- `rf-detr/src/rfdetr/detr.py` — `_load_classes` / `_detect_num_classes_for_training` use `resolve_roboflow_coco_paths`

**Why:** Roboflow-style training expects `train/_annotations.coco.json`. An alternate COCO tree keeps JSON under `annotations/` (`instances_train.json`, `instances_valid.json`) with images under `train/` and `valid/`. RF-DETR now resolves both layouts so `TrainConfig.dataset_dir` can point at the dataset root without copying JSON into each split folder.

**Rebase:** Re-apply this behavior if upstream `coco.py` / `__init__.py` / `detr.py` change the Roboflow detection or path constants.
