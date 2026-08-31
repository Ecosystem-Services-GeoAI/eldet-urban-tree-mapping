# ELDET

Reference implementation of **ELDET** (Early-Learning Distillation with Noisy Labels for Object Detection) for **YOLO** and **RF-DETR**. The shared logic lives in `eldet/`; trainers wrap the vendored stacks in `ultralytics/` and `rf-detr/`.

## Install

Python 3.10 or newer. From the repository root:

```bash
pip install -e .
pip install -e "./rf-detr[train]"
pip install -e "./ultralytics"
```

## Train

**YOLO** (downloads the small `coco8` sample if needed):

```bash
python examples/train_yolo_eldet.py --data coco8.yaml --epochs 3
```

**RF-DETR** (pass a COCO-style dataset root with `train/`, `valid/`, and annotations):

```bash
python examples/train_rfdetr_eldet.py --dataset-dir /path/to/coco --dry-run
python examples/train_rfdetr_eldet.py --dataset-dir /path/to/coco --epochs 10
```

Optional: `pytest eldet/tests -q` after `pip install -e ".[dev]"`.

## License

First-party code in `eldet/`, `examples/`, and `configs/` is **MIT**. Vendored **Ultralytics** is **AGPL-3.0**; vendored **RF-DETR** is **Apache-2.0**. See `VENDOR_VERSIONS.txt` for pinned revisions.
