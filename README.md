# ELDET — Urban Tree Mapping

**ELDET** (Early-Learning Distillation with Noisy Labels for Object Detection) trains detectors when labels are noisy. This repo is a reference implementation for **YOLO** (Ultralytics) and **RF-DETR** (Lightning), aimed at urban tree mapping and related GeoAI workflows (including RGB / RGBI GeoTIFF chips).

Shared math and schedule logic live in `eldet/`. Framework adapters live in `eldet/yolo/` and `eldet/rfdetr/`. Training entry points are under `examples/`.

Status: **Pre-Alpha** (`0.1.0`). APIs may change.

---

## Table of contents

1. [How ELDET works](#how-eldet-works)
2. [Repository layout](#repository-layout)
3. [Prerequisites](#prerequisites)
4. [Vendor setup](#vendor-setup)
5. [Install](#install)
6. [Data](#data)
7. [Train with YOLO](#train-with-yolo)
8. [Train with RF-DETR](#train-with-rf-detr)
9. [ELDET configuration](#eldet-configuration)
10. [Outputs and resume](#outputs-and-resume)
11. [Testing](#testing)
12. [Caveats](#caveats)
13. [License](#license)

---

## How ELDET works

ELDET assumes labels may be noisy and exploits the **early-learning** phase before the network memorizes bad annotations:

1. **Probe the train set** with CA (class-agnostic localization recall) and GTBA (ground-truth box allocation / classification accuracy).
2. **Detect transition epochs** `t_loc` and `t_cls` (exponential saturation fit, γ ≈ 0.9).
3. Maintain a **slow EMA teacher** (asymmetric: slower global decay; faster decay on classification parameters between `t_loc` and `t_cls`).
4. Add **knowledge distillation** from teacher → student (KL on logits + L1 on boxes).

Example scripts default to a **fixed teacher start at epoch 0** (metrics off). Pass `--metric-phases` on YOLO, or set `use_metric_phase_detection=True` in Python, for the CA/GTBA-driven schedule.

For **single-class** detection (typical for “tree” vs background), ELDET auto-collapses `t_cls` to `t_loc` when `nc == 1` / `num_classes == 1`.

---

## Repository layout

| Path | Role |
|------|------|
| `eldet/` | First-party package: metrics, transition, KD, EMA policy, noise helpers, YOLO/RF-DETR adapters, tests |
| `examples/` | CLI trainers: `train_yolo_eldet.py`, `train_rfdetr_eldet.py`, resume helpers |
| `configs/` | Documentation stubs for ELDET fields (not loaded as runtime YAML) |
| `patches/` | Notes for rebasing vendored RF-DETR patches (matcher indices, COCO layouts, multichannel TIFF) |
| `VENDOR_VERSIONS.txt` | Pinned upstream commit SHAs for `ultralytics/` and `rf-detr/` |
| `ultralytics/` | **Vendored** Ultralytics tree (you supply; not always in the git clone) |
| `rf-detr/` | **Vendored** RF-DETR tree (you supply; not always in the git clone) |

---

## Prerequisites

- **Python** 3.10–3.13
- **PyTorch** 1.13+ (CUDA recommended; CPU works for smoke runs)
- **Vendored trees** at the repo root: `ultralytics/` and `rf-detr/` (see below)
- **No API keys** required
- **Windows:** RF-DETR dataloader workers default to `0` (override with `--num-workers` if needed)

---

## Vendor setup

Training imports the **local** `ultralytics/` and `rf-detr/` directories. If they are missing from your clone, add them at the pins in `VENDOR_VERSIONS.txt`, then re-apply the behaviors documented under `patches/`.

Pinned revisions (see `VENDOR_VERSIONS.txt`):

```text
ultralytics  ec8a844f5a7fea50e339fc7230b85cd963c3c668
rf-detr      680b58607e20b3fd083fe4aa7c5cc63649afa21e
```

Example (from the repository root):

```bash
git clone https://github.com/ultralytics/ultralytics.git ultralytics
git -C ultralytics checkout ec8a844f5a7fea50e339fc7230b85cd963c3c668

git clone https://github.com/roboflow/rf-detr.git rf-detr
git -C rf-detr checkout 680b58607e20b3fd083fe4aa7c5cc63649afa21e
```

Then re-apply ELDET-related vendor patches described in:

- `patches/rfdetr-criterion-eldet-indices.md` — stash Hungarian matcher indices for KD
- `patches/rfdetr-datasets-coco-rgbi-layout.md` — alternate COCO / RGBI path layout
- `patches/rfdetr-coco-multichannel-tiff.md` — GeoTIFF loads when `num_channels > 3`

YOLO also depends on a forked train loop in `eldet/yolo/vendor_train_loop.py`; re-sync that file when rebasing Ultralytics.

---

## Install

From the repository root (after vendor trees are in place):

```bash
pip install -e .
pip install -e "./rf-detr[train]"
pip install -e "./ultralytics"
```

Optional development extras (pytest, ruff, Lightning):

```bash
pip install -e ".[dev]"
```

For multichannel GeoTIFF chips (`--num-channels` > 3 with RF-DETR):

```bash
pip install tifffile
```

Core package dependencies (`numpy`, `scipy`, `torch`) install with `pip install -e .`. Framework packages come from the editable vendor installs above.

---

## Data

### YOLO (Ultralytics dataset YAML)

Pass a dataset YAML with `--data`. The default `coco8.yaml` is downloaded automatically by Ultralytics if missing.

Point the YAML at your images and labels in the usual Ultralytics layout (paths relative to the YAML or absolute). There is no fixed `data/` folder in this repo.

### RF-DETR (COCO-style root)

Pass `--dataset-dir` to a dataset root. Two layouts are supported (with the vendored patches):

**Roboflow-style**

```text
dataset_root/
  train/
    _annotations.coco.json
    <images>
  valid/
    _annotations.coco.json
    <images>
```

**Alternate layout** (annotations under `annotations/`; useful for tree / RGBI chips)

```text
dataset_root/
  train/                 # images (may nest, e.g. scene/chip/*.tif)
  valid/
  annotations/
    instances_train.json
    instances_valid.json
```

Set `--num-classes` to the number of **foreground** classes. For a single “tree” class, use `--num-classes 1`. For RGBI or other `num_channels > 3`, install `tifffile` and pass `--num-channels` accordingly.

Labels are assumed **already noisy** by default. Synthetic noise knobs (`noise_*` in the ELDET configs) are for ablations only; leave them at `0` for real datasets.

---

## Train with YOLO

Requires `pip install -e "./ultralytics"`.

### Quick start

```bash
python examples/train_yolo_eldet.py --data coco8.yaml --epochs 3
```

### Common options

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `yolo11n.pt` | Ultralytics weights or hub name |
| `--data` | `coco8.yaml` | Dataset YAML |
| `--epochs` | `3` | Training epochs |
| `--batch-size` | `8` | Per-device batch size |
| `--imgsz` | `640` | Square train/val size |
| `--device` | all CUDA GPUs, else `cpu` | e.g. `0`, `0,1`, `cpu` |
| `--metric-phases` | off | CA/GTBA teacher schedule; without it, teacher starts at epoch 0 |
| `--project` | `runs/detect` | Ultralytics project directory |
| `--name` | auto | Run name under `--project` |
| `--fresh` | off | Do not auto-resume |
| `--resume [PATH]` | newest `last.pt` under `--project` | `.pt` file, run dir, or `weights/` |

By default the script **resumes** from the newest `last.pt` under `--project` when one exists. Use `--fresh` for a clean run.

### Paper-style metric phases

```bash
python examples/train_yolo_eldet.py --data coco8.yaml --epochs 100 --metric-phases
```

### Programmatic use

```python
from eldet.yolo import ELDETDetectionTrainer, ELDETYoloConfig

trainer = ELDETDetectionTrainer(
    overrides=dict(
        model="yolo11n.pt",
        data="coco8.yaml",
        epochs=3,
        eldet=ELDETYoloConfig(enabled=True),
    )
)
trainer.train()
```

See `configs/yolo_eldet.yaml` for a field checklist (documentation only).

---

## Train with RF-DETR

Requires `pip install -e "./rf-detr[train]"` and a COCO-style `--dataset-dir`.

The example launcher uses **`RFDETRLargeConfig`**.

### Dry-run (wiring only)

```bash
python examples/train_rfdetr_eldet.py --dataset-dir /path/to/coco --dry-run
```

Builds the Lightning trainer and attaches ELDET callbacks without loading a dataset or fitting.

### Train

```bash
python examples/train_rfdetr_eldet.py --dataset-dir /path/to/coco --epochs 10
```

### Common options

| Flag | Default | Meaning |
|------|---------|---------|
| `--dataset-dir` | **required** | COCO-style dataset root |
| `--output-dir` | `runs/eldet_rfdetr` | Run / checkpoint directory |
| `--epochs` | `1` | Training epochs |
| `--num-classes` | `80` | Foreground classes |
| `--num-channels` | `3` | Input channels (`>3` needs `tifffile`) |
| `--resolution` | `704` | Square input size |
| `--batch-size` | `8` | Per-device batch size |
| `--devices` | `auto` | Lightning device count / selector |
| `--num-workers` | OS-dependent | `0` on Windows; `min(8, CPU count)` elsewhere |
| `--allow-random-resize-crops` | off | Stock multi-scale crop path; default is direct resize |
| `--fast-dev-run` | off | Lightning few-step smoke |
| `--no-pretrain` | off | Random init (skip pretrained weights) |
| `--fresh` / `--resume [PATH]` | auto `last.ckpt` | Resume control |
| `--allow-stock-ema` | off | Override the EMA conflict guard (not recommended) |

**Critical:** keep `TrainConfig.use_ema=False` when ELDET is on (stock EMA conflicts with the ELDET teacher). The example script does this unless you pass `--allow-stock-ema`.

Multi-GPU DDP uses `find_unused_parameters=True` so frozen teacher / matcher KD works.

### Single-class tree mapping example

```bash
python examples/train_rfdetr_eldet.py \
  --dataset-dir /path/to/tree_coco \
  --num-classes 1 \
  --epochs 50 \
  --output-dir runs/tree_rfdetr
```

RGBI GeoTIFF chips:

```bash
python examples/train_rfdetr_eldet.py \
  --dataset-dir /path/to/rgbi_coco \
  --num-classes 1 \
  --num-channels 4 \
  --epochs 50
```

See `configs/rfdetr_eldet.yaml` for a field checklist (documentation only).

---

## ELDET configuration

Training is **Python-driven**, not merged from `configs/*.yaml`. Pass `ELDETYoloConfig` / `ELDETRFDETRConfig` into the trainers (as the example scripts do).

Shared knobs (defaults from the dataclasses):

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `False` (scripts set `True`) | Turn on ELDET |
| `lambda_kd` / `lambda_kd_cls` / `lambda_kd_box` | `1.0` | KD loss weights |
| `kd_temperature` | `1.0` | Softmax / sigmoid temperature for KL |
| `teacher_decay` | `0.999` | Slow EMA decay |
| `teacher_cls_decay` | `0.1` | Faster EMA on class params in the `t_loc`→`t_cls` window |
| `use_metric_phase_detection` | `True` in dataclasses; **example scripts often force `False`** | CA/GTBA transitions vs fixed start |
| `teacher_start_epoch` | `None` | 0-based fixed teacher start when metrics are off |
| `transition_gamma` | `0.9` | Memorization derivative threshold |
| `gtba_tau` | `0.1` | GTBA IoU threshold |
| `metric_max_train_batches` | `32` | Cap probe cost |
| `metric_every_n_epochs` | `1` | Probe cadence |
| `metric_topk_anchors` | `200` (YOLO only) | Probe top-k anchors |
| `series_csv_path` | `None` | Append CA / GTBA / `loss_kd` CSV |
| `freeze_transitions_after_cls` | `True` | Stop updating histories after `t_cls` |
| `single_class` | `None` | Auto-collapse `t_cls=t_loc` when `nc==1` |
| `noise_*` | `0` | Synthetic noise for ablations only |

RF-DETR-only:

| Field | Default | Meaning |
|-------|---------|---------|
| `kd_use_matcher_indices` | `True` | KD on Hungarian-matched queries |
| `export_teacher_pth` | `True` | Write `last_teacher.pth` / `checkpoint_best_teacher.pth` |
| `export_student_last_pth` | `False` | Also write `last_student.pth` each epoch |

To customize beyond CLI flags, edit the `ELDET*Config(...)` construction in the example scripts or call the trainers from your own Python.

---

## Outputs and resume

### YOLO (`runs/detect/...` by default)

- Standard Ultralytics artifacts: `weights/last.pt`, `weights/best.pt`, `results.csv`
- With ELDET: `results.csv` can include an `eldet_kd` column
- Optional series CSV (`series_csv_path`) columns:  
  `epoch, train_ca, train_gtba, t_loc_0based, t_cls_0based, loss_kd`
- Stock ModelEMA is replaced by a student shim; distillation uses `trainer.eldet_teacher` only

Resume: newest `last.pt` under `--project`, or `--resume PATH` / `--fresh`.

### RF-DETR (`runs/eldet_rfdetr` by default)

- Lightning resume checkpoint: `last.ckpt` (may include `eldet_series_state`)
- Upstream student best: `checkpoint_best_regular.pth`
- ELDET teacher exports (default): `last_teacher.pth`, `checkpoint_best_teacher.pth` (on val mAP improve)
- Optional: `last_student.pth`
- Lightning log: `train/eldet_kd`
- No `checkpoint_best_ema.pth` while `use_ema=False` (the ELDET teacher is the slow-weight track)

Resume: `{output_dir}/last.ckpt` by default, or `--resume PATH` / `--fresh`.

---

## Testing

```bash
pip install -e ".[dev]"
pytest eldet/tests -q
```

Core Phase-1 tests (metrics, transition, KD, EMA, noise) run without the vendor trees. Many RF-DETR / YOLO integration tests skip or need `ultralytics` / `rfdetr` importable. Pytest sets `pythonpath` so `import ultralytics` resolves to the vendored package when present.

---

## Caveats

1. **`ultralytics/` and `rf-detr/` must exist** at the repo root before train installs work.
2. After updating vendors, **re-apply** `patches/*.md` and re-sync `eldet/yolo/vendor_train_loop.py` as needed.
3. **Do not combine stock EMA with ELDET** (`use_ema=False` for RF-DETR; YOLO replaces ModelEMA).
4. Example CLIs default to **teacher at epoch 0**, not metric phases — use `--metric-phases` (YOLO) or config for the paper schedule.
5. **Single-class** runs auto-collapse classification transition to localization transition.
6. **Windows** RF-DETR workers default to `0`.
7. **License mix:** shipping Ultralytics brings **AGPL-3.0** obligations; see below.
8. This repo does not ship a tree dataset, notebooks, Docker, or CI yet — bring your own COCO / YOLO data.

---

## License

- First-party code in `eldet/`, `examples/`, and `configs/` — **MIT**
- Vendored **Ultralytics** — **AGPL-3.0**
- Vendored **RF-DETR** — **Apache-2.0**

Pinned revisions: `VENDOR_VERSIONS.txt`.
