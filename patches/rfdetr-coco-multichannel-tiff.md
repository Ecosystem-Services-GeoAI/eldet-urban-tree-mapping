# RF-DETR COCO — multichannel / GeoTIFF loads (`num_channels` > 3)

**Files:** `rf-detr/src/rfdetr/datasets/coco.py`

**Why:** `torchvision.datasets.CocoDetection._load_image` always does `Image.open(...).convert("RGB")`, which drops extra bands on multichannel GeoTIFF chips. The vendored `CocoDetection` subclass now:

- Accepts `num_channels` (passed from the merged train/model namespace, i.e. `ModelConfig.num_channels`).
- Overrides `_load_image` to call `_load_pil_image_for_num_channels`, which uses **`tifffile`** for `.tif` / `.tiff` when `num_channels > 3`.

**Dependency:** `pip install tifffile`.

**Rebase:** Re-apply if upstream `coco.py` changes `CocoDetection` or dataset builders.
