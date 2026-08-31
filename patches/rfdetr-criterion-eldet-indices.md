# RF-DETR `SetCriterion` — ELDET matcher index stash

**File:** `rf-detr/src/rfdetr/models/criterion.py`  
**Change:** Immediately after the main-layer Hungarian call, assign:

```python
indices = self.matcher(outputs_without_aux, targets, group_detr=group_detr)
self._eldet_last_indices = indices
```

**Why:** `ELDETRFDETRModelModule` runs `loss_dict = self.criterion(outputs, targets)` before KD. ELDET’s
`compute_rf_detr_kd_auto` reuses `_eldet_last_indices` so student–teacher KD aligns with **the same** matched queries
as the detection loss, without a second matcher pass.

When rebasing the `rf-detr` vendor subtree, re-apply this hunk if the upstream file changes.
