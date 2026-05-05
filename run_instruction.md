# HAT + Hierarchical Frame Memory — Run Instructions

This document covers how to run the modified HAT pipeline with the new
`HierarchicalMemoryUnit` (drop-in replacement for the flat `HistoryUnit`),
including config flags, ablations, and the unchanged training/eval contract.

---

## 1. What changed

| File | Change |
|---|---|
| `models.py` | Added `HierarchicalMemoryUnit`. `MYNET` now selects it when `use_hier_memory` is true (default), else falls back to original `HistoryUnit`. The attribute name `model.history_unit` is preserved so the existing optimizer parameter group keeps working. |
| `opts_egtea.py`, `opts_thumos.py`, `opts_epic.py`, `opts_muses.py` | Added CLI flags: `--use_hier_memory`, `--mem_num_levels`, `--mem_sizes`, `--mem_level_strides`, `--mem_merge_thresh`, `--mem_age_decay`. |

Untouched: `dataset.py`, `loss_func.py`, `eval.py`, `iou_utils.py`, `Evaluation/`,
training loops in `main.py` / `EGTEA main.py` / `Thumos main.py`,
`encoder → decoder → anchor → classifier/regressor` pipeline, mAP/tIoU protocol.

---

## 2. Environment

Use the project's existing environment (no new dependencies were added):

```bash
pip install -r requirements.txt
```

The new module uses only `torch`, `torch.nn`, `torch.nn.functional` —
already in `requirements.txt`.

---

## 3. CLI flags (new)

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--use_hier_memory` | int | `1` | `1` = `HierarchicalMemoryUnit`, `0` = original flat `HistoryUnit` |
| `--mem_num_levels` | int | `3` | number of hierarchy levels |
| `--mem_sizes` | int+ | `16 12 8` | per-level slot capacity (length must equal `mem_num_levels`) |
| `--mem_level_strides` | int+ | `1 2 4` | per-level temporal stride (length must equal `mem_num_levels`) |
| `--mem_merge_thresh` | float | `0.75` | cosine-sim threshold to merge vs. insert |
| `--mem_age_decay` | float | `0.01` | weight of age in `importance = usage − age_decay × age` |

---

## 4. Default training runs

The new mechanism is on by default — existing entry points work unchanged.

### THUMOS-14
```bash
python "Thumos main.py"
```

### EGTEA-Gaze+
```bash
python "EGTEA main.py"
# or, since main.py imports opts_egtea:
python main.py
```

### EPIC-Kitchens
```bash
# whichever main script you wire to opts_epic in your run setup
python main.py   # after switching the import to opts_epic
```

### MUSES
```bash
python main.py   # after switching the import to opts_muses
```

---

## 5. Toggle / configure the memory

### Turn the new memory off (back to flat HAT)
```bash
python main.py --use_hier_memory 0
```

### Change depth and capacity
```bash
python main.py \
  --mem_num_levels 3 \
  --mem_sizes 16 12 8 \
  --mem_level_strides 1 2 4
```

### Tune merge behaviour
```bash
python main.py --mem_merge_thresh 0.8 --mem_age_decay 0.02
```

### Two-level (recent + long only)
```bash
python main.py \
  --mem_num_levels 2 \
  --mem_sizes 16 12 \
  --mem_level_strides 1 4
```

### Aggressive long-range abstraction
```bash
python main.py \
  --mem_num_levels 3 \
  --mem_sizes 16 8 4 \
  --mem_level_strides 1 4 16
```

---

## 6. Evaluation

Eval and mAP/tIoU protocol are untouched. After training, the existing
flow runs through:

- `eval_one_epoch` → `eval_frame` → `eval_map_nms`
- `evaluation_detection` from `eval.py`
- writes `./output/result_proposal{exp}.json`

No flags or scripts to change.

---

## 7. Ablation matrix

All ablations are pure CLI changes — no code edits.

| # | Goal | Command |
|---|---|---|
| 1 | Headline on/off | `--use_hier_memory 0` vs `1` |
| 2 | Depth | `--mem_num_levels 1` / `2` / `3` / `4` (with matching sizes/strides) |
| 3 | Capacity | `--mem_sizes 8 6 4` vs `16 12 8` vs `24 18 12` |
| 4 | Merge threshold | `--mem_merge_thresh 0.5 / 0.65 / 0.75 / 0.85 / 0.95` |
| 5 | Eviction policy | `--mem_age_decay 0.0` (LFU) vs `1.0` (FIFO-ish) vs `0.01` (default) |
| 6 | Strides | `--mem_level_strides 1 2 4` vs `1 4 16` vs `1 1 1` |

For each ablation, report mAP at the dataset's standard tIoU set
(THUMOS: 0.3–0.7 step 0.1; EGTEA / EPIC / MUSES: their respective
protocols already implemented in `eval.py`).

---

## 8. Sanity check (recommended before a long run)

Quick forward / backward pass to confirm the wiring on your device:

```python
import torch
from models import MYNET

opt = {
    'feat_dim': 4096,
    'num_of_class': 21,
    'hidden_dim': 256,
    'enc_layer': 1, 'enc_head': 4,
    'dec_layer': 1, 'dec_head': 4,
    'segment_size': 16,
    'anchors': [16, 32, 64],
    'use_hier_memory': True,
    'mem_num_levels': 3,
    'mem_sizes': [16, 12, 8],
    'mem_level_strides': [1, 2, 4],
    'mem_merge_thresh': 0.75,
    'mem_age_decay': 0.01,
}

model = MYNET(opt).cuda()
print(type(model.history_unit).__name__)         # HierarchicalMemoryUnit

x = torch.randn(2, 64, 4096).cuda()              # 16 short + 48 long
act_cls, act_reg, snip_cls = model(x)
print(act_cls.shape, act_reg.shape, snip_cls.shape)

(act_cls.sum() + act_reg.sum() + snip_cls.sum()).backward()
print('OK')
```

Edge case to verify on your end:

```python
# When the input is exactly the short-window length, long_x is empty.
# The unit falls back to learnable per-level priors — should still run.
x_short = torch.randn(2, 16, 4096).cuda()
model(x_short)
```

---

## 9. Notes on stability / cost

- **Graph bound**: insert path calls `.detach()` on the displaced slot, so
  the autograd graph never spans the full long history.
- **Compute**: streaming update is a Python loop over the per-level
  sequence length (after stride aggregation). With defaults, total
  iterations across levels are `≈ L + L/2 + L/4 ≈ 1.75·L` per batch.
  For typical THUMOS / EGTEA clip lengths this is negligible vs. the
  transformer encoder/decoder cost.
- **Memory**: bank size is bounded by `sum(mem_sizes)` × batch × hidden_dim.
  Default 36 × B × 1024 ≈ same order as the original 16-token flat history.

---

## 10. Common gotchas

- `--mem_sizes` and `--mem_level_strides` must each have exactly
  `--mem_num_levels` entries. The module asserts this at construction.
- The flag is consumed as an int (0/1), not a Python bool, because
  `argparse`'s `type=bool` is a known footgun. The model converts via
  `bool(opt.get("use_hier_memory", True))`.
- The optimizer in `main.py` keys on `model.history_unit.parameters()`
  with a separate (smaller) learning rate; this is preserved. If you add
  more memory-related submodules at the top level, give them the same
  parameter-group treatment to keep gradients balanced.
