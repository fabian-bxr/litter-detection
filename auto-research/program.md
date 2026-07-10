# AutoResearcher — Litter Segmentation (Loop 5)

## Mission

You are an autonomous ML research agent. Your goal is to **maximise `val_iou`**
(intersection-over-union on the validation set) for a pixel-wise litter
segmentation model trained on the TACO dataset.

This is a **fresh research loop**. Previous loops (visible in MLflow history)
peaked at `val_iou ≈ 0.6738` with a ResNet34 U-Net trained for 108 epochs
(90-minute time budget). This loop returns to the **default 20-epoch budget**
so all runs are directly comparable, and re-establishes the baseline before
attacking the plateau with the directions below.

The plateau is almost certainly **data/regularisation-limited, not
capacity-limited** — ResNet34 / EfficientNet-B4 are oversized for ~1062
training images. The plan therefore moves *down* in backbone size and *up*
in decoder quality and augmentation richness.

---

## Rules

1. **Only edit `train.py`.** Never modify `prepare.py` or `analysis.ipynb`.
   `program.md` may be edited by the human; the agent should treat it as
   read-only.
2. **Do not change `EPOCHS`** (default 20 per run) unless the human instructs
   you to. A consistent epoch count is what makes experiments comparable.
3. Every experiment must be a distinct MLflow run with a descriptive
   `--run-name` that captures what changed (e.g. `smp-mitb0-deeplabv3plus`,
   `copy-paste-aug`, `lovasz-loss`).
4. Always read the latest `val_iou` from MLflow before deciding the next
   change.
5. **One change at a time** — isolate variables so you know what caused any
   improvement or regression.
6. After each run, log the result to `auto-research/results.md` following
   the workflow in **Logging each run** below. Never edit `program.md`
   itself — only the human updates it.

---

## Setup (first time only)

```bash
# Install dependencies
uv sync

# Download and preprocess the TACO dataset (~10 min, one-time)
uv run python auto-research/prepare.py

# Launch MLflow UI (optional, for human review)
uv run mlflow ui
```

Many of the directions below depend on `segmentation_models_pytorch`, which
is **not yet installed**. Add it before the first non-baseline run:

```bash
uv add segmentation-models-pytorch
```

---

## Running an experiment

```bash
uv run python auto-research/train.py --run-name <descriptive-name> [--epochs N] [--seed N]
```

The script prints per-epoch metrics and logs everything to MLflow. The best
checkpoint is saved to `models/best_model.pth` and exported to
`models/best_model.onnx`.

---

## Logging each run

`program.md` is **read-only for the agent** — never edit it. The run log
lives in `auto-research/results.md` (gitignored). Each experiment follows
this loop:

1. Edit `train.py` — one logical change matching the next step of the plan.
2. Run: `uv run python auto-research/train.py --run-name <name>`
3. If the run **crashed**, append a row with `Status=crashed`, leave the
   `Commit` column empty, fix the issue, and retry. Otherwise continue.
4. Commit the `train.py` change with a short message naming the run and its
   `val_iou`, e.g.:
   ```bash
   git commit -am "smp-mitb1-deeplabv3plus: val_iou=0.7015"
   ```
5. Read the 7-char commit hash: `git rev-parse --short HEAD`.
6. Append one row to the table in `auto-research/results.md` with that hash.
7. Decide **kept** or **discarded**:
   - **kept** — improved on previous best (or is the first row of the loop).
     Leave the change in `train.py` for the next run.
   - **discarded** — regressed. Revert `train.py` to the previous state
     (`git revert <hash>` or a manual revert commit) before the next step.
8. Continue to the next step of the plan.

### `results.md` schema

Markdown table with one row per run. Columns, in order:

| Column      | Description                                                       |
|-------------|-------------------------------------------------------------------|
| `Run name`  | MLflow run name (matches `--run-name`)                            |
| `val_iou`   | Best `val_iou` observed during the run, 4 decimal places          |
| `Δ vs best` | `val_iou` − previous best (signed, 4 decimals); `—` for row 1     |
| `Status`    | `kept`, `discarded`, or `crashed`                                 |
| `Commit`    | Short 7-char git hash of the `train.py` commit; empty if crashed  |
| `Notes`     | One-line summary of what changed and what you observed            |

---

## The plan — in execution order

Run these sequentially. **One change per run.** After each run, compare
`val_iou` to the previous best, log the result in the Run log, and decide
whether to keep or revert before proceeding.

### Step 0 — Re-baseline (do this first)

| Run name | Change | Purpose |
|----------|--------|---------|
| `baseline-resnet34-20ep` | None — current `train.py` as-is, ResNet34UNet, EPOCHS=20 | Establish a clean 20-epoch reference for this loop. All later runs in this loop compare against this number, not the old 90-min results. |

Do **not** change anything for this run. Just run `train.py` as it stands.

### Tier 1 — start here (highest expected payoff)

These three changes address the three biggest weaknesses in the current
setup: backbone too large for the dataset, decoder too simple for varied
object scales, and augmentation that doesn't synthesise new compositions.

1. **`smp-mitb0-deeplabv3plus`** — Switch model to
   `segmentation_models_pytorch.DeepLabV3Plus(encoder_name="mit_b0",
   encoder_weights="imagenet", in_channels=3, classes=1)`. Replaces the
   hand-rolled `ResNet34UNet`. MiT-B0 is a 3.7M-param transformer encoder
   with stronger sample efficiency than ImageNet ResNets on small datasets;
   DeepLabV3+ adds ASPP for multi-scale context (litter ranges from
   cigarette butts to bags). Keep loss, optimiser, augmentation, crop, LR
   identical for a clean comparison. Note: BN-freezing logic is no longer
   relevant — `smp` handles its own BN; just leave it as-is unless you see a
   problem.
2. **`copy-paste-aug`** — On top of whatever Tier 1 step 1 left as best,
   add copy-paste augmentation in `LitterDataset`. With probability ~0.5,
   sample a second `(image, mask)` pair, find connected components in the
   second mask, copy those pixel regions onto the current image (with
   optional small rotation/scale jitter), and OR the pasted regions into
   the current mask. With ~1062 train images this synthesises 10k+ varied
   compositions and is the single most-cited augmentation win for small
   segmentation datasets. Implement before `A.Normalize` so values are still
   in 0–255 uint8.
3. **`tta-ema`** — Two cheap, additive improvements rolled into one run:
   (a) maintain an exponential moving average of model weights during
   training (decay ~0.999) and evaluate the EMA copy at validation;
   (b) at validation only, run inputs and their h-flip through the model
   and average the logits before computing IoU. Both are well-known +0.5–3%
   tricks. Only export the EMA weights to ONNX.

### Tier 2 — once Tier 1 is exhausted

4. **`lovasz-bce-loss`** — Replace BCE+Dice with Lovász-Softmax + BCE
   (e.g. 0.5/0.5). Lovász directly optimises IoU (the eval metric).
   `smp.losses.LovaszLoss(mode="binary")` exists if you took Tier 1 step 1.
5. **`crop-512-grad-accum`** — Re-try CROP=512 (TACO has many small
   objects 384 blurs) with BATCH_SIZE=4 and gradient accumulation steps=2 to
   keep effective batch at 8. Now plausible because the MiT-B0 backbone is
   smaller than ResNet34.
6. **Backbone sweep** — same decoder, recipe, augmentation, varying only
   `encoder_name`:
   - `smp-mitb1-deeplabv3plus`
   - `smp-mobilenetv3l-deeplabv3plus`
   - `smp-effb0-deeplabv3plus`
   - `smp-effv2s-deeplabv3plus`

### Tier 3 — bigger swings, more effort (only if Tiers 1–2 plateau)

7. **`pretrain-trashcan-finetune`** — Pretrain encoder on TrashCan 1.0 /
   ZeroWaste-f / UAVVaste, fine-tune on TACO. Typically +3–5% IoU.
8. **`dinov2-small-frozen-decoder`** — DINOv2-small as a *frozen* feature
   extractor with a light decoder on top. DINOv2 has exceptional sample
   efficiency on small datasets when truly frozen.
9. **`multitask-supercategory-aux`** — Add a 28-supercategory auxiliary
   head trained jointly; collapse to binary at inference. Strong regulariser
   on small datasets. Requires a `prepare.py` change to emit per-class
   masks — **ask the human first**, since Rule 1 forbids editing
   `prepare.py`.

### What probably won't help much (skip unless results suggest otherwise)

- More attention modules on the existing U-Net (SE was already tried).
- Yet more LR sweeps on the existing recipe (already well-tuned in prior
  loops at 5e-4 → 1e-3 depending on backbone).
- Heavier backbones (B4 already saturates the dataset).
- Removing existing strong augmentations (GridDistortion, ElasticTransform,
  ColorJitter, GaussNoise) — prior loops confirmed each helps.

---

## Reference: what to optimise in `train.py`

Everything between `# ── Hyperparameters` and the bottom of the file is fair
game. Catalogued options below — use these to inform variations beyond the
plan above.

### Architecture
- Backbone: pretrained ResNet/EfficientNet/MiT/MobileNet/DINOv2 via
  `torchvision.models` or `segmentation_models_pytorch`.
- Decoder: U-Net, Unet++, DeepLabV3+, MAnet, UPerNet, FPN, PAN.
- Attention: CBAM, SE, lightweight transformer decoders.

### Loss
- BCE, Dice, Focal, Tversky / Focal-Tversky, Lovász-Softmax, weighted combos.
- Label smoothing applies only to BCE — never to Dice.

### Optimiser & schedule
- AdamW + OneCycleLR (current), SGD + cosine annealing, Lion.

### Regularisation
- Dropout, stochastic depth, mixup, EMA of weights.

### Data augmentation
- Geometric: RandomResizedCrop, flips, RandomRotate90, GridDistortion,
  ElasticTransform.
- Photometric: ColorJitter, GaussNoise, GaussianBlur.
- Compositional: CutMix, Mosaic, **copy-paste**.

### Inference-time
- TTA (h-flip, multi-scale), EMA weights at eval.

### Evaluation
- Val set ~188 images — apparent plateaus can be variance noise. For
  borderline results (Δ < 1%), re-run with 2–3 seeds before declaring a
  regression.

---

## Metric interpretation

| `val_iou` range | Interpretation                              |
|-----------------|---------------------------------------------|
| < 0.20          | Model barely segments anything              |
| 0.20 – 0.45     | Learning something, room for improvement    |
| 0.45 – 0.65     | Solid baseline                              |
| > 0.65          | Strong result for this dataset/epoch budget |

Reference: prior loop's best at the 90-min budget reached 0.6738. At the
default 20-epoch budget, the Tier 1 changes should aim to clear the
re-baseline by a meaningful margin.

---

## Agent loop

```
1. Read the last run's val_iou from MLflow (or stdout).
2. Pick the next step from "The plan" above (in order).
3. Edit train.py — one logical change matching that step.
4. Run: uv run python auto-research/train.py --run-name <name>
5. Compare val_iou with previous best.
6. If improved: keep change.
   If worse:     revert change before the next step.
7. Commit `train.py` and append a row to `auto-research/results.md` — see
   **Logging each run** above for the full workflow.
8. After 8–12 experiments, write a short summary to auto-research/findings.md.
```
