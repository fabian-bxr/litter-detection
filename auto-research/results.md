# Run log

One row per experiment. Append a new row after each run — see **Logging each
run** in `program.md` for the workflow.

| Run name                | val_iou  | Δ vs best | Status    | Commit | Notes                                                                                                                                     |
|-------------------------|----------|-----------|-----------|--------|-------------------------------------------------------------------------------------------------------------------------------------------|
| baseline-resnet34-20ep  | 0.5600   | —         | kept      |        | Re-baseline ResNet34UNet, 20 epochs, `train.py` unchanged. 161s on CUDA, 24.3M params. Reference for this loop.                           |
| smp-mitb0-deeplabv3plus | 0.5868   | +0.0268   | kept      |        | smp.DeepLabV3Plus(mit_b0, imagenet). 4.1M params (~6× smaller), 107s. Same loss/optim/aug/crop/LR.                                        |
| copy-paste-aug          | 0.5930   | +0.0062   | kept      |        | Per-component copy-paste w/ rotation+scale+translation jitter, p=0.5. Borderline (<1%) but train_iou rose (0.675 vs 0.643).               |
| tta-ema                 | 0.6037   | +0.0107   | kept      |        | EMA decay=0.999, h-flip TTA at val. Best at epoch 8. Note: val_iou≈0 in epochs 1-3 due to EMA warmup.                                     |
| lovasz-bce-loss         | 0.6477   | +0.0440   | kept      |        | 0.5·BCE + 0.5·LovaszLoss(binary) replaces BCE+Dice. val_iou climbed steadily through all 20 epochs. Big win.                              |
| crop-512-grad-accum     | 0.6305   | -0.0172   | discarded |        | CROP=512, BS=4, accum=2 (eff. batch 8). 257s (2.3× slower). Underconverges in 20ep — val_iou still climbing at end. Reverted.             |
| smp-mitb1-deeplabv3plus | 0.7015   | +0.0538   | kept      |        | mit_b1 encoder (14.3M params, 3.5× larger than mit_b0). 183s. Beats prior loop's 90-min ResNet34 (0.6738). val_iou still climbing at end. |
| smp-mobilenetv3l-deeplabv3plus | 0.6660 | -0.0355  | discarded | ce6a821 | tu-mobilenetv3_large_100 encoder (~4.7M params). 98s (~2× faster). Peaked at ep17. Backbone too weak vs mit_b1. Reverted.                  |
| smp-effb0-deeplabv3plus | 0.6714 | -0.0301   | discarded | 8c4ef60 | efficientnet-b0 encoder (~6.3M params). 126s. val_iou still climbing at ep20 (0.6714). Under mit_b1 by ~3%. Reverted.                      |
| smp-effv2s-deeplabv3plus | 0.7377 | +0.0362  | kept      | 73e5363 | tu-tf_efficientnetv2_s encoder (~20.7M params). 271s. val_iou still climbing at ep20 (0.7377). New best; clears mit_b1 by 3.6%.           |
| smp-effv2s-unetpp      | 0.7203   | -0.0174   | discarded | 088d3d6 | UnetPlusPlus decoder (22.6M params) + effv2s. 334s. Slower aha-moment (val_iou=0 until ep9 under EMA warmup). DLv3+ ASPP wins. Reverted. |
| smp-effv2s-manet       | 0.7234   | -0.0143   | discarded | 4c9a46b | MAnet decoder (24.2M params) + effv2s. 242s. Even slower EMA warmup (val_iou=0 until ep13). ASPP still wins multi-scale. Reverted.     |
| faster-ema-0p995       | 0.7253   | -0.0124   | discarded | 6f21d83 | EMA_DECAY 0.999→0.995. EMA warms up by ep2 (val_iou=0.22 vs ~0) but noisier; peaks at ep16 then fluctuates 0.69–0.72. 0.999 is right.  |
| lovasz-heavy-0p7       | 0.7466   | +0.0089   | kept      | f5725c7 | Loss weighting 0.5/0.5 → 0.3 BCE / 0.7 Lovász. Still climbing at ep20. New best — heavier IoU-surrogate weighting pays off directly.  |
| lovasz-heavy-0p9       | 0.7486   | +0.0020   | kept      | db88866 | 0.3/0.7 → 0.1 BCE / 0.9 Lovász. Still climbing at ep20 (0.7407→0.7433→0.7486). Borderline (<1%); diminishing returns from this direction. |
| effv2s-dlv3p-60ep      | 0.7729   | +0.0243   | ablation  | db88866 | Human-authorized longer-budget run: same train.py as lovasz-heavy-0p9, `--epochs 60`. Peaks at ep39 then plateaus/oscillates 0.74–0.77 through ep60. 20-ep cap was binding; true ceiling is ~0.77. Train_iou 0.80→0.86 while val_iou flat → mild overfit after ep40. 778s. |
| wd-3e-4                | 0.7404   | -0.0082   | discarded | 86af696 | WEIGHT_DECAY 1e-4→3e-4 at 20 ep. Convergence slowed (val_iou 0.73 at ep19 vs 0.74 baseline); no higher peak. Reverted. |
| cp-prob-0p75           | 0.7292   | -0.0194   | discarded | 6a56dc9 | COPY_PASTE_PROB 0.5→0.75. Too much augmentation noise; convergence slowed, peak lower. 0.5 was calibrated right. Reverted. |
| lr-5e-4                | 0.6756   | -0.0730   | discarded | ba6a521 | LR 2e-4→5e-4. Large regression; EMA warmup extended to ep8-10. 2e-4 right for this backbone+recipe despite program's 5e-4-1e-3 note. |
| pct-start-0p1          | 0.7439   | -0.0047   | discarded | 475a2cb | OneCycle pct_start 0.05→0.1. Borderline (<1%). Slightly slower warmup, no peak gain. Reverted. |
| lion-5e-5              | 0.7220   | -0.0266   | discarded | 037931d | AdamW→Lion, LR 2e-4→5e-5. Slower warmup (val_iou≈0 through ep7), underconverges. AdamW well-tuned. Reverted. |
| mixup-0p3              | 0.7435   | -0.0051   | discarded | b88d906 | Batch-level mixup (p=0.3, α=0.2) on top of copy-paste. Borderline (<1%); still climbing at ep20. Copy-paste already covers that axis. Reverted. |
| effv2s-dlv3p-40ep      | 0.7614   | +0.0128   | ablation  | 1eaafbc | Human-authorized 40-ep run at best recipe (no code change). Peaks at ep38 (0.7614); val_iou plateau 0.75–0.76 from ep27 onward. 493s. Beats 20-ep best by +1.3%, but trails 60-ep peak (0.7729 at ep39) by -1.1% — OneCycleLR shape differs with total epochs (pct_start=0.05 → 2ep warmup vs 3ep), so not directly comparable. 40 epochs is *not* a strict subset of 60. |
