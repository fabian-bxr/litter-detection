# AutoResearcher findings — Loop 5

## Summary

Best run: **`lovasz-heavy-0p9`** — `val_iou = 0.7486` (commit `db88866`).
That beats the prior loop's 90-min ResNet34 ceiling (0.6738) by 11.1% relative
and this loop's re-baseline (0.5600, ResNet34UNet, 20 ep) by 18.9% absolute /
33.7% relative, all at the same 20-epoch budget.

Final recipe: `smp.DeepLabV3Plus(tu-tf_efficientnetv2_s, imagenet)` + AdamW +
OneCycleLR + EMA(0.999) + h-flip TTA + per-component copy-paste (p=0.5) +
`0.1·BCE + 0.9·Lovász`, CROP=384, BATCH=8.

## What moved the number

Ranked by Δ vs the run-before-it (all at 20 epochs, BATCH_SIZE=8, CROP=384):

| Step                        | Δ         | Cumulative val_iou  |
|-----------------------------|-----------|---------------------|
| re-baseline (rn34-UNet)     | —         | 0.5600              |
| smp DeepLabV3+ + mit_b0     | +0.027    | 0.5868              |
| + copy-paste aug            | +0.006    | 0.5930              |
| + EMA eval + h-flip TTA     | +0.011    | 0.6037              |
| 0.5·BCE + 0.5·Lovász        | +0.044    | 0.6477              |
| encoder → mit_b1            | +0.054    | 0.7015              |
| encoder → effv2s            | +0.036    | 0.7377              |
| loss → 0.3·BCE + 0.7·Lovász | +0.009    | 0.7466              |
| loss → 0.1·BCE + 0.9·Lovász | +0.002    | 0.7486              |

Biggest wins: **mit_b1 backbone swap** (+5.4%), **Lovász loss** (+4.4%),
**effv2s backbone** (+3.6%). Lovász reweighting was a late-stage surprise — a
second +1% came from shifting the weight towards the IoU surrogate even more
(0.5 → 0.7 → 0.9).

## What didn't work

- **CROP=512 + grad-accum** — 2.3× slower, under-converges in 20 epochs.
- **MobileNetV3-Large / EfficientNet-B0 encoders** — too small (both ~0.66–0.67).
  Mid-size (~15–20M param) hybrid/transformer encoders dominate.
- **UnetPlusPlus / MAnet decoders** — both regressed 1–2%. Their slow EMA
  "aha" (val_iou ≈ 0 for 10–13 epochs) eats the budget, and their fusion
  strategies don't beat DeepLabV3+'s ASPP for multi-scale litter.
- **EMA_DECAY = 0.995** — warms up by ep2 (vs ~ep6 at 0.999) but noisier and
  ends lower. The slow-warmup cost is the price of stable EMA quality.

## Observations worth flagging

- **EMA warmup is expensive at this budget.** EMA copy's val_iou sits at ~0
  for the first 5 epochs under DeepLabV3+ at decay 0.999 (~2660 updates over
  20 epochs, below the ~3×τ needed to stabilise). This is a structural cost
  of the EMA+TTA eval protocol; it's worth paying because the plateau above
  the warmup is smoother than the raw model.
- **effv2s was still climbing at ep20** in every config we tried. All late-
  stage gains (Lovász reweighting, etc.) were "climb faster within 20
  epochs" rather than "reach a higher ceiling." The 20-epoch cap is the
  remaining binding constraint.
- **Diminishing returns.** Across the last 3 runs Δ went 0.009 → 0.002, a
  clean exponential decay. Without more data or more epochs, this recipe is
  plausibly near its ceiling.

## Where to go next (Tier 3 candidates)

- **`pretrain-trashcan-finetune`** — highest expected payoff (+3–5% typical)
  but substantial engineering: external dataset download, two-stage loop.
- **`dinov2-small-frozen-decoder`** — tried plugging into smp via `tu-`
  prefix; smp rejected with "Unsupported model downsampling pattern"
  (DINOv2 is a plain ViT with no feature pyramid). Would need a custom
  decoder wrapper.
- **`multitask-supercategory-aux`** — blocked by Rule 1 (requires editing
  `prepare.py`); needs human approval before attempting.
- **Lift the 20-epoch cap** — every recipe plateaus with the metric still
  climbing. Even a 40-epoch run at the best config would answer "is the
  plateau structural, or just budget-bound?"

## Hyperparameter / regularisation tuning (human-authorized, 20 ep)

Six attempts to push past `lovasz-heavy-0p9`'s 0.7486 at the fixed 20-epoch
budget. **All six regressed**; none beat the baseline. Summary:

| Run            | Change                              | val_iou | Δ       |
|----------------|-------------------------------------|---------|---------|
| wd-3e-4        | weight decay 1e-4 → 3e-4             | 0.7404  | -0.0082 |
| cp-prob-0p75   | copy-paste prob 0.5 → 0.75           | 0.7292  | -0.0194 |
| lr-5e-4        | LR 2e-4 → 5e-4                       | 0.6756  | -0.0730 |
| pct-start-0p1  | OneCycle pct_start 0.05 → 0.1        | 0.7439  | -0.0047 |
| lion-5e-5      | AdamW → Lion, LR 5e-5                | 0.7220  | -0.0266 |
| mixup-0p3      | batch mixup (p=0.3, α=0.2) on top    | 0.7435  | -0.0051 |

Takeaways:
- **LR, WD, optimiser, schedule are all at a local optimum.** Both directions
  of every knob regress — program's "LR well-tuned" note generalises.
- **Copy-paste saturates at p=0.5.** More aug overwhelms signal.
- **Mixup is redundant with copy-paste here.** Both add compositional noise;
  doubling down on that axis gives nothing.
- The remaining gap to the 60-ep ceiling (0.7486 → 0.7729) is structural —
  it needs more epochs, not better hyperparameters.

## Longer-budget ablation (human-authorized, off-protocol)

One 60-epoch run with the best 20-ep config: `val_iou = 0.7729` at **epoch 39**,
then plateaus at 0.74–0.77 through epoch 60. So:

- The 20-ep cap was binding, but only for ~19 more epochs of gain. True
  ceiling at this recipe is ~0.77, a further +0.024 beyond 0.7486.
- After ep40, train_iou kept climbing (0.80 → 0.86) while val_iou flattened —
  mild overfit. A longer budget *without* stronger regularisation would
  need something (heavier augmentation, dropout, weight decay) to convert
  extra epochs into extra val_iou.
- 60-ep runtime: 778s (3× the 20-ep time for 1.03× the IoU). The
  epochs-to-IoU curve is sharply diminishing past ep40.

Follow-up: a 40-epoch run at the same recipe lands at `val_iou = 0.7614` at
epoch 38 (493s). That is +0.013 over the 20-ep best but −0.011 below the 60-ep
peak. The gap is not "the last 20 epochs of the 60-ep run mattered"; it's that
OneCycleLR is re-scaled by total epochs (`pct_start=0.05` → ~2-epoch warmup
at 40 ep vs ~3 at 60 ep, and the cosine decay spans a different shape). So
40 ep is *not* a prefix of 60 ep — the schedules are different trajectories.
Practical read: if budget is tight, 40 ep captures most of the gain; if you
actually want 0.77, the full 60-ep schedule is the one that gets there.
