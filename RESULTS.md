# Results: SubspaceAD vs. AnomalyDINO on VisA (small-scale baseline)

Setup (identical for both): VisA, 12 categories, 1-shot, one seed (SubspaceAD seed 42, AnomalyDINO seed 0), DINOv2 **small** at 448 px,
CPU only (4 cores). This is a pipeline check and a baseline, **not** the paper setting (the paper uses the giant model at 672 px
and averages several seeds). Image-level AUROC, in %.

| Category | SubspaceAD | AnomalyDINO | Better by |
|---|---|---|---|
| candle | 87.4 | 87.7 | -0.4 |
| capsules | 90.8 | **98.2** | -7.5 |
| cashew | **97.3** | 78.5 | +18.8 |
| chewinggum | 97.9 | 98.3 | -0.3 |
| fryum | **96.9** | 95.5 | +1.4 |
| macaroni1 | **86.3** | 85.4 | +0.8 |
| macaroni2 | **70.7** | 60.2 | +10.4 |
| pcb1 | **94.1** | 88.1 | +6.0 |
| pcb2 | 77.4 | **85.2** | -7.8 |
| pcb3 | 71.6 | **84.7** | -13.1 |
| pcb4 | **95.7** | 86.1 | +9.6 |
| pipe_fryum | **96.3** | 88.7 | +7.5 |
| **Average** | **88.5** | **86.4** | **+2.1** |

SubspaceAD (also: pixel AUROC 96.8, AU-PRO 90.0 on average) is ahead on 7 of 12 categories. The winner changes strongly by category
(SubspaceAD: cashew +19, macaroni2 +10, pcb4 +10; AnomalyDINO: pcb3 +13, capsules +8, pcb2 +8), so the two methods fail on different
kinds of defects. For reference, the papers report one-shot VisA image AUROC of 93.2 (SubspaceAD) and 87.4 (AnomalyDINO) with the giant model.

Speed on 4 CPU cores: SubspaceAD about 0.69 s per test image; AnomalyDINO about 0.56 s per test image (small model, 448 px).
DINOv2 forward pass at 672 px: base 4.0 s, giant 42 s per image (CPU), so the paper setting needs a GPU.

Reproduce: `bash scripts/get_visa.sh && bash scripts/smoke_visa.sh` (or the manual GitHub Actions workflow `smoke-visa`).
Known quirks handled in `scripts/compat/`: NumPy 2 (`np.trapz`), AnomalyDINO hardcoding `cuda`.

## Paper setting (DINOv2-with-registers-giant, 672 px, seed 42), VisA, SubspaceAD

| Shots | Image AUROC (paper) | Image AUROC (ours) | Pixel AUROC (ours) | AU-PRO (ours) |
|---|---|---|---|---|
| 1 | 93.2 | 93.0 | 98.3 | 95.4 |
| 2 | 93.8 | 94.1 | 98.4 | 95.7 |
| 4 | 94.7 | **94.7** | 98.5 | 96.0 |

Run on a Kaggle T4. Only the saliency layer's attention map is kept (`scripts/compat/sitecustomize.py`) to fit memory; scores are unaffected.

## Improvement experiment: PCA residual + nearest-neighbour fusion (small backbone, 448 px, 4-shot, VisA)

Own compact implementation (`fsad/`): 6 flip/rotation augmentations per shot, mean of layers -4/-5, image score = mean of top 1% patch scores.
Variants were compared on 4 tuning categories (candle, capsules, macaroni1, pcb1); the chosen variant is reported on the 8 held-out categories.
Scores are rescaled per category from the test-patch score distribution (no labels used).

| Variant | Tuning image AUROC | Held-out image AUROC | Held-out pixel AUROC |
|---|---|---|---|
| PCA residual (99% variance) | 91.6 | 88.5 | 95.6 |
| Nearest neighbour (k=1) | 90.9 | 89.4 | 94.5 |
| **Fusion PCA 99% + kNN 1** | **92.3** | **89.8** | **95.6** |

Fusion is +1.3 over PCA alone and +0.4 over kNN alone on held-out categories. Raw numbers: `results_exp/small_k4.json`.
Caveats: single seed, small model, simplified augmentation; not yet confirmed with the giant backbone.

## Confirmation with the giant backbone (448 px, 4-shot, 3 seeds, all 12 VisA categories)

Same `fsad/` pipeline, DINOv2-with-registers-giant, layers -12..-18. Mean over seeds 42/43/44 (± std across seeds). Raw: `results_exp/giant/`.

| Variant | Image AUROC (12 cats) | Image AUROC (8 held-out) | Pixel AUROC (8 held-out) |
|---|---|---|---|
| PCA residual (99%) | 94.1 ± 0.3 | 93.8 ± 0.3 | **96.8** |
| Nearest neighbour (k=3) | **95.4 ± 0.4** | **95.5 ± 0.5** | 96.0 |
| Fusion PCA 99% + kNN 1 | 95.1 ± 0.4 | 95.1 ± 0.4 | 96.5 |

Findings:
- The small-model result did not carry over as a win: with the giant backbone, nearest neighbour alone is best at image level (+1.3 over PCA) and fusion is slightly below it (-0.3).
- PCA gives the best pixel-level localisation; kNN the best image-level detection; fusion sits within 0.3-0.5 of the best on both, i.e. the best balance rather than the best on either.
- Not directly comparable to the paper's table: 448 px instead of 672, 6 augmentations instead of 30, own scoring code.
