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
| 4 | 94.7 | running | | |

Run on a Kaggle T4. Only the saliency layer's attention map is kept (`scripts/compat/sitecustomize.py`) to fit memory; scores are unaffected.
