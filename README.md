# Few-shot industrial anomaly detection: SubspaceAD vs. AnomalyDINO

Reproduction and comparison of two training-free, few-shot anomaly detectors that use the same frozen
DINOv2 patch features:

- **SubspaceAD** — Lendering et al., *CVPR 2026* ([arXiv 2602.23013](https://arxiv.org/abs/2602.23013), Apache-2.0): fits PCA to normal patches, scores by reconstruction residual.
- **AnomalyDINO** — Damm et al., *WACV 2025* ([arXiv 2405.14529](https://arxiv.org/abs/2405.14529), Apache-2.0): nearest-neighbour distance to a memory bank of normal patches.

Both official implementations are included unmodified as git submodules in `third_party/`.

## Run in a GitHub Codespace (no GPU needed)

1. **Code → Codespaces → Create codespace** (4-core / 16 GB). Setup runs automatically (`scripts/setup.sh`).
2. `bash scripts/get_visa.sh` — downloads VisA (Amazon, CC BY 4.0, ~1.9 GB) and builds the 1-class split.
3. `bash scripts/smoke_visa.sh` — quick CPU check: both methods, 1-shot, small DINOv2.

Datasets and results stay in the Codespace (`datasets/`, `results/` are git-ignored).

## Status

- [ ] Smoke test (small backbone) — pipeline runs, numbers sensible
- [ ] Reproduction at paper settings (giant backbone, 672 px) on VisA and MVTec-AD
- [ ] Analysis: where PCA beats nearest-neighbour and why; speed/memory; MVTec AD 2 public split

Dataset licences: VisA CC BY 4.0; MVTec-AD CC BY-NC-SA 4.0 (download from MVTec, not redistributed here).
