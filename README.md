# Few-shot defect inspection

Show the model **a handful of photos of a good part**. It flags defects in new photos and shows **where**. No training, no defect examples.

**[Live demo (Hugging Face Space)](https://huggingface.co/spaces/Rakif215/defect-inspection)** · built on frozen DINOv2 features · reproduces the CVPR 2026 SubspaceAD results on VisA

## What is in here

| | |
|---|---|
| `fsad/` | the detector (PCA residual + nearest-neighbour score, fused), a FastAPI service, a Gradio demo |
| `third_party/` | the two official implementations (SubspaceAD, AnomalyDINO), unmodified, used for the reproduction |
| `kaggle*/`, `scripts/` | GPU runs on Kaggle, dataset download, experiment and example scripts |
| `tests/` | 13 tests, including real-photo checks of the demo |
| `RESULTS.md` | every table below with the setup that produced it |

## Results

**1. Reproduction of the paper** (VisA, image AUROC, DINOv2-with-registers-giant, 672 px, seed 42, official code)

| Shots | Paper | This reproduction |
|---|---|---|
| 1 | 93.2 | 93.0 |
| 2 | 93.8 | 94.1 |
| 4 | 94.7 | 94.7 |

Within 0.3 points on all three. Pixel AUROC is 98.3-98.5 and AU-PRO 95.4-96.0.

**2. PCA vs nearest neighbour vs a fusion of both** (own implementation, 448 px, 4-shot, 12 VisA categories, giant backbone, 3 seeds)

| Scorer | Image AUROC | Pixel AUROC (8 held-out categories) |
|---|---|---|
| PCA residual | 94.1 | **96.8** |
| Nearest neighbour | **95.4** | 96.0 |
| Fusion | 95.1 | 96.5 |

No single scorer wins both tasks: nearest neighbour detects best, PCA localises best, the fusion sits within about 0.5 of the best on each. (An earlier small-model run suggested the fusion was a clear win; the giant-model confirmation did not support that.) Not directly comparable to table 1: lower resolution, fewer augmentations, own scoring code.

**3. The demo detector on real parts** (small backbone, 448 px, 4 reference photos, every VisA test image of the part)

| Part | Image AUROC | Defects caught | Good parts flagged |
|---|---|---|---|
| Capsules | 0.95 | 73% | 5% |
| Macaroni | 0.94 | 96% | 24% |
| Circuit board | 0.89 | 5% | 2% |

## Limitations (read before trusting a verdict)

- **The ranking is good, the cut-off is not always.** AUROC is 0.89-0.95, but the decision threshold is calibrated from only the reference photos, so where it lands varies by part (circuit board: very strict; macaroni: loose). The demo has a sensitivity slider for this.
- **Results depend on which reference photos you show.** On the circuit board, a different random choice of 4 references caught 84% of defects instead of 5%. More and more varied references help.
- The demo runs the small model on CPU so it answers in seconds; the paper-setting numbers use the giant model.
- Example photos are from VisA and were chosen to show typical cases, including one miss per part.

## Run it

```bash
pip install -r requirements.txt -r requirements-api.txt gradio
python app.py                                   # Gradio demo at http://127.0.0.1:7860
uvicorn fsad.api:app --port 8000                # JSON API + simple web page
docker build -t fsad . && docker run -p 8000:8000 fsad
python -m pytest tests -q
```

API: `POST /models` (upload 1-16 good photos, get a model id), `POST /models/{id}/inspect` (score, verdict, heatmap overlay).

Reproduce the paper numbers: `bash scripts/get_visa.sh`, then push `kaggle/` to a Kaggle GPU kernel (about 3.5 hours per shot count on a T4).

## Credits and licences

SubspaceAD (Lendering et al., CVPR 2026, [arXiv 2602.23013](https://arxiv.org/abs/2602.23013), Apache-2.0) and AnomalyDINO (Damm et al., WACV 2025, [arXiv 2405.14529](https://arxiv.org/abs/2405.14529), Apache-2.0) are the methods; DINOv2 is Apache-2.0. VisA (Amazon) is CC BY 4.0. MVTec AD is not used (non-commercial licence).
This repository is an independent reproduction and demo; it does not claim the methods as its own.
