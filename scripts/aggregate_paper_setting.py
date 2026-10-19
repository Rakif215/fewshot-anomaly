"""Builds the paper-setting table (RESULTS.md) from results_exp/giant_paper_672/<category>_k<shots>.json.

Each file holds {category: {variant: [image_auroc, pixel_auroc]}}; the table is the mean over the 12 VisA categories.
    python scripts/aggregate_paper_setting.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent / "results_exp" / "giant_paper_672"
CATS = "candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum".split()
VARIANTS = [("PCA residual (99% variance)", "pca_ev0.99"), ("Nearest neighbour (k=1)", "knn_k1"), ("Nearest neighbour (k=3)", "knn_k3"),
            ("Fusion PCA 99% + kNN 1", "fusion_pca99+knn1")]


def load(cat, k):
    return json.loads((ROOT / f"{cat}_k{k}.json").read_text())[cat]


def mean(xs):
    return sum(xs) / len(xs)


print("| Scorer | 1-shot image / pixel | 2-shot image / pixel | 4-shot image / pixel |")
print("|---|---|---|---|")
for label, key in VARIANTS:
    cells = []
    for k in (1, 2, 4):
        vals = [load(c, k)[key] for c in CATS]
        cells.append(f"{mean([v[0] for v in vals]) * 100:.1f} / {mean([v[1] for v in vals]) * 100:.1f}")
    print(f"| {label} | " + " | ".join(cells) + " |")

print("\nImage AUROC at 4 shots, per category: PCA 99% | kNN k=3 | fusion")
wins = 0
for c in CATS:
    r = load(c, 4)
    p, n, f = (r[k][0] * 100 for k in ("pca_ev0.99", "knn_k3", "fusion_pca99+knn1"))
    wins += n > p
    print(f"  {c:11s} {p:5.1f} | {n:5.1f} | {f:5.1f}")
print(f"kNN k=3 beats PCA in {wins} of {len(CATS)} categories at 4 shots")
