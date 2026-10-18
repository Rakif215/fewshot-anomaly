"""Patch-level anomaly scorers. Each takes (N, C) normal patches and returns a function scoring (M, C) patches."""
import os

import numpy as np


def pca_scorer(train: np.ndarray, ev: float = 0.99):
    """Residual of the projection onto the principal subspace that explains `ev` of the variance."""
    x = train.astype(np.float32)
    mu = x.mean(0)
    xc = x - mu
    cov = (xc.T @ xc) / max(len(xc) - 1, 1)
    vals, vecs = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(vals)[::-1]
    vals, vecs = np.clip(vals[order], 0, None), vecs[:, order]
    k = int(np.searchsorted(np.cumsum(vals) / vals.sum(), ev) + 1)
    comps = vecs[:, :k].T.astype(np.float32)

    def score(x_new):
        z = x_new.astype(np.float32) - mu
        res = z - (z @ comps.T) @ comps
        return np.linalg.norm(res, axis=1)

    return score


def knn_scorer(train: np.ndarray, k: int = 1):
    """Mean distance to the k nearest normal patches. Uses the GPU when there is one.

    The bank stays on the CPU and moves to the GPU inside each call, so the scorer also survives
    environments where the GPU is only attached for the duration of a call (Hugging Face ZeroGPU).
    """
    import torch
    bank_cpu = torch.from_numpy(train).float()

    def score(x):
        dev = os.environ.get("FSAD_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        bank = bank_cpu.to(dev)
        chunk = 2048 if dev == "cuda" else 4096
        out = []
        for i in range(0, len(x), chunk):  # chunk the queries: a full distance matrix does not fit in memory
            q = torch.from_numpy(x[i:i + chunk]).float().to(dev)
            out.append(torch.cdist(q, bank).topk(k, largest=False).values.mean(1).cpu().numpy())
        return np.concatenate(out)

    return score


def normalise(s: np.ndarray, lo_q: float = 50, hi_q: float = 99.9):
    """Unsupervised rescale using the score distribution of the images being scored (no labels)."""
    lo, hi = np.percentile(s, [lo_q, hi_q])
    return np.clip((s - lo) / (hi - lo + 1e-9), 0, None)


def image_score(patch_scores: np.ndarray, top: float = 0.01):
    flat = np.sort(patch_scores.reshape(-1))[::-1]
    return float(flat[: max(1, int(len(flat) * top))].mean())
