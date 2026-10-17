"""Patch-level anomaly scorers. Each takes (N, C) normal patches and returns a function scoring (M, C) patches."""
import numpy as np


def pca_scorer(train: np.ndarray, ev: float = 0.99, whiten: bool = False):
    mu = train.mean(0)
    u, s, vt = np.linalg.svd(train - mu, full_matrices=False)
    var = s**2 / (s**2).sum()
    k = int(np.searchsorted(np.cumsum(var), ev) + 1)
    comps = vt[:k]
    scale = (s[:k] / np.sqrt(len(train))) if whiten else None

    def score(x):
        z = x - mu
        proj = z @ comps.T
        if whiten:
            proj = proj / (scale + 1e-6)  # unused in the residual; kept for Mahalanobis-style variants
        res = z - (z @ comps.T) @ comps
        return np.linalg.norm(res, axis=1)

    return score


def knn_scorer(train: np.ndarray, k: int = 1):
    import torch
    bank = torch.from_numpy(train).float()

    def score(x):
        out = []
        for i in range(0, len(x), 4096):  # chunk the queries: a full distance matrix does not fit in RAM
            q = torch.from_numpy(x[i:i + 4096]).float()
            out.append(torch.cdist(q, bank).topk(k, largest=False).values.mean(1).numpy())
        return np.concatenate(out)

    return score


def normalise(s: np.ndarray, lo_q: float = 50, hi_q: float = 99.9):
    """Unsupervised rescale using the score distribution of the images being scored (no labels)."""
    lo, hi = np.percentile(s, [lo_q, hi_q])
    return np.clip((s - lo) / (hi - lo + 1e-9), 0, None)


def image_score(patch_scores: np.ndarray, top: float = 0.01):
    flat = np.sort(patch_scores.reshape(-1))[::-1]
    return float(flat[: max(1, int(len(flat) * top))].mean())
