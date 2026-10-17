"""Cache features once per category, then compare scorers cheaply.
Usage: python -m fsad.experiments --data DATA --out OUT --k 4 --cats candle capsules ...
"""
import argparse, glob, json, os, random, time
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from .features import Extractor
from .scorers import pca_scorer, knn_scorer, normalise, image_score

AUG = [lambda i: i, lambda i: i.transpose(Image.FLIP_LEFT_RIGHT), lambda i: i.transpose(Image.FLIP_TOP_BOTTOM),
       lambda i: i.rotate(90), lambda i: i.rotate(180), lambda i: i.rotate(270)]


def load(cat_dir):
    train = sorted(glob.glob(f"{cat_dir}/train/good/*"))
    good = sorted(glob.glob(f"{cat_dir}/test/good/*"))
    bad = sorted(glob.glob(f"{cat_dir}/test/bad/*"))
    return train, good, bad


def gt_mask(cat_dir, path, size):
    m = f"{cat_dir}/ground_truth/bad/{os.path.splitext(os.path.basename(path))[0]}.png"
    if not os.path.exists(m):
        return np.zeros((size, size), bool)
    return np.asarray(Image.open(m).convert("L").resize((size, size), Image.NEAREST)) > 0


def cache_features(ext, cat_dir, cat, k, seed, cache):
    f = f"{cache}/{cat}_k{k}_s{seed}.npz"
    if os.path.exists(f):
        d = np.load(f)
        return d["train"], d["test"], d["labels"], d["masks"]
    train, good, bad = load(cat_dir)
    random.Random(seed).shuffle(train)
    tr = np.concatenate([ext(a(Image.open(p))).reshape(-1, ext.model.config.hidden_size) for p in train[:k] for a in AUG])
    test = np.stack([ext(Image.open(p)) for p in good + bad])
    labels = np.array([0] * len(good) + [1] * len(bad))
    masks = np.stack([gt_mask(cat_dir, p, 128) for p in good + bad])
    np.savez_compressed(f, train=tr.astype(np.float16), test=test.astype(np.float16), labels=labels, masks=masks)
    return tr, test, labels, masks


def up(pm, size=128):
    from scipy.ndimage import zoom
    return zoom(pm, size / pm.shape[0], order=1)


def evaluate(train, test, labels, masks, variants):
    train, test = train.astype(np.float32), test.astype(np.float32)
    n, h, w, c = test.shape
    flat = test.reshape(-1, c)
    base = {}
    out = {}
    for name, spec in variants.items():
        parts = []
        for kind, kw in spec:
            key = (kind, tuple(sorted(kw.items())))
            if key not in base:
                fn = pca_scorer(train, **kw) if kind == "pca" else knn_scorer(train, **kw)
                base[key] = normalise(fn(flat))
            parts.append(base[key])
        s = np.mean(parts, axis=0).reshape(n, h, w)
        img = [image_score(x) for x in s]
        pix = np.stack([up(x) for x in s])
        out[name] = (roc_auc_score(labels, img), roc_auc_score(masks.reshape(-1), pix.reshape(-1)))
    return out


VARIANTS = {
    "pca_ev0.90": [("pca", {"ev": 0.90})],
    "pca_ev0.95": [("pca", {"ev": 0.95})],
    "pca_ev0.99": [("pca", {"ev": 0.99})],
    "knn_k1": [("knn", {"k": 1})],
    "knn_k3": [("knn", {"k": 3})],
    "fusion_pca99+knn1": [("pca", {"ev": 0.99}), ("knn", {"k": 1})],
    "fusion_pca95+knn1": [("pca", {"ev": 0.95}), ("knn", {"k": 1})],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cats", nargs="+", required=True)
    ap.add_argument("--ckpt", default="facebook/dinov2-with-registers-small"); ap.add_argument("--res", type=int, default=448)
    ap.add_argument("--layers", default="-4,-5")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/cache", exist_ok=True)
    ext = Extractor(a.ckpt, a.res, [int(x) for x in a.layers.split(",")])
    res = {}
    for cat in a.cats:
        t = time.time()
        tr, te, lb, mk = cache_features(ext, f"{a.data}/{cat}", cat, a.k, a.seed, f"{a.out}/cache")
        res[cat] = evaluate(tr, te, lb, mk, VARIANTS)
        print(cat, f"{time.time()-t:.0f}s", {k: f"{v[0]:.3f}/{v[1]:.3f}" for k, v in res[cat].items()}, flush=True)
        json.dump(res, open(f"{a.out}/results_k{a.k}.json", "w"), indent=1)
