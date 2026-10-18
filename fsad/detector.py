"""Few-shot anomaly detector: fit on a handful of normal images, then score new images and produce a heatmap.

Image score = mean of the top 1% patch scores (decides "defect or not"); the heatmap shows where.
Patch score = fusion of the PCA residual and the nearest-neighbour distance (see RESULTS.md for why).
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

from .features import Extractor
from .scorers import image_score, knn_scorer, normalise, pca_scorer

AUGMENTATIONS = (
    lambda i: i,
    lambda i: i.transpose(Image.FLIP_LEFT_RIGHT),
    lambda i: i.transpose(Image.FLIP_TOP_BOTTOM),
    lambda i: i.rotate(90),
    lambda i: i.rotate(180),
    lambda i: i.rotate(270),
)


class FewShotDetector:
    def __init__(self, ckpt="facebook/dinov2-with-registers-small", res=448, layers=(-4, -5), ev=0.99, knn_k=1, augment=True, extractor=None):
        self.extractor = extractor or Extractor(ckpt, res, layers)
        self.ev, self.knn_k, self.augment = ev, knn_k, augment
        self._pca = self._knn = None
        self._feats = None
        self._scale = None        # (mean, std) of the raw scores of NORMAL held-out patches, per scorer
        self.threshold = None     # image-score level above which an image is called defective
        self.n_reference = 0

    @property
    def fitted(self) -> bool:
        return self._pca is not None

    def _scorers(self, feats):
        return pca_scorer(feats, ev=self.ev), knn_scorer(feats, k=self.knn_k)

    @staticmethod
    def _raw_with(scorers, flat):
        return np.stack([scorers[0](flat), scorers[1](flat)], axis=1)

    def fit(self, images: list[Image.Image], margin: float = 1.0) -> "FewShotDetector":
        """Learn "normal" from the reference images and calibrate the decision threshold by leave-one-out:
        every reference image is scored by a model built from the OTHER references, so the scale and the
        threshold reflect how normal images that were not seen actually score."""
        if not images:
            raise ValueError("at least one reference image is required")
        hid = self.extractor.model.config.hidden_size
        augs = AUGMENTATIONS if self.augment else AUGMENTATIONS[:1]
        per_image = [[self.extractor(a(img.convert("RGB"))).reshape(-1, hid).astype(np.float32) for a in augs] for img in images]
        self._feats = np.concatenate([v for views in per_image for v in views])
        self.n_reference = len(images)

        # Held-out normal samples: (training features, views to score). Every view (original + flips/rotations) of the
        # held-out image is scored, so the threshold rests on many scores, not on the maximum of a handful.
        if len(images) >= 2:
            units = [(np.concatenate([v for j, views in enumerate(per_image) if j != i for v in views]), per_image[i])
                     for i in range(len(images))]
        else:   # one reference: train on the original, hold out its flipped / rotated views
            units = [(per_image[0][0], per_image[0][1:] or [per_image[0][0]])]
        raws = []
        for train, held in units:
            sc = self._scorers(train)
            raws.append([self._raw_with(sc, h) for h in held])
        flat = np.concatenate([r for unit in raws for r in unit])
        self._scale = (flat.mean(0), flat.std(0) + 1e-9)
        scores = [image_score(((r - self._scale[0]) / self._scale[1]).mean(1)) for unit in raws for r in unit]
        self.threshold = float(max(scores) * margin) if max(scores) > 0 else 3.0
        self._pca, self._knn = self._scorers(self._feats)
        return self

    def state(self) -> dict:
        """Plain arrays and numbers only (picklable), so a fitted detector can cross process boundaries."""
        return {"feats": self._feats, "scale": self._scale, "threshold": self.threshold, "ev": self.ev, "knn_k": self.knn_k, "n": self.n_reference}

    @classmethod
    def from_state(cls, st: dict, extractor) -> "FewShotDetector":
        det = cls(ev=st["ev"], knn_k=st["knn_k"], extractor=extractor)
        det._feats, det._scale, det.threshold, det.n_reference = st["feats"], st["scale"], st["threshold"], st["n"]
        det._pca, det._knn = det._scorers(det._feats)
        return det

    def _raw(self, flat: np.ndarray) -> np.ndarray:
        return np.stack([self._pca(flat), self._knn(flat)], axis=1)

    def score(self, image: Image.Image, out_size: int | None = None) -> dict:
        """Return {"score": float, "heatmap": float32 array (H, W) in 0..1-ish, "grid": (h, w)}."""
        if not self.fitted:
            raise RuntimeError("fit() the detector on reference images first")
        f = self.extractor(image.convert("RGB"))
        h, w, c = f.shape
        raw = self._raw(f.reshape(-1, c))
        mu, sd = self._scale
        z = ((raw - mu) / sd).mean(1)                       # fused, in units of "reference standard deviations"
        patch = z.reshape(h, w).astype(np.float32)
        size = out_size or image.size[0]
        heat = gaussian_filter(zoom(patch, size / h, order=1), sigma=max(1.0, size / 64))
        return {"score": image_score(patch), "heatmap": heat, "grid": (h, w)}

    def threshold_hint(self, images=None) -> float:
        """Kept for compatibility: the leave-one-out threshold computed in fit()."""
        return float(self.threshold)
