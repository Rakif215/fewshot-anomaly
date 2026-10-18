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
        self.n_reference = 0
        # Patch-score scale learnt from the reference images themselves (leave-one-out would be better; see fit()).
        self._ref_scale = None

    @property
    def fitted(self) -> bool:
        return self._pca is not None

    def fit(self, images: list[Image.Image]) -> "FewShotDetector":
        if not images:
            raise ValueError("at least one reference image is required")
        hid = self.extractor.model.config.hidden_size
        views = [aug(img.convert("RGB")) for img in images for aug in (AUGMENTATIONS if self.augment else AUGMENTATIONS[:1])]
        self._feats = np.concatenate([self.extractor(v).reshape(-1, hid) for v in views]).astype(np.float32)
        # The un-augmented originals calibrate the score scale.
        self._ref = np.concatenate([self.extractor(i.convert("RGB")).reshape(-1, hid) for i in images]).astype(np.float32)
        self.n_reference = len(images)
        return self._build()

    def _build(self) -> "FewShotDetector":
        self._pca = pca_scorer(self._feats, ev=self.ev)
        self._knn = knn_scorer(self._feats, k=self.knn_k)
        raw = self._raw(self._ref)
        self._ref_scale = (raw.mean(0), raw.std(0) + 1e-9)
        return self

    def state(self) -> dict:
        """Plain arrays and numbers only (picklable), so a fitted detector can cross process boundaries."""
        return {"feats": self._feats, "ref": self._ref, "ev": self.ev, "knn_k": self.knn_k, "n": self.n_reference}

    @classmethod
    def from_state(cls, st: dict, extractor) -> "FewShotDetector":
        det = cls(ev=st["ev"], knn_k=st["knn_k"], extractor=extractor)
        det._feats, det._ref, det.n_reference = st["feats"], st["ref"], st["n"]
        return det._build()

    def _raw(self, flat: np.ndarray) -> np.ndarray:
        return np.stack([self._pca(flat), self._knn(flat)], axis=1)

    def score(self, image: Image.Image, out_size: int | None = None) -> dict:
        """Return {"score": float, "heatmap": float32 array (H, W) in 0..1-ish, "grid": (h, w)}."""
        if not self.fitted:
            raise RuntimeError("fit() the detector on reference images first")
        f = self.extractor(image.convert("RGB"))
        h, w, c = f.shape
        raw = self._raw(f.reshape(-1, c))
        mu, sd = self._ref_scale
        z = ((raw - mu) / sd).mean(1)                       # fused, in units of "reference standard deviations"
        patch = z.reshape(h, w).astype(np.float32)
        size = out_size or image.size[0]
        heat = gaussian_filter(zoom(patch, size / h, order=1), sigma=max(1.0, size / 64))
        return {"score": image_score(patch), "heatmap": heat, "grid": (h, w)}

    def threshold_hint(self, images: list[Image.Image]) -> float:
        """A starting decision threshold: the highest score among known-good images plus a 10% margin."""
        return float(max(self.score(i)["score"] for i in images) * 1.1)
