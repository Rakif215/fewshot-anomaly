"""Smoke test with synthetic images: a textured 'good' surface vs the same surface with a dark blob."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

pytest.importorskip("torch")
from fsad.detector import FewShotDetector  # noqa: E402


def good(seed):
    rng = np.random.default_rng(seed)
    base = (np.indices((224, 224)).sum(0) // 8 % 2) * 40 + 120          # checker texture
    img = np.clip(base + rng.normal(0, 4, (224, 224)), 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([img] * 3, -1))


def defective(seed):
    im = good(seed)
    ImageDraw.Draw(im).ellipse((90, 90, 140, 140), fill=(20, 20, 20))
    return im


@pytest.fixture(scope="module")
def detector():
    return FewShotDetector(res=224).fit([good(i) for i in range(4)])


def test_defect_scores_higher_than_good(detector):
    assert detector.score(defective(10))["score"] > detector.score(good(11))["score"]


def test_heatmap_peaks_on_the_defect(detector):
    heat = detector.score(defective(12))["heatmap"]
    y, x = np.unravel_index(heat.argmax(), heat.shape)
    assert 70 < x < 160 and 70 < y < 160


def test_unfitted_detector_raises():
    with pytest.raises(RuntimeError):
        FewShotDetector(res=224).score(good(0))
