"""Turn a heatmap into an overlay image for the demo and the API."""
import numpy as np
from PIL import Image


def overlay(image: Image.Image, heat: np.ndarray, vmax: float | None = None, alpha: float = 0.55) -> Image.Image:
    base = image.convert("RGB").resize((heat.shape[1], heat.shape[0]))
    v = float(vmax if vmax is not None else max(heat.max(), 1e-6))
    t = np.clip(heat / v, 0, 1)
    # simple blue -> yellow -> red ramp, no matplotlib dependency
    r = np.clip(2 * t, 0, 1)
    g = np.clip(2 * t, 0, 1) * np.clip(2 - 2 * t, 0, 1)
    b = np.clip(1 - 2 * t, 0, 1) * 0.6
    color = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    mix = (np.asarray(base, np.float32) * (1 - alpha * t[..., None]) + color * alpha * t[..., None]).astype(np.uint8)
    return Image.fromarray(mix)
