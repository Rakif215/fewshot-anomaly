"""Frozen DINOv2 (with registers) patch features."""
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


class Extractor:
    def __init__(self, ckpt="facebook/dinov2-with-registers-small", res=448, layers=(-4, -5), device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(ckpt).to(self.device).eval()
        self.res, self.layers = res, list(layers)
        cfg = self.model.config
        self.drop = 1 + getattr(cfg, "num_register_tokens", 0)
        self.grid = res // cfg.patch_size

    @torch.no_grad()
    def __call__(self, img: Image.Image) -> np.ndarray:
        """Return an (H, W, C) grid of patch features, mean over the chosen layers."""
        a = (np.asarray(img.convert("RGB").resize((self.res, self.res), Image.BICUBIC), np.float32) / 255 - MEAN) / STD
        x = torch.from_numpy(a).permute(2, 0, 1)[None].to(self.device)
        hs = self.model(pixel_values=x, output_hidden_states=True).hidden_states
        f = torch.stack([hs[i] for i in self.layers]).mean(0)[0, self.drop:]
        return f.reshape(self.grid, self.grid, -1).float().cpu().numpy()
