import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

pytest.importorskip("torch")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import fsad.api as api  # noqa: E402

api.RES = 224
client = TestClient(api.app)


def png(im):
    b = io.BytesIO(); im.save(b, "PNG"); return b.getvalue()


def good(seed):
    rng = np.random.default_rng(seed)
    base = (np.indices((224, 224)).sum(0) // 8 % 2) * 40 + 120
    a = np.clip(base + rng.normal(0, 4, (224, 224)), 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([a] * 3, -1))


def bad(seed):
    im = good(seed); ImageDraw.Draw(im).ellipse((90, 90, 140, 140), fill=(20, 20, 20)); return im


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_flow_good_vs_defect():
    r = client.post("/models", files=[("files", (f"g{i}.png", png(good(i)), "image/png")) for i in range(4)])
    assert r.status_code == 200, r.text
    mid = r.json()["model_id"]
    ok = client.post(f"/models/{mid}/inspect", files={"file": ("a.png", png(good(20)), "image/png")}).json()
    ng = client.post(f"/models/{mid}/inspect", files={"file": ("b.png", png(bad(21)), "image/png")}).json()
    assert ng["score"] > ok["score"]
    assert ng["defect"] is True and "overlay_png_base64" in ng
    assert client.delete(f"/models/{mid}").status_code == 200


def test_errors():
    assert client.post("/models", files=[("files", ("x.txt", b"not an image", "text/plain"))]).status_code == 400
    assert client.post("/models/nope/inspect", files={"file": ("a.png", png(good(1)), "image/png")}).status_code == 404
