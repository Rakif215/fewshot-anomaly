"""FastAPI service: upload a few good reference images, then inspect new images.

Run:  uvicorn fsad.api:app --host 0.0.0.0 --port 8000
Env:  FSAD_CKPT (default facebook/dinov2-with-registers-small), FSAD_RES (default 448)
"""
from __future__ import annotations

import base64
import io
import os
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .detector import FewShotDetector
from .render import overlay

CKPT = os.environ.get("FSAD_CKPT", "facebook/dinov2-with-registers-small")
RES = int(os.environ.get("FSAD_RES", "448"))
MAX_REFERENCES, MAX_BYTES = 16, 15 * 1024 * 1024

app = FastAPI(title="Few-shot anomaly inspection", version="0.1.0")
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
_models: dict[str, dict] = {}   # id -> {"detector", "threshold", "n"}


def _read(upload: UploadFile, data: bytes) -> Image.Image:
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"{upload.filename}: image larger than {MAX_BYTES // 1_000_000} MB")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, f"{upload.filename}: not a readable image")


@app.get("/", include_in_schema=False)
def demo_page():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "checkpoint": CKPT, "resolution": RES, "models": len(_models)}


@app.post("/models")
async def create_model(files: list[UploadFile] = File(...)):
    """Fit a detector on 1-16 images of normal (defect-free) parts. Returns a model id and a starting threshold."""
    if not 1 <= len(files) <= MAX_REFERENCES:
        raise HTTPException(400, f"send between 1 and {MAX_REFERENCES} reference images")
    images = [_read(f, await f.read()) for f in files]
    det = FewShotDetector(CKPT, RES).fit(images)
    mid = uuid.uuid4().hex[:10]
    _models[mid] = {"detector": det, "threshold": det.threshold, "n": len(images)}
    return {"model_id": mid, "references": len(images), "threshold": _models[mid]["threshold"]}


@app.post("/models/{model_id}/inspect")
async def inspect(model_id: str, file: UploadFile = File(...), threshold: float | None = None, include_overlay: bool = True):
    m = _models.get(model_id)
    if m is None:
        raise HTTPException(404, "unknown model_id (models live in memory; create one with POST /models)")
    img = _read(file, await file.read())
    out = m["detector"].score(img, out_size=min(img.size[0], 512))
    limit = m["threshold"] if threshold is None else threshold
    body = {"score": out["score"], "threshold": limit, "defect": bool(out["score"] > limit)}
    if include_overlay:
        buf = io.BytesIO()
        overlay(img, out["heatmap"], vmax=max(limit * 1.5, 1e-6)).save(buf, "PNG")
        body["overlay_png_base64"] = base64.b64encode(buf.getvalue()).decode()
    return body


@app.post("/models/{model_id}/overlay.png")
async def overlay_png(model_id: str, file: UploadFile = File(...)):
    m = _models.get(model_id)
    if m is None:
        raise HTTPException(404, "unknown model_id")
    img = _read(file, await file.read())
    out = m["detector"].score(img, out_size=min(img.size[0], 512))
    buf = io.BytesIO()
    overlay(img, out["heatmap"], vmax=m["threshold"] * 1.5).save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.delete("/models/{model_id}")
def delete_model(model_id: str):
    if _models.pop(model_id, None) is None:
        raise HTTPException(404, "unknown model_id")
    return {"deleted": model_id}
