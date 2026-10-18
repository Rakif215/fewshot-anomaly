"""Gradio demo logic and UI. Used by app.py (Hugging Face Space) and runnable locally: python app.py"""
import json
import os
from pathlib import Path

from PIL import Image

from .detector import FewShotDetector
from .features import Extractor
from .render import overlay

try:  # Hugging Face ZeroGPU: the GPU is attached only inside functions decorated with spaces.GPU
    import spaces
    gpu = spaces.GPU(duration=60)
except Exception:  # not on a Space (or no spaces package): run as a plain function
    def gpu(fn):
        return fn

EXAMPLES = Path(__file__).parent / "static" / "examples"
CKPT = os.environ.get("FSAD_CKPT", "facebook/dinov2-with-registers-small")
RES = int(os.environ.get("FSAD_RES", "448"))
_extractor: Extractor | None = None


def extractor() -> Extractor:
    global _extractor
    if _extractor is None:
        _extractor = Extractor(CKPT, RES)
    return _extractor


def example_meta() -> dict:
    return json.loads((EXAMPLES / "index.json").read_text())


@gpu
def learn(images: list[Image.Image]) -> dict:
    """Fit on the reference photos. Returns plain data only (no functions), because ZeroGPU pickles what crosses the call."""
    det = FewShotDetector(CKPT, RES, extractor=extractor()).fit(images)
    return {"state": det.state(), "threshold": det.threshold, "n": len(images)}


@gpu
def inspect(model: dict, image: Image.Image):
    """Return (verdict markdown, overlay image, score text)."""
    det = FewShotDetector.from_state(model["state"], extractor())
    out = det.score(image, out_size=min(image.size[0], 512))
    limit = model["threshold"]
    defect = out["score"] > limit
    verdict = f"## {'🔴 Defect likely' if defect else '🟢 Looks good'}"
    info = f"score **{out['score']:.2f}**, threshold **{limit:.2f}**"
    return verdict, overlay(image, out["heatmap"], vmax=limit * 1.5), info


def example_images(cat: str):
    refs = [Image.open(EXAMPLES / cat / f"ref{i}.jpg") for i in range(1, 5)]
    tests = [(str(EXAMPLES / cat / f"{f}.jpg"), label) for f, label in example_meta()[cat]["tests"]]
    return refs, tests


def build():
    import gradio as gr

    meta = example_meta()
    labels = {m["label"]: cat for cat, m in meta.items()}

    def pick_example(label):
        try:
            cat = labels[label]
            refs, tests = example_images(cat)
            model = learn(refs)
            return model, [t[0] for t in tests], [(r, "reference (good)") for r in refs], f"Learned from {len(refs)} good photos. Now click a test photo below."
        except gr.Error:
            raise
        except Exception as e:
            raise gr.Error(f"{type(e).__name__}: {e}")

    def pick_test(model, paths, evt: gr.SelectData):
        if model is None:
            raise gr.Error("Choose an example part first.")
        try:
            return inspect(model, Image.open(paths[evt.index]).convert("RGB"))
        except gr.Error:
            raise
        except Exception as e:
            raise gr.Error(f"{type(e).__name__}: {e}")

    def learn_upload(files):
        if not files:
            raise gr.Error("Upload 1-8 photos of a good part.")
        try:
            imgs = [Image.open(f if isinstance(f, str) else f.name).convert("RGB") for f in files[:8]]
            return learn(imgs), f"Learned from {len(imgs)} photos. Upload a photo to inspect."
        except Exception as e:
            raise gr.Error(f"{type(e).__name__}: {e}")

    def inspect_upload(model, img):
        if model is None:
            raise gr.Error("Learn from your good photos first.")
        if img is None:
            raise gr.Error("Upload a photo to inspect.")
        try:
            return inspect(model, img.convert("RGB"))
        except Exception as e:
            raise gr.Error(f"{type(e).__name__}: {e}")

    with gr.Blocks(title="Few-shot defect inspection") as demo:
        gr.Markdown("# Few-shot defect inspection\nShow the model a few photos of a good part. It flags defects in new photos and shows where. "
                    "No training and no defect examples.")
        with gr.Tab("Try an example"):
            choice = gr.Radio(list(labels), label="Example part (VisA dataset, CC BY 4.0)")
            status = gr.Markdown()
            refs_g = gr.Gallery(label="Good reference photos", columns=4, height=140, interactive=False)
            tests_g = gr.Gallery(label="Test photos (click one)", columns=3, height=200, allow_preview=False)
            model_state, paths_state = gr.State(None), gr.State([])
            with gr.Row():
                verdict, overlay_img = gr.Markdown(), gr.Image(label="Where the model looks", type="pil")
            info = gr.Markdown()
            choice.change(pick_example, choice, [model_state, paths_state, refs_g, status]).then(
                lambda p: [(x, None) for x in p], paths_state, tests_g)
            tests_g.select(pick_test, [model_state, paths_state], [verdict, overlay_img, info])
        with gr.Tab("Use your own photos"):
            files = gr.File(file_count="multiple", file_types=["image"], label="1-8 photos of a good part")
            learn_btn = gr.Button("Learn this part")
            status2 = gr.Markdown()
            my_model = gr.State(None)
            test_img = gr.Image(type="pil", label="Photo to inspect")
            go = gr.Button("Inspect")
            with gr.Row():
                v2, o2 = gr.Markdown(), gr.Image(label="Where the model looks", type="pil")
            i2 = gr.Markdown()
            learn_btn.click(learn_upload, files, [my_model, status2])
            go.click(inspect_upload, [my_model, test_img], [v2, o2, i2])
        gr.Markdown("**Method:** frozen DINOv2 features; a PCA-residual score and a nearest-neighbour score fused. "
                    "Paper reproduction on VisA (image AUROC, 1/2/4-shot): 93.0 / 94.1 / 94.7 vs the paper's 93.2 / 93.8 / 94.7. "
                    "[Code and results](https://github.com/Rakif215/fewshot-anomaly) · [SubspaceAD paper](https://arxiv.org/abs/2602.23013). "
                    "Uploaded images are processed in memory and not stored.")
    return demo
