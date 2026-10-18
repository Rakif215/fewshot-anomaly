"""Gradio demo logic and UI. Used by app.py (Hugging Face Space) and runnable locally: python app.py"""
import json
import os
from pathlib import Path

from PIL import Image

from .detector import FewShotDetector
from .features import Extractor
from .render import overlay

# Hugging Face ZeroGPU Spaces refuse to start unless one function is marked with spaces.GPU, but every visitor has a small
# GPU quota that a demo would burn through in a few clicks. The small model is fast enough on CPU, so the real work runs on
# CPU (FSAD_DEVICE=cpu, set in app.py) and the marked function below is only a placeholder that is never called.
try:
    import spaces

    @spaces.GPU
    def _gpu_placeholder():
        return True
except Exception:  # not on a Space (or no spaces package)
    pass

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


def learn(images: list[Image.Image]) -> dict:
    """Fit on the reference photos. Returns plain data only (no functions), because ZeroGPU pickles what crosses the call."""
    det = FewShotDetector(CKPT, RES, extractor=extractor()).fit(images)
    return {"state": det.state(), "threshold": det.threshold, "n": len(images)}


def inspect(model: dict, image: Image.Image, sensitivity: float = 1.0):
    """Return (verdict markdown, overlay image, score text). sensitivity < 1 lowers the threshold (more alerts)."""
    det = FewShotDetector.from_state(model["state"], extractor())
    out = det.score(image, out_size=min(image.size[0], 512))
    limit = model["threshold"] * sensitivity
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
            st = example_meta()[cat].get("stats")
            facts = (f" On all {st['n_good']} good and {st['n_defect']} defective test photos of this part: image AUROC **{st['auroc']:.2f}**, "
                     f"**{st['defects_caught']:.0%}** of defects caught, **{st['good_flagged']:.0%}** of good parts falsely flagged. "
                     "The photos below were picked to show typical cases, including one miss.") if st else ""
            return model, [t[0] for t in tests], [(r, "reference (good)") for r in refs], f"Learned from {len(refs)} good photos." + facts + " Click a test photo below."
        except gr.Error:
            raise
        except Exception as e:
            raise gr.Error(f"{type(e).__name__}: {e}")

    def pick_test(model, paths, sens, evt: gr.SelectData):
        if model is None:
            raise gr.Error("Choose an example part first.")
        try:
            return inspect(model, Image.open(paths[evt.index]).convert("RGB"), sens)
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

    def inspect_upload(model, img, sens):
        if model is None:
            raise gr.Error("Learn from your good photos first.")
        if img is None:
            raise gr.Error("Upload a photo to inspect.")
        try:
            return inspect(model, img.convert("RGB"), sens)
        except Exception as e:
            raise gr.Error(f"{type(e).__name__}: {e}")

    with gr.Blocks(title="Few-shot defect inspection") as demo:
        gr.Markdown("# Few-shot defect inspection\nShow the model a few photos of a good part. It flags defects in new photos and shows where. "
                    "No training and no defect examples.")
        with gr.Tab("Try an example"):
            choice = gr.Radio(list(labels), label="Example part (VisA dataset, CC BY 4.0)")
            status = gr.Markdown()
            refs_g = gr.Gallery(label="Good reference photos", columns=4, height=140, interactive=False)
            sens1 = gr.Slider(0.7, 1.5, value=1.0, step=0.05, label="Sensitivity: lower = more alerts (and more false alarms)")
            tests_g = gr.Gallery(label="Test photos (click one)", columns=3, height=200, allow_preview=False)
            model_state, paths_state = gr.State(None), gr.State([])
            with gr.Row():
                verdict, overlay_img = gr.Markdown(), gr.Image(label="Where the model looks", type="pil")
            info = gr.Markdown()
            choice.change(pick_example, choice, [model_state, paths_state, refs_g, status]).then(
                lambda p: [(x, None) for x in p], paths_state, tests_g)
            tests_g.select(pick_test, [model_state, paths_state, sens1], [verdict, overlay_img, info])
        with gr.Tab("Use your own photos"):
            files = gr.File(file_count="multiple", file_types=["image"], label="1-8 photos of a good part")
            learn_btn = gr.Button("Learn this part")
            status2 = gr.Markdown()
            my_model = gr.State(None)
            sens2 = gr.Slider(0.7, 1.5, value=1.0, step=0.05, label="Sensitivity: lower = more alerts (and more false alarms)")
            test_img = gr.Image(type="pil", label="Photo to inspect")
            go = gr.Button("Inspect")
            with gr.Row():
                v2, o2 = gr.Markdown(), gr.Image(label="Where the model looks", type="pil")
            i2 = gr.Markdown()
            learn_btn.click(learn_upload, files, [my_model, status2])
            go.click(inspect_upload, [my_model, test_img, sens2], [v2, o2, i2])
        gr.Markdown("**Method:** frozen DINOv2 features; a PCA-residual score and a nearest-neighbour score fused. "
                    "Paper reproduction on VisA (image AUROC, 1/2/4-shot): 93.0 / 94.1 / 94.7 vs the paper's 93.2 / 93.8 / 94.7. "
                    "[Code and results](https://github.com/Rakif215/fewshot-anomaly) · [SubspaceAD paper](https://arxiv.org/abs/2602.23013). "
                    "Results depend on which good photos you show: 4 well-chosen references work better than 1. Uploaded images are processed in memory and not stored.")
    return demo
