"""Hugging Face Space / local entry point: python app.py"""
import os

os.environ.setdefault("FSAD_DEVICE", "cpu")   # run on CPU: avoids the per-visitor ZeroGPU quota (see fsad/demo.py)

from fsad.demo import build  # noqa: E402

demo = build()

if __name__ == "__main__":
    demo.launch()
