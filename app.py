"""Hugging Face Space / local entry point: python app.py"""
from fsad.demo import build

demo = build()

if __name__ == "__main__":
    demo.launch()
