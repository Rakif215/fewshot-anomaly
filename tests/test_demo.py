import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("gradio")
import fsad.demo as demo  # noqa: E402

demo.RES = 224


def test_example_flow_scores_defect_higher():
    cat = next(iter(demo.example_meta()))
    refs, tests = demo.example_images(cat)
    model = demo.learn(refs)
    good = demo.inspect(model, Image.open(tests[0][0]).convert("RGB"))
    bad = demo.inspect(model, Image.open(tests[1][0]).convert("RGB"))
    assert "Defect" in bad[0] or "score" in bad[2]
    assert float(bad[2].split("**")[1]) > float(good[2].split("**")[1])


def test_ui_builds():
    assert demo.build() is not None


def test_everything_crossing_the_gpu_boundary_is_picklable():
    import pickle
    cat = next(iter(demo.example_meta()))
    refs, tests = demo.example_images(cat)
    model = demo.learn(refs)
    pickle.loads(pickle.dumps(model))                      # what learn() returns
    result = demo.inspect(model, Image.open(tests[1][0]).convert("RGB"))
    pickle.loads(pickle.dumps(result))                     # what inspect() returns
