import pytest
from pathlib import Path
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


def test_ui_builds_without_gradio_warnings():
    """Gradio warns when it cannot recognise an event argument (e.g. gr.SelectData hidden by string annotations)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        demo.build()


def test_good_images_pass_and_defects_are_flagged_on_every_example_set():
    """The decision threshold must separate real good and defective photos, not only rank them."""
    meta = demo.example_meta()
    results = {}
    for cat in meta:
        refs, tests = demo.example_images(cat)
        model = demo.learn(refs)
        scores = {Path(p).stem: float(demo.inspect(model, Image.open(p).convert("RGB"))[2].split("**")[1]) for p, _ in tests}
        results[cat] = (scores, model["threshold"])
    print(results)
    for cat, (sc, thr) in results.items():
        assert sc["test_good"] < thr, (cat, sc, thr)
        assert max(sc["test_defect1"], sc["test_defect2"]) > thr, (cat, sc, thr)
