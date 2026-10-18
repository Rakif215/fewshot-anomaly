"""Rebuild the demo's example photos and per-part statistics from the detector's own scores.
Uses the same 4 reference photos the demo uses (the first 4 sorted training images), scores every VisA test image, then picks:
  - a good part near the median good score,
  - two defects the model catches (60th-75th percentile of the caught ones, not the easiest),
  - one subtle defect it misses (just under the threshold).
Writes fsad/static/examples/<cat>/*.jpg and index.json with real AUROC / catch rate / false-alarm rate."""
import glob, json, os, shutil, sys
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from fsad.detector import FewShotDetector

D, OUT = "datasets/VisA_pytorch/1cls", sys.argv[1]
CATS = [("pcb4", "Circuit board"), ("macaroni1", "Macaroni"), ("capsules", "Capsules")]
shutil.rmtree(OUT, ignore_errors=True)
meta = {}
for cat, label in CATS:
    os.makedirs(f"{OUT}/{cat}")
    refs = sorted(glob.glob(f"{D}/{cat}/train/good/*"))[:4]
    det = FewShotDetector(res=448).fit([Image.open(p).convert("RGB") for p in refs])
    good = sorted(glob.glob(f"{D}/{cat}/test/good/*")); bad = sorted(glob.glob(f"{D}/{cat}/test/bad/*"))
    sg = np.array([det.score(Image.open(p).convert("RGB"))["score"] for p in good])
    sb = np.array([det.score(Image.open(p).convert("RGB"))["score"] for p in bad])
    thr = det.threshold
    base = thr / 1.0   # threshold = highest leave-one-out score of a held-out normal reference (margin 1.0)
    print(cat, "for information, other margins:", {m: (round(float(np.mean(sb > base * m)), 2), round(float(np.mean(sg > base * m)), 2)) for m in (0.8, 0.9, 1.0, 1.2)}, "(caught, false alarms)", flush=True)
    auc = roc_auc_score([0] * len(sg) + [1] * len(sb), np.r_[sg, sb])
    caught = np.where(sb > thr)[0]; caught = caught[np.argsort(sb[caught])]
    missed = np.where(sb <= thr)[0]; missed = missed[np.argsort(-sb[missed])]
    pick_good = int(np.argsort(sg)[len(sg) // 2])
    c1, c2 = caught[int(len(caught) * .60)], caught[min(int(len(caught) * .75), len(caught) - 1)]
    subtle = missed[0] if len(missed) else caught[0]
    def save(p, name):
        im = Image.open(p).convert("RGB"); im.thumbnail((448, 448)); im.save(f"{OUT}/{cat}/{name}.jpg", quality=85)
    for i, p in enumerate(refs): save(p, f"ref{i+1}")
    save(good[pick_good], "test_good"); save(bad[c1], "test_defect1"); save(bad[c2], "test_defect2"); save(bad[subtle], "test_subtle")
    meta[cat] = {"label": label, "refs": 4,
                 "tests": [["test_good", "Good part"], ["test_defect1", "Defective part"], ["test_defect2", "Defective part"], ["test_subtle", "Subtle defect (often missed)"]],
                 "stats": {"auroc": round(float(auc), 3), "defects_caught": round(float(np.mean(sb > thr)), 2), "good_flagged": round(float(np.mean(sg > thr)), 2),
                           "n_good": len(sg), "n_defect": len(sb)}}
    print(cat, meta[cat]["stats"], "thr", round(thr, 2), flush=True)
json.dump(meta, open(f"{OUT}/index.json", "w"), indent=1)
