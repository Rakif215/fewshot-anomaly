"""Evaluate FewShotDetector (the demo's detector) on all VisA test images of some categories: AUROC and the error rates at its own threshold."""
import glob, random, sys, time
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from fsad.detector import FewShotDetector

D = "datasets/VisA_pytorch/1cls"
k = int(sys.argv[1]); cats = sys.argv[2:]
for cat in cats:
    t = time.time()
    train = sorted(glob.glob(f"{D}/{cat}/train/good/*")); random.Random(42).shuffle(train)
    det = FewShotDetector(res=448).fit([Image.open(p).convert("RGB") for p in train[:k]])
    good = sorted(glob.glob(f"{D}/{cat}/test/good/*")); bad = sorted(glob.glob(f"{D}/{cat}/test/bad/*"))
    sg = np.array([det.score(Image.open(p).convert("RGB"))["score"] for p in good])
    sb = np.array([det.score(Image.open(p).convert("RGB"))["score"] for p in bad])
    auc = roc_auc_score([0] * len(sg) + [1] * len(sb), np.r_[sg, sb])
    print(f"{cat} k={k}: AUROC {auc:.3f} | threshold {det.threshold:.2f} | good flagged {np.mean(sg > det.threshold):.0%} | defects caught {np.mean(sb > det.threshold):.0%} "
          f"| median good {np.median(sg):.2f} median defect {np.median(sb):.2f} ({time.time()-t:.0f}s)", flush=True)
