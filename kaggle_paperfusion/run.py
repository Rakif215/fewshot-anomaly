"""Kaggle GPU run: PCA vs kNN vs fused score at the PAPER setting (DINOv2-with-registers-giant, 672 px,
30 random-rotation augmentations, layers -12..-18, seed 42) on VisA, k = 1, 2, 4. Same fsad code and flags as the earlier runs.
Categories are the outer loop so each category's cached features can be deleted after its three shot counts."""
import json, os, shutil, subprocess, time

def sh(cmd):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True)

T0 = time.time()
REPO, OUT, TMP = "/kaggle/tmp/repo", "/kaggle/working/results", "/kaggle/tmp/cache"
os.makedirs(OUT, exist_ok=True)
subprocess.run(f"git clone --recursive -q https://github.com/Rakif215/fewshot-anomaly.git {REPO}", shell=True, check=True)
os.chdir(REPO)
subprocess.run("pip install -q scikit-learn scipy pillow", shell=True, check=True)
subprocess.run("bash scripts/get_visa.sh", shell=True, check=True)
DATA = f"{REPO}/datasets/VisA_pytorch/1cls"
CATS = "candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum".split()
for cat in CATS:
    for k in (1, 2, 4):
        t = time.time()
        r = sh(f"python -u -m fsad.experiments --data {DATA} --out {TMP}/{cat} --k {k} --seed 42 --aug paper --aug_count 30 "
               f"--ckpt facebook/dinov2-with-registers-giant --res 672 --layers=-12,-13,-14,-15,-16,-17,-18 --cats {cat}")
        src = f"{TMP}/{cat}/results_k{k}.json"
        if os.path.exists(src):
            shutil.copy(src, f"{OUT}/{cat}_k{k}.json")
        print(f"{cat} k={k} rc={r.returncode} {(time.time()-t)/60:.1f} min (total {(time.time()-T0)/60:.0f} min)", flush=True)
    shutil.rmtree(f"{TMP}/{cat}", ignore_errors=True)    # free disk
print("ALL DONE", flush=True)
