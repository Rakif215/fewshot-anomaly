"""Kaggle GPU run: PCA vs kNN vs fusion with the giant DINOv2-with-registers backbone, 4-shot, three seeds, all 12 VisA categories."""
import os, subprocess, time

def sh(cmd, **kw):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)

T0 = time.time()
REPO, OUT = "/kaggle/tmp/repo", "/kaggle/working/results"
os.makedirs(OUT, exist_ok=True)
sh(f"git clone --recursive -q https://github.com/Rakif215/fewshot-anomaly.git {REPO}")
os.chdir(REPO)
sh("pip install -q scikit-learn scipy pillow")
sh("bash scripts/get_visa.sh")
DATA = f"{REPO}/datasets/VisA_pytorch/1cls"
CATS = "candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"
for seed in (42, 43, 44):
    sh(f"python -u -m fsad.experiments --data {DATA} --out {OUT}/giant_s{seed} --k 4 --seed {seed} "
       f"--ckpt facebook/dinov2-with-registers-giant --res 448 --layers=-12,-13,-14,-15,-16,-17,-18 --cats {CATS}")
    sh(f"rm -rf {OUT}/giant_s{seed}/cache")   # keep only the result json in the saved output
    print(f"seed {seed} done at {(time.time()-T0)/60:.1f} min", flush=True)
print("ALL DONE", flush=True)
