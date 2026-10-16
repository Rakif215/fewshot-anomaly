"""Kaggle GPU run: SubspaceAD at the PAPER setting (DINOv2-with-registers giant, 672 px) on VisA, k = 1, 2, 4 shots, seed 42.
Same flags as third_party/SubspaceAD/scripts/benchmark_few_shot.sh. Results are written to /kaggle/working/results."""
import os, shutil, subprocess, sys, time

def sh(cmd, **kw):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)

T0 = time.time()
REPO = "/kaggle/tmp/repo"          # /kaggle/tmp is not part of the saved output
OUT = "/kaggle/working/results"
os.makedirs(OUT, exist_ok=True)
sh(f"git clone --recursive -q https://github.com/Rakif215/fewshot-anomaly.git {REPO}")
os.chdir(REPO)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh("pip install -q -r requirements.txt")                    # torch is already installed (with CUDA) on Kaggle
sh("pip install -q -e third_party/SubspaceAD --no-deps")
sh("bash scripts/get_visa.sh")
env = dict(os.environ, SUBSPACEAD_KEEP_ATTN_LAYER="6", PYTHONPATH=f"{REPO}/scripts/compat:" + os.environ.get("PYTHONPATH", ""))
DATA = f"{REPO}/datasets/VisA_pytorch/1cls"
for k in (4,):
    t = time.time()
    out = f"{OUT}/visa_k{k}_dinov2G"
    cmd = (f"python -u third_party/SubspaceAD/main.py --dataset_name visa --dataset_path {DATA} --image_res 672 --k_shot {k} "
           f'--layers="-12,-13,-14,-15,-16,-17,-18" --model_ckpt facebook/dinov2-with-registers-giant --aug_count 30 '
           f"--pca_ev 0.99 --agg_method mean --seed 42 --outdir {out} 2>&1 | tee {OUT}/visa_k{k}.log | tail -n 40")
    subprocess.run(cmd, shell=True, env=env)                # a failed shot count must not stop the others
    print(f"k={k} finished in {(time.time()-t)/60:.1f} min (total {(time.time()-T0)/60:.1f} min)", flush=True)
print("ALL DONE", flush=True)
