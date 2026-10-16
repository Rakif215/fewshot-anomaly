#!/usr/bin/env bash
# Usage: bash scripts/smoke_visa.sh            (both methods)
#        ONLY=anomalydino bash scripts/smoke_visa.sh   (or ONLY=subspacead)
# Smoke test on CPU: both methods, VisA (all 12 categories), 1-shot, ONE seed, SMALL DINOv2 at 448 px.
# Purpose: prove the pipeline runs end to end and the numbers are sensible — NOT the paper setting
# (the paper uses dinov2-with-registers-giant at 672 px; that is scripts/repro_visa.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/smoke_visa_1shot_vits
mkdir -p "$OUT"
DATA="$(pwd)/datasets/VisA_pytorch/1cls"
# NOTE: AnomalyDINO always plots examples after evaluation, so makes it crash at the very end.
# Compatibility shims for newer library versions (third_party code stays unmodified).
export PYTHONPATH="$(pwd)/scripts/compat${PYTHONPATH:+:$PYTHONPATH}"

# Default layers (-12..-18) only exist in the giant model; -4,-5 is the authors' own setting for 12-layer
# S/B backbones (scripts/backbone_ablation.sh: same 55-70% relative depth).
if [ "${ONLY:-}" != "anomalydino" ]; then
echo "== SubspaceAD (dinov2-with-registers-small, layers -4,-5, 448 px, 1-shot)"
time python third_party/SubspaceAD/main.py \
  --dataset_name visa --dataset_path "$DATA" \
  --model_ckpt facebook/dinov2-with-registers-small --image_res 448 \
  --layers=-4,-5 \
  --k_shot 1 --aug_count 30 --pca_ev 0.99 \
  --outdir "$OUT/subspacead" 2>&1 | tee "$OUT/subspacead.log"
fi

if [ "${ONLY:-}" != "subspacead" ]; then
echo "== AnomalyDINO (dinov2_vits14, 448 px, 1-shot, CPU FAISS)"
( cd third_party/AnomalyDINO && time python run_anomalydino.py \
  --dataset VisA --data_root "$DATA/" --model_name dinov2_vits14 --resolution 448 \
  --shots 1 --num_seeds 1 --device cpu --faiss_on_cpu \
  --tag smoke_vits ) 2>&1 | tee "$OUT/anomalydino.log"
fi

echo "Done. Logs in $OUT/"
