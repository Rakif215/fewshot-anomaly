#!/usr/bin/env bash
# Download VisA (Amazon, CC BY 4.0, ~1.9 GB) and build the official 1-class split used by both methods.
# Output: datasets/VisA_pytorch/1cls/<category>/{train,test,ground_truth}
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p datasets/VisA_20220922
if [ ! -f datasets/VisA_20220922.tar ] && [ ! -d datasets/VisA_20220922/candle ]; then
  curl -L --fail -o datasets/VisA_20220922.tar https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
fi
if [ ! -d datasets/VisA_20220922/candle ]; then
  tar -xf datasets/VisA_20220922.tar -C datasets/VisA_20220922
  rm -f datasets/VisA_20220922.tar   # keep the Codespace disk small
fi
python third_party/SubspaceAD/tools/prepare_visa.py \
  --split-type 1cls \
  --data-folder datasets/VisA_20220922 \
  --save-folder datasets/VisA_pytorch \
  --split-file datasets/VisA_20220922/split_csv/1cls.csv
du -sh datasets/VisA_pytorch
