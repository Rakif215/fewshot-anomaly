#!/usr/bin/env bash
# One-time environment setup (runs automatically when the Codespace is created).
# CPU-only PyTorch: Codespaces have no GPU, and the CPU wheels are much smaller.
set -euo pipefail
cd "$(dirname "$0")/.."

git submodule update --init --recursive
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
# SubspaceAD as a package, without its listed-but-unused heavy dependency (anomalib).
python -m pip install -e third_party/SubspaceAD --no-deps
python -c "import torch, transformers, faiss; print('ready: torch', torch.__version__, '| transformers', transformers.__version__)"
