#!/usr/bin/env bash
# Publish the Gradio demo to a Hugging Face Space (SDK: Gradio).
# Usage: bash scripts/deploy_space.sh <hf-username>/<space-name>
# Needs a Hugging Face account you are logged in to (huggingface-cli login, or git credentials) and an empty Gradio Space.
set -euo pipefail
SPACE="${1:?usage: deploy_space.sh <user>/<space>}"
cd "$(dirname "$0")/.."
TMP="$(mktemp -d)"
git clone "https://huggingface.co/spaces/$SPACE" "$TMP/space"
cp hf-space/README.md hf-space/requirements.txt app.py "$TMP/space/"
rm -rf "$TMP/space/fsad" && cp -r fsad "$TMP/space/fsad"
find "$TMP/space/fsad" -name __pycache__ -prune -exec rm -rf {} +
cd "$TMP/space" && git add -A && git commit -m "Deploy demo" && git push
echo "Done: https://huggingface.co/spaces/$SPACE"
