#!/usr/bin/env bash
# One-time setup on a RunPod GPU pod (PyTorch template, CUDA 12.8+, 24 GB+ VRAM).
# Usage:  cd /workspace/atlaskit && bash deploy/runpod/setup.sh
#
# Python packages go in a venv on the container disk, where imports are fast. RunPod
# wipes that disk when a pod stops, so start.sh re-runs this script when it is missing.
# Model weights go to /workspace (the persistent volume) and only download once.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export HF_HOME=${HF_HOME:-/workspace/hf-cache}
VENV=${ATLAS_VENV:-/opt/atlaskit-venv}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/workspace/.uv-cache} UV_LINK_MODE=copy   # wheels survive restarts, so reinstalls are fast

if [ -z "${HF_TOKEN:-}" ]; then
  echo "!! Set HF_TOKEN (a Hugging Face token whose account has accepted the N-ATLaS terms) and re-run."; exit 1
fi

command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null; }  # decode WhatsApp .ogg / .m4a voice notes
python3 -m pip install -q uv
[ -x "$VENV/bin/python" ] || uv venv -q --python "$(command -v python3)" "$VENV"
uv pip install -q --python "$VENV/bin/python" vllm librosa soundfile huggingface_hub -e gateway -e sdk/python

"$VENV/bin/python" deploy/runpod/download_models.py
echo "Setup complete. Start with: bash deploy/runpod/start.sh"
