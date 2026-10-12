#!/usr/bin/env bash
# Start vLLM (N-ATLaS) on :8000 (internal) and the AtlasKit gateway on :8080 (public).
# Safe to re-run: it stops any previous instance first.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export HF_HOME=${HF_HOME:-/workspace/hf-cache}
export ATLAS_LOG_PATH=${ATLAS_LOG_PATH:-/workspace/atlaskit-logs/usage.jsonl}
: "${ATLAS_API_KEYS:?Set ATLAS_API_KEYS to one or more comma-separated keys you will give to testers}"
mkdir -p "$(dirname "$ATLAS_LOG_PATH")"

PY=${ATLAS_PYTHON:-${ATLAS_VENV:-/opt/atlaskit-venv}/bin/python}
if ! "$PY" -c "import vllm" 2>/dev/null; then
  echo "Python environment not found (RunPod resets the container disk on restart); running setup…"
  bash deploy/runpod/setup.sh
fi

GPU_UTIL=${GPU_UTIL:-0.80}          # leave headroom for the Whisper ASR models
MAX_LEN=${MAX_LEN:-8192}
export PORT=${PORT:-8080}

wait_for() {  # wait_for URL PID LOG: poll until URL answers; fail fast if the process dies
  until curl -sf "$1" >/dev/null; do
    if ! kill -0 "$2" 2>/dev/null; then
      echo "!! Process exited during startup. Last lines of $3:"; tail -n 40 "$3"; exit 1
    fi
    sleep 3
  done
}

pkill -f -- "-m atlaskit_gateway" || true
pkill -f -- "vllm.entrypoints.openai.api_server" || true
while pgrep -f -- "vllm.entrypoints.openai.api_server" >/dev/null; do sleep 1; done

nohup "$PY" -m vllm.entrypoints.openai.api_server \
  --model NCAIR1/N-ATLaS --served-model-name NCAIR1/N-ATLaS \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL" \
  > /workspace/vllm.log 2>&1 &
echo "vLLM starting (log: /workspace/vllm.log)…"
wait_for http://127.0.0.1:8000/v1/models $! /workspace/vllm.log
echo "vLLM ready."

export ATLAS_UPSTREAM_URL=http://127.0.0.1:8000/v1
nohup "$PY" -m atlaskit_gateway > /workspace/gateway.log 2>&1 &
wait_for "http://127.0.0.1:$PORT/health" $! /workspace/gateway.log
echo "AtlasKit gateway ready on :$PORT"
echo "Public URL: https://${RUNPOD_POD_ID:-<pod-id>}-$PORT.proxy.runpod.net"
