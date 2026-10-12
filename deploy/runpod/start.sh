#!/usr/bin/env bash
# Start vLLM (N-ATLaS) on :8000 (internal) and the AtlasKit gateway on :8080 (public).
# Safe to re-run: it stops any previous instance first. GATEWAY_ONLY=1 restarts only the
# gateway (a few seconds, the model stays loaded), e.g. after adding API keys or updating code.
#
# Fine-tuned LoRA adapters (see finetune/README.md) are served alongside the base model:
#   ATLAS_LORA_MODULES="natlas-mt=/workspace/adapters/natlas-mt" bash deploy/runpod/start.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export HF_HOME=${HF_HOME:-/workspace/hf-cache}
LOG_DIR=${ATLAS_LOG_DIR:-/workspace}   # use a separate directory per pod when pods share a network volume
export ATLAS_LOG_PATH=${ATLAS_LOG_PATH:-$LOG_DIR/atlaskit-logs/usage.jsonl}
: "${ATLAS_API_KEYS:?Set ATLAS_API_KEYS to one or more comma-separated keys you will give to testers}"
mkdir -p "$LOG_DIR" "$(dirname "$ATLAS_LOG_PATH")"

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

LORA_ARGS=()
if [ -n "${ATLAS_LORA_MODULES:-}" ]; then
  read -ra modules <<< "${ATLAS_LORA_MODULES//,/ }"
  LORA_ARGS=(--enable-lora --max-lora-rank "${MAX_LORA_RANK:-64}" --lora-modules "${modules[@]}")
  names=(); for m in "${modules[@]}"; do names+=("${m%%=*}"); done
  export ATLAS_ADAPTERS=$(IFS=,; echo "${names[*]}")
fi

pkill -f -- "-m atlaskit_gateway" || true
while pgrep -f -- "-m atlaskit_gateway" >/dev/null; do sleep 1; done   # uvicorn drains before exiting
if [ "${GATEWAY_ONLY:-0}" = 1 ] && curl -sf http://127.0.0.1:8000/v1/models >/dev/null; then
  echo "GATEWAY_ONLY=1: keeping the running vLLM, restarting the gateway only."
else
  pkill -f -- "vllm.entrypoints.openai.api_server" || true
  while pgrep -f -- "vllm.entrypoints.openai.api_server" >/dev/null; do sleep 1; done
  nohup "$PY" -m vllm.entrypoints.openai.api_server \
    --model NCAIR1/N-ATLaS --served-model-name NCAIR1/N-ATLaS \
    --host 127.0.0.1 --port 8000 --dtype bfloat16 \
    --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL" "${LORA_ARGS[@]}" \
    > "$LOG_DIR/vllm.log" 2>&1 &
  echo "vLLM starting (log: $LOG_DIR/vllm.log)${ATLAS_ADAPTERS:+ with adapters: $ATLAS_ADAPTERS}…"
  wait_for http://127.0.0.1:8000/v1/models $! "$LOG_DIR/vllm.log"
  echo "vLLM ready."
fi

export ATLAS_UPSTREAM_URL=http://127.0.0.1:8000/v1
nohup "$PY" -m atlaskit_gateway > "$LOG_DIR/gateway.log" 2>&1 &
wait_for "http://127.0.0.1:$PORT/health" $! "$LOG_DIR/gateway.log"
echo "AtlasKit gateway ready on :$PORT"
echo "Public URL: https://${RUNPOD_POD_ID:-<pod-id>}-$PORT.proxy.runpod.net"
