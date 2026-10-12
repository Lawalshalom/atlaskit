#!/usr/bin/env bash
# Entrypoint for a RunPod Serverless load-balancing endpoint. One worker runs vLLM (N-ATLaS plus any
# LoRA adapters) and the AtlasKit gateway. The gateway starts first and answers RunPod's /ping probe
# with 204 while N-ATLaS loads, then 200, so traffic only reaches a ready worker. Weights, adapters,
# compile caches and the usage log live on the endpoint's network volume, mounted at /runpod-volume.
set -euo pipefail
VOL=${ATLAS_VOLUME:-/runpod-volume}
export HF_HOME=${HF_HOME:-$VOL/hf-cache}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$VOL/vllm-cache}   # reuse torch.compile results across cold starts
export ATLAS_LOG_PATH=${ATLAS_LOG_PATH:-$VOL/atlaskit-logs/usage.jsonl}
export ATLAS_UPSTREAM_URL=http://127.0.0.1:8000/v1
export PORT=${PORT:-8080}
: "${ATLAS_API_KEYS:?Set ATLAS_API_KEYS on the endpoint}"
mkdir -p "$(dirname "$ATLAS_LOG_PATH")" "$VLLM_CACHE_ROOT"

LORA_ARGS=()
if [ -n "${ATLAS_LORA_MODULES:-}" ]; then
  read -ra modules <<< "${ATLAS_LORA_MODULES//,/ }"
  LORA_ARGS=(--enable-lora --max-lora-rank "${MAX_LORA_RANK:-64}" --lora-modules "${modules[@]}")
  names=(); for m in "${modules[@]}"; do names+=("${m%%=*}"); done
  export ATLAS_ADAPTERS=$(IFS=,; echo "${names[*]}")
fi
EXTRA=()
if [ "${ENFORCE_EAGER:-0}" = 1 ]; then EXTRA+=(--enforce-eager); fi   # faster start, slower generation

python3 -m atlaskit_gateway &
python3 -m vllm.entrypoints.openai.api_server --model NCAIR1/N-ATLaS --served-model-name NCAIR1/N-ATLaS \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len "${MAX_LEN:-8192}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.80}" "${LORA_ARGS[@]}" "${EXTRA[@]}" &
wait -n   # if either process exits, stop the worker so RunPod replaces it
exit 1
