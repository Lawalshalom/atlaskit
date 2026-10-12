# Deploying on RunPod (about 30 minutes)

## 0. Before you start (Hugging Face)

1. Sign in at huggingface.co and open each of these pages. Click **Agree and access repository** on each:
   - https://huggingface.co/NCAIR1/N-ATLaS
   - https://huggingface.co/NCAIR1/Yoruba-ASR
   - https://huggingface.co/NCAIR1/Hausa-ASR
   - https://huggingface.co/NCAIR1/Igbo-ASR
   - https://huggingface.co/NCAIR1/NigerianAccentedEnglish
2. Create a **read** token under Settings → Access Tokens.

## 1. Create the pod

- **GPU:** any 24 GB+ card. RTX 4090, RTX A5000, L4 and A40 all work, and A40/L40S give more headroom. N-ATLaS in bf16 is about 16 GB, and the ASR models add about 1 GB per language once used.
- **Template:** *RunPod PyTorch 2.x* (CUDA 12.x).
- **Volume disk:** 60 GB mounted at `/workspace`, so model weights survive restarts.
- **Expose HTTP port:** `8080`. If you forget, use **Edit Pod → Expose HTTP Ports** later. That restarts the pod, but everything on `/workspace` is kept.
- **Environment variables:**
  - `HF_TOKEN` = your Hugging Face token
  - `ATLAS_API_KEYS` = comma-separated keys, one per tester. Generate them with `python -c "import secrets;print(secrets.token_urlsafe(24))"`.

  Instead of pod environment variables, you can put the same `KEY=value` lines in a `.env` file at the repo root. `setup.sh` and `start.sh` both read it.

## 2. Install and start (pod web terminal)

```bash
cd /workspace
git clone https://github.com/Lawalshalom/atlaskit.git atlaskit
cd atlaskit
bash deploy/runpod/setup.sh             # ~10–15 min the first time (downloads ~20 GB)
bash deploy/runpod/start.sh             # ~2–3 min for vLLM to load N-ATLaS
```

Your public endpoint is `https://<POD_ID>-8080.proxy.runpod.net`. Opening it in a browser shows the playground.

## 3. Smoke test

```bash
URL=https://<POD_ID>-8080.proxy.runpod.net KEY=<one of your keys>
curl -s $URL/health
curl -s $URL/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"n-atlas","atlas_language":"yo","messages":[{"role":"user","content":"Kí ni olú ìlú Nàìjíríà?"}]}'
python eval/run_eval.py --data eval/data/smoke.jsonl --endpoint base=$URL --api-key $KEY
```

## 4. Restarting

The `/workspace` volume keeps the weights. RunPod clears the container disk whenever a pod stops, and that is where the Python environment lives (in `/opt/atlaskit-venv`, for fast imports). After a restart, run only `bash deploy/runpod/start.sh`: it notices that the environment is missing and reinstalls it first (a few minutes, with no model re-download). Re-running `start.sh` at any time stops the old processes and starts fresh ones, for example after adding tester keys.

Logs are in:

- `/workspace/vllm.log`
- `/workspace/gateway.log`
- `/workspace/atlaskit-logs/usage.jsonl`

## Troubleshooting

| Symptom | Fix |
|---|---|
| `401` / `403` while downloading | The HF account has not accepted the terms on that model page, or `HF_TOKEN` is not set |
| vLLM out of memory | `GPU_UTIL=0.75 MAX_LEN=4096 bash deploy/runpod/start.sh`, or use a larger GPU |
| ASR returns "Could not decode audio" | `apt-get install -y ffmpeg` (setup.sh does this) |
| `/health` returns `degraded` | vLLM is still loading; check `tail -f /workspace/vllm.log` |
