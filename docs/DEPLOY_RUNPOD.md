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
| `403` with body `error code: 1010` | RunPod's Cloudflare proxy blocks Python's default `Python-urllib` user agent. The AtlasKit SDKs, the OpenAI SDK, `requests`, curl and browsers are unaffected. In raw `urllib` code, set a `User-Agent` header |
| Added tester keys or updated code | `GATEWAY_ONLY=1 bash deploy/runpod/start.sh` restarts only the gateway in a few seconds; N-ATLaS stays loaded |
| Two pods share one network volume | Give each its own logs with `ATLAS_LOG_DIR=/root/logs bash deploy/runpod/start.sh`, so usage stats are not mixed |

## Scale to zero (RunPod Serverless)

An always-on pod bills every hour. For a demo or a low-traffic app, run the same stack on a RunPod Serverless **load-balancing** endpoint instead: workers start on demand and stop after an idle period.

1. **Prepare a network volume.** Run `setup.sh` once on a pod that has the volume attached (steps 1–2 above), so that `hf-cache/` holds N-ATLaS and the ASR models. If you use a fine-tuned adapter, put it under `adapters/` on the same volume.
2. **Image.** `.github/workflows/serverless-image.yml` builds `deploy/runpod/serverless/Dockerfile` (vLLM 0.31.0, CUDA 13.0) and publishes `ghcr.io/lawalshalom/atlaskit-serverless:<commit>`. The entrypoint starts the gateway first, answering `/ping` with 204 while N-ATLaS loads and 200 when it is ready. It reads weights, adapters, compile caches and the usage log from the volume at `/runpod-volume`.
3. **Template** (serverless). Use that image, expose `8080/http`, and set these environment variables:
   - `ATLAS_API_KEYS`
   - `PORT=8080`
   - `ATLAS_LORA_MODULES=natlas-mt=/runpod-volume/adapters/natlas-mt` (optional)
   - `GPU_UTIL=0.80`
4. **Endpoint.** In the console, choose **Endpoint Type → Load Balancer**. Through the API, use GraphQL `saveEndpoint(input: {..., type: "LB"})`; the REST API cannot set the type. Attach the volume, choose 24–48 GB GPUs, and use these settings:
   - **Workers:** min 0, max 1
   - **Idle timeout:** 600 s
   - **FlashBoot:** on
   - **Minimum CUDA version:** 13.0
5. **Public access.** RunPod requires its own API key on every request, and that key must never reach a browser. Deploy `deploy/cloudflare/worker.js` as a Cloudflare Worker with these bindings:
   - `BACKEND=https://<endpoint-id>.api.runpod.ai`
   - `PLAYGROUND=<your playground URL>`
   - a secret `RUNPOD_KEY`, holding a RunPod key restricted to this endpoint

   The Worker forwards each caller's AtlasKit key as `X-API-Key`.
6. **Point the playground at it.** Put the Worker URL in `playground/hosted-api-url.txt`. The GitHub Pages workflow republishes the playground.

**Cold starts.** Measured on an RTX 4090, where the host already had the image:

| Stage | Time |
|---|---|
| Engine set-up | 34 s |
| Loading weights from the volume | 46 s |
| One-time `torch.compile` (cached on the volume afterwards) | 34 s |
| GPU graph capture | 31 s |

That totals about 3 minutes the first time and about 2–2.5 minutes after that. A machine that has never pulled the 10 GB image adds about 5–10 minutes. RunPod drops requests that wait more than 2 minutes for a worker, so clients should retry. The hosted playground waits for up to 6 minutes and shows a notice.

**Cost.** You pay only for the time workers run, plus volume storage (about $0.07 per GB per month).
