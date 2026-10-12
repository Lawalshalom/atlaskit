# Architecture

```
            ┌──────────────────────── GPU pod (RunPod, 24 GB+) ────────────────────────┐
            │                                                                          │
 apps ──►   │  AtlasKit gateway :8080 (Starlette/uvicorn)        vLLM :8000 (private)  │
 SDKs       │  ├─ auth (API keys)                                serving               │
 OpenAI SDK │  ├─ /v1/chat/completions ── normalise ───────────► NCAIR1/N-ATLaS (8B)    │
 playground │  │    • model allow-list (N-ATLaS only)             bf16, Llama-3 template │
 curl       │  │    • default system prompt + date_string                               │
            │  │    • atlas_language → reply-language instruction                       │
            │  │    • model-card generation defaults                                    │
            │  │    • SSE relay, special-token cleanup                                  │
            │  ├─ /v1/audio/transcriptions ─► NCAIR1/{Yoruba,Hausa,Igbo}-ASR,           │
            │  │                              NCAIR1/NigerianAccentedEnglish (Whisper)  │
            │  │                              lazy-loaded per language, 16 kHz decode   │
            │  ├─ /v1/atlas/voice-chat = ASR → N-ATLaS                                  │
            │  ├─ /v1/atlas/{translate,summarize} = task prompts → N-ATLaS              │
            │  ├─ /v1/atlas/detect-language  (dependency-free heuristic)                │
            │  ├─ usage log (JSONL, no prompt text) → /v1/atlas/stats                   │
            │  └─ / playground (static HTML)                                            │
            └──────────────────────────────────────────────────────────────────────────┘
```

## Design decisions

- **vLLM for inference.** It provides continuous batching and paged attention, so one 24 GB GPU can serve many concurrent users. The gateway talks to vLLM's OpenAI server, which keeps the inference engine swappable (for example TGI or SGLang) without changing the SDKs.
- **OpenAI wire format.** It is the de facto standard, so existing tools work against N-ATLaS unchanged. N-ATLaS-specific behaviour is added through optional fields (`atlas_language`) and `/v1/atlas/*` routes, never by breaking the standard ones.
- **N-ATLaS only.** `build_chat_request` rejects any model that is not N-ATLaS. This guarantees that apps built on AtlasKit really run on the sovereign model.
- **Framework-free core.** All request shaping lives in `gateway/atlaskit_gateway/core.py` (pure Python) and is unit-tested without a GPU. The HTTP layer is thin.
- **Lazy ASR.** Each Whisper-small checkpoint (~1 GB in fp16) loads on first use for its language. That is why vLLM is capped at 80% of GPU memory.
- **Zero-dependency SDKs.** The Python SDK uses only `urllib` and the JS SDK only `fetch`, so they install anywhere, including low-spec machines and serverless functions.
- **Privacy.** The usage log records endpoint, language, latency, token counts and a hashed client id. It never records prompts, completions or audio.

## Repository layout

```
gateway/atlaskit_gateway/   core.py (logic) · app.py (HTTP) · asr.py (Whisper) · usage.py (log/stats)
sdk/python/atlaskit/        client.py · cli.py
sdk/js/src/index.js         JS SDK          sdk/js/test/   node:test suite
playground/index.html       browser playground (served at /)
eval/                       run_eval.py · data/smoke.jsonl
deploy/runpod/              setup.sh · start.sh · Dockerfile
tests/                      pytest: core + end-to-end (mock vLLM) + JS SDK + eval harness
examples/dev_server.py      gateway against a mock model, no GPU
```
