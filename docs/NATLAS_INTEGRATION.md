# N-ATLaS integration evidence

AtlasKit integrates directly with the official N-ATLaS release from NCAIR. No other foundation model is used anywhere in the stack.

## Models used

| Purpose | Model | Where it is wired in |
|---|---|---|
| Text generation (chat, translation, summarisation, voice-chat replies) | `NCAIR1/N-ATLaS` (Llama-3 8B fine-tune, bf16) | `deploy/runpod/start.sh` launches vLLM with `--model NCAIR1/N-ATLaS`; `core.py` → `DEFAULT_MODEL_ID` |
| Yoruba ASR | `NCAIR1/Yoruba-ASR` | `core.py` → `LANGUAGES["yo"]`, loaded in `asr.py` |
| Hausa ASR | `NCAIR1/Hausa-ASR` | `LANGUAGES["ha"]` |
| Igbo ASR | `NCAIR1/Igbo-ASR` | `LANGUAGES["ig"]` |
| Nigerian-accented English ASR | `NCAIR1/NigerianAccentedEnglish` | `LANGUAGES["en"]` |

## Faithfulness to the model card

The gateway reproduces the usage that NCAIR documents for N-ATLaS:

- **Chat template:** the Llama-3 template bundled with the N-ATLaS tokenizer is applied by vLLM. The `date_string` variable the template expects is supplied on every request (`chat_template_kwargs.date_string`, format `12 Oct 2026`).
- **Default system prompt:** the N-ATLaS card's system prompt is injected when the caller does not supply one.
- **Generation defaults:** `temperature=0.1`, `repetition_penalty=1.12` and `max_tokens=1000`, taken from the card's example. Callers can override any of them.
- **ASR input:** audio is decoded and resampled to 16 kHz mono, as the ASR cards recommend.

## Guard against substitution

`build_chat_request()` rejects any `model` value that is not N-ATLaS (HTTP 404, `model_not_found`). This is covered by `tests/test_core.py::test_rejects_other_models_and_bad_input` and `tests/test_integration.py::test_other_model_rejected`.

## How to verify on a live deployment

```bash
curl -s $URL/health                  # "model": "NCAIR1/N-ATLaS", "upstream": true
curl -s -H "Authorization: Bearer $KEY" $URL/v1/models
tail /workspace/vllm.log             # vLLM log shows the NCAIR1/N-ATLaS weights loading
nvidia-smi                           # GPU memory held by the vLLM process
```
