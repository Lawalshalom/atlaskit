# Benchmark results, 12 October 2026

All numbers come from live runs of `eval/run_eval.py` against an AtlasKit gateway serving `NCAIR1/N-ATLaS` through vLLM. The raw per-item outputs are in the JSON files next to this README.

**Setup:**
- **Hardware:** one NVIDIA RTX PRO 4500 Blackwell GPU (32 GB) on RunPod.
- **Software:** vLLM 0.31.0, transformers 5.17.0, torch 2.13.0 (CUDA 13.0), peft 0.21.2.
- **Generation settings:** the gateway defaults taken from the N-ATLaS model card: temperature 0.1, repetition penalty 1.12. Translations use the `/v1/atlas/translate` prompt.

## 1. Translation: base N-ATLaS vs the AtlasKit LoRA fine-tune (FLORES+)

This uses the full FLORES+ **devtest** split: 1,012 human-translated sentences for each language pair. Scores are corpus chrF (higher is better). The adapter was trained only on the separate **dev** split, so no test sentence was seen in training.

| Direction | Base N-ATLaS | Fine-tuned (`natlas-mt`) | Δ chrF | 95% interval (paired bootstrap) | Fine-tune better in |
|---|---|---|---|---|---|
| English → Hausa | 48.89 | 50.10 | +1.21 | [+0.68, +1.75] | 100.0% of 1,000 resamples |
| English → Igbo | 40.45 | 41.06 | +0.61 | [+0.12, +1.11] | 99.6% of 1,000 resamples |
| English → Yoruba | 26.55 | 27.75 | +1.20 | [+0.60, +1.82] | 100.0% of 1,000 resamples |

Neither model had any errors across the 6,072 requests. Median latency was 1.33 s for base and 1.45 s for the adapter, with 16 concurrent requests.

The gains are modest but statistically significant in all three directions. That fits a small adapter trained for 14 minutes on 2,991 sentence pairs. Yoruba is clearly the hardest of the three directions for N-ATLaS, which makes it the best candidate for a larger fine-tuning dataset.

**Reproduce:**

```bash
python finetune/prepare_flores.py --langs yo,ha,ig --split dev --out finetune/data/flores-dev.jsonl
python finetune/train_lora.py --data finetune/data/flores-dev.jsonl --out /workspace/adapters/natlas-mt
ATLAS_LORA_MODULES="natlas-mt=/workspace/adapters/natlas-mt" bash deploy/runpod/start.sh
python eval/run_eval.py --flores yo,ha,ig --limit 1012 --endpoint base=$URL --endpoint "tuned=$URL#natlas-mt" \
  --api-key $KEY --workers 16
```

**Training run:**
- **Data and duration:** 2,991 examples, 2 epochs, 374 steps, 14.1 minutes on the same GPU type.
- **LoRA settings:** rank 16 and alpha 32 on all attention and MLP projections, which is 41.9M trainable parameters (0.52% of N-ATLaS).
- **Loss:** fell from 2.01 to 0.94.
- **Files:** `train-natlas-mt.log`, `atlaskit_training.json` and `adapter_config.json`.

Generation runs at temperature 0.1, so rerunning can shift individual scores by a small amount.

## 2. Speech recognition: the official N-ATLaS ASR models (FLEURS)

This uses [FLEURS](https://huggingface.co/datasets/google/fleurs) test recordings (read speech with human transcripts). It takes the first 100 sentences per language, with one recording per sentence, and sends each through `/v1/audio/transcriptions`, which routes to `NCAIR1/{Yoruba,Hausa,Igbo}-ASR`. Scores are corpus error rates (lower is better), measured after lowercasing and removing punctuation.

| Language | Model | WER | CER | WER with diacritics ignored |
|---|---|---|---|---|
| Hausa | NCAIR1/Hausa-ASR | 33.7% | 15.4% | 33.7% |
| Igbo | NCAIR1/Igbo-ASR | 36.9% | 15.1% | 32.2% |
| Yoruba | NCAIR1/Yoruba-ASR | 57.5% | 22.4% | 30.2% |

All 300 requests succeeded, with a median latency of 3.7 s per clip. For Yoruba, almost half of the word errors are tone marks or under-dots alone: WER drops from 57.5% to 30.2% when diacritics are ignored. FLEURS is out of domain for these models, so treat these numbers as a baseline, not a ceiling.

**Reproduce:**

```bash
python eval/run_eval.py --fleurs yo,ha,ig --limit 100 --endpoint base=$URL --api-key $KEY
```

## 3. Smoke test

The 14 items in `eval/data/smoke.jsonl` cover language ID and factual QA in all four languages. Both base and fine-tuned scored 100% on every group, with 0 errors. This test checks that a deployment works end to end; it is not a quality benchmark.
