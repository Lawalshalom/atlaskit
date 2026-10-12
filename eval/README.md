# Evaluation harness

`run_eval.py` benchmarks one or more AtlasKit gateways and writes `eval/reports/report-*.{json,md}`.

* `data/smoke.jsonl` — 14-item smoke test (language ID + factual QA in all four languages). Checks that a deployment works end to end; it is **not** a quality benchmark.
* `--flores yo,ha,ig` — English→Yoruba/Hausa/Igbo translation on [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus) devtest (human-translated, 1,012 sentences per language), scored with corpus chrF. Needs `pip install datasets` and a Hugging Face token that has accepted the FLORES+ terms.
* `--fleurs yo,ha,ig` — speech recognition on [FLEURS](https://huggingface.co/datasets/google/fleurs) test recordings (read speech with human transcripts), one recording per sentence, sent through `/v1/audio/transcriptions`. It reports corpus WER, CER and WER with diacritics ignored, which shows how much of the error is tone marks.
* Pass `--endpoint` more than once to compare deployments side by side. `name=URL#model` selects a fine-tuned adapter served by that gateway (see `finetune/README.md`).
* `--workers` (default 4) sets how many requests run concurrently against each endpoint. vLLM batches them.

```bash
python eval/run_eval.py --data eval/data/smoke.jsonl --endpoint base=http://localhost:8080
python eval/run_eval.py --flores yo,ha,ig --limit 200 --endpoint base=$ATLASKIT_BASE_URL --api-key $ATLASKIT_API_KEY
python eval/run_eval.py --flores yo,ha,ig --endpoint base=$URL --endpoint tuned=$URL#natlas-mt --api-key $KEY
python eval/run_eval.py --fleurs yo,ha,ig --limit 100 --endpoint base=$URL --api-key $KEY
```
