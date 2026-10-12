# Evaluation harness

`run_eval.py` benchmarks one or more AtlasKit gateways and writes `eval/reports/report-*.{json,md}`.

* `data/smoke.jsonl` — 14-item smoke test (language ID + factual QA in all four languages). Checks that a deployment works end to end; it is **not** a quality benchmark.
* `--flores yo,ha,ig` — English→Yoruba/Hausa/Igbo translation on [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus) devtest (human-translated, 1,012 sentences per language), scored with corpus chrF. Needs `pip install datasets` and a Hugging Face token that has accepted the FLORES+ terms.
* Pass `--endpoint` more than once to compare base N-ATLaS with a fine-tuned N-ATLaS side by side.

```bash
python eval/run_eval.py --data eval/data/smoke.jsonl --endpoint base=http://localhost:8080
python eval/run_eval.py --flores yo,ha,ig --limit 200 --endpoint base=$ATLASKIT_BASE_URL --api-key $ATLASKIT_API_KEY
```
