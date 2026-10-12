# Fine-tuning N-ATLaS with LoRA

AtlasKit's fine-tuning kit takes you from a dataset to a fine-tuned N-ATLaS served behind the same API and scored against the base model. It has four steps:

1. **Data.** Use JSONL in the OpenAI fine-tuning format (`{"messages": [...]}`), so existing datasets work as they are. `prepare_flores.py` builds a translation set from FLORES+.
2. **Train.** `train_lora.py` trains a LoRA adapter on top of `NCAIR1/N-ATLaS`. Each example is rendered with N-ATLaS's own chat template, including the `date_string` it expects, exactly as the gateway renders requests at inference time. The loss covers only the assistant's reply.
3. **Serve.** vLLM loads the adapter next to the base weights, and the gateway exposes it by name. Any other model is still refused.
4. **Evaluate.** `eval/run_eval.py` compares the adapter with base N-ATLaS on held-out data.

## 1. Prepare data

```bash
# English → Yoruba/Hausa/Igbo pairs from the FLORES+ dev split (997 sentences each).
# The evaluation uses the separate devtest split, so test sentences are never trained on.
python finetune/prepare_flores.py --langs yo,ha,ig --split dev --out finetune/data/flores-dev.jsonl
```

For your own task, write one conversation per line. Only the last assistant message is trained on:

```json
{"messages": [{"role": "system", "content": "You are a helpful agronomist."}, {"role": "user", "content": "Kí ni mo lè ṣe sí kòkòrò àgbàdo?"}, {"role": "assistant", "content": "..."}]}
```

## 2. Train

This step needs a GPU with about 24 GB free, plus `pip install peft` in the environment that `deploy/runpod/setup.sh` creates.

```bash
python finetune/train_lora.py --data finetune/data/flores-dev.jsonl --out /workspace/adapters/natlas-mt
```

The defaults are LoRA rank 16 on all attention and MLP projections, 2 epochs, learning rate 2e-4 and an effective batch of 16. Run with `--help` for the options. The adapter folder also holds `atlaskit_training.json`, which records the data, settings, loss and run time.

Train on a different GPU (or pod) from the one serving users: vLLM reserves most of the memory on a serving GPU.

## 3. Serve

```bash
ATLAS_LORA_MODULES="natlas-mt=/workspace/adapters/natlas-mt" bash deploy/runpod/start.sh
```

vLLM starts with `--enable-lora`, and the gateway accepts `natlas-mt` as a model name. A LoRA adapter can only run on top of the base model vLLM has loaded, so everything the gateway serves is still N-ATLaS.

```python
atlas.translate("Wash your hands before eating.", "yo", "en", model="natlas-mt")
atlas.chat("...", model="natlas-mt")
client.chat.completions.create(model="natlas-mt", messages=[...])   # OpenAI SDK
```

## 4. Evaluate

```bash
python eval/run_eval.py --flores yo,ha,ig --limit 100 \
  --endpoint base=$URL --endpoint tuned=$URL#natlas-mt --api-key $KEY
```

The report puts base and fine-tuned corpus chrF side by side for each language.

## Results

Results are added here after each run, with the exact command and the report file.
