"""Fine-tune N-ATLaS with LoRA.

Trains a LoRA adapter on chat examples in the OpenAI fine-tuning format
(``{"messages": [...]}``, one JSON object per line). Each example is rendered
with N-ATLaS's own chat template, including the ``date_string`` the template
expects, exactly as vLLM renders the gateway's requests. The loss covers only
the final assistant reply.

    python finetune/train_lora.py --data finetune/data/flores-dev.jsonl --out /workspace/adapters/natlas-mt

Needs a GPU with about 24 GB free (bf16 base weights plus LoRA) and
``pip install peft``. Serve the adapter with vLLM's LoRA support; see
finetune/README.md.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway"))
from atlaskit_gateway.core import DEFAULT_MODEL_ID, today_string  # noqa: E402

IGNORE_INDEX = -100
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_conversations(path: str) -> list[list[dict]]:
    convs = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            messages = json.loads(line).get("messages")
            if not messages or messages[-1].get("role") != "assistant":
                raise ValueError(f"{path}:{n}: each example needs `messages` ending with an assistant reply")
            convs.append(messages)
    return convs


def encode(tokenizer, messages: list[dict], date_string: str, max_len: int) -> dict | None:
    """Token ids and labels for one conversation; only the last assistant reply is trained on.

    The prompt is tokenized on its own, as vLLM does at inference time, and the
    reply (which ends with the template's end-of-turn token) is appended.
    Returns None when the prompt alone exceeds ``max_len``.
    """
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True,
                                           date_string=date_string)
    full = tokenizer.apply_chat_template(messages, tokenize=False, date_string=date_string)
    if not full.startswith(prompt):
        raise ValueError("chat template output is not prefix-stable, so the prompt cannot be masked")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    reply_ids = tokenizer(full[len(prompt):], add_special_tokens=False)["input_ids"]
    if len(prompt_ids) >= max_len:
        return None
    return {"input_ids": (prompt_ids + reply_ids)[:max_len],
            "labels": ([IGNORE_INDEX] * len(prompt_ids) + reply_ids)[:max_len]}


def collate(examples: list[dict], pad_id: int) -> dict:
    import torch

    width = max(len(e["input_ids"]) for e in examples)
    ids = torch.full((len(examples), width), pad_id)
    labels = torch.full((len(examples), width), IGNORE_INDEX)
    mask = torch.zeros((len(examples), width), dtype=torch.long)
    for row, e in enumerate(examples):
        n = len(e["input_ids"])
        ids[row, :n] = torch.tensor(e["input_ids"])
        labels[row, :n] = torch.tensor(e["labels"])
        mask[row, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="JSONL of {\"messages\": [...]} examples")
    ap.add_argument("--out", required=True, help="Directory for the adapter")
    ap.add_argument("--base", default=DEFAULT_MODEL_ID)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--date-string", default=today_string(), help="Value for the template's date_string")
    args = ap.parse_args(argv)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    convs = load_conversations(args.data)
    examples = [e for e in (encode(tokenizer, m, args.date_string, args.max_len) for m in convs) if e]
    print(f"{len(examples)} training examples ({len(convs) - len(examples)} skipped as too long)", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             target_modules=TARGET_MODULES, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_steps = math.ceil(len(examples) / args.batch_size) * args.epochs // args.grad_accum
    if total_steps < 1:
        raise SystemExit("Not enough examples for one optimizer step; lower --batch-size or --grad-accum")
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, total_steps // 20), total_steps)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    rng = random.Random(args.seed)
    model.train()
    step, micro, losses, started = 0, 0, [], time.time()
    for epoch in range(args.epochs):
        rng.shuffle(examples)
        for i in range(0, len(examples), args.batch_size):
            batch = collate(examples[i:i + args.batch_size], pad_id)
            loss = model(**{k: v.to(model.device) for k, v in batch.items()}).loss
            (loss / args.grad_accum).backward()
            losses.append(loss.item())
            micro += 1
            if micro % args.grad_accum or step >= total_steps:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == total_steps:
                recent = losses[-2 * args.grad_accum * 10:]
                print(f"epoch {epoch + 1} step {step}/{total_steps} loss {sum(recent) / len(recent):.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e} {time.time() - started:.0f}s", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    recent = losses[-2 * args.grad_accum * 10:]
    meta = {"base_model": args.base, "data": args.data, "examples": len(examples), "epochs": args.epochs,
            "steps": step, "lr": args.lr, "rank": args.rank, "batch_size": args.batch_size,
            "grad_accum": args.grad_accum, "first_loss": round(losses[0], 4),
            "final_loss": round(sum(recent) / len(recent), 4),
            "minutes": round((time.time() - started) / 60, 1), "date_string": args.date_string}
    (out / "atlaskit_training.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved adapter to {out}: {meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
