"""Build a translation fine-tuning set from FLORES+ (English -> Yoruba/Hausa/Igbo).

Writes chat examples in the OpenAI fine-tuning format, one JSON object per line:

    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

The prompts come from the gateway's own ``translation_messages()``, so the
adapter learns exactly the request that /v1/atlas/translate sends at inference
time. Train on the ``dev`` split: the evaluation harness scores ``devtest``.

    python finetune/prepare_flores.py --langs yo,ha,ig --split dev --out finetune/data/flores-dev.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gateway"), str(ROOT / "eval")]
from atlaskit_gateway.core import translation_messages  # noqa: E402


def to_example(item: dict) -> dict:
    """One FLORES+ translation item -> one chat example ending with the reference translation."""
    messages = translation_messages(item["input"], item["target"], item.get("source"))
    return {"messages": messages + [{"role": "assistant", "content": item["reference"]}]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default="yo,ha,ig", help="Comma-separated target languages")
    ap.add_argument("--split", default="dev", help="FLORES+ split (keep devtest for evaluation)")
    ap.add_argument("--limit", type=int, default=10_000, help="Max sentences per language")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from run_eval import load_flores

    items = load_flores([l.strip() for l in args.langs.split(",")], args.limit, split=args.split)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(to_example(item), ensure_ascii=False) + "\n")
    print(f"Wrote {len(items)} examples to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
