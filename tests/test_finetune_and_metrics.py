"""Fine-tuning data encoding and eval metrics, without a GPU or model download."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "finetune"), str(ROOT / "eval")]

from prepare_flores import to_example  # noqa: E402
from run_eval import asr_errors, edit_distance, normalise_transcript  # noqa: E402
from train_lora import IGNORE_INDEX, encode  # noqa: E402


class StubTokenizer:
    """Character-level stand-in for a Llama-3 style chat template."""

    def __init__(self, stable=True):
        self.stable = stable

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, date_string=None):
        out = f"<bos>[{date_string}]" + "".join(f"<{m['role']}>{m['content']}<eot>" for m in messages[:-1] or [])
        last = messages[-1]
        if last["role"] == "assistant" and not add_generation_prompt:
            return out + f"<assistant>{last['content']}<eot>" if self.stable else "!" + out
        return out + f"<{last['role']}>{last['content']}<eot>" + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def conversation():
    return to_example({"input": "Good morning", "target": "yo", "source": "en", "reference": "Ẹ kú àárọ̀"})["messages"]


def test_to_example_uses_gateway_translation_prompt():
    msgs = conversation()
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert "from Nigerian English to Yoruba" in msgs[1]["content"] and msgs[2]["content"] == "Ẹ kú àárọ̀"


def test_encode_trains_only_on_the_reply():
    e = encode(StubTokenizer(), conversation(), "12 Oct 2026", 4096)
    trained = "".join(chr(t) for t, l in zip(e["input_ids"], e["labels"]) if l != IGNORE_INDEX)
    assert trained == "Ẹ kú àárọ̀<eot>"
    prompt = "".join(chr(t) for t, l in zip(e["input_ids"], e["labels"]) if l == IGNORE_INDEX)
    assert prompt.startswith("<bos>[12 Oct 2026]") and prompt.endswith("<assistant>")


def test_encode_truncates_and_rejects():
    msgs = conversation()
    full = encode(StubTokenizer(), msgs, "d", 4096)
    prompt_len = full["labels"].count(IGNORE_INDEX)
    short = encode(StubTokenizer(), msgs, "d", prompt_len + 3)
    assert len(short["input_ids"]) == prompt_len + 3 and short["labels"][-3:] == full["labels"][prompt_len:prompt_len + 3]
    assert encode(StubTokenizer(), msgs, "d", prompt_len) is None
    with pytest.raises(ValueError):
        encode(StubTokenizer(stable=False), msgs, "d", 4096)


def test_asr_error_rates():
    assert edit_distance("kitten".split(), "sitting".split()) == 1
    assert edit_distance("kitten", "sitting") == 3
    assert normalise_transcript("Ẹ KÚ, àárọ̀!") == normalise_transcript("ẹ kú àárọ̀")
    assert normalise_transcript("Ẹ kú àárọ̀", keep_tones=False) == "e ku aaro"
    errs = asr_errors("e ku aaro", "Ẹ kú àárọ̀.")
    assert errs["words"] == 3 and errs["word_errors"] == 3 and errs["bare_word_errors"] == 0


def test_corpus_chrf_and_paired_bootstrap():
    from run_eval import chrf, chrf_stats, corpus_chrf, paired_bootstrap
    refs = ["Ẹ kú àárọ̀", "Fọ ọwọ́ rẹ kí o tó jẹun", "Owó ọjà ti gòkè"]
    good = list(refs)
    bad = ["E ku aaro", "Wẹ ọwọ́", "Owo oja"]
    assert corpus_chrf(list(zip(good, refs))) == 100.0
    assert corpus_chrf([(good[0], refs[0])]) == chrf(good[0], refs[0])
    a = [chrf_stats(h, r) for h, r in zip(bad, refs)]
    b = [chrf_stats(h, r) for h, r in zip(good, refs)]
    res = paired_bootstrap(a, b, resamples=200)
    assert res["delta"] > 0 and res["ci95"][0] > 0 and res["b_better_share"] == 1.0
    assert paired_bootstrap(a, a, resamples=50)["delta"] == 0
