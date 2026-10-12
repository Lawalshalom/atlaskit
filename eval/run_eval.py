#!/usr/bin/env python3
"""AtlasKit evaluation harness for N-ATLaS.

Runs a JSONL benchmark against one or more AtlasKit gateways (for example base
N-ATLaS vs. a fine-tuned N-ATLaS) and writes a JSON + Markdown report.

Task types (one JSON object per line):
  {"task": "translate", "source": "en", "target": "yo", "input": "...", "reference": "..."}
  {"task": "lid",       "input": "...", "label": "ha"}
  {"task": "qa",        "input": "...", "answers": ["Abuja"], "language": "en"}
  {"task": "asr",       "audio": "path.wav", "language": "yo", "reference": "..."}

An endpoint is name=URL, optionally with #model to pick a fine-tuned N-ATLaS
adapter served by that gateway (see finetune/README.md).

Usage:
  python eval/run_eval.py --data eval/data/smoke.jsonl --endpoint base=http://localhost:8080
  python eval/run_eval.py --flores yo,ha,ig --limit 100 --endpoint base=URL --endpoint tuned=URL#natlas-mt
  python eval/run_eval.py --fleurs yo,ha,ig --limit 100 --endpoint base=URL
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdk" / "python"))
from atlaskit import AtlasKit, AtlasKitError  # noqa: E402

# ---------------------------------------------------------------------------
# chrF (character n-gram F-score, Popović 2015) — dependency-free implementation
# matching sacrebleu's defaults (char order 6, beta 2, whitespace removed).
# ---------------------------------------------------------------------------

def _ngrams(text: str, n: int) -> Counter:
    text = "".join(text.split())
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def chrf(hypothesis: str, reference: str, order: int = 6, beta: float = 2.0) -> float:
    precisions, recalls = [], []
    for n in range(1, order + 1):
        hyp, ref = _ngrams(hypothesis, n), _ngrams(reference, n)
        if not hyp or not ref:
            continue
        overlap = sum((hyp & ref).values())
        precisions.append(overlap / sum(hyp.values()))
        recalls.append(overlap / sum(ref.values()))
    if not precisions:
        return 0.0
    p, r = statistics.mean(precisions), statistics.mean(recalls)
    if p + r == 0:
        return 0.0
    b2 = beta ** 2
    return 100 * (1 + b2) * p * r / (b2 * p + r)


def chrf_stats(hypothesis: str, reference: str, order: int = 6) -> list[int]:
    """Per n-gram order: matches, hypothesis n-grams, reference n-grams. Summed over segments they give corpus chrF."""
    row = []
    for n in range(1, order + 1):
        h, r = _ngrams(hypothesis, n), _ngrams(reference, n)
        row += [sum((h & r).values()), sum(h.values()), sum(r.values())]
    return row


def chrf_from_stats(stats: list[int], beta: float = 2.0) -> float:
    precisions, recalls = [], []
    for k in range(0, len(stats), 3):
        overlap, hyp_total, ref_total = stats[k:k + 3]
        if hyp_total and ref_total:
            precisions.append(overlap / hyp_total)
            recalls.append(overlap / ref_total)
    if not precisions:
        return 0.0
    p, r = statistics.mean(precisions), statistics.mean(recalls)
    b2 = beta ** 2
    return 0.0 if p + r == 0 else 100 * (1 + b2) * p * r / (b2 * p + r)


def _column_sums(rows) -> list[int]:
    return [sum(col) for col in zip(*rows)]


def corpus_chrf(pairs: list[tuple[str, str]], order: int = 6, beta: float = 2.0) -> float:
    """Corpus-level chrF: n-gram statistics summed over all segments (as sacrebleu does)."""
    return chrf_from_stats(_column_sums(chrf_stats(h, r, order) for h, r in pairs), beta)


def paired_bootstrap(stats_a: list[list[int]], stats_b: list[list[int]], resamples: int = 1000,
                     seed: int = 0) -> dict:
    """Paired bootstrap resampling (Koehn, 2004) of corpus chrF for system B minus system A.

    Both systems are scored on the same resampled sentences each round; the
    result gives the observed difference, a 95% interval and how often B wins.
    """
    rng = random.Random(seed)
    n = len(stats_a)
    deltas = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(chrf_from_stats(_column_sums(stats_b[i] for i in idx))
                      - chrf_from_stats(_column_sums(stats_a[i] for i in idx)))
    deltas.sort()
    observed = chrf_from_stats(_column_sums(stats_b)) - chrf_from_stats(_column_sums(stats_a))
    return {"delta": round(observed, 2),
            "ci95": [round(deltas[int(0.025 * resamples)], 2), round(deltas[int(0.975 * resamples) - 1], 2)],
            "b_better_share": round(sum(d > 0 for d in deltas) / resamples, 3), "resamples": resamples}


def compare_translations(items: list[dict], results_a: list[dict], results_b: list[dict]) -> dict:
    """Paired bootstrap per translation direction, for two endpoints run on the same items."""
    groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for item, a, b in zip(items, results_a, results_b):
        if item["task"] == "translate" and "output" in a and "output" in b:
            ga, gb = groups[a["group"]]
            ga.append(chrf_stats(a["output"], item["reference"]))
            gb.append(chrf_stats(b["output"], item["reference"]))
    return {g: paired_bootstrap(sa, sb) for g, (sa, sb) in sorted(groups.items())}


def normalise(s: str) -> str:
    return " ".join(s.lower().split())


# ---------------------------------------------------------------------------
# Speech recognition: word and character error rates
# ---------------------------------------------------------------------------

def normalise_transcript(s: str, keep_tones: bool = True) -> str:
    """Lowercase, NFC, punctuation removed. With keep_tones=False, diacritics are stripped too."""
    s = unicodedata.normalize("NFD" if not keep_tones else "NFC", s.lower())
    if not keep_tones:
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = "".join(" " if unicodedata.category(c).startswith("P") else c for c in s)
    return " ".join(s.split())


def edit_distance(a: list | str, b: list | str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def asr_errors(hypothesis: str, reference: str) -> dict:
    """Edit counts for corpus WER/CER, plus WER with tone marks and other diacritics ignored."""
    hyp, ref = normalise_transcript(hypothesis), normalise_transcript(reference)
    hyp_bare, ref_bare = normalise_transcript(hypothesis, False), normalise_transcript(reference, False)
    return {"word_errors": edit_distance(hyp.split(), ref.split()), "words": len(ref.split()),
            "char_errors": edit_distance(hyp, ref), "chars": len(ref),
            "bare_word_errors": edit_distance(hyp_bare.split(), ref_bare.split())}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


FLORES_CODES = {"en": "eng_Latn", "yo": "yor_Latn", "ha": "hau_Latn", "ig": "ibo_Latn"}


def load_flores(langs: list[str], limit: int, split: str = "devtest") -> list[dict]:
    """English -> {yo,ha,ig} translation items from FLORES+ (requires `datasets` and HF access)."""
    from datasets import load_dataset

    def column(code: str) -> list[str]:
        ds = load_dataset("openlanguagedata/flores_plus", FLORES_CODES[code], split=split)
        return [row["text"] for row in ds]

    en = column("en")[:limit]
    items = []
    for lang in langs:
        refs = column(lang)[:limit]
        items += [{"task": "translate", "source": "en", "target": lang, "input": s, "reference": r}
                  for s, r in zip(en, refs)]
    return items


FLEURS_CODES = {"yo": "yo_ng", "ha": "ha_ng", "ig": "ig_ng"}


def load_fleurs(langs: list[str], limit: int, split: str = "test") -> list[dict]:
    """Speech items from FLEURS (read speech with human transcripts), one recording per sentence."""
    import tarfile

    from huggingface_hub import hf_hub_download

    cache = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "atlaskit-fleurs"
    items = []
    for lang in langs:
        code = FLEURS_CODES[lang]
        tsv = hf_hub_download("google/fleurs", f"data/{code}/{split}.tsv", repo_type="dataset")
        refs: dict[str, str] = {}
        seen: set[str] = set()
        with open(tsv, encoding="utf-8") as f:
            for line in f:
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 4 or cols[0] in seen:
                    continue
                seen.add(cols[0])
                refs[cols[1]] = cols[3]          # file name -> normalised transcription
                if len(refs) >= limit:
                    break
        out_dir = cache / f"{code}-{split}"
        if any(not (out_dir / name).exists() for name in refs):
            archive = hf_hub_download("google/fleurs", f"data/{code}/audio/{split}.tar.gz", repo_type="dataset")
            out_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive) as tar:
                for member in tar:
                    name = Path(member.name).name
                    if member.isfile() and name in refs:
                        (out_dir / name).write_bytes(tar.extractfile(member).read())
        items += [{"task": "asr", "language": lang, "audio": str(out_dir / name), "reference": ref}
                  for name, ref in refs.items()]
    return items


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_item(client: AtlasKit, item: dict, model: str | None = None) -> dict:
    t0 = time.perf_counter()
    task = item["task"]
    out: dict = {"task": task}
    try:
        if task == "translate":
            hyp = client.translate(item["input"], item["target"], item.get("source"), model=model)
            out.update(output=hyp, score=chrf(hyp, item["reference"]), group=f"{item.get('source', '?')}->{item['target']}")
        elif task == "lid":
            pred = client.detect_language(item["input"])["language"]
            out.update(output=pred, score=float(pred == item["label"]), group=f"lid:{item['label']}")
        elif task == "qa":
            params = {"model": model} if model else {}
            reply = client.chat(item["input"], language=item.get("language"), max_tokens=200, **params).text
            hit = any(normalise(a) in normalise(reply) for a in item["answers"])
            out.update(output=reply, score=float(hit), group=f"qa:{item.get('language') or 'auto'}")
        elif task == "asr":
            text = client.transcribe(item["audio"], item["language"])["text"]
            errs = asr_errors(text, item["reference"])
            out.update(output=text, score=errs["word_errors"] / max(errs["words"], 1), group=f"asr:{item['language']}", **errs)
        else:
            raise ValueError(f"unknown task {task}")
    except (AtlasKitError, ValueError, KeyError) as exc:
        out.update(error=str(exc), score=0.0, group=f"{task}:error")
    out["latency_s"] = round(time.perf_counter() - t0, 3)
    return out


def summarise(items: list[dict], results: list[dict]) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    asr: dict[str, Counter] = defaultdict(Counter)
    for item, res in zip(items, results):
        groups[res["group"]].append(res["score"])
        if item["task"] == "translate" and "output" in res:
            pairs[res["group"]].append((res["output"], item["reference"]))
        if item["task"] == "asr" and "words" in res:
            asr[res["group"]].update({k: res[k] for k in ("word_errors", "words", "char_errors", "chars",
                                                           "bare_word_errors")})
    summary = {}
    for g, scores in sorted(groups.items()):
        entry = {"n": len(scores), "mean": round(statistics.mean(scores), 4)}
        if g in pairs:
            entry["corpus_chrf"] = round(corpus_chrf(pairs[g]), 2)
        if g in asr:
            c = asr[g]
            entry.update(wer=round(100 * c["word_errors"] / max(c["words"], 1), 1),
                         cer=round(100 * c["char_errors"] / max(c["chars"], 1), 1),
                         wer_no_diacritics=round(100 * c["bare_word_errors"] / max(c["words"], 1), 1))
        summary[g] = entry
    lat = [r["latency_s"] for r in results]
    summary["_overall"] = {"n": len(results), "errors": sum(1 for r in results if "error" in r),
                           "median_latency_s": statistics.median(lat) if lat else None}
    return summary


def to_markdown(report: dict) -> str:
    names = list(report["endpoints"])
    groups = sorted({g for n in names for g in report["endpoints"][n]["summary"] if not g.startswith("_")})
    lines = [f"# N-ATLaS evaluation — {report['dataset']}", "",
             f"Run at {report['started']} · {report['n_items']} items", "",
             "| Group | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for g in groups:
        cells, n = [], "-"
        for name in names:
            e = report["endpoints"][name]["summary"].get(g)
            if not e:
                cells.append("-")
                continue
            n = e["n"]
            if "corpus_chrf" in e:
                cells.append(f"chrF {e['corpus_chrf']}")
            elif "wer" in e:
                cells.append(f"WER {e['wer']}% · CER {e['cer']}% · WER without diacritics {e['wer_no_diacritics']}%")
            else:
                cells.append(f"{e['mean'] * 100:.1f}%")
        lines.append(f"| {g} | {n} | " + " | ".join(cells) + " |")
    lines.append("")
    for name in names:
        ep = report["endpoints"][name]
        o = ep["summary"]["_overall"]
        lines.append(f"- **{name}** (`{ep['model']}`): {o['errors']} errors, median latency {o['median_latency_s']}s")
    for label, groups in report.get("comparisons", {}).items():
        if not groups:
            continue
        lines += ["", f"## {label}: paired bootstrap of corpus chrF", "",
                  "| Group | Δ chrF | 95% interval | resamples where the second system is better |", "|---|---|---|---|"]
        for g, c in groups.items():
            lines.append(f"| {g} | {c['delta']:+.2f} | [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] | "
                         f"{100 * c['b_better_share']:.1f}% of {c['resamples']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="JSONL benchmark file")
    ap.add_argument("--flores", help="Comma-separated target languages to pull from FLORES+ (e.g. yo,ha,ig)")
    ap.add_argument("--fleurs", help="Comma-separated languages for FLEURS speech recognition (yo,ha,ig)")
    ap.add_argument("--limit", type=int, default=100, help="Items per FLORES/FLEURS language")
    ap.add_argument("--endpoint", action="append", required=True,
                    help="name=URL[#model] of an AtlasKit gateway; repeat to compare models")
    ap.add_argument("--workers", type=int, default=4, help="Concurrent requests per endpoint")
    ap.add_argument("--api-key")
    ap.add_argument("--out", default="eval/reports")
    args = ap.parse_args(argv)

    items: list[dict] = []
    if args.data:
        items += load_jsonl(args.data)
    if args.flores:
        items += load_flores([l.strip() for l in args.flores.split(",")], args.limit)
    if args.fleurs:
        items += load_fleurs([l.strip() for l in args.fleurs.split(",")], args.limit)
    if not items:
        ap.error("provide --data, --flores and/or --fleurs")

    sources = [s for s in (args.data, args.flores and f"flores_plus:{args.flores}",
                           args.fleurs and f"fleurs:{args.fleurs}") if s]
    report = {"dataset": " + ".join(sources), "n_items": len(items),
              "started": time.strftime("%Y-%m-%d %H:%M:%S"), "endpoints": {}}
    for spec in args.endpoint:
        name, _, url = spec.partition("=")
        if not url:
            name, url = "default", name
        url, _, model = url.partition("#")
        client = AtlasKit(url, api_key=args.api_key)
        done = 0

        def run(item: dict, client=client, model=model or None, name=name) -> dict:
            nonlocal done
            res = run_item(client, item, model)
            done += 1
            print(f"\r[{name}] {done}/{len(items)}", end="", file=sys.stderr, flush=True)
            return res

        with ThreadPoolExecutor(max(1, args.workers)) as pool:
            results = list(pool.map(run, items))
        print(file=sys.stderr)
        report["endpoints"][name] = {"url": url, "model": model or "n-atlas",
                                     "summary": summarise(items, results), "results": results}

    names = list(report["endpoints"])
    if len(names) > 1:
        first = report["endpoints"][names[0]]["results"]
        report["comparisons"] = {f"{other} vs {names[0]}": compare_translations(
            items, first, report["endpoints"][other]["results"]) for other in names[1:]}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out / f"report-{stamp}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = to_markdown(report)
    (out / f"report-{stamp}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
