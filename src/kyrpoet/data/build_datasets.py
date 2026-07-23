"""Phase 3 assembly (§6/§7): build CPT / SFT / DPO splits from artifacts.

* CPT: plain-LM text records from cleaned poems (+ general Kyrgyz elsewhere).
* SFT: deterministic train/val split of back-generated examples.
* DPO: a preference-pair helper used at Phase 4.3 (chosen = higher overall,
  rejected = lower, requiring a minimum score gap).
"""

from __future__ import annotations

import argparse
import hashlib

from ..jsonl import read_jsonl, write_jsonl


def _bucket(key: str) -> float:
    """Stable [0,1) hash bucket for deterministic, reproducible splitting."""
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split_examples(examples: list[dict], val_frac: float = 0.05):
    """Deterministically split by source_poem_id so a poem never straddles splits."""
    train, val = [], []
    for ex in examples:
        key = str(ex.get("source_poem_id") or ex.get("instruction", ""))
        (val if _bucket(key) < val_frac else train).append(ex)
    return train, val


def filter_by_lines(examples: list[dict], max_lines: int | None) -> list[dict]:
    """Drop SFTExamples whose poem exceeds ``max_lines`` (None = keep all).

    Very long poems make poor 7–8 syllable lyric targets and carry meaningless
    form specs (huge rhyme-scheme strings)."""
    if not max_lines:
        return list(examples)
    return [e for e in examples if e.get("form", {}).get("n_lines", 0) <= max_lines]


def poetry_repeats(poetry_chars: int, general_chars: int,
                   target_frac: float | None, max_repeats: int = 10) -> int:
    """How many times to repeat the poetry so it reaches ~``target_frac`` of the mix.

    Solves ``R*P / (R*P + G) = f`` for R. Capped at ``max_repeats`` — heavy
    repetition of a small corpus risks the model memorizing poems verbatim
    rather than learning the register.
    """
    if not target_frac or not poetry_chars or not general_chars:
        return 1
    if not 0 < target_frac < 1:
        raise ValueError("--poetry-frac must be between 0 and 1")
    r = round(target_frac * general_chars / ((1 - target_frac) * poetry_chars))
    return max(1, min(r, max_repeats))


def cpt_records(clean_poems):
    """Plain-LM records for continued pretraining."""
    for cp in clean_poems:
        yield {"text": cp["text"], "source_poem_id": cp.get("id", "")}


def make_preference_pair(prompt: str, cand_a: dict, cand_b: dict, min_gap: float = 0.15):
    """Build a DPO pair from two scored candidates, or None if the gap is too small.

    Each candidate: {"text": str, "overall": float}. Higher overall = chosen.
    """
    hi, lo = sorted((cand_a, cand_b), key=lambda c: c["overall"], reverse=True)
    if hi["overall"] - lo["overall"] < min_gap:
        return None
    return {"prompt": prompt, "chosen": hi["text"], "rejected": lo["text"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Assemble train datasets")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sft", help="split SFTExamples -> train/val")
    s.add_argument("--in", dest="inp", default="data/sft/examples.jsonl")
    s.add_argument("--train", default="data/sft/train.jsonl")
    s.add_argument("--val", default="data/sft/val.jsonl")
    s.add_argument("--val-frac", type=float, default=0.05)
    s.add_argument("--max-lines", type=int, default=None,
                   help="drop examples whose poem exceeds this many lines")

    c = sub.add_parser("cpt", help="cleaned poems (+ general prose) -> CPT text records")
    c.add_argument("--in", dest="inp", default="data/clean/poems.jsonl")
    c.add_argument("--out", default="data/cpt/train.jsonl")
    c.add_argument("--extra", nargs="*", default=[],
                   help="extra jsonl files with a 'text' field to merge in "
                        "(e.g. data/cpt/general_ky.jsonl from fetch_general_ky.py)")
    c.add_argument("--poetry-frac", type=float, default=None,
                   help="target share of the CPT mix that should be poetry "
                        "(e.g. 0.15); upsamples poetry to hit it")
    c.add_argument("--max-repeats", type=int, default=10,
                   help="cap on poetry upsampling (guards against memorization)")

    args = ap.parse_args(argv)
    if args.cmd == "sft":
        examples = list(read_jsonl(args.inp))
        kept = filter_by_lines(examples, args.max_lines)
        if len(kept) != len(examples):
            print(f"dropped {len(examples) - len(kept)} examples over {args.max_lines} lines")
        train, val = split_examples(kept, args.val_frac)
        write_jsonl(args.train, train)
        write_jsonl(args.val, val)
        print(f"SFT split: {len(train)} train / {len(val)} val")
    elif args.cmd == "cpt":
        poetry = list(cpt_records(read_jsonl(args.inp)))
        general = []
        for path in args.extra:
            recs = [{"text": r["text"], "source": r.get("source", path)}
                    for r in read_jsonl(path) if r.get("text", "").strip()]
            print(f"  + {len(recs)} records from {path}")
            general.extend(recs)

        p_chars = sum(len(r["text"]) for r in poetry)
        g_chars = sum(len(r["text"]) for r in general)
        repeats = poetry_repeats(p_chars, g_chars, args.poetry_frac, args.max_repeats)
        if repeats > 1:
            print(f"upsampling poetry {repeats}x toward {args.poetry_frac:.0%} of the mix")
            if repeats == args.max_repeats:
                print(f"  NOTE: hit --max-repeats {args.max_repeats}; target share not reached")

        records = poetry * repeats + general
        total = p_chars * repeats + g_chars
        frac = (p_chars * repeats) / total if total else 1.0
        n = write_jsonl(args.out, records)
        print(f"CPT: {n} text records -> {args.out} "
              f"({len(poetry) * repeats} poetry + {len(general)} general, "
              f"{total/1e6:.1f}M chars, poetry {frac:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
