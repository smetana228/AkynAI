#!/usr/bin/env python3
"""Fetch general Kyrgyz prose for continued pretraining (Phase 4.1 / spec §5.1).

CPT exists to raise base-language fluency, so it needs broad prose — the poetry
corpus alone is too small and too narrow. This pulls a HuggingFace text dataset
(Kyrgyz Wikipedia by default), normalizes it with the same cleaner used for the
poems, drops stubs/low-Cyrillic junk, and writes CPT text records.

Requires: pip install datasets

Usage:
    python scripts/fetch_general_ky.py                       # Kyrgyz Wikipedia
    python scripts/fetch_general_ky.py --max-docs 500        # quick sample
    python scripts/fetch_general_ky.py --dataset oscar-corpus/OSCAR-2301 \
        --config ky --license "CC0-ish (web crawl)" --out data/cpt/oscar_ky.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from kyrpoet.data.clean import normalize
from kyrpoet.jsonl import write_jsonl

CYR = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюяөүң")


def cyrillic_ratio(text: str) -> float:
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c in CYR) / len(letters)


def clean_doc(text: str, min_chars: int, min_cyrillic: float) -> str | None:
    """Normalize one document; return None if it's too short or not Kyrgyz.

    Unlike the poem cleaner this keeps long prose lines — prose is the point here.
    """
    text = normalize(text or "")
    # drop lines with no letters (nav junk, symbol rows)
    lines = [ln for ln in text.split("\n") if any(c.isalpha() for c in ln)]
    text = "\n".join(lines).strip()
    if len(text) < min_chars or cyrillic_ratio(text) < min_cyrillic:
        return None
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch general Kyrgyz prose for CPT")
    ap.add_argument("--dataset", default="wikimedia/wikipedia")
    ap.add_argument("--config", default="20231101.ky")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--out", default="data/cpt/general_ky.jsonl")
    ap.add_argument("--license", default="CC-BY-SA-4.0")
    ap.add_argument("--min-chars", type=int, default=200, help="drop stubs shorter than this")
    ap.add_argument("--min-cyrillic", type=float, default=0.8)
    ap.add_argument("--max-docs", type=int, default=None, help="cap for a quick sample")
    args = ap.parse_args(argv)

    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets is required:  pip install datasets", file=sys.stderr)
        return 1

    print(f"loading {args.dataset} [{args.config}] split={args.split} ...")
    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)

    records, seen, kept, chars = [], 0, 0, 0
    for row in ds:
        seen += 1
        text = clean_doc(row.get(args.text_field, ""), args.min_chars, args.min_cyrillic)
        if text:
            kept += 1
            chars += len(text)
            records.append({"text": text, "source": args.dataset, "license": args.license})
        if args.max_docs and kept >= args.max_docs:
            break
        if seen % 5000 == 0:
            print(f"  scanned {seen}, kept {kept} ({chars/1e6:.1f}M chars)")

    n = write_jsonl(args.out, records)
    print(f"\nWrote {n} documents -> {args.out}")
    print(f"  scanned {seen}, kept {kept}, dropped {seen - kept}")
    print(f"  total {chars/1e6:.1f}M characters (~{chars/2e6:.1f}M tokens, rough)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
