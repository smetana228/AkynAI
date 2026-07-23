#!/usr/bin/env python3
"""Walk the scraped buckets into a single RawPoem JSONL (Phase 2 ingest).

Layout expected:
    data/raw_src/classic/<author>/<poem>.txt
    data/raw_src/modern/<artist>/<song>.txt
    data/raw_src/manas/<file>.txt

Each file becomes one RawPoem; author = its folder, title = filename stem,
source = bucket, and meta.bucket records the bucket so Manas can be
down-sampled at dataset-build time.
"""

from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from kyrpoet.data.ingest import raw_poem
from kyrpoet.jsonl import write_jsonl

RAW_ROOT = "data/raw_src"
OUT = "data/raw/poems.jsonl"

# (bucket folder, source tag, license tag)
BUCKETS = [
    ("classic", "vostoka-classic", "public-domain"),
    ("modern", "kyrgyz-songs", "web-unknown"),
    ("manas", "eposmanas", "public-domain"),
]


def records():
    for bucket, source, lic in BUCKETS:
        base = os.path.join(RAW_ROOT, bucket)
        for path in sorted(glob.glob(os.path.join(base, "**", "*.txt"), recursive=True)):
            text = open(path, encoding="utf-8").read()
            if not text.strip():
                continue
            rel = os.path.relpath(path, base)
            parts = rel.split(os.sep)
            author = parts[0] if len(parts) > 1 else None
            title = os.path.splitext(os.path.basename(path))[0]
            yield raw_poem(text, source=source, license=lic, title=title,
                           author=author, meta={"bucket": bucket})


def main() -> int:
    recs = list(records())
    n = write_jsonl(OUT, recs)
    by_bucket: dict[str, int] = {}
    for r in recs:
        by_bucket[r["meta"]["bucket"]] = by_bucket.get(r["meta"]["bucket"], 0) + 1
    print(f"Wrote {n} RawPoem records -> {OUT}")
    for b, c in by_bucket.items():
        print(f"  {b:<10} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
