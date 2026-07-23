"""Phase 2 ingestion (§5.2): raw sources -> RawPoem JSONL.

RawPoem schema:
    {"id","source","title"|null,"author"|null,"text","license","meta":{}}
One record per poem; line breaks in ``text`` are preserved.

This CLI ingests a directory of UTF-8 ``.txt`` files, one poem per file (the
filename stem becomes the title unless a source adapter says otherwise). Add
source-specific adapters as new corpora are located via
github.com/alexeyev/awesome-kyrgyz-nlp.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os

from ..jsonl import write_jsonl


def make_id(source: str, text: str) -> str:
    h = hashlib.sha1(f"{source}\n{text}".encode("utf-8")).hexdigest()[:16]
    return f"{source}-{h}"


def raw_poem(text: str, source: str, license: str,
             title: str | None = None, author: str | None = None,
             meta: dict | None = None) -> dict:
    text = text.strip("\n")
    return {
        "id": make_id(source, text),
        "source": source,
        "title": title,
        "author": author,
        "text": text,
        "license": license,
        "meta": meta or {},
    }


def ingest_dir(directory: str, source: str, license: str):
    """Yield one RawPoem per ``.txt`` file in ``directory``."""
    for path in sorted(glob.glob(os.path.join(directory, "*.txt"))):
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        if not text.strip():
            continue
        title = os.path.splitext(os.path.basename(path))[0]
        yield raw_poem(text, source=source, license=license, title=title)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ingest raw poems -> RawPoem JSONL")
    ap.add_argument("--input-dir", required=True, help="dir of .txt poems")
    ap.add_argument("--source", required=True, help="source id (for provenance)")
    ap.add_argument("--license", required=True, help="license tag for this source")
    ap.add_argument("--out", default="data/raw/poems.jsonl")
    args = ap.parse_args(argv)

    n = write_jsonl(args.out, ingest_dir(args.input_dir, args.source, args.license))
    print(f"Wrote {n} RawPoem records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
