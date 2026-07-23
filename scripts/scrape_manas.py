#!/usr/bin/env python3
"""Scrape a page range of the Manas epic from eposmanas.ru into one .txt file.

The reader paginates each page number to a random URL; we start at the given
page and follow the "вперед" (next) link forward. Verse lines live in
<div class="book_content"> as <P> elements. The site is Windows-1251 encoded.

Excluded (per request): the bold <h1>/<h2> titles, and "*"/"***" symbols.

Usage:
    python scripts/scrape_manas.py \
        --start-url https://eposmanas.ru/manas_kg/-446/-450/23780696.html \
        --start 50 --end 200 --out data/raw_src/manas/manas.txt
"""

from __future__ import annotations

import argparse
import html as H
import os
import re
import ssl
import time
import urllib.request

HOST = "https://eposmanas.ru"
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 Chrome/120 Safari/537.36"})
    return urllib.request.urlopen(req, timeout=40, context=_CTX).read().decode("cp1251", "replace")


def page_number(raw: str) -> int | None:
    m = re.search(r"Стр\.\s*(\d+)", raw)
    return int(m.group(1)) if m else None


def next_url(raw: str) -> str | None:
    m = re.search(r'<a href="([^"]+)"[^>]*>\s*вперед', raw)
    return HOST + m.group(1) if m else None


def clean_page(raw: str) -> list[str]:
    """Verse lines from book_content, minus bold titles and asterisk symbols."""
    m = re.search(r'<div class="book_content"[^>]*>(.*?)</div>', raw, re.S)
    if not m:
        return []
    block = m.group(1)
    # Drop bold titles (the <h1>/<h2> chapter heading, and any <b>/<strong>).
    block = re.sub(r"<h[1-6][^>]*>.*?</h[1-6]>", "", block, flags=re.S | re.I)
    block = re.sub(r"<(b|strong)\b[^>]*>.*?</\1>", "", block, flags=re.S | re.I)
    # Each <P> / <br> is a verse line.
    block = re.sub(r"</p\s*>", "\n", block, flags=re.I)
    block = re.sub(r"<br\s*/?>", "\n", block, flags=re.I)
    block = re.sub(r"<[^>]+>", "", block)
    block = H.unescape(block)

    lines = []
    for ln in block.split("\n"):
        ln = ln.replace("*", "").strip()   # strip "*"/"***" symbols
        if ln and any(c.isalpha() for c in ln):
            lines.append(ln)
    return lines


def scrape(start_url: str, start: int, end: int, out: str, delay: float) -> None:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    url = start_url
    total_pages = end - start + 1
    collected: list[str] = []
    pages_done = 0
    last_pg = None

    while url and pages_done < total_pages:
        raw = get(url)
        pg = page_number(raw)
        lines = clean_page(raw)
        if collected:
            collected.append("")  # blank line between pages
        collected.extend(lines)
        pages_done += 1
        last_pg = pg
        if pages_done == 1 or pages_done % 25 == 0 or pages_done == total_pages:
            print(f"  page {pg} ({pages_done}/{total_pages})  +{len(lines)} lines")
        url = next_url(raw)
        time.sleep(delay)

    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(collected) + "\n")
    n_lines = sum(1 for x in collected if x)
    print(f"\nDone. pages {start}..{last_pg} -> {out}  ({n_lines} verse lines)")
    if last_pg != end:
        print(f"  NOTE: stopped at page {last_pg} (asked for {end}); next link ran out.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scrape Manas epic page range")
    ap.add_argument("--start-url", required=True, help="URL of the first page (--start)")
    ap.add_argument("--start", type=int, default=50)
    ap.add_argument("--end", type=int, default=200)
    ap.add_argument("--out", default="data/raw_src/manas/manas.txt")
    ap.add_argument("--delay", type=float, default=0.2)
    args = ap.parse_args(argv)
    scrape(args.start_url, args.start, args.end, args.out, args.delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
